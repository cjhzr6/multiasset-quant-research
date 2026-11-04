"""Four full-period raw-price accounts, with explicit unresolved-event publication gates."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import numpy as np
import pandas as pd
from account_engine import Account

ROOT = Path(__file__).resolve().parents[1]


def prepare_events(calendar):
    gm = pd.read_parquet(ROOT / 'data/processed/corporate_actions_unverified_2020_2025.parquet')
    bs = pd.read_parquet(ROOT / 'data/processed/baostock_corporate_dates_2020_2025.parquet')
    lookup = {}
    for row in bs.to_dict('records'):
        day = pd.to_datetime(row.get('dividOperateDate'), errors='coerce')
        if pd.notna(day):
            key = (row['symbol'], day)
            lookup.setdefault(key, []).append(row)
    def select_date_record(candidates, cash=None):
        if not candidates:
            return {}, False
        if cash is not None:
            matching = [r for r in candidates if pd.notna(pd.to_numeric(r.get('dividCashPsBeforeTax'), errors='coerce')) and abs(float(r['dividCashPsBeforeTax'])-cash) <= .001]
            if matching:
                candidates = matching
        selected = max(candidates, key=lambda r: sum(bool(str(v)) for v in r.values()))
        ambiguous = any(len({r.get(f) for r in candidates if r.get(f)}) > 1 for f in ['dividRegistDate','dividPayDate','dividStockMarketDate'])
        return selected, ambiguous
    rows = []
    for row in gm.to_dict('records'):
        key = (row['symbol'], row['source_created_date'])
        extra, ambiguous = select_date_record(lookup.pop(key, []), float(row['cash_div']))
        ex_date = row['source_created_date']
        cash = float(row['cash_div'])
        bs_cash = pd.to_numeric(extra.get('dividCashPsBeforeTax'), errors='coerce')
        source_conflict = bool(np.isfinite(bs_cash) and abs(bs_cash - cash) > .001)
        rows.append(dict(symbol=row['symbol'], ex_date=ex_date,
                         record_date=pd.to_datetime(extra.get('dividRegistDate'), errors='coerce'),
                         pay_date=pd.to_datetime(extra.get('dividPayDate'), errors='coerce'),
                         share_date=pd.to_datetime(extra.get('dividStockMarketDate'), errors='coerce'),
                         cash_ps=cash, bonus_ratio=float(row['share_div_ratio']), transfer_ratio=float(row['share_trans_ratio']),
                         unresolved=not bool(extra) or ambiguous or source_conflict or row['allotment_ratio'] != 0,
                         date_source='BaoStock' if extra else 'UNVERIFIED_GM_CREATED_AT', source_conflict=source_conflict))
    # Additional source events supplement GM gaps, without replacing existing observations.
    for (symbol, day), candidates in lookup.items():
        row, ambiguous = select_date_record(candidates)
        if not pd.Timestamp('2020-01-01') <= day < pd.Timestamp('2026-01-01'):
            continue
        def number(field):
            value = pd.to_numeric(row.get(field), errors='coerce')
            return float(value) if pd.notna(value) else 0.
        rows.append(dict(symbol=symbol, ex_date=day,
                         record_date=pd.to_datetime(row.get('dividRegistDate'), errors='coerce'),
                         pay_date=pd.to_datetime(row.get('dividPayDate'), errors='coerce'),
                         share_date=pd.to_datetime(row.get('dividStockMarketDate'), errors='coerce'),
                         cash_ps=number('dividCashPsBeforeTax'), bonus_ratio=number('dividStocksPs'),
                         transfer_ratio=number('dividReserveToStockPs'), unresolved=ambiguous,
                         date_source='BaoStock_SUPPLEMENT', source_conflict=False))
    events = pd.DataFrame(rows)
    override_path=ROOT/'results/corporate_dates/gap_overrides.json'
    if override_path.exists():
        overrides=json.loads(override_path.read_text(encoding='utf-8'))
        for correction in overrides['overrides']:
            mask=events.symbol.eq(correction['symbol']) & events.ex_date.eq(pd.Timestamp(correction['ex_date']))
            for field,value in correction['changes'].items():
                if field in ('record_date','pay_date','share_date'):
                    value=pd.Timestamp(value)
                events.loc[mask,field]=value
    confirmation_path = ROOT/'results/corporate_dates/issuer_confirmations.json'
    if confirmation_path.exists():
        confirmations = json.loads(confirmation_path.read_text(encoding='utf-8'))
        for confirmation in confirmations['confirmations']:
            source_file = ROOT/confirmation['source_file']
            assert hashlib.sha256(source_file.read_bytes()).hexdigest() == confirmation['sha256']
            mask = events.symbol.eq(confirmation['symbol']) & events.ex_date.eq(pd.Timestamp(confirmation['ex_date']))
            assert mask.sum() == 1
            assert abs(float(events.loc[mask, 'cash_ps'].iloc[0])-confirmation['cash_ps']) < 1e-6
            assert events.loc[mask, ['record_date','pay_date']].notna().all().all()
            events.loc[mask, 'source_conflict'] = False
            events.loc[mask, 'unresolved'] = False
            events.loc[mask, 'date_source'] = 'Issuer combined-dividend confirmation + verified dates'
    factor = pd.read_csv(ROOT / 'results/account_inputs_v1/factor_change_audit.csv', parse_dates=['trade_date'])
    keys = set(zip(events.symbol, events.ex_date))
    rights = json.loads((ROOT / 'results/corporate_dates/rights_reconciliation.json').read_text(encoding='utf-8'))
    keys.update((row['symbol'], pd.Timestamp(row['ex_date'])) for row in rights['rights'])
    unsupported = factor.loc[[ (r.symbol, r.trade_date) not in keys for r in factor.itertuples() ]]
    return events, unsupported


def make_rankings():
    panel = pd.read_parquet(ROOT / 'data/processed/training_panel_v1.parquet')
    panel = panel.loc[panel.signal_date.between('2020-01-01', '2025-12-31') & panel.prediction_ready].copy()
    panel['symbol'] = panel.symbol.astype(str)
    columns = ['ret60', 'ret120', 'momentum20_skip5', 'ret5', 'vol20', 'max_return20', 'log_amount20', 'illiquidity20']
    ranks = panel.groupby('signal_date', observed=True)[columns].rank(pct=True)
    panel['B01'] = (ranks[['ret60', 'ret120', 'momentum20_skip5']].mean(axis=1)
                    + (1 - ranks.ret5) + (2 - ranks.vol20 - ranks.max_return20) / 2
                    + (ranks.log_amount20 + 1 - ranks.illiquidity20) / 2) / 4
    rankings = {name: {} for name in ('B00', 'B01', 'E00', 'E10')}
    for day, group in panel.groupby('signal_date', observed=True):
        rankings['B00'][day] = sorted(group.symbol.tolist())
        rankings['B01'][day] = group.sort_values(['B01', 'symbol'], ascending=[False, True]).symbol.tolist()
    prediction = pd.read_parquet(ROOT / 'results/rolling_signal_v1/predictions_2020_2025.parquet', columns=['signal_date', 'symbol', 'model', 'score'])
    prediction['symbol'] = prediction.symbol.astype(str)
    for (name, day), group in prediction.groupby(['model', 'signal_date'], observed=True):
        rankings[name][day] = group.sort_values(['score', 'symbol'], ascending=[False, True]).symbol.tolist()
        assert set(rankings[name][day]) == set(rankings['B00'][day])
    return rankings


def performance(daily, trades):
    result = {}
    for label, lower, upper in [('development', '2020-01-01', '2024-01-01'), ('validation', '2024-01-01', '2026-01-01')]:
        rows = daily.loc[daily.date.ge(lower) & daily.date.lt(upper)].copy()
        if rows.empty:
            continue
        previous = daily.loc[daily.date.lt(lower), 'nav']
        initial = previous.iloc[-1] if len(previous) else 1_000_000.
        nav = np.r_[initial, rows.nav.to_numpy()]
        returns = nav[1:] / nav[:-1] - 1
        annual = (nav[-1] / initial) ** (252 / len(rows)) - 1
        volatility = returns.std(ddof=1) * np.sqrt(252)
        drawdown = nav / np.maximum.accumulate(nav) - 1
        selected = trades.loc[trades.date.ge(lower) & trades.date.lt(upper)] if len(trades) else trades
        result[label] = dict(total_return=float(nav[-1] / initial - 1), annualized_return=float(annual),
                             volatility=float(volatility), sharpe_zero_cash=float(returns.mean() * 252 / volatility) if volatility else None,
                             max_drawdown=float(drawdown.min()), mean_holdings=float(rows.holdings.mean()),
                             mean_cash_fraction=float((rows.cash / rows.nav).mean()),
                             trading_fees=float(selected.fee.sum()) if len(selected) else 0.,
                             realized_dividend_tax=float(selected.dividend_tax.sum()) if len(selected) else 0.,
                             slippage_cost=float(selected.slippage_cost.sum()) if len(selected) else 0.,
                             annualized_two_sided_turnover=float(selected.notional.sum() / rows.nav.mean() * 252 / len(rows)) if len(selected) else 0.)
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--cost-multiplier', type=float, default=1.)
    parser.add_argument('--slippage', type=float, default=.0005)
    parser.add_argument('--output-tag', default='account_backtest_v1')
    parser.add_argument('--start-date', choices=['2020-01-01','2024-01-01'], default='2020-01-01')
    args=parser.parse_args()
    if not re.fullmatch('[a-z0-9_]+',args.output_tag) or args.cost_multiplier<=0 or not 0<=args.slippage<=.01:
        raise ValueError('Invalid scenario specification')
    out = ROOT / 'results' / args.output_tag
    out.mkdir(parents=True, exist_ok=True)
    source = ROOT / 'data/processed/account_market_2020_2025.parquet'
    market = pd.read_parquet(source)
    market = market.loc[market.trade_date.ge(args.start_date)]
    calendar = pd.DatetimeIndex(sorted(market.trade_date.unique()))
    events, unsupported = prepare_events(calendar)
    events.to_parquet(out / 'reconciled_corporate_events.parquet', index=False)
    unsupported.to_csv(out / 'unsupported_factor_changes.csv', index=False)
    ranks = make_rankings()
    config = dict(capital=1000000, slippage_oneway=args.slippage, commission_rate=.0003, minimum_commission=5,
                  portfolio_count=80, retained_rank=120, cadence_sessions=5,
                  models=['B00', 'B01', 'E00', 'E10'], benchmark_count='all eligible current members',
                  period='2020-2025; 2026 not loaded', dividend_policy='gross receivable on ex-date; payment at verified date close; FIFO disposal tax, pending tax in NAV',
                  missing_date_policy='receivable/new shares frozen with publication failure; never invent credit date',
                  execution='next open, no fills at upper/lower limit, no future daily volume; board quantities, T+1, cash, 3% buy cap',
                  performance_gate='Unresolved held events or unexplained held-price gaps => diagnostic only, not accepted investment returns')
    config['rights_policy']='Do not subscribe to verified rights issues; retain actual dilution losses, never create free shares.'
    config['cost_multiplier']=args.cost_multiplier
    config['start_date']=args.start_date
    (out / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    accounts = {name: Account(slippage=args.slippage, commission_multiplier=args.cost_multiplier) for name in config['models']}
    mergers = [json.loads(path.read_text(encoding='utf-8')) for path in sorted((ROOT/'results/corporate_dates').glob('merger_*_2025.json'))]
    assert all(merger['content_verified'] for merger in mergers)
    config['merger_sources']=mergers
    (out / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    by_event = {day: group.to_dict('records') for day, group in events.groupby('ex_date')}
    by_unknown = {day: group.symbol.tolist() for day, group in unsupported.groupby('trade_date')}
    # Independent cash-flow reconciliation every session.
    for i, (day, frame) in enumerate(market.groupby('trade_date', sort=True), 1):
        quotes = frame.set_index('symbol').to_dict('index')
        next_day = calendar[i] if i < len(calendar) else day + pd.Timedelta(days=10)
        previous_day = calendar[i-2] if i > 1 else pd.Timestamp('2019-12-31')
        for name, account in accounts.items():
            before_cash, before_trades = account.cash, len(account.trades)
            for merger in mergers:
                if day == pd.Timestamp(merger['conversion_date']):
                    account.merger(merger, day, quotes)
            for symbol in by_unknown.get(day, []):
                if account.quantity(symbol):
                    account.issue(day, symbol, 'UNSUPPORTED_FACTOR_CHANGE_WHILE_HELD')
            for event in by_event.get(day, []):
                account.corporate_action(event, day, previous_day)
            account.execute(day, next_day, quotes)
            account.close(day, quotes)
            new_trades = account.trades[before_trades:]
            expected_cash = before_cash + account.daily[-1]['cash_paid']
            expected_cash += sum((t['notional'] if t['side'] == 'SELL' else -t['notional']) - t['fee'] - t['dividend_tax'] for t in new_trades)
            assert abs(account.cash - expected_cash) < 1e-5
            if (i - 1) % 5 == 0 and i < len(calendar):
                account.plan(day, ranks[name][day], quotes, benchmark=name=='B00')
        if i % 200 == 0:
            progress = dict(status='RUNNING', completed_sessions=i, expected_sessions=len(calendar), date=str(day.date()))
            (out / 'progress.json').write_text(json.dumps(progress), encoding='utf-8')
            print(f'ACCOUNTS sessions={i}/{len(calendar)} day={day.date()}', flush=True)
    report = dict(status='COMPLETE_ACCOUNT_SIMULATION_WITH_PUBLICATION_GATES', models={}, sealed_2026_evaluated=False,
                  market_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), cash_reconciliation='PASS every account/session',
                  corporate_source_sha256=hashlib.sha256((ROOT/'data/processed/baostock_corporate_dates_2020_2025.parquet').read_bytes()).hexdigest(),
                  config_sha256=hashlib.sha256((out/'config.json').read_bytes()).hexdigest(),
                  reconciliation_source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'results/corporate_dates').glob('*.json') if p.name != 'manifest.json'},
                  source_event_count=len(events), unsupported_factor_change_count=len(unsupported),
                  limitations=['Daily OHLC cannot establish order queue priority or partial fills.',
                               'Bonus-share fractions floored at account level; exact registrar rounding not independently verified.',
                               'Dividend tax assumes domestic individual FIFO lots; tax/credit chronology is modelled, not broker statement reconciliation.',
                               'Month-only historical pool and unavailable industry classifications remain limitations.'])
    for name, account in accounts.items():
        daily, trades = pd.DataFrame(account.daily), pd.DataFrame(account.trades)
        issues = pd.DataFrame(account.issues, columns=['date', 'symbol', 'reason'])
        daily.to_csv(out / f'{name}_daily.csv', index=False)
        trades.to_csv(out / f'{name}_trades.csv', index=False)
        issues.to_csv(out / f'{name}_issues.csv', index=False)
        pd.DataFrame(account.rejections).to_csv(out / f'{name}_rejections.csv', index=False)
        pd.DataFrame(account.events).to_csv(out / f'{name}_corporate_ledger.csv', index=False)
        pd.DataFrame([dict(symbol=s, quantity=l.quantity, acquired=l.acquired, available=l.available, dividend_base=l.dividend_base)
                      for s, lots in account.lots.items() for l in lots]).to_csv(out / f'{name}_closing_lots.csv', index=False)
        accepted = len(issues) == 0
        report['models'][name] = dict(status='ENGINEERING_PASS_WITH_ASSUMPTIONS' if accepted else 'DIAGNOSTIC_ONLY_UNRESOLVED_ACCOUNT_EVENTS',
                                     publishable_returns=accepted, sessions=len(daily), trades=len(trades),
                                     issue_count=len(issues), issues_by_reason=issues.reason.value_counts().to_dict(),
                                     diagnostic_performance=performance(daily, trades))
    (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    (out / 'progress.json').write_text(json.dumps(dict(status='COMPLETE', completed_sessions=len(calendar), expected_sessions=len(calendar))), encoding='utf-8')
    print(json.dumps({name: dict(issues=r['issue_count'], trades=r['trades'], status=r['status']) for name,r in report['models'].items()}), flush=True)


if __name__ == '__main__':
    main()
