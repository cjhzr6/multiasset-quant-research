"""Prepare and audit raw-account inputs. No synthetic cash dates or strategy P&L."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from build_training_panel import dates

ROOT = Path(__file__).resolve().parents[1]


def fee_components(amount, side, day, commission_rate=.0003, minimum=5.):
    """Commission is an explicit assumption including exchange/regulatory fees.

    Transfer and stamp tax are separate; dividend withholding is not trade fee.
    Rates apply to the 2020-2025 research interval, not all historical regimes.
    """
    day = pd.Timestamp(day)
    if amount <= 0 or side not in ('BUY', 'SELL') or not pd.Timestamp('2020-01-01') <= day < pd.Timestamp('2026-01-01'):
        raise ValueError('Unsupported amount, side or historical fee regime')
    commission = max(minimum, amount * commission_rate)
    transfer = amount * (.00001 if day >= pd.Timestamp('2022-04-29') else .00002)
    stamp = amount * (.0005 if day >= pd.Timestamp('2023-08-28') else .001) if side == 'SELL' else 0.
    return {'commission_including_exchange_regulatory': commission, 'transfer': transfer,
            'stamp': stamp, 'total': commission + transfer + stamp}


def buy_quantity(symbol, budget, price):
    """2020-25 ordinary mainland boards only; STAR minimum 200, increment 1."""
    if not np.isfinite(price) or price <= 0 or not np.isfinite(budget) or budget < 0:
        raise ValueError('Invalid order inputs')
    if symbol.startswith('SHSE.688'):
        quantity = int(np.floor(budget / price))
        return quantity if quantity >= 200 else 0
    # Specific verified code, not a guess that every 302-prefixed issue is ordinary.
    if symbol == 'SZSE.302132':
        return int(np.floor(budget / price / 100)) * 100
    if symbol.startswith(('SHSE.600', 'SHSE.601', 'SHSE.603', 'SHSE.605', 'SZSE.000', 'SZSE.001', 'SZSE.002', 'SZSE.003', 'SZSE.300', 'SZSE.301')):
        return int(np.floor(budget / price / 100)) * 100
    raise ValueError('Unverified security quantity rules')


def ex_reference(previous_close, cash, bonus, transfer, rights, rights_price):
    return (previous_close - cash + rights * rights_price) / (1 + bonus + transfer + rights)


def main():
    full = json.loads((ROOT / 'results/data_collection/full_manifest.json').read_text(encoding='utf-8'))
    dividend = json.loads((ROOT / 'results/gm_supplement/dividend_legacy_manifest.json').read_text(encoding='utf-8'))
    records = {(r['job']['kind'], r['job'].get('symbol')): r for r in full['records'] if r['status'] == 'PASS'}
    dividend_records = {r['symbol']: r for r in dividend['records']}
    benchmark = pd.read_parquet(ROOT / records[('bars', 'SHSE.000300')]['file'])
    calendar = dates(benchmark.eob).sort_values()
    execution_tables, event_tables, changes = [], [], []
    unknown_symbols, empty = [], []
    for n, symbol in enumerate(sorted(full['symbols']), 1):
        bars = pd.read_parquet(ROOT / records[('bars', symbol)]['file'])
        metrics = pd.read_parquet(ROOT / records[('instruments', symbol)]['file'])
        bars.index = dates(bars.eob)
        metrics.index = dates(metrics.trade_date)
        frame = bars.reindex(calendar).join(metrics[['pre_close', 'upper_limit', 'lower_limit', 'adj_factor', 'is_suspended', 'board', 'trade_n']], validate='one_to_one')
        frame['previous_raw_close'] = frame.close.shift(1)
        frame['factor_ratio'] = frame.adj_factor / frame.adj_factor.shift(1)
        keep = frame.index.to_series().between('2020-01-01', '2025-12-31')
        frame = frame.loc[keep].copy()
        frame['symbol'] = symbol
        frame.index.name = 'trade_date'
        try:
            buy_quantity(symbol, 10000, 10)
        except ValueError:
            unknown_symbols.append(symbol)
        execution_tables.append(frame.drop(columns=['eob'], errors='ignore').reset_index())
        record = dividend_records[symbol]
        if record['status'] == 'EMPTY':
            empty.append(symbol)
            events = pd.DataFrame()
        else:
            events = pd.read_parquet(ROOT / record['file'])
            events['source_created_date'] = dates(events.created_at)
            events = events.loc[events.source_created_date.between('2020-01-01', '2025-12-31')].copy()
        if len(events):
            events = events.merge(frame[['previous_raw_close', 'pre_close', 'factor_ratio']], left_on='source_created_date', right_index=True, how='left', validate='many_to_one')
            values = ['cash_div', 'share_div_ratio', 'share_trans_ratio', 'allotment_ratio', 'allotment_price']
            events['all_components_observed'] = events[values].notna().all(axis=1)
            events['theoretical_ex_reference'] = ex_reference(events.previous_raw_close, events.cash_div, events.share_div_ratio, events.share_trans_ratio, events.allotment_ratio, events.allotment_price)
            events['reference_abs_error'] = (events.pre_close - events.theoretical_ex_reference).abs()
            events['reference_match_011'] = events.all_components_observed & events.reference_abs_error.le(.011)
            events['factor_changed'] = events.factor_ratio.sub(1).abs().gt(1e-6)
            events['cash_payment_date'] = pd.NaT
            events['cash_payment_verified'] = False
            event_tables.append(events)
        change = frame.loc[frame.factor_ratio.sub(1).abs().gt(1e-6), ['symbol', 'factor_ratio']].reset_index()
        event_dates = set(events.source_created_date) if len(events) else set()
        change['has_source_event_on_date'] = change.trade_date.isin(event_dates)
        changes.append(change)
        if n % 150 == 0:
            print(f'ACCOUNT_INPUT_AUDIT symbols={n}/637', flush=True)
    out = ROOT / 'results/account_inputs_v1'
    out.mkdir(parents=True, exist_ok=True)
    execution = pd.concat(execution_tables, ignore_index=True)
    events = pd.concat(event_tables, ignore_index=True)
    factor_changes = pd.concat(changes, ignore_index=True)
    execution.to_parquet(ROOT / 'data/processed/account_market_2020_2025.parquet', index=False)
    events.to_parquet(ROOT / 'data/processed/corporate_actions_unverified_2020_2025.parquet', index=False)
    factor_changes.to_csv(out / 'factor_change_audit.csv', index=False)
    events[['symbol', 'source_created_date', 'cash_div', 'share_div_ratio', 'share_trans_ratio', 'allotment_ratio', 'reference_abs_error', 'reference_match_011', 'factor_changed']].to_csv(out / 'corporate_reference_audit.csv', index=False)
    observable = events.reference_abs_error.notna() & events.all_components_observed
    report = dict(status='ACCOUNT_INPUTS_PREPARED_WITH_GAPS', tradable_backtest_ready=False,
                  market_rows=len(execution), symbols=637, period='2020-2025 only; 2026 untouched',
                  corporate_event_rows=len(events), observable_reference_events=int(observable.sum()),
                  reference_matches=int(events.reference_match_011.sum()), reference_match_rate=float(events.loc[observable, 'reference_match_011'].mean()),
                  source_events_without_factor_change=int((~events.factor_changed).sum()),
                  factor_changes=len(factor_changes), factor_changes_without_event=int((~factor_changes.has_source_event_on_date).sum()),
                  rights_events=int(events.allotment_ratio.gt(0).sum()), source_date_duplicate_rows=int(events.duplicated(['symbol', 'source_created_date'], keep=False).sum()),
                  unknown_quantity_rules=unknown_symbols, dividend_empty_symbols=empty,
                  suspended_rows=int(execution.is_suspended.eq(1).sum()),
                  unsuspended_missing_open=int((execution.is_suspended.eq(0) & execution.open.isna()).sum()),
                  blockers=['created_at is documented as creation time; alignment evidence does not prove cash payment date.',
                            'No verified cash-payment/bonus-credit dates, dividend withholding or rights-subscription ledger.',
                            'Reference mismatches and factor changes lacking events must be explained before P&L publication.'],
                  sources=['https://emquant.18.cn/help/doc/python/python_select_api_old.html',
                           'https://www.mof.gov.cn/jrttts/202308/t20230828_3904235.htm',
                           'https://www.sse.com.cn/onething/gptz/',
                           'https://www.xcsc.com/main/a/20220429/1022871784.shtml',
                           'https://edu.sse.com.cn/tib/qa/c/4866268.shtml',
                           'https://www.szse.cn/marketServices/message/index/dynamic/t20250603_613877.html'])
    (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    policy = dict(period='2020-2025', commission_assumption=.0003, commission_min_assumption=5,
                  commission_includes=['exchange_handling', 'regulatory'], separate=['transfer', 'sell_stamp'],
                  slippage_oneway_scenarios=[.0005, .001, .002],
                  initial_capital=1000000, target_count=80, hold_rank_limit=120, rebalance_exchange_sessions=5,
                  corporate_cash_mode='BLOCK until payment dates/withholding verified; no invented cash credit',
                  status='PREPARED_NOT_AN_EXECUTED_ACCOUNT')
    (out / 'policy.json').write_text(json.dumps(policy, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=True), flush=True)


if __name__ == '__main__':
    main()
