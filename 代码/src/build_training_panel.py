"""Causal monthly-universe panel. No downloads, account returns or trades."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from build_price_features import FEATURES, transform, self_check

ROOT = Path(__file__).resolve().parents[1]
STATES = ['market_ret20', 'market_ma60_distance', 'market_vol20',
          'pool_breadth_ma20', 'pool_dispersion1']


def read_manifest(name):
    return json.loads((ROOT / 'results/data_collection' / name).read_text(encoding='utf-8'))


def dates(values):
    values = pd.to_datetime(values)
    if values.dt.tz is not None:
        values = values.dt.tz_convert('Asia/Shanghai').dt.tz_localize(None)
    return pd.DatetimeIndex(values.dt.normalize())


def effective_pool(calendar, snapshots):
    """The close-dated snapshot is usable strictly after its snapshot date."""
    out = {}
    for i, (day, members) in enumerate(sorted(snapshots.items())):
        begin = calendar.searchsorted(day, side='right')
        end = calendar.searchsorted(sorted(snapshots)[i + 1], side='right') if i + 1 < len(snapshots) else len(calendar)
        for session in calendar[begin:end]:
            out[session] = members
    return out


def market_states(panel, benchmark):
    """Breadth denominator includes members with observable current and MA20 prices.

    A suspension remains represented if today's prices are available. Missing bars
    are not filled, and coverage is reported separately from the market features.
    """
    groups = panel.groupby('signal_date', observed=True)
    breadth = groups.ma20_distance.agg(lambda x: x.dropna().gt(0).mean())
    state = pd.DataFrame(index=benchmark.index)
    state['market_ret20'] = benchmark.close.pct_change(20, fill_method=None)
    state['market_ma60_distance'] = benchmark.close / benchmark.close.rolling(60).mean() - 1
    state['market_vol20'] = benchmark.close.pct_change(fill_method=None).rolling(20).std()
    state['pool_breadth_ma20'] = breadth
    state['pool_dispersion1'] = groups.ret1.std()
    state['breadth_observed'] = groups.ma20_distance.count()
    state['return_observed'] = groups.ret1.count()
    state['pool_size'] = groups.size()
    return state


def main():
    self_check()
    full = read_manifest('full_manifest.json')
    membership = read_manifest('membership_manifest.json')
    records = {(r['job']['kind'], r['job'].get('symbol')): r for r in full['records'] if r['status'] == 'PASS'}
    index_record = records.get(('bars', 'SHSE.000300')) or next(r for r in membership['records'] if r['job'] == {'kind': 'bars', 'symbol': 'SHSE.000300'})
    benchmark = pd.read_parquet(ROOT / index_record['file'])
    benchmark.index = dates(benchmark.eob)
    benchmark = benchmark.sort_index()
    assert benchmark.index.is_unique and benchmark.close.gt(0).all()
    snapshots = {}
    for record in membership['records']:
        if record['job']['kind'] != 'membership':
            continue
        data = pd.read_parquet(ROOT / record['file'])
        assert len(data) == data.symbol.nunique() == 300
        snapshots[pd.Timestamp(record['job']['date'])] = set(data.symbol)
    pool = effective_pool(benchmark.index, snapshots)
    tables = []
    for n, symbol in enumerate(sorted(full['symbols']), 1):
        bars = pd.read_parquet(ROOT / records[('bars', symbol)]['file'])
        metrics = pd.read_parquet(ROOT / records[('instruments', symbol)]['file'])
        bars.index = dates(bars.eob)
        metrics.index = dates(metrics.trade_date)
        assert bars.index.is_unique and metrics.index.is_unique
        aligned = bars.reindex(benchmark.index).join(metrics[['adj_factor', 'is_suspended']], validate='one_to_one')
        output = transform(aligned, benchmark)
        mask = [symbol in pool.get(day, set()) for day in benchmark.index]
        output = output.loc[mask].copy()
        output['symbol'] = symbol
        output.index.name = 'signal_date'
        tables.append(output.reset_index())
        if n % 100 == 0:
            print(f'PANEL symbols={n}/{len(full["symbols"])}', flush=True)
    panel = pd.concat(tables, ignore_index=True).sort_values(['signal_date', 'symbol']).reset_index(drop=True)
    assert not panel.duplicated(['signal_date', 'symbol']).any()
    assert panel.groupby('signal_date').size().eq(300).all()
    state = market_states(panel, benchmark)
    panel = panel.merge(state[STATES], left_on='signal_date', right_index=True, validate='many_to_one')
    panel['prediction_ready'] = panel.feature_ready & panel[STATES].notna().all(axis=1)
    # label_ready deliberately absent from prediction eligibility.
    for name in FEATURES + STATES + ['label_5d_adjusted']:
        panel[name] = panel[name].astype('float32')
    panel['symbol'] = panel.symbol.astype('category')
    target = ROOT / 'data/processed/training_panel_v1.parquet'
    target.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(target, index=False)
    coverage = state.dropna(subset=['pool_size'])
    coverage.reset_index(names='signal_date').to_parquet(ROOT / 'data/processed/state_coverage_v1.parquet', index=False)
    report = dict(status='PASS', file=str(target.relative_to(ROOT)), sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                  rows=len(panel), symbols=int(panel.symbol.nunique()), sessions=int(panel.signal_date.nunique()),
                  feature_ready_rows=int(panel.feature_ready.sum()), prediction_ready_rows=int(panel.prediction_ready.sum()),
                  prediction_ready_without_label=int((panel.prediction_ready & ~panel.label_ready).sum()),
                  features=FEATURES, states=STATES, first_signal=str(panel.signal_date.min().date()), last_signal=str(panel.signal_date.max().date()),
                  breadth_coverage_min=float((coverage.breadth_observed / coverage.pool_size).min()),
                  breadth_coverage_median=float((coverage.breadth_observed / coverage.pool_size).median()),
                  calendar_source='SHSE.000300 daily bars; aligned sessions, no stock-row compression',
                  universe_rule='latest snapshot dated strictly before signal date; 300 per day',
                  label='t+1 open to t+6 open, adjusted; endpoints available and unsuspended',
                  limitation='Labels are research price returns; no fees, dividend cash ledger or executable account. 2026 included for data integrity only.')
    (ROOT / 'results/training_panel_v1.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'PANEL_PASS rows={len(panel)} prediction_ready={report["prediction_ready_rows"]}', flush=True)


if __name__ == '__main__':
    main()
