"""Replay restored major stock strategies from frozen scores; never train or download."""
import argparse,sys,json,time,hashlib
from pathlib import Path
import numpy as np
import pandas as pd

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--archive-root',required=True)
    ap.add_argument('--output-dir',required=True)
    ap.add_argument('--cost',choices=['standard','stress'],default='standard')
    ap.add_argument('--models',nargs='*',help='Optional strategy subset; default includes all restored main stock routes')
    args=ap.parse_args();root=Path(args.archive_root).resolve();out=Path(args.output_dir).resolve()
    if (out/'summary.json').exists():raise FileExistsError('Choose a fresh output directory')
    assert (root/'scripts/account_engine.py').is_file()
    sys.path.insert(0,str(root/'scripts'))
    import run_account_backtest as base
    from account_engine import Account
    import run_trend_state_trial_v1 as trend
    import temporal_attention_accounts_v1 as temporal
    started=time.perf_counter();ranks=base.make_rankings();sources=[]
    def add_scores(path):
        if not path.is_file():raise FileNotFoundError(path)
        d=pd.read_parquet(path);sources.append(path)
        for (tag,day),g in d.groupby(['model','signal_date'],observed=True):
            ranks.setdefault(str(tag),{})[pd.Timestamp(day)]=g.sort_values(['score','symbol'],ascending=[False,True]).symbol.astype(str).tolist()
    for rel in ['information_screen_v2','qlib_joint_screen_v1','qlib_joint_monthly_v1','jf05_signal_v1','jf20_p20_full_v1']:
        add_scores(root/'results'/rel/'predictions_2024_2025.parquet')
    panel=pd.read_parquet(root/'data/processed/training_panel_v1.parquet')
    pred=pd.read_parquet(root/'results/rolling_signal_v1/predictions_2020_2025.parquet')
    tr,_=trend.build_rankings(panel,pred)
    ranks.update({k:v for k,v in tr.items() if k in ['T01','S01']})
    state_path=root/'results/regime_sequence_remaining_months_v1_1/predictions_2024_2025.parquet'
    state=pd.read_parquet(state_path);sources.append(state_path)
    for tag in ['RG01','SQ01']:
        ranks[tag]={pd.Timestamp(day):g.sort_values([tag+'_score','symbol'],ascending=[False,True]).symbol.astype(str).tolist() for day,g in state.groupby('signal_date',observed=True)}
    pilot=root/'results/temporal_attention_pilot_v1'
    keys=pd.read_csv(pilot/'data_cache/keys.csv',dtype={'symbol':str},parse_dates=['signal_date'],low_memory=False)
    evaluation=keys.loc[keys.split.eq('evaluation')]
    for tag in [f'{model}_seed{seed}' for model in ['ATTN','MLP'] for seed in [11,23,37]]:
        path=pilot/'predictions'/f'{tag}.npy';values=np.load(path,mmap_mode='r');assert values.shape==(len(keys),)
        sources.append(path)
        ranks[tag]={pd.Timestamp(day):temporal.ordered(g,values[g.key_id.to_numpy()]) for day,g in evaluation.groupby('signal_date')}
    static_path=root/'results/n01_static_e00_v1/static_e00_predictions.npy'
    if static_path.is_file():
        static=np.load(static_path,mmap_mode='r');assert static.shape==(len(keys),)
        sources.append(static_path)
        ranks['N01_STATIC']={pd.Timestamp(day):temporal.ordered(g,static[g.key_id.to_numpy()]) for day,g in evaluation.groupby('signal_date')}
        for seed in [11,23,37]:
            attn=np.load(pilot/'predictions'/f'ATTN_seed{seed}.npy',mmap_mode='r');daily={}
            for day,g in evaluation.groupby('signal_date'):
                ids=g.key_id.to_numpy();ready=g.sequence_ready.to_numpy(bool);score=np.full(len(g),np.nan)
                score[ready]=.5*(pd.Series(static[ids][ready]).rank(pct=True).to_numpy()+pd.Series(attn[ids][ready]).rank(pct=True).to_numpy())
                # Original N02 persists float32 blends before ranking. Without
                # this cast, mathematically tied fractions can sort differently.
                daily[pd.Timestamp(day)]=temporal.ordered(g,score.astype(np.float32))
            ranks[f'N02_seed{seed}']=daily
    from r01_risk_controls_v1 import R01Account,apply_valid_decision
    risk_dir=root/'results/r01_development_calibration_v1_1'
    risk_cache=pd.read_parquet(risk_dir/'decision_cache.parquet')
    risk_cache={pd.Timestamp(d):g.iloc[0].to_dict() for d,g in risk_cache.groupby('signal_date')}
    static_fraction=json.loads((risk_dir/'calibration.json').read_text(encoding='utf-8'))['static_stock_fraction_candidate']
    for tag in ['R01_DYNAMIC','R01_STATIC']:ranks[tag]=ranks['E00']
    if args.models:
        assert set(args.models).issubset(ranks),set(args.models)-set(ranks)
        ranks={k:v for k,v in ranks.items() if k in args.models}
    market=pd.read_parquet(root/'data/processed/account_market_2020_2025.parquet')
    market=market.loc[market.trade_date.between('2024-01-01','2025-12-31')]
    calendar=pd.DatetimeIndex(sorted(market.trade_date.unique()));assert len(calendar)==485
    events,unknown=base.prepare_events(calendar)
    by_event={d:g.to_dict('records') for d,g in events.groupby('ex_date')}
    by_unknown={d:g.symbol.tolist() for d,g in unknown.groupby('trade_date')}
    mergers=[json.loads(f.read_text(encoding='utf-8')) for f in sorted((root/'results/corporate_dates').glob('merger_*_2025.json'))]
    mult,slip=(1.,.0005) if args.cost=='standard' else (2.,.001)
    accounts={tag:(R01Account if tag.startswith('R01_') else Account)(slippage=slip,commission_multiplier=mult) for tag in ranks}
    checks={tag:0. for tag in ranks}
    # Refuse silently missing signal days, even if this could shorten the run.
    for tag in ranks:assert all(d in ranks[tag] for d in calendar[::5]),tag
    out.mkdir(parents=True,exist_ok=True)
    print(f'Loaded {len(accounts)} frozen-score accounts; replaying {len(calendar)} sessions',flush=True)
    for i,(day,frame) in enumerate(market.groupby('trade_date',sort=True)):
        quotes=frame.set_index('symbol').to_dict('index');nextday=calendar[i+1] if i+1<len(calendar) else day+pd.Timedelta(days=10)
        prev=calendar[i-1] if i else pd.Timestamp('2023-12-29')
        for tag,a in accounts.items():
            before=a.cash;n=len(a.trades)
            for m in mergers:
                if day==pd.Timestamp(m['conversion_date']):a.merger(m,day,quotes)
            for s in by_unknown.get(day,[]):
                if a.quantity(s):a.issue(day,s,'UNSUPPORTED_FACTOR_CHANGE_WHILE_HELD')
            for e in by_event.get(day,[]):a.corporate_action(e,day,prev)
            a.execute(day,nextday,quotes);a.close(day,quotes)
            expected=before+a.daily[-1]['cash_paid']+sum((t['notional'] if t['side']=='SELL' else -t['notional'])-t['fee']-t['dividend_tax'] for t in a.trades[n:])
            checks[tag]=max(checks[tag],abs(expected-a.cash));assert checks[tag]<1e-5
            if i%5==0 and i+1<len(calendar):
                if tag.startswith('R01_'):
                    row=risk_cache[day];fraction=row['g_dynamic'] if tag=='R01_DYNAMIC' else static_fraction
                    if row['risk_valid']:apply_valid_decision(a,day,ranks[tag][day],quotes,row,float(fraction))
                else:a.plan(day,ranks[tag][day],quotes,benchmark=tag=='B00')
        if (i+1)%100==0:print(f'{i+1}/485, {time.perf_counter()-started:.1f}s',flush=True)
    results=[]
    for tag,a in accounts.items():
        dest=out/tag;dest.mkdir();d=pd.DataFrame(a.daily);t=pd.DataFrame(a.trades)
        d.to_csv(dest/'daily.csv',index=False);t.to_csv(dest/'fills.csv',index=False)
        metrics=temporal.metrics(d)
        metrics.update(fills=len(t),fees=float(t.fee.sum()) if len(t) else 0,issues=len(a.issues),cash_identity_max_error=checks[tag])
        identity=d.cash+d.market_value+d.dividend_receivable-d.estimated_dividend_tax
        metrics['nav_identity_max_error']=float(np.max(np.abs(d.nav-identity)))
        assert metrics['nav_identity_max_error']<1e-5 and d.cash.min()>-1e-6
        results.append({'strategy':tag,'metrics':metrics,'publication_status':'ASSUMPTION_LIMITED_ACCOUNT_PASS' if not a.issues else 'DIAGNOSTIC_UNRESOLVED_EVENTS'})
    record={'status':'ACCOUNT_REPLAY_COMPLETE','scope':'Frozen historical scores plus original deterministic rules; no retraining','period':['2024-01-02','2025-12-31'],'sessions':485,'cost':args.cost,'results':results,'elapsed_seconds':time.perf_counter()-started,'prediction_sha256':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}
    (out/'summary.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'status':record['status'],'strategies':len(results),'elapsed_seconds':record['elapsed_seconds'],'issues':{r['strategy']:r['metrics']['issues'] for r in results}},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
