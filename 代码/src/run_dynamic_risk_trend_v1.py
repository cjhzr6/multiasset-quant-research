from __future__ import annotations
import json, time, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize

import run_joint_stock_etf_allocation_search_fast_v1 as base
ROOT=base.ROOT
OUT=ROOT/'results/joint_stock_etf_allocation_search_fast_v1/dynamic_risk_trend'
ASSETS=base.ASSETS; CAPS=base.CAPS
RISK={'RISK_B1':np.array([.45,.30,.20,.05]),'RISK_B2':np.array([.55,.25,.15,.05])}
W0=np.full(4,.25)

def bounded(x,total): return base.project_bounded_simplex(x,total,CAPS)

def make_schedules(rets,calendar):
    schedules={k:{} for k in [*RISK,'TREND_FIXED_ETA05_RHO05']}
    prior={k:W0.copy() for k in RISK}
    for i,d in enumerate(calendar):
        d=pd.Timestamp(d)
        if i%20: continue
        hist=rets.loc[:d].tail(60).to_numpy(float)
        if len(hist)<40 or not np.isfinite(hist).all():
            for k in RISK: schedules[k][d]=prior[k].copy()
            schedules['TREND_FIXED_ETA05_RHO05'][d]=W0.copy()
            continue
        cov=np.cov(hist,rowvar=False,ddof=1)
        cov=.5*cov+.5*np.diag(np.diag(cov))
        cov+=np.eye(4)*1e-12
        for name,budget in RISK.items():
            prev=prior[name]
            def obj(w):
                sig=float(np.sqrt(max(w@cov@w,1e-18)))
                rc=w*(cov@w)/sig
                shares=rc/(rc.sum()+1e-18)
                return float(np.sum((shares-budget)**2)+.02*np.sum(np.abs(w-prev))**2)
            res=minimize(obj,prev,method='SLSQP',bounds=list(zip(np.zeros(4),CAPS)),
                         constraints=[{'type':'eq','fun':lambda w:np.sum(w)-1.}],
                         options={'maxiter':100,'ftol':1e-11})
            w=bounded(res.x,1.) if res.success and np.isfinite(res.x).all() else prev.copy()
            schedules[name][d]=w; prior[name]=w
        # Frozen formula: trend z uses compounded 60-session return over trailing daily volatility.
        z=np.clip(np.prod(1+hist,axis=0)-1, -2*np.sqrt(60)*hist.std(axis=0,ddof=1), 2*np.sqrt(60)*hist.std(axis=0,ddof=1))
        denom=hist.std(axis=0,ddof=1)*np.sqrt(60)
        z=np.divide(z,denom,out=np.zeros_like(z),where=denom>1e-12); z=np.clip(z,-2,2)
        a=W0*np.exp(.5*z)
        if a.sum()>1: a=a/a.sum()
        a=bounded(a,float(a.sum()))
        tw=bounded(.5*W0+.5*a,float((.5*W0+.5*a).sum()))
        schedules['TREND_FIXED_ETA05_RHO05'][d]=tw
    return schedules

def run(policy,weights,start,end,mode,schedule):
    import joint_stock_etf_fast_real_v1 as real
    real.START=pd.Timestamp(start); real.END=pd.Timestamp(end)
    market,calendar=real.load_stock_market(); ranks=base.build_e00_rank(calendar); prices,events,lookup=real.load_etf()
    _,idx,_,_,_,ret,_,_,_=__import__('run_joint_stock_etf_fast_v1').load_panel(); ret.index=idx
    ret=ret.reindex(calendar).ffill().fillna(0.)
    # With precomputed target schedule, weights are keyed on the account signal calendar and never peek forward.
    sch={pd.Timestamp(d):np.asarray(w,float) for d,w in schedule.items() if pd.Timestamp(d) in calendar}
    if len(sch)!=sum(i%20==0 for i in range(len(calendar))): raise RuntimeError(f'{policy} schedule coverage {len(sch)}')
    dummy=pd.DataFrame(index=calendar)
    daily,fills,metrics=real.run_mixed_account(ranks,mode,calendar,market,prices,events,lookup,
      policy,dummy,ret,target_weight_schedule=sch)
    folder=OUT/'accounts'/policy/f'{start[:4]}_{end[:4]}'/mode; folder.mkdir(parents=True,exist_ok=True)
    daily.to_csv(folder/'daily.csv',index=False); fills.to_csv(folder/'fills.csv',index=False)
    pd.DataFrame(daily.attrs['decisions']).to_csv(folder/'target_decisions.csv',index=False)
    pd.DataFrame(daily.attrs['etf_orders']).to_csv(folder/'etf_orders.csv',index=False)
    pd.DataFrame(daily.attrs['etf_events']).to_csv(folder/'etf_events.csv',index=False)
    return metrics,daily,fills,folder

def main():
    t0=time.perf_counter(); OUT.mkdir(parents=True,exist_ok=True)
    # Strategy risk proxy deliberately uses CSI500, not the E00 stock sleeve, for ex-ante risk and trend signals.
    _,idx,_,_,_,ret,_,_,_=__import__('run_joint_stock_etf_fast_v1').load_panel(); ret.index=idx
    proxy=ret[ASSETS].copy()
    rows=[]; schedules_cache={}
    # Generate full-history schedule from historical returns, then slice per account period.
    for policy,b in RISK.items():
        # Risk optimization is recalculated in shared routine; trend sequence computed once against available market history.
        pass
    fullcal=pd.DatetimeIndex(sorted(pd.to_datetime(pd.read_parquet(base.ROOT/'data/processed/account_market_2020_2025.parquet',columns=['trade_date']).trade_date.unique()).normalize()))
    proxy=proxy.reindex(fullcal).ffill().fillna(0.)
    # Rebalance phase restarts at each account period, matching the existing 20-session account clock.
    allsch={}
    for start,end in [('2023-01-01','2023-12-31'),('2024-01-01','2025-12-31')]:
        import joint_stock_etf_fast_real_v1 as real
        real.START=pd.Timestamp(start); real.END=pd.Timestamp(end)
        _,cal=real.load_stock_market()
        allsch[(start,end)]=make_schedules(proxy,cal)
    # Convert each period's pre-close schedules and run real shared-cash ledgers.
    per2023={}; per2425={}
    for policy in [*RISK,'TREND_FIXED_ETA05_RHO05']:
        sc23=allsch[('2023-01-01','2023-12-31')][policy]
        m,daily,fills,path=run(policy,None,'2023-01-01','2023-12-31','standard',sc23)
        per2023[policy]=m; rows.append({'period':'2023_validation','policy':policy,'mode':'standard','metrics':m,'path':str(path.relative_to(ROOT))})
        if policy.startswith('RISK_'): per2425[policy]=(m['sharpe_zero_cash'],policy)
    risk_winner=max(per2425.values())[1]
    for policy in [risk_winner,'TREND_FIXED_ETA05_RHO05']:
        sc=allsch[('2024-01-01','2025-12-31')][policy]
        for mode in ['standard','stress']:
            m,daily,fills,path=run(policy,None,'2024-01-01','2025-12-31',mode,sc)
            rows.append({'period':'2024_2025_exploratory','policy':policy,'mode':mode,'metrics':m,'path':str(path.relative_to(ROOT))})
    # Reconcile fills independently against recorded ETF order fills; stock/ETF detailed totals are preserved.
    reconciliation=[]
    for row in rows:
        fp=ROOT/row['path']/'fills.csv'; op=ROOT/row['path']/'etf_orders.csv'
        fills=pd.read_csv(fp); orders=pd.read_csv(op)
        etf=orders[(orders.side.astype(str).str.lower().eq('sell')) & (pd.to_numeric(orders.get('filled_units',0),errors='coerce').fillna(0)>0)] if len(orders) and 'side' in orders else pd.DataFrame()
        reconciliation.append({'policy':row['policy'],'period':row['period'],'mode':row['mode'],'fill_rows':len(fills),
           'etf_sell_filled_order_rows':len(etf),'etf_sell_units':float(pd.to_numeric(etf.get('filled_units',pd.Series(dtype=float)),errors='coerce').fillna(0).sum()),
           'ledger_fee_sum':float(pd.to_numeric(fills.get('fee',pd.Series(dtype=float)),errors='coerce').fillna(0).sum()),
           'fill_etf_sell_rows':int(((fills.get('asset_class',pd.Series(dtype=str)).astype(str).eq('ETF')) &
                                      (fills.get('side',pd.Series(dtype=str)).astype(str).str.lower().eq('sell'))).sum()),
           'summary_fee':row['metrics'].get('fees')})
    result={'status':'DYNAMIC_RISK_TREND_COMPLETE_EXPLORATORY_REVIEW_REQUIRED',
      'proxy':'CSI500 + SP500 ETF + gold + government bond daily event-adjusted returns, 60 sessions, 50% diagonal covariance shrink; actual validation/evaluation uses E00 ranking and shared-cash N04 account.',
      'risk_budget':{'targets':{k:v.tolist() for k,v in RISK.items()},'method':'long-only SLSQP; equal sum weights=1, caps [.8,.7,.6,.6]; squared relative risk-contribution deviation + .02 squared turnover; 20-session rebalance'},
      'trend':{'formula':'w0=.25 each; z=clip(ret60/(vol60*sqrt(60)),-2,2); a=w0*exp(.5*z); normalize a if sum>1; cap-project; w=.5*w0+.5*a; rebalance every 20 sessions','parameters_frozen_before_2023':True},
      'validation_2023':per2023,'risk_budget_winner_2023':risk_winner,'results':rows,'fill_fee_reconciliation':reconciliation,
      'limitations':['2024-25 is previously viewed exploratory history, not a blind test.','Risk/trend signals use CSI500 as stock proxy; executed stock sleeve uses E00 rankings.','Stress doubles fees/slippage under the N04 scenario.','No model training/refitting was done.'],
      'elapsed_seconds':time.perf_counter()-t0}
    (OUT/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
    print(json.dumps({'status':result['status'],'risk_budget_winner_2023':risk_winner,'results':rows,'fill_fee_reconciliation':reconciliation,'elapsed_seconds':result['elapsed_seconds']},ensure_ascii=False,indent=2,default=str))
if __name__=='__main__': main()
