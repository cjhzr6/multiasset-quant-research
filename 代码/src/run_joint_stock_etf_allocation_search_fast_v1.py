from __future__ import annotations
import json, hashlib, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/joint_stock_etf_allocation_search_fast_v1'
E00_DAILY=ROOT/'results/account_backtest_v1/E00_daily.csv'
PRED=ROOT/'results/rolling_signal_v1/predictions_2020_2025.parquet'
PRICE=ROOT/'data/processed/multi_asset_handoff_v2/daily_prices.parquet'
EVENT=ROOT/'data/processed/multi_asset_etf_trial_v1/corporate_actions.csv'
ASSETS=['csi500','sp500','gold','gov_bond']
CAPS=np.array([.8,.7,.6,.6]); SEED=20261010; COST=.0005; CADENCE=20

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def project_bounded_simplex(x,total,caps=CAPS):
 x=np.asarray(x,dtype=float)
 lo=float(np.min(x-caps)-1.); hi=float(np.max(x)+1.)
 for _ in range(100):
  mid=(lo+hi)/2; w=np.clip(x-mid,0,caps)
  if w.sum()>total: lo=mid
  else: hi=mid
 w=np.clip(x-(lo+hi)/2,0,caps)
 # correct floating residual on available coordinates
 rem=total-w.sum()
 for j in np.argsort(-(caps-w)):
  add=min(rem,caps[j]-w[j]); w[j]+=add; rem-=add
  if rem<=1e-12: break
 return w

def candidates():
 rng=np.random.default_rng(SEED); rows=[]; labels=[]
 for i in range(96):
  x=rng.dirichlet(np.ones(4)); rows.append(project_bounded_simplex(x,1.)); labels.append(f'DIR_FULL_{i:03d}')
 for i in range(32):
  cash=float(rng.uniform(0,.30)); x=rng.dirichlet(np.ones(4))*(1-cash)
  rows.append(project_bounded_simplex(x,1-cash)); labels.append(f'DIR_CASH_{i:03d}')
 rows.append(np.ones(4)*.25); labels.append('ANCHOR_EQUAL')
 for i,x in enumerate([[.4,.3,.2,.1],[.3,.4,.2,.1],[.3,.25,.35,.1]]):
  rows.append(np.asarray(x,dtype=float)); labels.append(f'ANCHOR_DIRECTION_{i+1}')
 out=[]; seen=set()
 for label,w in zip(labels,rows):
  k=tuple(np.round(w,12))
  if k in seen: continue
  seen.add(k); out.append((label,w))
 return out

def module_returns():
 e=pd.read_csv(E00_DAILY,parse_dates=['date']).set_index('date').sort_index()
 stock=e.nav.astype(float).pct_change().fillna(0.).rename('stock')
 from run_joint_stock_etf_fast_v1 import load_panel
 _,idx,_,_,_,ret,_,_,_=load_panel()
 # The common labelled stock sleeve is net E00; the other columns are event-adjusted available-gross returns.
 etf=ret[['sp500','gold','gov_bond']].copy(); etf.index=idx
 panel=pd.concat([stock,etf],axis=1).reindex(stock.index)
 # Missing ETF sessions are held at the last valid mark in this proxy: return 0 and disclose; never backfill from the future.
 panel.loc[:,['sp500','gold','gov_bond']]=panel[['sp500','gold','gov_bond']].fillna(0.)
 return panel

def proxy_path(returns,w):
 x=returns.to_numpy(float); nav=1.; cur=np.zeros(4); pending=None; daily=[]; turnovers=[]
 for i,r in enumerate(x):
  cost=0.
  if pending is not None:
   cost=nav*COST*np.abs(pending-cur).sum(); nav-=cost; cur=pending.copy(); pending=None
  pr=float(cur@r); nav*=1+pr
  cur=cur*(1+r)/(1+pr) if 1+pr>0 else cur
  daily.append(nav); turnovers.append(cost)
  if i%CADENCE==0 and i+1<len(x): pending=np.asarray(w,dtype=float)
 d=pd.DataFrame({'date':returns.index,'nav':daily,'cost':turnovers})
 d['ret']=d.nav.pct_change().fillna(0.)
 return d

def sharpe(r):
 sd=float(np.std(r,ddof=1)) if len(r)>1 else 0.
 return float(np.mean(r)/sd*np.sqrt(252)) if sd>0 else None

def score_training(panel,grid):
 tr=panel.loc['2020-01-01':'2022-12-31']; rows=[]
 for label,w in grid:
  d=proxy_path(tr,w); years={}; sh=[]
  for y,g in d.groupby(d.date.dt.year):
   s=sharpe(g.ret.to_numpy()); mean=float(g.ret.mean()); years[str(y)]={'sharpe':s,'return':float((1+g.ret).prod()-1),'sessions':len(g)}
   if s is None or mean<=0: sh=[]; break
   sh.append(s)
  q=float(np.mean(sh)-.5*np.std(sh,ddof=0)) if len(sh)==3 else None
  rows.append({'candidate_id':label,'weights':dict(zip(ASSETS,w.tolist())),'cash':float(1-sum(w)),
   'training_score':q,'annual_metrics':years,'proxy_cost_rate_one_way':COST,'daily_count':len(d)})
 return sorted(rows,key=lambda z:(-1e9 if z['training_score'] is None else -z['training_score'],z['candidate_id']))

def build_e00_rank(calendar):
 p=pd.read_parquet(PRED,columns=['signal_date','symbol','model','score']); p.signal_date=pd.to_datetime(p.signal_date).dt.normalize()
 p=p[p.model.eq('E00') & p.signal_date.isin(calendar)]
 return {d:g.sort_values(['score','symbol'],ascending=[False,True],kind='stable').symbol.astype(str).tolist()
  for d,g in p.groupby('signal_date',observed=True)}

def run_account(policy,w,start,end,mode):
 import joint_stock_etf_fast_real_v1 as real
 real.START=pd.Timestamp(start); real.END=pd.Timestamp(end)
 market,calendar=real.load_stock_market(); ranks=build_e00_rank(calendar); prices,events,lookup=real.load_etf()
 _,idx,_,_,_,ret,_,_,_=__import__('run_joint_stock_etf_fast_v1').load_panel()
 ret.index=idx
 if len(ranks)!=len(calendar) or not calendar.isin(ranks.keys()).all(): raise RuntimeError('E00 rank coverage mismatch')
 if not prices.trade_date.between(calendar[0],calendar[-1]).any(): raise RuntimeError('ETF prices miss account period')
 fixed=np.asarray(w,dtype=float)
 dummy=pd.DataFrame(index=calendar)
 daily,fills,metrics=real.run_mixed_account(ranks,mode,calendar,market,prices,events,lookup,
  policy,dummy,ret,fixed_target_weights=fixed)
 out=OUT/'accounts'/policy/mode; out.mkdir(parents=True,exist_ok=True)
 daily.to_csv(out/'daily.csv',index=False); fills.to_csv(out/'fills.csv',index=False)
 pd.DataFrame(daily.attrs['decisions']).to_csv(out/'target_decisions.csv',index=False)
 pd.DataFrame(daily.attrs['etf_events']).to_csv(out/'etf_events.csv',index=False)
 pd.DataFrame(daily.attrs['stock_rejections']).to_csv(out/'stock_rejections.csv',index=False)
 pd.DataFrame(daily.attrs['etf_orders']).to_csv(out/'etf_orders.csv',index=False)
 return metrics,{'calendar':calendar,'daily':daily}

def main():
 OUT.mkdir(parents=True,exist_ok=True); t0=time.perf_counter()
 panel=module_returns(); grid=candidates(); scored=score_training(panel,grid)
 pd.DataFrame(scored).to_json(OUT/'training_candidate_scores.json',orient='records',force_ascii=False,indent=2)
 if len(scored)<132: raise RuntimeError(f'only {len(scored)} unique candidates; phi says do not replace duplicates')
 viable=[x for x in scored if x['training_score'] is not None]
 selected=[]
 for c in viable:
  w=np.array([c['weights'][a] for a in ASSETS])
  if not selected or all(np.abs(w-np.array([s['weights'][a] for a in ASSETS])).sum()>=.20-1e-12 for s in selected): selected.append(c)
  if len(selected)==2: break
 if len(selected)<2: raise RuntimeError('could not find two training candidates with L1 distance >=.20')
 # 2023 uses real E00+ETF shared accounts; it only chooses among these two and C1 equal split.
 validation=[]; vp=[]
 choices=[*selected,{'candidate_id':'ANCHOR_EQUAL_VALIDATION','weights':dict(zip(ASSETS,[.25]*4)),'cash':0.}]
 for c in choices:
  w=np.array([c['weights'][a] for a in ASSETS]); m,o=run_account(c['candidate_id'],w,'2023-01-01','2023-12-31','standard')
  row={'candidate_id':c['candidate_id'],'weights':c['weights'],'metrics':m}
  validation.append(row); vp.append((m['sharpe_zero_cash'],m['total_return'],c))
 pd.DataFrame(validation).to_json(OUT/'validation_2023.json',orient='records',force_ascii=False,indent=2)
 viable_v=[v for v in vp if v[0] is not None and v[1]>0]
 if not viable_v: raise RuntimeError('no positive-return 2023 validation candidate; stop before 2024 selection')
 winner=max(viable_v,key=lambda v:(v[0],-sum(v[2]['weights'][a] for a in ASSETS)))
 win=winner[2]; w=np.array([win['weights'][a] for a in ASSETS])
 standard,_=run_account(win['candidate_id'],w,'2024-01-01','2025-12-31','standard')
 stress,_=run_account(win['candidate_id'],w,'2024-01-01','2025-12-31','stress')
 # Reuse the previously executed C1 fixed25 control; no account is rerun here.
 base=json.loads((ROOT/'results/joint_stock_etf_fast_v1/real_accounts/real_account_summary.json').read_text(encoding='utf-8'))
 base_stress=json.loads((ROOT/'results/joint_stock_etf_fast_v1/real_accounts/stress_account_summary.json').read_text(encoding='utf-8'))
 report={'status':'FIXED_WEIGHT_SEARCH_COMPLETE_REQUIRES_INDEPENDENT_REVIEW','candidate_count':len(scored),
  'train_selection':{'period':['2020-01-02','2022-12-30'],'score':'mean annual zero-cash Sharpe - 0.5*population SD; reject any year nonpositive return/undefined Sharpe',
   'selected_two':[{'candidate_id':c['candidate_id'],'training_score':c['training_score'],'weights':c['weights'],'cash':c['cash']} for c in selected]},
  'validation_2023':validation,'winner':{'candidate_id':win['candidate_id'],'weights':win['weights'],'cash':float(1-w.sum()),
   'validation_sharpe':winner[0],'validation_return':winner[1]},
  'holdout_review_2024_2025':{'standard':standard,'stress':stress,
   'C1_fixed25_standard':base['accounts']['C1_fixed25'],'C1_fixed25_stress':base_stress['accounts']['C1_fixed25']},
  'cost_mode':'real standard/stress shared cash N04 runner; no weight/model refit after 2023 selection',
  'limitations':['training proxy uses daily net E00 module NAV and event-adjusted ETF gross returns; stock internal E00 costs are not subtracted twice',
   'training proxy uses close-to-close fractional paths with approximate 5bp external turnover cost; candidate selection is not a real account result',
   '2023 winner selection on one viewed year is a small validation sample; 2024-25 was already viewed exploratory history, not a blind test',
   'E00 actual ranked equities and true lots/fees/events are used only in validation/evaluation accounts'],
  'source_sha256':{str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in [E00_DAILY,PRED,PRICE,EVENT]},
  'elapsed_seconds':time.perf_counter()-t0}
 (OUT/'allocation_search_summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
 print(json.dumps({'status':report['status'],'winner':report['winner'],'selected':report['train_selection']['selected_two'],
  'validation':validation,'holdout_standard':standard,'holdout_stress':stress,'elapsed_seconds':report['elapsed_seconds']},ensure_ascii=False,indent=2,default=str))
if __name__=='__main__': main()
