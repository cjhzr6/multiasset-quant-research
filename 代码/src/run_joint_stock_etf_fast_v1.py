from __future__ import annotations
import argparse, hashlib, json, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/joint_stock_etf_fast_v1'
PRICE=ROOT/'data/processed/multi_asset_handoff_v2/daily_prices.parquet'
EVENT=ROOT/'data/processed/multi_asset_etf_trial_v1/corporate_actions.csv'
ASSETS=['csi500','sp500','gold','gov_bond']
CAPS=np.array([.8,.7,.6,.6]); H=20; GAMMA=5.; ALPHA=1.

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''): h.update(b)
 return h.hexdigest()

def load_panel():
 d=pd.read_parquet(PRICE); d['trade_date']=pd.to_datetime(d.trade_date).dt.normalize()
 d=d[d.asset_id.isin(ASSETS)].sort_values(['asset_id','trade_date'])
 if d.duplicated(['asset_id','trade_date']).any(): raise ValueError('duplicate price keys')
 ev=pd.read_csv(EVENT,encoding='utf-8-sig'); ev['asset_id']=ev.asset_id.replace({'physical_gold':'gold','rmb_gov_bond':'gov_bond'}); ev=ev[ev.asset_id.isin(ASSETS)].copy()
 # Build daily gross economic factors from raw close and the locally audited event snapshot.
 rows=[]; idx=pd.DatetimeIndex(sorted(d.trade_date.unique()))
 prices=d.pivot(index='trade_date',columns='asset_id',values='close_cny').reindex(idx).ffill()
 opens=d.pivot(index='trade_date',columns='asset_id',values='open_cny').reindex(idx).ffill()
 masks=d.pivot(index='trade_date',columns='asset_id',values='close_cny').reindex(idx).notna()
 prices=prices[ASSETS]; opens=opens[ASSETS]; masks=masks[ASSETS]
 gross=prices.pct_change().fillna(0.)+1.
 for r in ev.itertuples(index=False):
  if r.verification_status not in ('official_verified','secondary_verified_primary_pdf_missing'): continue
  dt=pd.Timestamp(r.ex_date); a=r.asset_id
  if dt not in gross.index: continue
  prev=prices.index[prices.index.get_loc(dt)-1] if gross.index.get_loc(dt)>0 else None
  if prev is None: continue
  ratio=float(r.ratio_old_to_new) if pd.notna(r.ratio_old_to_new) else 1.
  cash=float(r.cash_cny_per_unit) if pd.notna(r.cash_cny_per_unit) else 0.
  gross.loc[dt,a]=(ratio*prices.loc[dt,a]+cash)/prices.loc[prev,a]
 # Use total-return gross series for signals, labels, and proxy wealth. Current-day close signal only.
 ret=gross-1.
 feat=[]
 for a in ASSETS:
  x=ret[a]
  feat += [((1+x).rolling(n).apply(np.prod,raw=True)-1).rename(f'{a}_r{n}') for n in (5,20,60)]
  feat += [x.rolling(20).std().rename(f'{a}_v20'),
           ((1+x).cumprod()/((1+x).cumprod().rolling(60).max())-1).rename(f'{a}_dd60'),
           (~masks[a]).astype(float).rename(f'{a}_missing')]
 X=pd.concat(feat,axis=1)
 # Pairwise trailing correlations are contemporaneous-as-of only.
 for i,a in enumerate(ASSETS):
  for b in ASSETS[i+1:]: X[f'corr_{a}_{b}']=ret[a].rolling(60).corr(ret[b])
 # Labels: signal at t close, enter at t+1 open, exit at t+21 open. Proxy close adjustment covers distributions/conversions only when locally sourced.
 lab=pd.DataFrame(index=idx)
 for a in ASSETS:
  lab[a]=(opens[a].shift(-21)/opens[a].shift(-1)-1.)
 return d,idx,prices,opens,masks,ret,X,lab,ev

def fit_models(X,lab,idx):
 train=(idx<=pd.Timestamp('2022-12-30')) & (lab.notna().all(axis=1).to_numpy()) & X.notna().all(axis=1).to_numpy()
 valid=(idx>pd.Timestamp('2022-12-30'))&(idx<=pd.Timestamp('2023-12-29'))&lab.notna().all(axis=1).to_numpy()&X.notna().all(axis=1).to_numpy()
 # Explicitly retain only labels with a fully matured t+21 endpoint by each cutoff.
 end_pos={d:i for i,d in enumerate(idx)}
 matured_train=np.array([bool(train[i]) and i+21<len(idx) and idx[i+21]<=pd.Timestamp('2022-12-30') for i in range(len(idx))])
 matured_val=np.array([bool(valid[i]) and i+21<len(idx) and idx[i+21]<=pd.Timestamp('2023-12-29') for i in range(len(idx))])
 xmat=X.to_numpy(dtype=float)
 impute=np.nanmedian(xmat[matured_train],axis=0)
 xmat=np.where(np.isfinite(xmat),xmat,impute)
 scaler=StandardScaler().fit(xmat[matured_train])
 xs=scaler.transform(xmat)
 ysc=StandardScaler().fit(lab.to_numpy()[matured_train])
 ys=ysc.transform(lab.to_numpy()[matured_train])
 ridge=Ridge(alpha=ALPHA).fit(xs[matured_train],ys)
 with warnings.catch_warnings():
  warnings.simplefilter('ignore')
  mlp=MLPRegressor(hidden_layer_sizes=(32,),activation='relu',solver='adam',alpha=1e-4,
   learning_rate_init=1e-3,max_iter=40,random_state=11,shuffle=False,tol=0.,n_iter_no_change=40)
  mlp.fit(xs[matured_train],ys)
 pred={k:ysc.inverse_transform(m.predict(xs)) for k,m in [('P',ridge),('MLP',mlp)]}
 # 2023 is report-only; no parameter selection.
 v={}
 if matured_val.any():
  y=lab.to_numpy()[matured_val]
  for k in pred:
   p=pred[k][matured_val]
   ic=[]
   for j in range(4):
    if np.std(y[:,j]) and np.std(p[:,j]): ic.append(float(np.corrcoef(y[:,j],p[:,j])[0,1]))
   v[k]={'n':int(matured_val.sum()),'mae_20d':float(np.mean(np.abs(p-y))),
         'asset_pearson_ic':dict(zip(ASSETS,ic+[None]*(4-len(ic))))}
 return pred,{'train_n':int(matured_train.sum()),'validation_n':int(matured_val.sum()),
  'train_end_label_date':'2022-12-30','validation_end_label_date':'2023-12-29',
  'ridge_alpha':ALPHA,'mlp':{'hidden_layer_sizes':[32],'seed':11,'max_iter':40,'actual_iter':int(mlp.n_iter_)},
  'validation_report_only':v,
  'scaler_mean':scaler.mean_.tolist(),'scaler_scale':scaler.scale_.tolist(),
  'label_mean':ysc.mean_.tolist(),'label_scale':ysc.scale_.tolist()},scaler,ysc,ridge,mlp

def weights(mu,cov,prev):
 sigma=H*(.5*cov+.5*np.diag(np.diag(cov)))
 vals,vec=np.linalg.eigh((sigma+sigma.T)/2); sigma=vec@np.diag(np.maximum(vals,1e-10))@vec.T
 fee=.0005
 def obj(w): return -float(mu@w-.5*GAMMA*w@sigma@w-fee*np.abs(w-prev).sum())
 res=minimize(obj,np.clip(prev,0,CAPS),method='SLSQP',bounds=list(zip(np.zeros(4),CAPS)),
  constraints=[{'type':'ineq','fun':lambda w:1.-w.sum()}],options={'maxiter':100,'ftol':1e-6,'disp':False})
 if res.success and np.isfinite(res.x).all() and res.x.sum()<=1+1e-6: return np.clip(res.x,0,CAPS),str(res.message)
 return prev.copy(),'FALLBACK_PREVIOUS:'+str(res.message)

def backtest(name, pred, idx, ret, prices, opens, out, ev):
 start=pd.Timestamp('2024-01-02'); end=pd.Timestamp('2026-09-21')
 ids=np.flatnonzero((idx>=start)&(idx<=end)); dates=idx[ids]
 if len(dates)<100: raise ValueError('insufficient report calendar')
 nav=1_000_000.; w=np.zeros(4); pending=None; pending_status='INITIAL_CASH'; last_target=np.zeros(4); rows=[]; turnover=0.; costtotal=0.; failures=0
 rets=ret.to_numpy(float); daily_mat=rets
 event_lookup={}
 for r0 in ev.itertuples(index=False):
  dt=pd.Timestamp(r0.ex_date); a=r0.asset_id
  ratio=float(r0.ratio_old_to_new) if pd.notna(r0.ratio_old_to_new) else 1.
  cash=float(r0.cash_cny_per_unit) if pd.notna(r0.cash_cny_per_unit) else 0.
  prev=event_lookup.get((dt,a),(1.,0.)); event_lookup[(dt,a)]=(prev[0]*ratio,prev[1]+cash)
 for k,ix in enumerate(ids):
  date=idx[ix]
  # Legacy holdings carry the close-to-open move. New signals are formed after close t
  # and filled at open t+1, so overnight P&L belongs only to positions held at t.
  traded=0.; cost=0.; msg='HOLD'; executed=False
  if k>0 and pending is not None:
   prev_date=idx[ix-1]
   overnight=[]
   for a0 in ASSETS:
    ratio,cash0=event_lookup.get((date,a0),(1.,0.))
    prior=float(prices.loc[prev_date,a0]); op=float(opens.loc[date,a0])
    overnight.append((ratio*op+cash0)/prior-1. if prior>0 else 0.)
   overnight=np.asarray(overnight); overnight_port=float(w@overnight); nav*=1+overnight_port
   w=w*(1+overnight)/(1+overnight_port) if 1+overnight_port>0 else w
  if pending is not None:
   target,pending_status=pending; msg=pending_status; pending=None; executed=True
   change=np.abs(target-w); traded=float(change.sum())
   cost=nav*traded*.0005; nav-=cost; costtotal+=cost; turnover+=nav*traded
   w=target.copy(); last_target=target.copy()
  # On execution days new targets earn only open-to-close; on other days the
  # event-adjusted prior-close to close total return is applied.
  if executed:
   intra=np.array([prices.loc[date,a0]/opens.loc[date,a0]-1. if opens.loc[date,a0]>0 else 0. for a0 in ASSETS])
   dayr=intra
  elif k==0:
   dayr=np.zeros(4)
  else: dayr=rets[ix]
  port=float(w@dayr); nav*=1+port
  w=w*(1+dayr)/(1+port) if (1+port)>0 else w
  cash=max(0.,1-w.sum()); rows.append({'date':date,'nav':nav,'daily_return':nav/(rows[-1]['nav'] if rows else 1_000_000.)-1,
   **{f'w_{a}':float(w[j]) for j,a in enumerate(ASSETS)},
   **{f'target_{a}':float(last_target[j]) for j,a in enumerate(ASSETS)},
   'w_cash':float(cash),'gross_rebalance_notional_cny':float(nav*traded),
   'cost_cny':cost,'optimizer_status':msg})
  # Form today's signal only after close; execute no earlier than the next session.
  if k%H==0 and k+1<len(ids):
   if name=='C1_fixed25': pending=(np.array([.25]*4),'FIXED25')
   else:
    hist=daily_mat[max(0,ix-59):ix+1]
    if len(hist)>=40:
     cov=np.cov(hist,rowvar=False,ddof=1)
     pending=weights(pred[ix],cov,w)
     failures+=int(pending[1].startswith('FALLBACK'))
 daily=pd.DataFrame(rows); daily.to_csv(out/f'{name}_daily.csv',index=False)
 r=daily.daily_return.to_numpy(); sd=float(np.std(r,ddof=1)); peak=np.maximum.accumulate(np.r_[1e6,daily.nav.to_numpy()]); dd=(np.r_[1e6,daily.nav.to_numpy()]/peak-1)
 metrics={'name':name,'sessions':len(daily),'total_return':float(daily.nav.iloc[-1]/1e6-1),
  'annualized_return':float((daily.nav.iloc[-1]/1e6)**(252/len(daily))-1),
  'annualized_volatility':float(sd*np.sqrt(252)),'sharpe_zero_cash':float(np.mean(r)/sd*np.sqrt(252)) if sd else None,
  'max_drawdown':float(dd.min()),'ending_nav':float(daily.nav.iloc[-1]),
  'mean_weights':{a:float(daily[f'w_{a}'].mean()) for a in ASSETS},'mean_cash':float(daily.w_cash.mean()),
  'gross_rebalance_notional_cny':float(daily.gross_rebalance_notional_cny.sum()),'approx_cost_cny':float(daily.cost_cny.sum()),
  'optimizer_fallbacks':failures,'proxy_scope':'fractional close-mark portfolio; no real fills, integer lots, fees by instrument, taxes, full event receivables, or E00 selected-stock execution'}
 # Continuous account is segmented only for reporting; NAV never resets.
 for label,a,b in [('2024','2024-01-02','2024-12-31'),('2025','2025-01-01','2025-12-31'),('2026_partial','2026-01-01','2026-09-21')]:
  g=daily[pd.to_datetime(daily.date).between(a,b)]
  if len(g):
   before=daily[pd.to_datetime(daily.date)<a].nav
   base=float(before.iloc[-1]) if len(before) else 1e6
   metrics.setdefault('period_returns',{})[label]=float(g.nav.iloc[-1]/base-1)
 return metrics

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--force',action='store_true'); a=ap.parse_args()
 OUT.mkdir(parents=True,exist_ok=True)
 t0=time.perf_counter()
 d,idx,prices,opens,masks,ret,X,lab,ev=load_panel()
 good=X.notna().all(axis=1)&lab.notna().all(axis=1)
 pred_path=OUT/'predictions.csv'; old_summary=OUT/'fast_results.json'
 if not old_summary.exists(): old_summary=OUT/'pre_overnight_correction_attempt'/'fast_results.json'
 if pred_path.exists() and old_summary.exists():
  # Reuse the already-fitted fixed model outputs; do not fit again for account-clock fixes.
  predframe=pd.read_csv(pred_path,parse_dates=['trade_date']); modelmeta=json.loads(old_summary.read_text(encoding='utf-8'))['model']
  pred={k:np.full((len(idx),4),np.nan) for k in ('P','MLP')}
  pos={pd.Timestamp(d):i for i,d in enumerate(idx)}
  for row in predframe.itertuples(index=False):
   i=pos[pd.Timestamp(row.trade_date)]
   for k in pred:
    pred[k][i]=np.asarray([getattr(row,f'{k}_{a0}') for a0 in ASSETS],dtype=float)
 else:
  pred,modelmeta,xsc,ysc,ridge,mlp=fit_models(X,lab,idx)
  pred_idx=np.flatnonzero(X.notna().all(axis=1).to_numpy())
  predframe=pd.DataFrame({'trade_date':idx[pred_idx]})
  for m in pred:
   for j,a0 in enumerate(ASSETS): predframe[f'{m}_{a0}']=pred[m][pred_idx,j]
  predframe.to_csv(pred_path,index=False)
 reports=[]
 for tag in ('P','MLP','C1_fixed25'):
  pm=pred['P'] if tag=='P' else pred.get(tag, pred['MLP'])
  reports.append(backtest(tag,pm,idx,ret,prices,opens,OUT,ev))
 manifest={'status':'PROXY_ACCOUNTS_COMPLETE_REQUIRES_REVIEW','created_local':time.strftime('%Y-%m-%d %H:%M:%S'),
  'assets':ASSETS,'stock_module':'STOCK_TARGET_PROXY=CSI500 ETF 510500 price/event gross return; E00 actual constituent execution not included',
  'input_sha256':{str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in [PRICE,EVENT]},
  'price_rows':len(d),'calendar_sessions':len(idx),'first_date':str(idx.min().date()),'last_date':str(idx.max().date()),
  'event_rows_used':len(ev),'event_verification_counts':ev.verification_status.value_counts().to_dict(),
  'model':modelmeta,'backtests':reports,'limitations':['Proxy only, not real shared-cash trading account or actual E00 stock picks.',
   'Fractional weights and approximate 5bp one-way cost; no exact lots, instrument commissions, slippage by asset, T+1/capacity/limit rules, taxes or spendability timing.',
   'Daily total-return factors use available event snapshot; snapshot is incomplete for csi500 and has one secondary-verified 2026 bond event.',
   'Rebalance approximation uses close-derived weights and open-close on rebalance date; H labels use t+1 to t+21 open but daily wealth is event-adjusted proxy.',
   'Historical period is explored/repeatedly viewed; no claim of out-of-sample superiority. 2023 reported only, not used for tuning.'],
  'elapsed_seconds':time.perf_counter()-t0}
 (OUT/'fast_results.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
 print(json.dumps(manifest,ensure_ascii=False,indent=2,default=str))

if __name__=='__main__': main()
