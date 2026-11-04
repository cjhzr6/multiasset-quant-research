from __future__ import annotations
import argparse, hashlib, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
 return h.hexdigest()

def resolve(p,base):
 p=Path(p); return p if p.is_absolute() else (base/p).resolve()

def configure(data_root,code_root,output):
 sys.path.insert(0,str(code_root/'src'))
 import joint_stock_etf_fast_real_v1 as real
 import multi_asset_etf_trial_v1 as etfs
 import stock_etf_stage_trial_v1 as primitives
 import run_account_backtest as account_backtest
 import audit_account_inputs, build_training_panel, account_engine
 import stock_etf_stage_account_engine_v1 as stock_engine
 import run_joint_stock_etf_fast_v1 as fast
 import run_joint_stock_etf_allocation_search_fast_v1 as allocation
 paths={
  'market':data_root/'data/processed/account_market_2023_2025.parquet',
  'panel':data_root/'data/processed/training_panel_v1.parquet',
  'pred':data_root/'results/rolling_signal_v1/e00_frozen_predictions_2023_2025.parquet',
  'etf_price':data_root/'data/processed/multi_asset_etf_trial_v1/four_etf_daily_prices.parquet',
  'etf_event':data_root/'data/processed/multi_asset_etf_trial_v1/corporate_actions.csv',
  'proxy_price':data_root/'data/processed/multi_asset_handoff_v2/daily_prices.parquet',
  'proxy_event':data_root/'data/processed/multi_asset_etf_trial_v1/corporate_actions.csv',
  'contract':data_root/'configs/multi_asset_etf_trial_v1.contract.json'}
 real.ROOT=data_root; real.SCRIPTS=code_root/'src'; real.MARKET_PATH=paths['market']; real.PANEL_PATH=paths['panel']; real.PRED_PATH=paths['pred']; real.PRICE_PATH=paths['etf_price']; real.EVENT_PATH=paths['etf_event']; real.OUT=output
 etfs.ROOT=data_root; etfs.CONTRACT_PATH=paths['contract']; etfs.PRICE_PATH=paths['etf_price']; etfs.EVENTS_READINESS_PATH=data_root/'results/multi_asset_etf_trial_v1/event_readiness.json'; etfs.PRODUCTS_PATH=data_root/'results/multi_asset_etf_trial_v1/products.json'
 primitives.ROOT=data_root; primitives.ETF_PARENT_CONTRACT=paths['contract']; primitives.ETF_PRICE=paths['etf_price']; primitives.ETF_EVENTS=paths['etf_event']
 account_backtest.ROOT=data_root; audit_account_inputs.ROOT=data_root; build_training_panel.ROOT=data_root; account_engine.ROOT=data_root; stock_engine.ROOT=data_root
 fast.ROOT=data_root; fast.PRICE=paths['proxy_price']; fast.EVENT=paths['proxy_event']
 allocation.ROOT=data_root; allocation.OUT=output; allocation.PRED=paths['pred']; allocation.PRICE=paths['proxy_price']; allocation.EVENT=paths['proxy_event']; allocation.E00_DAILY=data_root/'results/account_backtest_v1/E00_daily.csv'
 real.OUT=output/'accounts'
 return real,fast,allocation,paths

def ranking_map(path,calendar):
 pred=pd.read_parquet(path,columns=['signal_date','symbol','model','score']); pred['signal_date']=pd.to_datetime(pred.signal_date).dt.normalize(); pred['symbol']=pred.symbol.astype(str)
 pred=pred.loc[pred.model.eq('E00')&pred.signal_date.isin(calendar)]
 ranks={d:g.sort_values(['score','symbol'],ascending=[False,True],kind='stable').symbol.tolist() for d,g in pred.groupby('signal_date',observed=True)}
 if len(ranks)!=len(calendar): raise RuntimeError(f'E00 rankings cover {len(ranks)}/{len(calendar)} account dates')
 return ranks

def account_run(real, calendar, market, ranks, prices, events, lookup, returns, policy, start, end, mode, schedule=None, fixed=None, output=None):
 dummy=pd.DataFrame(index=calendar)
 daily,fills,metrics=real.run_mixed_account(ranks,mode,calendar,market,prices,events,lookup,policy,dummy,returns,
   fixed_target_weights=fixed,target_weight_schedule=schedule)
 period='2023_validation' if start.startswith('2023') else '2024_2025_exploratory'
 directory=output/'accounts'/policy/period/mode; directory.mkdir(parents=True,exist_ok=True)
 daily.to_csv(directory/'daily.csv',index=False); fills.to_csv(directory/'fills.csv',index=False)
 pd.DataFrame(daily.attrs['decisions']).to_csv(directory/'target_decisions.csv',index=False)
 pd.DataFrame(daily.attrs['etf_orders']).to_csv(directory/'etf_orders.csv',index=False)
 pd.DataFrame(daily.attrs['etf_events']).to_csv(directory/'etf_events.csv',index=False)
 pd.DataFrame(daily.attrs['stock_rejections']).to_csv(directory/'stock_rejections.csv',index=False)
 return {'period':period,'policy':policy,'mode':mode,'metrics':metrics,'path':directory.relative_to(output).as_posix()}

def main():
 ap=argparse.ArgumentParser(description='Reproduce frozen E00 stock ranking with RISK_B1 and equal-weight shared-cash accounts.')
 ap.add_argument('--data-root',default='数据'); ap.add_argument('--output-dir',default='本地复现输出'); ap.add_argument('--ranking-mode',choices=['frozen','retrain'],default='frozen')
 args=ap.parse_args(); package_root=Path(__file__).resolve().parents[1]; data_root=resolve(args.data_root,package_root); output=resolve(args.output_dir,package_root); output.mkdir(parents=True,exist_ok=True)
 code_root=Path(__file__).resolve().parent; real,fast,allocation,paths=configure(data_root,code_root,output)
 started=time.perf_counter(); pred_path=paths['pred']
 training_report=None
 if args.ranking_mode=='retrain':
  from train_e00 import run as train
  pred_path=train(data_root,output/'training')
  # Record score parity on the complete common E00 key set; ranking is produced from the fresh scores.
  frozen=pd.read_parquet(paths['pred'],columns=['signal_date','symbol','score']); fresh=pd.read_parquet(pred_path,columns=['signal_date','symbol','score'])
  frozen['signal_date']=pd.to_datetime(frozen.signal_date).dt.normalize(); fresh['signal_date']=pd.to_datetime(fresh.signal_date).dt.normalize()
  fresh=fresh.loc[fresh.signal_date.between('2023-01-01','2025-12-31')].copy()
  joined=frozen.merge(fresh,on=['signal_date','symbol'],suffixes=('_frozen','_retrained'),validate='one_to_one')
  frozen_full=pd.read_parquet(paths['pred'],columns=['signal_date','symbol']); fresh_keys=fresh[['signal_date','symbol']]
  key_equal=len(frozen)==len(fresh) and len(joined)==len(frozen) and not frozen_full.merge(fresh_keys,on=['signal_date','symbol'],how='left',indicator=True)._merge.ne('both').any()
  rank_equal=True
  for _,g in joined.groupby('signal_date',observed=True):
   if g.sort_values(['score_frozen','symbol'],ascending=[False,True],kind='stable').symbol.tolist()!=g.sort_values(['score_retrained','symbol'],ascending=[False,True],kind='stable').symbol.tolist(): rank_equal=False; break
  training_report={'common_rows':len(joined),'frozen_rows':len(frozen),'retrained_rows_2023_2025':len(fresh),'keys_complete_equal':bool(key_equal),'max_abs_score_difference':float(np.max(np.abs(joined.score_frozen-joined.score_retrained))),'rankings_exact':bool(rank_equal)}
 panel=fast.load_panel(); ret=panel[5]; idx=panel[1]; ret.index=idx
 assets=allocation.ASSETS; proxy=ret[assets].copy()
 results=[]
 for start,end in [('2023-01-01','2023-12-31'),('2024-01-01','2025-12-31')]:
  real.START=pd.Timestamp(start); real.END=pd.Timestamp(end); market,calendar=real.load_stock_market(); ranks=ranking_map(pred_path,calendar); prices,events,lookup=real.load_etf(); account_returns=proxy.reindex(calendar).ffill().fillna(0.)
  schedules=__import__('run_dynamic_risk_trend_v1').make_schedules(proxy,calendar)
  for mode in (['standard'] if start.startswith('2023') else ['standard','stress']):
   for policy in ['RISK_B1','FIXED25']:
    sch=schedules['RISK_B1'] if policy=='RISK_B1' else None; fixed=np.full(4,.25) if policy=='FIXED25' else None
    results.append(account_run(real,calendar,market,ranks,prices,events,lookup,account_returns,policy,start,end,mode,sch,fixed,output))
 summary={'status':'ACCOUNT_REPRODUCTION_COMPLETE','ranking_mode':args.ranking_mode,'training_parity':training_report,'results':results,
  'periods':'2023 standard validation; 2024-2025 standard/stress previously viewed exploration','inputs_sha256':{(str(p.relative_to(data_root)).replace('\\','/') if p.is_relative_to(data_root) else 'retrained_predictions/'+p.name):sha(p) for p in [paths['market'],paths['panel'],pred_path,paths['etf_price'],paths['etf_event'],paths['proxy_price']]},
  'limitations':['2024-2025 is not blind out-of-sample; 2026 was not run in this package workflow.','RISK_B1 risk signals use CSI500 proxy; executed equity ranks are E00; shared cash and lots/fees/events are handled by the account ledger.','Retrain mode refits monthly E00 rankings, not all research branches.','This package does not certify redistribution rights for source market data.'], 'elapsed_seconds':time.perf_counter()-started}
 (output/'reproduction_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
 print(json.dumps({'status':summary['status'],'ranking_mode':args.ranking_mode,'training_parity':training_report,'results':[{**{k:r[k] for k in ('period','policy','mode','path')},'metrics':{k:r['metrics'].get(k) for k in ('total_return','sharpe_zero_cash','max_drawdown','fills','gross_notional','fees','issues','nav_recon_max_abs')}} for r in results],'elapsed_seconds':summary['elapsed_seconds']},ensure_ascii=False,indent=2,default=str))
if __name__=='__main__': main()
