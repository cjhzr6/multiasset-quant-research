from __future__ import annotations
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb

FEATURES=['ret1','ret5','ret20','ret60','ret120','momentum20_skip5','vol20','vol60','downside20','range20','ma20_distance','ma60_distance','volume_ratio5_20','log_amount20','illiquidity20','beta60','residual_vol60','overnight20','intraday20','max_return20']
PARAMS=dict(objective='regression',num_leaves=15,learning_rate=.05,min_data_in_leaf=100,lambda_l2=5.0,seed=20261006,num_threads=4,verbosity=-1,deterministic=True,force_col_wise=True)

def training_mask(panel,month,cutoff):
 return (panel.signal_date.ge(month-pd.DateOffset(months=36)) & panel.signal_date.lt(month) & panel.prediction_ready & panel.label_ready & panel.label_end_date.le(cutoff))

def run(data_root:Path,output:Path):
 t0=time.perf_counter(); output.mkdir(parents=True,exist_ok=True); models=output/'models'; models.mkdir(exist_ok=True)
 p=data_root/'data/processed/training_panel_v1.parquet'
 columns=['signal_date','symbol',*FEATURES,'entry_date','label_end_date','label_5d_adjusted','feature_ready','label_ready','prediction_ready']
 panel=pd.read_parquet(p,columns=columns); panel=panel.loc[panel.signal_date.lt('2026-01-01')].copy()
 # Do not let a label whose endpoint lies in 2026 enter the historical training or evaluation file.
 sealed=panel.label_end_date.ge('2026-01-01'); panel.loc[sealed,'label_5d_adjusted']=np.nan; panel.loc[sealed,'label_ready']=False
 predictions=[]; audits=[]
 periods=pd.date_range('2020-01-01','2025-12-01',freq='MS')
 for n,month in enumerate(periods,1):
  prev=panel.loc[panel.signal_date.lt(month),'signal_date']; cutoff=prev.max()
  train=panel.loc[training_mask(panel,month,cutoff)]
  test=panel.loc[panel.signal_date.ge(month)&panel.signal_date.lt(month+pd.DateOffset(months=1))&panel.prediction_ready]
  if len(train)<=10000 or len(test)==0 or train.label_end_date.max()>cutoff or cutoff>=test.signal_date.min(): raise RuntimeError(f'causal data gate failed {month.date()}')
  target=train.groupby('signal_date',observed=True).label_5d_adjusted.rank(pct=True)-.5
  weights=1/train.groupby('signal_date',observed=True).symbol.transform('size'); weights=weights/weights.mean()
  model=lgb.train(PARAMS,lgb.Dataset(train[FEATURES],label=target,weight=weights),num_boost_round=100)
  model_path=models/f'E00_{month:%Y%m}.txt'; model_path.write_text(model.model_to_string(),encoding='utf-8')
  out=test[['signal_date','symbol','entry_date','label_end_date','label_ready','label_5d_adjusted']].copy(); out['score']=model.predict(test[FEATURES]); out['model']='E00'; predictions.append(out)
  audits.append({'month':str(month.date()),'cutoff':str(cutoff.date()),'train_rows':len(train),'train_first':str(train.signal_date.min().date()),'train_last':str(train.signal_date.max().date()),'max_label_end':str(train.label_end_date.max().date()),'prediction_rows':len(test),'model_sha256':hashlib.sha256(model_path.read_bytes()).hexdigest()})
  print(f'TRAIN {n}/{len(periods)} {month:%Y-%m}',flush=True)
 pred=pd.concat(predictions,ignore_index=True); pred.to_parquet(output/'e00_predictions_2020_2025.parquet',index=False,compression='zstd')
 config={'features':FEATURES,'params':PARAMS,'boosting_rounds':100,'window_months':36,'fit_schedule':'monthly 2020-01 through 2025-12','target':'within-date percentile rank of 5-session adjusted opening-price return minus 0.5','date_weight':'each training date equal total weight','cutoff':'last globally available signal date before test month; train label_end_date <= cutoff','sealed_2026_evaluated':False}
 (output/'training_config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
 report={'status':'E00_CAUSAL_TRAINING_COMPLETE','completed_months':len(audits),'fitted_models':len(audits),'prediction_rows':len(pred),'panel_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'elapsed_seconds':time.perf_counter()-t0,'monthly_audit':audits}
 (output/'training_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps({k:v for k,v in report.items() if k!='monthly_audit'},ensure_ascii=False,indent=2))
 return output/'e00_predictions_2020_2025.parquet'

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--data-root',type=Path,required=True); ap.add_argument('--output-dir',type=Path,required=True); a=ap.parse_args(); run(a.data_root,a.output_dir)
if __name__=='__main__': main()
