from __future__ import annotations
import argparse,json
from pathlib import Path
import pandas as pd

def resolve(p,base):
 p=Path(p); return p if p.is_absolute() else (base/p).resolve()
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--data-root',default='数据'); ap.add_argument('--output-dir',default='本地复现输出'); a=ap.parse_args()
 root=Path(__file__).resolve().parents[1]; data=resolve(a.data_root,root); out=resolve(a.output_dir,root)
 ref=json.loads((data/'results/expected_core_accounts.json').read_text(encoding='utf-8'))['accounts']; summary=json.loads((out/'reproduction_summary.json').read_text(encoding='utf-8')); checks=[]
 for row in summary['results']:
  key='|'.join((row['policy'],row['period'],row['mode'])); expected=ref.get(key)
  if expected is None: checks.append({'key':key,'pass':False,'error':'no frozen reference'}); continue
  d=pd.read_csv(out/row['path']/'daily.csv'); f=pd.read_csv(out/row['path']/'fills.csv'); m=row['metrics']
  nav_error=float((d.nav-(d.cash+d.market_value+d.dividend_receivable-d.estimated_dividend_tax)).abs().max())
  comparisons={}
  for k,tol in [('total_return',1e-8),('sharpe_zero_cash',1e-8),('max_drawdown',1e-8),('fees',.01),('gross_notional',.01)]:
   if k not in expected: continue
   comparisons[k]={'actual':float(m[k]),'expected':float(expected[k]),'abs_diff':abs(float(m[k])-float(expected[k])),'tolerance':tol,'pass':abs(float(m[k])-float(expected[k]))<=tol}
  checks.append({'key':key,'sessions':len(d),'fills_actual':len(f),'fills_metric':m['fills'],'fees_actual':float(pd.to_numeric(f.fee,errors='coerce').fillna(0).sum()),'fees_metric':m['fees'],'gross_actual':float(pd.to_numeric(f.notional,errors='coerce').fillna(0).sum()),'gross_metric':m['gross_notional'],'nav_error':nav_error,'cash_min':float(d.cash.min()),'issues':m['issues'],'comparisons':comparisons,
   'pass':bool(len(d)==expected['sessions'] and len(f)==m['fills'] and abs(float(pd.to_numeric(f.fee,errors='coerce').fillna(0).sum())-m['fees'])<.01 and abs(float(pd.to_numeric(f.notional,errors='coerce').fillna(0).sum())-m['gross_notional'])<.01 and nav_error<1e-6 and d.cash.min()>=-1e-6 and m['issues']==0 and all(x['pass'] for x in comparisons.values()))})
 result={'status':'PASS' if checks and all(x['pass'] for x in checks) else 'FAIL','checked_accounts':len(checks),'checks':checks}
 (out/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
