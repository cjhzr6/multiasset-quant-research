"""Frozen N03 weights -> CPU NumPy inference -> original ETF account engine.

No fitting/Torch installation. Restricted checkpoint decoder supports only the
plain tensor state dictionaries used in this project, not arbitrary pickles.
"""
import argparse,ast,sys,json,pickle,zipfile,io,collections,hashlib,time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import erf

def load_plain_tensor_state(path):
    with zipfile.ZipFile(path) as z:
        names=z.namelist();entry=next(n for n in names if n.endswith('/data.pkl'));prefix=entry[:-len('data.pkl')]
        arrays={}
        def rebuild(storage,offset,shape,strides,*ignored):
            return np.ndarray(shape=tuple(shape),dtype=storage.dtype,buffer=storage,offset=offset*storage.dtype.itemsize,strides=tuple(s*storage.dtype.itemsize for s in strides)).copy()
        class Reader(pickle.Unpickler):
            def find_class(self,module,name):
                if (module,name)==('collections','OrderedDict'):return collections.OrderedDict
                if module=='torch._utils' and name in {'_rebuild_tensor','_rebuild_tensor_v2'}:return rebuild
                if module=='torch' and name in {'FloatStorage','DoubleStorage','LongStorage'}:return {'FloatStorage':np.dtype('<f4'),'DoubleStorage':np.dtype('<f8'),'LongStorage':np.dtype('<i8')}[name]
                raise pickle.UnpicklingError(f'Unsupported checkpoint global {module}.{name}')
            def persistent_load(self,pid):
                kind,dtype,key,device,count=pid
                if kind!='storage':raise pickle.UnpicklingError('Not plain storage')
                if key not in arrays:
                    arrays[key]=np.frombuffer(z.read(prefix+'data/'+key),dtype=dtype).copy()
                    assert len(arrays[key])==count
                return arrays[key]
        state=Reader(io.BytesIO(z.read(entry))).load()
        assert all(isinstance(v,np.ndarray) for v in state.values())
        return state

def forward(x,w,attention):
    def linear(a,p):return a@w[p+'.weight'].T+w[p+'.bias']
    def norm(a,p):return (a-a.mean(-1,keepdims=True))/np.sqrt(a.var(-1,keepdims=True)+np.float32(1e-5))*w[p+'.weight']+w[p+'.bias']
    def gelu(a):return np.float32(.5)*a*(1+erf(a/np.float32(np.sqrt(2.))))
    h=linear(x,'encoder')
    if attention:
        q,k,v=np.split(h@w['attn.in_proj_weight'].T+w['attn.in_proj_bias'],3,axis=-1)
        def split(a):return a.reshape(len(a),4,2,16).transpose(0,2,1,3)
        q,k,v=split(q),split(k),split(v)
        logits=(q/np.float32(4.))@k.transpose(0,1,3,2)
        logits-=logits.max(-1,keepdims=True);prob=np.exp(logits);prob/=prob.sum(-1,keepdims=True)
        a=(prob@v).transpose(0,2,1,3).reshape(len(x),4,32)
        h=norm(h+linear(a,'attn.out_proj'),'norm1')
        h=norm(h+linear(gelu(linear(h,'ff.0')),'ff.2'),'norm2')
    else:h=linear(gelu(linear(h,'ff.0')),'ff.2')
    return linear(h,'head').squeeze(-1)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--archive-root',required=True);ap.add_argument('--team-root',required=True);ap.add_argument('--output-dir',required=True);args=ap.parse_args()
    root=Path(args.archive_root).resolve();team=Path(args.team_root).resolve();out=Path(args.output_dir).resolve()
    if out.exists():raise FileExistsError('Use fresh output directory')
    out.mkdir(parents=True);started=time.perf_counter()
    # Reuse only the reviewed pure feature/sample functions from the original file.
    source=team/'代码/n03_cross_asset_attention_v1.py'
    tree=ast.parse(source.read_text(encoding='utf-8-sig'))
    body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in {'gross_factors','build_features','make_samples'}]
    ns={'np':np,'pd':pd,'ASSET_ORDER':['csi300','sp500_qdii','physical_gold','rmb_gov_bond'],'HORIZON':20}
    exec(compile(ast.Module(body=body,type_ignores=[]),str(source),'exec'),ns)
    price_path=root/'data/processed/multi_asset_etf_trial_v1/four_etf_daily_prices.parquet'
    event_path=root/'data/processed/multi_asset_etf_trial_v1/corporate_actions.csv'
    prices=pd.read_parquet(price_path);prices.trade_date=pd.to_datetime(prices.trade_date)
    raw_events=pd.read_csv(event_path,dtype={'asset_id':str})
    X,y,split,meta=ns['make_samples'](ns['build_features'](prices,raw_events))
    train=X[split['train']].reshape(-1,X.shape[-1]);mu=train.mean(0);sd=train.std(0);sd=np.where(sd==0,1.,sd)
    inputs=((X[split['eval']]-mu)/sd).astype(np.float32)
    dates=np.asarray(meta['dates'])[split['eval']].tolist()
    sys.path.insert(0,str(root/'scripts'));sys.path.insert(0,str(team/'代码'))
    import n03_cross_asset_accounts_v1 as accounts
    accounts.PRICE_PATH=price_path;accounts.EVENT_PATH=event_path
    events=accounts.prepare_events()
    wide=prices.pivot(index='trade_date',columns='asset_id',values='close_cny').dropna(subset=accounts.ASSET_ORDER)
    cal=wide.index[(wide.index>='2024-01-01')&(wide.index<='2025-12-31')]
    ckpt_manifest=json.loads((team/'模型权重/N03/manifest.json').read_text(encoding='utf-8'))['models']
    predictions={};weights_sha={}
    for name in ckpt_manifest:
        path=team/'模型权重/N03'/f'{name}.pt';actual=hashlib.sha256(path.read_bytes()).hexdigest();assert actual==ckpt_manifest[name]['sha256']
        weights_sha[name]=actual;predictions[name]=forward(inputs,load_plain_tensor_state(path),name.startswith('ATTN'))
    records=[]
    for name in ['static_25pct']+list(predictions):
        values=np.ones((len(dates),4),dtype=np.float32) if name=='static_25pct' else predictions[name]
        by_date={d:{a:float(values[j,k]) for k,a in enumerate(accounts.ASSET_ORDER)} for j,d in enumerate(dates)}
        for cost in ['standard','stress']:
            ledger,orders,fills=accounts.simulate_account(prices,events,cal,cost,by_date)
            dest=out/name/cost;dest.mkdir(parents=True);ledger.to_csv(dest/'daily.csv',index=False);fills.to_csv(dest/'fills.csv',index=False)
            ref=team/'结果/n03_daily'/f'{name}_{cost}_daily.csv'
            old=pd.read_csv(ref);assert ledger.date.astype(str).tolist()==old.date.astype(str).tolist()
            diff=float(np.max(np.abs(ledger.nav_cny.to_numpy()-old.nav_cny.to_numpy())))
            records.append({'strategy':name,'cost':cost,'sessions':len(ledger),'max_nav_difference':diff,'matches_original_ledger':diff<1e-5})
    np.savez(out/'recovered_predictions.npz',**predictions)
    result={'status':'ALL_ORIGINAL_ACCOUNT_NAVS_MATCH' if all(r['matches_original_ledger'] for r in records) else 'DIFFERENCES_REQUIRE_REVIEW','backend':'NumPy CPU inference of saved weights; no retraining; original prediction arrays not recovered','models':6,'accounts':records,'weights_sha256':weights_sha,'elapsed_seconds':time.perf_counter()-started,'scaler_rebuilt_from_original_training_split_only':True}
    (out/'recovery_report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(result,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
