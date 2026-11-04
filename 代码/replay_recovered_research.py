"""Offline NLP refit or stage-account replay from recovered immutable input/actions."""
import argparse,sys,importlib,json,time
from pathlib import Path
import pandas as pd

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--archive-root',required=True)
    ap.add_argument('--output-dir',required=True)
    ap.add_argument('--experiment',choices=['nlp','stage','four-etf','stock-etf-stage','risk-allocation','fixed-sleeves','learned-allocation'],required=True)
    args=ap.parse_args();root=Path(args.archive_root).resolve();out=Path(args.output_dir).resolve()
    if out.exists():raise FileExistsError('Use a fresh output directory')
    out.mkdir(parents=True);sys.path.insert(0,str(root/'scripts'));started=time.perf_counter()
    if args.experiment=='risk-allocation':
        m=importlib.import_module('run_dynamic_risk_trend_v1')
        # ROOT is used only for output-path metadata; data loaders retain their
        # original archive root in imported modules.
        m.ROOT=out.parent;m.OUT=out;m.main()
        return
    if args.experiment=='fixed-sleeves':
        m=importlib.import_module('run_fixed_sleeves_complementarity_v1')
        # Original runner requires it to create a nonexistent output directory.
        out.rmdir();m.OUT=out;m.main()
        return
    if args.experiment=='learned-allocation':
        m=importlib.import_module('joint_stock_etf_fast_real_v1')
        m.OUT=out
        for cost in ['standard','stress']:
            sys.argv=['joint_stock_etf_fast_real_v1','--mode',cost];m.main()
        return
    if args.experiment=='nlp':
        m=importlib.import_module('run_macro_news_contrast_v1');m.OUT=out
        sys.argv=['run_macro_news_contrast_v1','--preflight-only'];m.main()
        sys.argv=['run_macro_news_contrast_v1'];m.main()
        return
    if args.experiment=='four-etf':
        m=importlib.import_module('multi_asset_etf_trial_v1')
        binding_path=root/'results/multi_asset_etf_trial_v1/execution_binding.json'
        prices,events,contract,_=m.load_and_validate_inputs(root/'data/processed/multi_asset_etf_trial_v1/corporate_actions.csv',binding_path)
        binding=json.loads(binding_path.read_text(encoding='utf-8'))
        binding['binding_file']=str(binding_path)
        m.RUN_OUT=out;m.CALENDAR_OUT=out/'common_calendar.csv'
        m.run_economic_accounts(prices,events,contract,binding)
        return
    if args.experiment=='stock-etf-stage':
        m=importlib.import_module('stock_etf_stage_trial_v1')
        contract=json.loads(m.CONTRACT.read_text(encoding='utf-8'))
        stocks,etfs,engine=m._runtime_modules()
        market,calendar,stock_events,unsupported,mergers=stocks.load_market_and_events()
        rankings,panel=stocks.build_rankings()
        ma=pd.read_parquet(stocks.PANEL,columns=['signal_date','symbol','ma60_distance'],filters=[('signal_date','>=',stocks.START),('signal_date','<=',stocks.END)])
        ma['symbol']=ma.symbol.astype(str)
        panel=panel.merge(ma,on=['signal_date','symbol'],how='left',validate='one_to_one')
        # Original full states file is absent. Reconstruct only the M/B fields
        # used by this rule from contemporaneous frozen predictor inputs.
        state_rows=[]
        for day,g in panel.groupby('signal_date',observed=True):
            values=pd.to_numeric(g.ma60_distance,errors='coerce')
            M=float(g.market_ma60_distance.iloc[0])
            state_rows.append({'signal_date':day,'M_csi300_ma60_distance':M,'B_pool_share_above_ma60':float(values.gt(0).sum()/values.notna().sum()),'B_coverage':float(values.notna().mean()),'M_present':bool(pd.notna(M)),'pool_size':len(g)})
        states=pd.DataFrame(state_rows).set_index('signal_date').sort_index()
        states.reset_index().to_parquet(out/'rebuilt_M_B_states.parquet',index=False)
        prices=pd.read_parquet(m.ETF_PRICE);prices.trade_date=pd.to_datetime(prices.trade_date)
        events=etfs.normalize_event_ledger(pd.read_csv(m.ETF_EVENTS,dtype={'asset_id':str}))
        factors={a:etfs._gross_factors_for_asset(prices,events,a) for a in etfs.ASSET_ORDER}
        inputs=(stocks,etfs,engine,market,calendar,stock_events,unsupported,mergers,rankings,panel,states,prices,events,factors)
        picks,outer=m._stock_state_schedule(calendar,states,contract)
        scores,score_table,invalid=m.build_causal_score_table(calendar,factors,contract,etfs)
        outer.to_csv(out/'outer_selection_log.csv',index=False);score_table.to_csv(out/'etf_scores.csv',index=False)
        results={};logs=[]
        for name,policy,cost in [('STAGE_STANDARD','stage','standard'),('FIXED_STANDARD','fixed','standard'),('STAGE_STRESS','stage','stress'),('FIXED_STRESS','fixed','stress')]:
            run=m.run_hybrid_account(name,policy,cost,inputs,contract,picks,scores,logs,out)
            results[name]=run['metrics']
        result={'status':'FOUR_ACCOUNTS_REPLAYED_FROM_REBUILT_M_B','original_full_state_parquet_restored':False,'state_source':'M and B rebuilt from original contemporaneous feature rows; no parameter search','accounts':results,'invalid_etf_fraction':invalid,'elapsed_seconds':time.perf_counter()-started}
        (out/'recovery_report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(result,ensure_ascii=False),flush=True)
        return
    m=importlib.import_module('run_strategy_switch_trial_v1')
    rankings,_=m.build_rankings();market,calendar,events,unsupported,mergers=m.load_market_and_events()
    comparators,_=m.run_comparators(out,rankings,market,calendar,events,unsupported,mergers)
    shadows=m.load_shadow_navs(out)
    history=m.run_history60(out,rankings,market,calendar,events,unsupported,mergers,shadows)
    # The original full state Parquet was not found in snapshots. Recover the
    # actually executed choice path from the contemporaneous switch inventory,
    # without inventing M/B/P values or claiming reconstruction of state features.
    log=pd.read_csv(root/'results/strategy_switch_trial_v1/selector_switch_inventory.csv',parse_dates=['decision_date'])
    selectors={};target=out/'selectors';target.mkdir()
    for tag,mode,name in [('MAIN_20D','standard','MAIN_20D_standard'),('MAIN_20D','stress','MAIN_20D_stress'),('SENSITIVITY_40D','standard','SENSITIVITY_40D_standard')]:
        changes=log.loc[log.account.eq(name)].sort_values('decision_date')
        assert not changes.empty,name
        changes_by_day={d:g.iloc[0] for d,g in changes.groupby('decision_date')}
        current=changes.iloc[0].previous_strategy;schedule={}
        for day in calendar:
            if day in changes_by_day:
                row=changes_by_day[day];assert current==row.previous_strategy
                current=row.new_strategy
            schedule[day]=current
        _,selectors[name]=m.run_account(tag,rankings,market,calendar,events,unsupported,mergers,mode,schedule,target)
    result={'status':'ACCOUNT_REPLAY_COMPLETE','scope':'comparators and history selector recalculated; stage selector uses recovered original action path, NOT rebuilt original state features','full_original_state_parquet_restored':False,'sessions':len(calendar),'comparators':comparators,'history60':history,'selectors':selectors,'elapsed_seconds':time.perf_counter()-started}
    original=json.loads((root/'results/strategy_switch_trial_v1/execution_audit.json').read_text(encoding='utf-8'))['accounts']
    comparisons={}
    for group in [comparators,history,selectors]:
        for name,item in group.items():
            if name in original:
                metrics=item.get('metrics',item)
                comparisons[name]={k:abs(float(metrics[k])-float(original[name][k])) for k in ['total_return','sharpe_zero_cash','max_drawdown'] if k in metrics}
    result['old_metrics_comparison']=comparisons
    (out/'recovery_replay.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
    print(json.dumps({'status':result['status'],'elapsed_seconds':result['elapsed_seconds'],'comparison':comparisons},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
