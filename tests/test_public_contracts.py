from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"代码"/"src"))
import run_joint_stock_etf_allocation_search_fast_v1 as base
import run_dynamic_risk_trend_v1 as risk

def test_bounded_simplex_respects_sum_and_caps():
    caps=np.array([.8,.7,.6,.6]); w=base.project_bounded_simplex(np.array([4.,3.,2.,1.]),1.,caps)
    assert np.all(w>=-1e-10) and np.all(w<=caps+1e-10)
    assert abs(w.sum()-1.)<1e-9

def test_risk_budget_schedule_has_frozen_cadence_and_valid_weights():
    import pandas as pd
    rng=np.random.default_rng(7); cols=base.ASSETS
    rets=pd.DataFrame(rng.normal(0,[.01,.015,.012,.003],size=(100,4)),index=pd.bdate_range("2024-01-02",periods=100),columns=cols)
    out=risk.make_schedules(rets,rets.index)
    for policy in ("RISK_B1","RISK_B2","TREND_FIXED_ETA05_RHO05"):
        assert len(out[policy])==5
        for w in out[policy].values():
            assert np.isfinite(w).all() and (w>=-1e-9).all() and (w<=np.array([.8,.7,.6,.6])+1e-9).all()
            if policy.startswith("RISK_"):
                assert abs(w.sum()-1.)<1e-8
            else:
                assert w.sum()<=1.+1e-8  # residual weight is intentional cash
