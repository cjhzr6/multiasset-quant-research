"""Synthetic teaching demo for the project's frozen risk-budget allocator.
All returns below are simulated; they are not historical performance or a strategy recommendation.
Run from GitHub项目: python examples/synthetic_risk_budget.py
"""
import sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"代码"/"src"))
import run_dynamic_risk_trend_v1 as allocator

rng=np.random.default_rng(20261105)
vol=np.array([.010,.014,.012,.003])
corr=np.array([[1,.45,.05,-.05],[.45,1,.08,-.10],[.05,.08,1,.02],[-.05,-.10,.02,1]])
cov=np.outer(vol,vol)*corr
returns=rng.multivariate_normal(np.zeros(4),cov,size=260)
# Trading-day index; only 20-session checkpoints are used by the existing allocator.
import pandas as pd
rets=pd.DataFrame(returns,index=pd.bdate_range("2024-01-02",periods=260),columns=allocator.ASSETS)
schedules=allocator.make_schedules(rets,rets.index)
for policy in ("RISK_B1","RISK_B2","TREND_FIXED_ETA05_RHO05"):
    d,w=next(reversed(schedules[policy].items()))
    assert np.isfinite(w).all() and np.all(w>=0) and np.all(w<=allocator.CAPS+1e-9)
    if policy.startswith("RISK_"):
        assert abs(w.sum()-1)<1e-8 and not np.allclose(w,allocator.W0,atol=1e-3)
    print(f"{policy} {d.date()} assets={allocator.ASSETS}")
    print("  initial_target="+", ".join(f"{x:.3f}" for x in allocator.W0))
    print("  final_target="+", ".join(f"{x:.3f}" for x in w)+f" total={w.sum():.3f}")
print("Synthetic returns only; no real market data or realised strategy return used.")
