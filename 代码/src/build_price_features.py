"""Price-only feature/label engineering on audited samples, not a strategy backtest."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ["ret1", "ret5", "ret20", "ret60", "ret120", "momentum20_skip5",
            "vol20", "vol60", "downside20", "range20", "ma20_distance", "ma60_distance",
            "volume_ratio5_20", "log_amount20", "illiquidity20", "beta60", "residual_vol60",
            "overnight20", "intraday20", "max_return20"]


def transform(frame, benchmark):
    """Inputs already aligned to the full exchange calendar, including missing stock bars."""
    data = frame.copy()
    price = data.close * data.adj_factor
    opening = data.open * data.adj_factor
    ret = price.pct_change(fill_method=None)
    market = benchmark.close.pct_change(fill_method=None)
    out = pd.DataFrame(index=data.index)
    for horizon in (1, 5, 20, 60, 120):
        out[f"ret{horizon}"] = price.pct_change(horizon, fill_method=None)
    out["momentum20_skip5"] = price.shift(5) / price.shift(20) - 1
    out["vol20"] = ret.rolling(20).std()
    out["vol60"] = ret.rolling(60).std()
    out["downside20"] = ret.clip(upper=0).pow(2).rolling(20).mean().pow(0.5)
    out["range20"] = ((data.high - data.low) / data.close).rolling(20).mean()
    for horizon in (20, 60):
        out[f"ma{horizon}_distance"] = price / price.rolling(horizon).mean() - 1
    out["volume_ratio5_20"] = data.volume.rolling(5).mean() / data.volume.rolling(20).mean()
    out["log_amount20"] = np.log1p(data.amount.rolling(20).mean())
    out["illiquidity20"] = (ret.abs() / data.amount.replace(0, np.nan)).rolling(20).mean()
    market_var = market.rolling(60).var()
    out["beta60"] = ret.rolling(60).cov(market) / market_var.replace(0, np.nan)
    out["residual_vol60"] = (ret.rolling(60).var() - out.beta60.pow(2) * market_var).clip(lower=0).pow(0.5)
    out["overnight20"] = (opening / price.shift(1) - 1).rolling(20).mean()
    out["intraday20"] = (price / opening - 1).rolling(20).mean()
    out["max_return20"] = ret.rolling(20).max()
    out = out.replace([np.inf, -np.inf], np.nan)
    dates = pd.Series(data.index, index=data.index)
    out["entry_date"] = dates.shift(-1)
    out["label_end_date"] = dates.shift(-6)
    # Fixed exchange sessions, not next six available stock observations.
    out["label_5d_adjusted"] = opening.shift(-6) / opening.shift(-1) - 1
    out["feature_ready"] = out[FEATURES].notna().all(axis=1) & data.is_suspended.eq(0)
    out["label_ready"] = out.label_5d_adjusted.notna() & data.is_suspended.shift(-1).eq(0) & data.is_suspended.shift(-6).eq(0)
    out.loc[~out.label_ready, "label_5d_adjusted"] = np.nan
    return out


def self_check():
    dates = pd.bdate_range("2020-01-01", periods=200)
    close = 100 * np.exp(np.arange(200) / 1000 + np.sin(np.arange(200)) / 100)
    frame = pd.DataFrame({"open": close * 0.999, "close": close, "high": close * 1.02,
                          "low": close * 0.98, "volume": 1000., "amount": 100000.,
                          "adj_factor": 1., "is_suspended": 0}, index=dates)
    baseline = transform(frame, frame)
    future_changed = frame.copy()
    future_changed.loc[dates[160]:, ["open", "close", "high", "low"]] *= 2
    altered = transform(future_changed, future_changed)
    pd.testing.assert_frame_equal(baseline.loc[:dates[159], FEATURES], altered.loc[:dates[159], FEATURES])
    assert np.isclose(baseline.loc[dates[150], "label_5d_adjusted"], frame.open.iloc[156] / frame.open.iloc[151] - 1)
    assert baseline.label_ready.iloc[-6:].eq(False).all()
    missing = frame.copy()
    missing.loc[dates[151], ["open", "close"]] = np.nan
    assert not transform(missing, frame).loc[dates[150], "label_ready"]
    print("PASS: future-price invariance, fixed-calendar label, maturity and missing-entry checks")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return
    records = json.loads((ROOT / "results/data_collection/sample_manifest.json").read_text(encoding="utf-8"))["records"]
    benchmark_record = next(r for r in records if r["job"] == {"kind": "bars", "symbol": "SHSE.000300"} and r["status"] == "PASS")
    benchmark = pd.read_parquet(ROOT / benchmark_record["file"])
    benchmark.index = pd.to_datetime(benchmark.eob).dt.tz_localize(None).dt.normalize()
    benchmark = benchmark.sort_index()
    tables = []
    for record in records:
        if record.get("status") != "PASS" or record["job"]["kind"] != "instruments":
            continue
        symbol = record["job"]["symbol"]
        bars_record = next(r for r in records if r["job"] == {"kind": "bars", "symbol": symbol} and r["status"] == "PASS")
        bars = pd.read_parquet(ROOT / bars_record["file"])
        metrics = pd.read_parquet(ROOT / record["file"])
        bars.index = pd.to_datetime(bars.eob).dt.tz_localize(None).dt.normalize()
        metrics.index = pd.to_datetime(metrics.trade_date).dt.tz_localize(None).dt.normalize()
        aligned = bars.reindex(benchmark.index).join(metrics[["adj_factor", "is_suspended"]], validate="one_to_one")
        output = transform(aligned, benchmark)
        output["symbol"] = symbol
        output.index.name = "signal_date"
        tables.append(output.reset_index())
    result = pd.concat(tables, ignore_index=True)
    target = ROOT / "data/processed/price_feature_engineering_sample.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(target, index=False)
    summary = {"purpose": "三只股票的数据工程样本；未训练、未回测、不是最终股票池。",
               "rows": len(result), "features": FEATURES, "feature_ready_rows": int(result.feature_ready.sum()),
               "label_ready_rows": int(result.label_ready.sum()), "file": str(target.relative_to(ROOT)),
               "label_note": "五个交易日复权开盘收益；不含交易成本，未扣基准，须审计公司行为口径。"}
    (ROOT / "results/price_feature_engineering.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"ENGINEERING_SAMPLE_READY rows={len(result)}, features={len(FEATURES)}")


if __name__ == "__main__":
    main()
