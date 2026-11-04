from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def resolve(path: str, base: Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (base / value).resolve()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay the frozen 2026 E00 plus multi-asset accounts."
    )
    parser.add_argument("--data-root", default="数据")
    parser.add_argument("--output-dir", default="本地2026复现输出")
    args = parser.parse_args()

    code_root = Path(__file__).resolve().parent
    package_root = code_root.parent
    data_root = resolve(args.data_root, package_root)
    output = resolve(args.output_dir, package_root)
    output.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(code_root / "src"))
    import joint_stock_etf_fast_real_v1 as real
    import run_joint_stock_etf_fast_v1 as fast
    import run_dynamic_risk_trend_v1 as risk
    from fees_2026_v1 import fee_components

    paths = {
        "market": data_root / "data/processed/sealed_2026/account_market_2026_candidate.parquet",
        "predictions": data_root / "results/e00_2026_final_v1/e00_predictions_2026.parquet",
        "events": data_root / "results/sealed_2026_event_inputs_v1/canonical_events.parquet",
        "etf_prices": data_root / "data/processed/multi_asset_etf_trial_v1/four_etf_daily_prices.parquet",
        "etf_events": data_root / "data/processed/multi_asset_etf_trial_v1/corporate_actions.csv",
        "proxy_prices": data_root / "data/processed/multi_asset_handoff_v2/daily_prices.parquet",
        "proxy_events": data_root / "data/processed/multi_asset_etf_trial_v1/corporate_actions.csv",
        "contract": data_root / "configs/multi_asset_etf_trial_v1.contract.json",
    }
    missing = [str(p) for p in paths.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing packaged inputs: " + ", ".join(missing))

    real.ROOT = data_root
    real.START = pd.Timestamp("2026-01-01")
    real.END = pd.Timestamp("2026-09-21")
    real.MARKET_PATH = paths["market"]
    real.PRICE_PATH = paths["etf_prices"]
    real.EVENT_PATH = paths["etf_events"]
    real.engine.fee_components = fee_components

    events = pd.read_parquet(paths["events"])
    for col in ("ex_date", "record_date", "pay_date", "share_date"):
        if col in events:
            events[col] = pd.to_datetime(events[col])
    if events.unresolved.fillna(True).any():
        raise RuntimeError("Canonical corporate events contain unresolved rows")
    if events.duplicated(["symbol", "ex_date"]).any():
        raise RuntimeError("Canonical corporate events contain duplicate keys")
    if events.rights_ratio.fillna(0).gt(0).any():
        raise RuntimeError("Rights issues are outside this replay contract")
    real.prepare_events = lambda calendar: (
        events.copy(), pd.DataFrame(columns=["trade_date", "symbol"])
    )

    predictions = pd.read_parquet(paths["predictions"])
    if "signal_date" not in predictions:
        predictions = predictions.rename(columns={"trade_date": "signal_date"})
    predictions["signal_date"] = pd.to_datetime(predictions.signal_date).dt.normalize()
    if predictions.duplicated(["signal_date", "symbol"]).any():
        raise RuntimeError("Frozen E00 predictions contain duplicate date-symbol keys")

    market, calendar = real.load_stock_market()
    ranks = {
        date: group.sort_values(
            ["score", "symbol"], ascending=[False, True], kind="stable"
        ).symbol.astype(str).tolist()
        for date, group in predictions.groupby("signal_date", sort=True)
    }
    if set(calendar) != set(ranks):
        raise RuntimeError(f"Prediction calendar mismatch: {len(ranks)} scores for {len(calendar)} sessions")

    prices, etf_events, lookup = real.load_etf()
    fast.ROOT = data_root
    fast.PRICE = paths["proxy_prices"]
    fast.EVENT = paths["proxy_events"]
    proxy = fast.load_panel()[5]
    if not all(pd.Timestamp(day) in proxy.index for day in calendar):
        raise RuntimeError("Risk proxy does not cover the 2026 account calendar")
    if not all((asset, day) in lookup for asset in real.etfs.ASSET_ORDER for day in calendar):
        raise RuntimeError("ETF prices do not cover all account assets and sessions")

    schedules = risk.make_schedules(proxy, calendar)
    started = time.perf_counter()
    results = []
    for mode in ("standard", "stress"):
        for policy in ("RISK_B1", "FIXED25"):
            daily, fills, metrics = real.run_mixed_account(
                ranks,
                mode,
                calendar,
                market,
                prices,
                etf_events,
                lookup,
                policy,
                pd.DataFrame(index=calendar),
                proxy.reindex(calendar),
                fixed_target_weights=np.full(4, 0.25) if policy == "FIXED25" else None,
                target_weight_schedule=schedules["RISK_B1"] if policy == "RISK_B1" else None,
            )
            directory = output / policy / mode
            directory.mkdir(parents=True, exist_ok=True)
            daily.to_csv(directory / "daily.csv", index=False)
            fills.to_csv(directory / "fills.csv", index=False)
            for key in ("decisions", "etf_orders", "etf_events", "stock_rejections"):
                pd.DataFrame(daily.attrs[key]).to_csv(directory / f"{key}.csv", index=False)
            nav_error = float(
                (daily.nav - (daily.cash + daily.market_value + daily.dividend_receivable
                              - daily.estimated_dividend_tax)).abs().max()
            )
            fee_error = abs(float(pd.to_numeric(fills.fee).sum()) - metrics["fees"])
            checks = {
                "nav_error": nav_error,
                "fee_error": fee_error,
                "fills_match": len(fills) == metrics["fills"],
                "cash_min": float(daily.cash.min()),
                "issues": metrics["issues"],
            }
            checks["pass"] = bool(
                nav_error < 1e-6 and fee_error < 0.01 and checks["fills_match"]
                and checks["cash_min"] >= -1e-6 and metrics["issues"] == 0
            )
            results.append({"policy": policy, "mode": mode, "metrics": metrics, "checks": checks})

    summary = {
        "status": "PASS" if all(row["checks"]["pass"] for row in results) else "REVIEW_REQUIRED",
        "period": [str(calendar[0].date()), str(calendar[-1].date())],
        "sessions": len(calendar),
        "initial_capital_each": 1_000_000,
        "results": results,
        "inputs_sha256": {name: sha256(path) for name, path in paths.items()},
        "monthly_model_files": sorted(p.name for p in (data_root / "models/E00_2026").glob("*.txt")),
        "elapsed_seconds": time.perf_counter() - started,
        "limitations": [
            "Fresh 2026 capital; holdings are not inherited from 2025.",
            "Observation ends 2026-09-21 and is not a full calendar year.",
            "Frozen monthly E00 scores are replayed; this entry does not refit models.",
            "2026 results are a previously viewed continuation, not a blind test.",
            "The replay applies the packaged 2026 fee adapter and frozen risk schedule.",
        ],
    }
    (output / "reproduction_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps({
        "status": summary["status"],
        "period": summary["period"],
        "sessions": summary["sessions"],
        "accounts": [{
            "policy": row["policy"], "mode": row["mode"],
            "return": row["metrics"]["total_return"],
            "sharpe_zero_cash": row["metrics"]["sharpe_zero_cash"],
            "max_drawdown": row["metrics"]["max_drawdown"],
            "checks": row["checks"],
        } for row in results],
        "elapsed_seconds": summary["elapsed_seconds"],
    }, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
