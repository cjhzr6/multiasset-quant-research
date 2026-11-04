"""Isolated stock/ETF stage-trial scaffold; economic runs require a frozen binding.

The stage contract is deliberately external. This module only defines shared
portfolio accounting primitives and preflight checks; it does not fit, download,
or run an economic evaluation without a coordinator-bound contract and inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass, field
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs" / "stock_etf_stage_trial_v1.contract.json"
OUT = ROOT / "results" / "stock_etf_stage_trial_v1"
RUN_OUT = OUT / "account_run"
STOCK_HIST = ROOT / "data" / "processed" / "account_market_2020_2025.parquet"
STOCK_2026 = ROOT / "data" / "processed" / "sealed_2026" / "account_market_2026_candidate.parquet"
ETF_PRICE = ROOT / "data" / "processed" / "multi_asset_etf_trial_v1" / "four_etf_daily_prices.parquet"
ETF_EVENTS = ROOT / "data" / "processed" / "multi_asset_etf_trial_v1" / "corporate_actions.csv"
ETF_PRODUCTS = ROOT / "results" / "multi_asset_etf_trial_v1" / "products.json"
ETF_PARENT_BINDING = ROOT / "results" / "multi_asset_etf_trial_v1" / "execution_binding.json"
ETF_PARENT_CONTRACT = ROOT / "configs" / "multi_asset_etf_trial_v1.contract.json"
STATE_FILE = ROOT / "results" / "strategy_switch_trial_v2" / "states_daily_v2.parquet"


class GateError(ValueError):
    pass


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


@dataclass
class HybridAccount:
    """One real inventory/cash ledger shared by stock and ETF sleeves."""

    cash: float
    units: dict[tuple[str, str], int] = field(default_factory=dict)
    receivables: dict[str, float] = field(default_factory=dict)

    def nav(self, close: Mapping[tuple[str, str], float]) -> float:
        missing = [key for key, qty in self.units.items() if qty and key not in close]
        if missing:
            raise GateError(f"missing close marks for held assets: {missing}")
        if self.cash < -1e-7 or any(q < 0 for q in self.units.values()):
            raise GateError("long-only account has negative cash or inventory")
        return (self.cash + sum(self.receivables.values())
                + sum(qty * float(close[key]) for key, qty in self.units.items()))

    def target_deltas(self, target_units: Mapping[tuple[str, str], int]) -> list[dict]:
        """Return net deltas against owned units; no sleeve NAV transfer/splice."""
        keys = set(self.units) | set(target_units)
        orders = []
        for asset_class, symbol in sorted(keys):
            current = int(self.units.get((asset_class, symbol), 0))
            target = int(target_units.get((asset_class, symbol), 0))
            if current < 0 or target < 0:
                raise GateError("short positions/targets are outside this scaffold")
            delta = target - current
            if delta:
                orders.append({"asset_class": asset_class, "symbol": symbol,
                               "side": "BUY" if delta > 0 else "SELL",
                               "quantity": abs(delta), "current_units": current,
                               "target_units": target})
        # Actual execution adapters must process sells before buys and debit/credit
        # this same cash balance. The order here is deterministic for auditability.
        return sorted(orders, key=lambda x: (x["side"] != "SELL", x["asset_class"], x["symbol"]))


def require_run_binding(contract_path: Path, binding_path: Path) -> dict:
    if not contract_path.is_file():
        raise GateError(f"frozen stage contract not found: {contract_path}")
    if not binding_path.is_file():
        raise GateError(f"coordinator run binding not found: {binding_path}")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    if contract.get("status") != "FROZEN_FOR_RUN":
        raise GateError("stage contract is not frozen for an economic run")
    expected = {"status": "BOUND_FOR_RUN", "contract_sha256": sha256_file(contract_path),
                "runner_sha256": sha256_file(Path(__file__).resolve())}
    mismatch = {key: (binding.get(key), value) for key, value in expected.items()
                if binding.get(key) != value}
    if mismatch:
        raise GateError(f"binding does not match frozen inputs/code: {mismatch}")
    accounts = contract.get("budget", {}).get("accounts", [])
    if not accounts or len(accounts) > 4 or len(set(accounts)) != len(accounts):
        raise GateError("contract must bind one to four unique evaluation accounts")
    if contract.get("budget", {}).get("model_fits", 0) != 0:
        raise GateError("this bounded stage trial does not authorize model fitting")
    return {"contract": contract, "binding": binding}


def smoke_check() -> dict:
    acct = HybridAccount(cash=100.0,
                         units={("stock", "AAA"): 2, ("etf", "ETF1"): 10},
                         receivables={"ETF1-DIV-1": 3.0})
    close = {("stock", "AAA"): 20.0, ("etf", "ETF1"): 5.0}
    nav = acct.nav(close)
    orders = acct.target_deltas({("stock", "BBB"): 1, ("etf", "ETF1"): 10})
    assert nav == 193.0
    assert [o["side"] for o in orders] == ["SELL", "BUY"]
    assert {(o["asset_class"], o["symbol"], o["quantity"]) for o in orders} == {
        ("stock", "AAA", 2), ("stock", "BBB", 1)}
    _, etfs, _ = _runtime_modules()
    from types import SimpleNamespace
    dates = pd.bdate_range("2025-01-01", periods=21)
    lookup = {}
    for asset in etfs.ASSET_ORDER:
        for idx, day in enumerate(dates):
            px = 5.0 if asset == "csi300" and idx == 20 else 10.0
            lookup[(asset, pd.Timestamp(day))] = SimpleNamespace(
                open_cny=px + .001, close_cny=px, high_cny=px + .1, low_cny=px - .1,
                volume_lots=1000.0)
    conv = {"event_id": "SMOKE-CONV", "asset_id": "csi300", "event_type": "unit_conversion",
            "ex_date": dates[-1], "cash_per_old_unit": 0.0, "new_units_per_old_unit": 2.0}
    synthetic = SimpleNamespace(fund_units={a: 0 for a in etfs.ASSET_ORDER},
                                fund_buy_date={a: None for a in etfs.ASSET_ORDER})
    template = _etf_order_templates(
        dates[-1], {"csi300": 10000, "sp500_qdii": 0, "physical_gold": 0, "rmb_gov_bond": 0},
        synthetic, lookup, dates, 20, [conv], [conv], "standard", {}, etfs)
    csi = next(x for x in template if x["asset_id"] == "csi300")
    assert csi["status"] == "READY" and csi["planned_units"] == 2000, csi
    return {"status": "SCAFFOLD_SMOKE_PASS", "nav_includes_cash_positions_receivables": nav,
            "net_orders": orders, "event_day_order_status": csi["status"],
            "converted_prior20_capacity_units": csi["planned_units"], "economic_run": False}


def _runtime_modules():
    scripts = ROOT / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    import run_strategy_switch_trial_v1 as stock_src
    import multi_asset_etf_trial_v1 as etf_src
    import stock_etf_stage_account_engine_v1 as stock_engine
    from audit_account_inputs import fee_components as fee_2025
    from fees_2026_v1 import fee_components as fee_2026

    def cross_year_fees(amount, side, day):
        day = pd.Timestamp(day)
        if pd.Timestamp("2025-01-01") <= day < pd.Timestamp("2026-01-01"):
            return fee_2025(amount, side, day)
        if pd.Timestamp("2026-01-01") <= day <= pd.Timestamp("2026-09-21"):
            return fee_2026(amount, side, day)
        raise GateError(f"unsupported stock fee date {day.date()}")

    stock_engine.fee_components = cross_year_fees
    return stock_src, etf_src, stock_engine


def _sources() -> dict[str, Path]:
    return {
        "contract": CONTRACT,
        "stock_engine_clone": ROOT / "scripts" / "stock_etf_stage_account_engine_v1.py",
        "runner": Path(__file__).resolve(),
        "stock_engine_reference": ROOT / "scripts" / "account_engine_2026_v1.py",
        "stock_fee_2025": ROOT / "scripts" / "audit_account_inputs.py",
        "stock_fee_2026": ROOT / "scripts" / "fees_2026_v1.py",
        "stock_runner_reference": ROOT / "scripts" / "run_strategy_switch_trial_v1.py",
        "stock_inputs_2020_25": STOCK_HIST,
        "stock_inputs_2026": STOCK_2026,
        "stock_panel": ROOT / "data" / "processed" / "training_panel_v1.parquet",
        "stock_predictions_2020_25": ROOT / "results" / "rolling_signal_v1" / "predictions_2020_2025.parquet",
        "stock_predictions_2026": ROOT / "results" / "e00_2026_final_v1" / "e00_predictions_2026.parquet",
        "stock_events_2026": ROOT / "results" / "sealed_2026_event_inputs_v1" / "canonical_events.parquet",
        "stock_event_review_2026": ROOT / "results" / "sealed_2026_final_event_review_v1" / "report_v2.json",
        "stock_states": STATE_FILE,
        "etf_parent_contract": ETF_PARENT_CONTRACT,
        "etf_parent_binding": ETF_PARENT_BINDING,
        "etf_prices": ETF_PRICE,
        "etf_events": ETF_EVENTS,
        "etf_products": ETF_PRODUCTS,
        "etf_event_readiness": ROOT / "results" / "multi_asset_etf_trial_v1" / "event_readiness.json",
        "calendar": ROOT / "results" / "multi_asset_etf_trial_v1" / "common_calendar.csv",
    }


def _sha_map() -> dict[str, str]:
    absent = [name for name, path in _sources().items() if not path.is_file()]
    if absent:
        raise GateError(f"required bound input files missing: {absent}")
    return {name: sha256_file(path) for name, path in _sources().items()}


def prepare_binding() -> dict:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    if contract.get("status") != "DESIGN_FROZEN_IMPLEMENTATION_AND_INPUT_BINDING_REQUIRED":
        raise GateError(f"unexpected stage contract status: {contract.get('status')}")
    expected_sha = "614f21e33e88923560eda70f076bf4fefc2e1a67dce8d13db5323b9761539082"
    contract_sha = sha256_file(CONTRACT)
    if contract_sha != expected_sha:
        raise GateError(f"frozen contract SHA differs from phi-reviewed copy: {contract_sha}")
    stocks, etfs, _ = _runtime_modules()
    market, calendar, _, _, _ = stocks.load_market_and_events()
    prices = pd.read_parquet(ETF_PRICE)
    prices["trade_date"] = pd.to_datetime(prices.trade_date).dt.normalize()
    start = pd.Timestamp(contract["period"]["start"])
    end = pd.Timestamp(contract["period"]["end"])
    etf_calendar = pd.DatetimeIndex(sorted(prices.pivot(index="trade_date", columns="asset_id", values="close_cny")
                                           .dropna(subset=etfs.ASSET_ORDER).loc[lambda x: x.index.to_series().between(start, end)].index))
    bound_calendar = pd.to_datetime(pd.read_csv(_sources()["calendar"]).iloc[:, 0]).dt.normalize()
    if not pd.DatetimeIndex(calendar).equals(pd.DatetimeIndex(etf_calendar)):
        raise GateError("stock and four-ETF common calendars differ")
    if not pd.DatetimeIndex(calendar).equals(pd.DatetimeIndex(bound_calendar)):
        raise GateError("common calendar differs from parent verified calendar")
    if len(calendar) != 418 or pd.Timestamp(calendar[0]) != pd.Timestamp("2025-01-02") or pd.Timestamp(calendar[-1]) != pd.Timestamp("2026-09-21"):
        raise GateError(f"unexpected frozen evaluation calendar: {len(calendar)} sessions")
    hashes = _sha_map()
    binding = {
        "status": "BOUND_FOR_RUN", "contract_sha256": contract_sha,
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "stock_engine_clone_sha256": hashes["stock_engine_clone"],
        "input_sha256": hashes,
        "calendar_sha256": hashlib.sha256(pd.util.hash_pandas_object(pd.Series(calendar), index=False).values.tobytes()).hexdigest(),
        "calendar_sessions": len(calendar), "download_count": 0, "model_fit_count": 0,
        "evaluated_account_ids": contract["budget"]["account_ids"],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "execution_binding.json"
    binding["binding_file"] = str(path)
    path.write_text(json.dumps(binding, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return binding


def load_binding(path: Path) -> tuple[dict, dict]:
    binding = json.loads(path.read_text(encoding="utf-8"))
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    if contract.get("status") != "DESIGN_FROZEN_IMPLEMENTATION_AND_INPUT_BINDING_REQUIRED":
        raise GateError("contract status changed after binding")
    expected = {"contract_sha256": sha256_file(CONTRACT),
                "runner_sha256": sha256_file(Path(__file__).resolve()),
                "stock_engine_clone_sha256": sha256_file(ROOT / "scripts" / "stock_etf_stage_account_engine_v1.py"),
                "input_sha256": _sha_map()}
    mismatch = {k: (binding.get(k), v) for k, v in expected.items() if binding.get(k) != v}
    if binding.get("status") != "BOUND_FOR_RUN" or mismatch:
        raise GateError(f"execution binding mismatch: {mismatch or binding.get('status')}")
    if int(contract["budget"]["new_model_fits"]) != 0 or int(contract["budget"]["new_downloads"]) != 0:
        raise GateError("contract unexpectedly authorizes fits/downloads; runner stays bounded")
    if binding.get("evaluated_account_ids") != contract["budget"]["account_ids"]:
        raise GateError("account identity list differs from frozen four-account budget")
    return binding, contract


def _load_inputs(contract: dict):
    stocks, etfs, engine = _runtime_modules()
    market, calendar, stock_events, unsupported, mergers = stocks.load_market_and_events()
    rankings, panel = stocks.build_rankings()
    state_raw = pd.read_parquet(STATE_FILE)
    state_raw["signal_date"] = pd.to_datetime(state_raw.signal_date).dt.normalize()
    state_raw = state_raw.set_index("signal_date").sort_index()
    state_sha = sha256_file(STATE_FILE)
    want_state_sha = contract["selector"]["state_source_sha256_reference"]
    if state_sha != want_state_sha:
        raise GateError(f"causal state source SHA mismatch: {state_sha}")
    prices = pd.read_parquet(ETF_PRICE)
    prices["trade_date"] = pd.to_datetime(prices.trade_date).dt.normalize()
    raw_events = pd.read_csv(ETF_EVENTS, dtype={"asset_id": str})
    events = etfs.normalize_event_ledger(raw_events)
    parent_contract = json.loads(ETF_PARENT_CONTRACT.read_text(encoding="utf-8"))
    products = json.loads(ETF_PRODUCTS.read_text(encoding="utf-8"))
    parent_binding = json.loads(ETF_PARENT_BINDING.read_text(encoding="utf-8"))
    etfs.validate_binding(parent_binding, sha256_file(ETF_PARENT_CONTRACT), sha256_file(ETF_PRICE),
                          sha256_file(ETF_PRODUCTS), sha256_file(ETF_EVENTS))
    bound_products = {str(p["asset_id"]): str(p["code"]) for p in products["products"]}
    expected_products = {str(k): str(v) for k, v in contract["candidates"]["ETF_R60_TOP2"]["products"].items()}
    if bound_products != expected_products:
        raise GateError(f"bound ETF products differ from frozen mixed-stage contract: {bound_products}")
    readiness = json.loads((ROOT / "results" / "multi_asset_etf_trial_v1" / "event_readiness.json").read_text(encoding="utf-8"))
    if readiness.get("status") != "PARTIAL_EVENTS_READY_PRIMARY_PDF_MISSING_511010_2026_09":
        raise GateError("parent verified ETF event readiness changed")
    if int(readiness.get("ledger_rows", -1)) != len(events):
        raise GateError("ETF action row count differs from accepted readiness")
    etf_calendar = pd.DatetimeIndex(sorted(prices.pivot(index="trade_date", columns="asset_id", values="close_cny")
                                           .dropna(subset=etfs.ASSET_ORDER)
                                           .loc[lambda x: x.index.to_series().between(pd.Timestamp(contract["period"]["start"]),
                                                                                     pd.Timestamp(contract["period"]["end"]))].index))
    if not pd.DatetimeIndex(calendar).equals(etf_calendar):
        raise GateError("four-ETF prices are not aligned to all 418 stock dates")
    factors = {asset: etfs._gross_factors_for_asset(prices, events, asset) for asset in etfs.ASSET_ORDER}
    return stocks, etfs, engine, market, calendar, stock_events, unsupported, mergers, rankings, panel, state_raw, prices, events, factors


def _outer_pick(day, state, rule):
    valid = False
    if state is not None:
        m = pd.to_numeric(pd.Series([state.get("M_csi300_ma60_distance")]), errors="coerce").iloc[0]
        b = pd.to_numeric(pd.Series([state.get("B_pool_share_above_ma60")]), errors="coerce").iloc[0]
        cov = pd.to_numeric(pd.Series([state.get("B_coverage")]), errors="coerce").iloc[0]
        valid = (pd.notna(m) and pd.notna(b) and pd.notna(cov) and np.isfinite(float(m))
                 and np.isfinite(float(b)) and float(cov) >= float(rule["B_coverage_min"])
                 and bool(state.get("M_present", False))
                 and float(state.get("pool_size", 0) or 0) > 0)
    if valid and float(m) > 0 and float(b) > float(rule["B_threshold"]):
        pick = "STOCK_S01"
    else:
        pick = "ETF_R60_TOP2"
    return pick, {"decision_date": pd.Timestamp(day).strftime("%Y-%m-%d"), "M": float(m) if valid else None,
                  "B": float(b) if valid else None, "B_coverage": float(cov) if valid else None,
                  "state_valid": bool(valid), "selected_candidate": pick,
                  "fallback_to_etf": not bool(valid)}


def _etf_scores(date, factors, contract):
    values = {}
    for asset, series in factors.items():
        window = series.loc[:date].tail(60).to_numpy(dtype=float)
        if len(window) != 60 or not np.isfinite(window).all() or np.any(window <= 0):
            values[asset] = None
            continue
        returns = window - 1.0
        vol = float(np.std(returns, ddof=1))
        ret = float(np.prod(window) - 1.0)
        score = ret / (math.sqrt(60.0) * vol) if vol > 0 else None
        values[asset] = {"R60": ret, "s60": vol, "Q": score if score is not None and np.isfinite(score) else None}
    return values


def _stock_state_schedule(calendar, states, contract):
    rule = contract["selector"]
    picks, rows = {}, []
    for i in range(0, len(calendar), int(rule["hold_sessions"])):
        day = pd.Timestamp(calendar[i])
        raw = states.loc[day].to_dict() if day in states.index else None
        pick, row = _outer_pick(day, raw, {"B_coverage_min": .80, "B_threshold": .50})
        row.update({"index": i, "stage_end_exclusive": min(i + 40, len(calendar)),
                    "full_40_session_stage": i + 40 <= len(calendar)})
        picks[i] = pick
        rows.append(row)
    invalid = sum(not x["state_valid"] for x in rows)
    rate = invalid / max(1, len(rows))
    if rate > .10:
        raise GateError(f"outer selector has invalid state coverage at {rate:.1%} nodes (>10%)")
    active = "ETF_R60_TOP2"
    day_picks = {}
    for i, day in enumerate(calendar):
        if i in picks:
            active = picks[i]
        day_picks[pd.Timestamp(day)] = active
    return day_picks, pd.DataFrame(rows)


def _etf_targets(date, nav, fraction, score_map, closes, contract, etfs):
    ranked = []
    code_map = {"csi300": "159919", "sp500_qdii": "513500", "physical_gold": "518800", "rmb_gov_bond": "511010"}
    for asset, score in score_map.items():
        if score is not None and score["Q"] is not None and score["Q"] > 0:
            ranked.append((asset, float(score["Q"]), code_map[asset]))
    ranked.sort(key=lambda x: (-x[1], x[2]))
    chosen = ranked[:2]
    targets, weights = {}, {}
    for asset in etfs.ASSET_ORDER:
        w = fraction * .5 if asset in {x[0] for x in chosen} else 0.0
        price = float(closes[(asset, pd.Timestamp(date))])
        lot = etfs.PRODUCT_RULES[asset]["buy_lot"]
        units = int((w * nav / price) // lot) * lot if w > 0 else 0
        targets[asset] = units
        weights[asset] = w
    return targets, {"selected": [x[0] for x in chosen], "ranked": ranked, "target_weights": weights, "target_units": targets}


def _etf_order_templates(date, targets, account, lookup, calendar, index, events_today, all_events, mode, contract, etfs):
    fee_spec = json.loads(ETF_PARENT_CONTRACT.read_text(encoding="utf-8"))["fees_and_tax"][mode]
    template = []
    for asset in etfs.ASSET_ORDER:
        target = int(targets.get(asset, 0))
        held = int(account.fund_units.get(asset, 0))
        row = lookup.get((asset, pd.Timestamp(date)))
        side = "sell" if held > target else "buy"
        requested = abs(target - held)
        rule = etfs.PRODUCT_RULES[asset]
        if row is None:
            status, qty = "NO_ROW", 0
            raw_open = fp = np.nan
        else:
            raw_open = float(row.open_cny)
            qty = (requested // rule["buy_lot"] * rule["buy_lot"] if side == "buy"
                   else etfs._round_sell_qty(held, requested, rule["sell_lot"]))
            prior_vol = []
            history_dates = sorted(d for a, d in lookup if a == asset and d < pd.Timestamp(date))[-20:]
            for pd_day in history_dates:
                prior_row = lookup.get((asset, pd.Timestamp(pd_day)))
                if prior_row is not None:
                    ratio_to_current = math.prod(
                        float(e["new_units_per_old_unit"]) for e in all_events
                        if str(e["asset_id"]) == asset and e["event_type"] == "unit_conversion"
                        and pd.Timestamp(e["ex_date"]) > pd.Timestamp(pd_day)
                        and pd.Timestamp(e["ex_date"]) <= pd.Timestamp(date))
                    prior_vol.append(float(prior_row.volume_lots) * 100 * ratio_to_current)
            cap = int(math.floor((float(np.mean(prior_vol)) if prior_vol else 0.0) * .01 /
                                 rule["buy_lot"]) * rule["buy_lot"])
            qty = min(qty, cap)
            qty = (etfs._round_sell_qty(held, qty, rule["sell_lot"]) if side == "sell"
                   else qty // rule["buy_lot"] * rule["buy_lot"])
            prev = lookup.get((asset, pd.Timestamp(calendar[index - 1]))) if index else None
            prev_close = float(prev.close_cny) if prev is not None else np.nan
            ratios = [float(e["new_units_per_old_unit"]) for e in events_today
                      if str(e["asset_id"]) == asset and e["event_type"] == "unit_conversion"]
            cashes = [float(e["cash_per_old_unit"]) for e in events_today
                      if str(e["asset_id"]) == asset and e["event_type"] == "cash_distribution"]
            if np.isfinite(prev_close) and (ratios or cashes):
                prev_close = etfs.reference_price_after_actions(prev_close, rule["tick"],
                                                                 math.prod(ratios) if ratios else 1.0,
                                                                 sum(cashes))
            slip = float(fee_spec["slippage_rate_each_side"])
            if qty <= 0 or float(row.volume_lots) <= 0:
                status = "NO_LIQUIDITY_OR_LOT"
            elif not np.isfinite(prev_close):
                status = "NO_PREVIOUS_CLOSE"
            elif side == "sell" and rule["t_plus"] == 1 and account.fund_buy_date.get(asset) is not None and pd.Timestamp(date) <= account.fund_buy_date[asset]:
                status = "T1_NOT_SELLABLE"
            else:
                upper = etfs.price_limit(prev_close, rule["tick"], rule["limit_pct"], "up")
                lower = etfs.price_limit(prev_close, rule["tick"], rule["limit_pct"], "down")
                if (side == "buy" and raw_open >= upper - 1e-9) or (side == "sell" and raw_open <= lower + 1e-9):
                    status, qty = "OPEN_AT_PRICE_LIMIT", 0
                else:
                    fp = etfs.round_tick(raw_open * (1 + slip if side == "buy" else 1 - slip),
                                         rule["tick"], side)
                    status = "SLIPPAGE_OUTSIDE_DAILY_RANGE" if fp > float(row.high_cny) + 1e-9 or fp < float(row.low_cny) - 1e-9 else "READY"
                    if status != "READY":
                        qty = 0
        template.append({"date": pd.Timestamp(date), "asset_id": asset, "side": side,
                         "requested_units": requested, "planned_units": int(qty), "filled_units": 0,
                         "status": status, "raw_open": raw_open,
                         "fill_price": fp if row is not None and status == "READY" else np.nan,
                         "fee_spec": fee_spec})
    return template


def _etf_fill_sells(orders, account, date, etfs):
    fee_spec = orders[0]["fee_spec"] if orders else {}
    fills = []
    for order in orders:
        if order["side"] != "sell" or order["status"] != "READY":
            continue
        asset, qty = order["asset_id"], int(order["planned_units"])
        qty = min(qty, int(account.fund_units.get(asset, 0)))
        if not qty:
            order["status"] = "NO_AVAILABLE_UNITS"
            continue
        notional = qty * float(order["fill_price"])
        fee = etfs.fund_fee(notional, fee_spec)
        account.cash += notional - fee
        account.fund_units[asset] = int(account.fund_units.get(asset, 0)) - qty
        order["filled_units"] = qty
        order["status"] = "FILLED" if qty == order["planned_units"] else "PARTIAL_CAPACITY"
        fills.append({"date": pd.Timestamp(date), "asset_class": "ETF", "asset_id": asset, "side": "SELL",
                      "units": qty, "raw_open": order["raw_open"], "fill_price": order["fill_price"],
                      "notional": notional, "fee": fee, "stamp_tax": 0.0,
                      "slippage_cost": qty * abs(float(order["fill_price"]) - float(order["raw_open"]))})
    return fills


def _buy_options_stock(account, quotes, date, opening_nav, stock_engine, fee_mult):
    order = account.orders
    if not order:
        return {}
    options = {}
    for symbol in order["ranking"]:
        held = account.quantity(symbol)
        diff = int(order["targets"].get(symbol, 0)) - held
        if diff <= 0:
            continue
        row = quotes.get(symbol, {})
        op = row.get("open", np.nan)
        if row.get("is_suspended", np.nan) != 0 or not np.isfinite(op) or op <= 0:
            continue
        up, low = row.get("upper_limit", np.nan), row.get("lower_limit", np.nan)
        if not np.isfinite(up) or not np.isfinite(low) or (up > 0 and op >= up - 1e-5):
            continue
        price = op * (1 + account.slippage)
        if up > 0 and price > up + 1e-5:
            continue
        cap = max(0.0, opening_nav * float(order.get("stock_fraction", 1.0)) * .03 - held * op)
        budget = min(cap, float(order["budgets"].get(symbol, 0.0)))
        qty = min(diff, stock_engine.buy_quantity(symbol, budget, price))
        qty = stock_engine.buy_quantity(symbol, qty * price + 1e-6, price)
        if qty > 0:
            fee = stock_engine.fee_components(qty * price, "BUY", date)["total"] * fee_mult
            options[symbol] = {"max_units": int(qty), "price": price, "all_in": qty * price + fee}
    return options


def _buy_options_etf(orders):
    out = {}
    if not orders:
        return out
    spec = orders[0]["fee_spec"]
    for o in orders:
        if o["side"] == "buy" and o["status"] == "READY" and o["planned_units"] > 0:
            qty, px = int(o["planned_units"]), float(o["fill_price"])
            out[o["asset_id"]] = {"max_units": qty, "price": px,
                                  "fee_spec": spec, "all_in": qty * px}
    return out


def _scaled_buy_quantities(options, scale, kind, etfs, stock_engine):
    result = {}
    for symbol, info in options.items():
        qty = int(info["max_units"] * scale)
        if kind == "etf":
            lot = etfs.PRODUCT_RULES[symbol]["buy_lot"]
            qty = qty // lot * lot
        else:
            qty = stock_engine.buy_quantity(symbol, qty * info["price"], info["price"])
        if qty > 0:
            result[symbol] = qty
    return result


def _ration_buys(stock_options, etf_options, cash, date, fee_mult, stock_engine, etfs):
    def estimate(scale):
        s = _scaled_buy_quantities(stock_options, scale, "stock", etfs, stock_engine)
        e = _scaled_buy_quantities(etf_options, scale, "etf", etfs, stock_engine)
        total = 0.0
        for symbol, qty in s.items():
            amount = qty * stock_options[symbol]["price"]
            total += amount + stock_engine.fee_components(amount, "BUY", date)["total"] * fee_mult
        for symbol, qty in e.items():
            amount = qty * etf_options[symbol]["price"]
            total += amount + etfs.fund_fee(amount, etf_options[symbol]["fee_spec"])
        return total, s, e
    if estimate(1.0)[0] <= cash + 1e-6:
        _, s, e = estimate(1.0)
        return 1.0, s, e
    lo, hi = 0.0, 1.0
    for _ in range(48):
        mid = (lo + hi) / 2
        if estimate(mid)[0] <= cash + 1e-6:
            lo = mid
        else:
            hi = mid
    _, s, e = estimate(lo)
    return lo, s, e


def _etf_fill_buys(orders, quantities, account, date, etfs):
    fills = []
    fee_spec = orders[0]["fee_spec"] if orders else {}
    for order in orders:
        if order["side"] != "buy" or order["status"] != "READY":
            continue
        asset = order["asset_id"]
        qty = int(quantities.get(asset, 0))
        lot = etfs.PRODUCT_RULES[asset]["buy_lot"]
        qty = qty // lot * lot
        if qty <= 0:
            order["status"] = "CASH_RATIONED_TO_ZERO"
            continue
        price = float(order["fill_price"])
        notional = qty * price
        fee = etfs.fund_fee(notional, fee_spec)
        if notional + fee > account.cash + 1e-6:
            max_qty = int(max(0, account.cash / price) // lot) * lot
            while max_qty and max_qty * price + etfs.fund_fee(max_qty * price, fee_spec) > account.cash + 1e-6:
                max_qty -= lot
            qty = max_qty
            notional = qty * price
            fee = etfs.fund_fee(notional, fee_spec) if qty else 0.0
        if qty <= 0:
            order["status"] = "INSUFFICIENT_SHARED_CASH"
            continue
        account.cash -= notional + fee
        account.fund_units[asset] = int(account.fund_units.get(asset, 0)) + qty
        account.fund_buy_date[asset] = pd.Timestamp(date)
        order["filled_units"] = qty
        order["status"] = "FILLED" if qty == order["planned_units"] else "PARTIAL_CASH_RATION"
        fills.append({"date": pd.Timestamp(date), "asset_class": "ETF", "asset_id": asset, "side": "BUY",
                      "units": qty, "raw_open": order["raw_open"], "fill_price": price,
                      "notional": notional, "fee": fee, "stamp_tax": 0.0,
                      "slippage_cost": qty * abs(price - float(order["raw_open"]))})
    return fills


def _metrics(daily, trades):
    nav = daily.nav.to_numpy(float)
    prior = np.r_[1_000_000.0, nav[:-1]]
    rets = nav / prior - 1.0
    sd = float(np.std(rets, ddof=1)) if len(rets) > 1 else 0.0
    sharpe = float(np.sqrt(252.0) * np.mean(rets) / sd) if sd > 0 else None
    peak = np.maximum.accumulate(np.r_[1_000_000.0, nav])
    years = len(nav) / 252.0
    return {"sessions": len(nav), "starting_capital": 1_000_000.0, "ending_nav": float(nav[-1]),
            "total_return": float(nav[-1] / 1_000_000.0 - 1),
            "annualized_return": float((nav[-1] / 1_000_000.0) ** (1 / years) - 1) if years else None,
            "annualized_volatility": float(sd * np.sqrt(252.0)), "sharpe_zero_cash": sharpe,
            "max_drawdown": float((np.r_[1_000_000.0, nav] / peak - 1).min()),
            "fills": len(trades), "gross_notional": float(trades.notional.sum()) if len(trades) else 0.0,
            "fees": float(trades.fee.sum()) if len(trades) else 0.0,
            "stock_dividend_tax": float(trades.dividend_tax.sum()) if len(trades) and "dividend_tax" in trades else 0.0,
            "mean_cash_fraction": float((daily.cash / daily.nav).mean()),
            "mean_stock_fraction": float((daily.stock_market_value / daily.nav).mean()),
            "mean_etf_fraction": float((daily.fund_market_value / daily.nav).mean())}


def run_hybrid_account(account_id, policy, mode, inputs, contract, day_picks, score_by_day,
                       outer_logs, output_dir):
    stocks, etfs, engine, market, calendar, stock_events, unsupported, mergers, rankings, panel, states, prices, events, factors = inputs
    fee_spec = json.loads(ETF_PARENT_CONTRACT.read_text(encoding="utf-8"))["fees_and_tax"][mode]
    fee_mult = float(contract["execution_costs_events"]["stock"][mode]["trade_fee_multiplier"])
    stock_slip = float(contract["execution_costs_events"]["stock"][mode]["one_way_slippage"])
    account = engine.Account(capital=float(contract["period"]["capital_cny"]),
                             slippage=stock_slip, commission_multiplier=fee_mult)
    account.fund_units = {a: 0 for a in etfs.ASSET_ORDER}
    account.fund_prices = {a: 0.0 for a in etfs.ASSET_ORDER}
    account.fund_receivables = {}
    account.fund_entitlements = {}
    account.fund_buy_date = {a: None for a in etfs.ASSET_ORDER}
    p = prices.copy()
    lookup = {(r.asset_id, pd.Timestamp(r.trade_date)): r for r in p.itertuples(index=False)}
    p_events = events.copy().reset_index(drop=True)
    p_events["event_id"] = p_events.index.astype(str)
    for col in ("ex_date", "record_date", "payment_date", "announcement_date"):
        p_events[col] = pd.to_datetime(p_events[col], errors="coerce").dt.normalize()
    by_ex = {d: g.to_dict("records") for d, g in p_events.groupby("ex_date", dropna=False)}
    by_record = {d: g.to_dict("records") for d, g in p_events.groupby("record_date", dropna=False)}
    pay_dates = {str(r.event_id): pd.Timestamp(r.payment_date) for r in p_events.itertuples(index=False)}
    by_stock_event = {d: g.to_dict("records") for d, g in stock_events.groupby("ex_date", observed=True)}
    by_unsupported = {d: g.symbol.astype(str).tolist() for d, g in unsupported.groupby("trade_date", observed=True)}
    merger_dates = {pd.Timestamp(m["conversion_date"]): m for m in mergers}
    pending_etf_targets = {}
    daily_rows, stock_orders, etf_orders, fills, etf_events_log, inventory = [], [], [], [], [], []
    stock_rejects = []
    for i, date in enumerate(calendar):
        date = pd.Timestamp(date)
        next_day = pd.Timestamp(calendar[i + 1]) if i + 1 < len(calendar) else date + pd.Timedelta(days=10)
        quotes_frame = market.loc[market.trade_date == date]
        quotes = quotes_frame.set_index("symbol").to_dict("index")
        # Carry forward cash distributions whose official pay date fell on a
        # non-session; they become spendable before this next common open.
        for event_id in list(account.fund_receivables):
            if pay_dates[event_id] < date:
                amount = account.fund_receivables.pop(event_id)
                account.cash += amount
                etf_events_log.append({"date": date, "event_id": event_id, "type": "DISTRIBUTION_CASH_AVAILABLE_NEXT_OPEN", "cash": amount})
        # ETF unit conversions precede the opening execution; preserve holder-level
        # upward rounding and adjust only a due target for the same asset.
        events_today = by_ex.get(date, [])
        conversion_ratio = {}
        for e in events_today:
            if e["event_type"] != "unit_conversion":
                continue
            asset = str(e["asset_id"])
            ratio = float(e["new_units_per_old_unit"])
            old = int(account.fund_units.get(asset, 0))
            new = etfs.convert_holder_units_rounded_up(old, ratio)
            account.fund_units[asset] = new
            conversion_ratio[asset] = conversion_ratio.get(asset, 1.0) * ratio
            if date in pending_etf_targets:
                old_target = int(pending_etf_targets[date].get(asset, 0))
                pending_etf_targets[date][asset] = int(old_target * ratio) // etfs.PRODUCT_RULES[asset]["buy_lot"] * etfs.PRODUCT_RULES[asset]["buy_lot"]
            etf_events_log.append({"date": date, "event_id": str(e["event_id"]), "asset_id": asset,
                                   "type": "UNIT_CONVERSION", "old_units": old, "new_units": new, "ratio": ratio})
        # Opening marks let the stock adapter enforce its cap on total portfolio NAV.
        for asset in etfs.ASSET_ORDER:
            row = lookup.get((asset, date))
            if row is not None and np.isfinite(row.open_cny) and row.open_cny > 0:
                account.fund_prices[asset] = float(row.open_cny)
        if date in merger_dates:
            account.merger(merger_dates[date], date, quotes)
        for symbol in by_unsupported.get(date, []):
            if account.quantity(symbol):
                account.issue(date, symbol, "UNSUPPORTED_FACTOR_CHANGE_WHILE_HELD")
        for event in by_stock_event.get(date, []):
            if event.get("event_type") == "REFERENCE_ADJUSTMENT_ONLY_NO_OLD_HOLDER_SHARES":
                continue
            account.corporate_action(event, date, pd.Timestamp(calendar[i - 1]) if i else date - pd.Timedelta(days=1))
        for symbol in account.lots:
            row = quotes.get(symbol, {})
            if np.isfinite(row.get("open", np.nan)) and row["open"] > 0:
                account.prices[symbol] = float(row["open"])
        opening_nav = account.nav(date)
        # Actual stock sells and then ETF sells are completed before any buys.
        trade_start, reject_start = len(account.trades), len(account.rejections)
        if account.orders:
            account.execute(date, next_day, quotes, side_filter="SELL", opening_nav_override=opening_nav)
        stock_rejects += account.rejections[reject_start:]
        if date in pending_etf_targets:
            due_target = pending_etf_targets.pop(date)
            etf_order_rows = _etf_order_templates(date, due_target, account, lookup, calendar, i,
                                                  events_today, p_events.to_dict("records"), mode, contract, etfs)
        else:
            etf_order_rows = []
        etf_fills = _etf_fill_sells(etf_order_rows, account, date, etfs)
        # Build a combined proportional cash scale from eligible stock and ETF
        # buys after all net sells have credited the shared cash balance.
        stock_opts = _buy_options_stock(account, quotes, date, opening_nav, engine, fee_mult)
        etf_opts = _buy_options_etf(etf_order_rows)
        scale, stock_qty, etf_qty = _ration_buys(stock_opts, etf_opts, account.cash,
                                                  date, fee_mult, engine, etfs)
        if account.orders:
            account.execute(date, next_day, quotes, side_filter="BUY", buy_scale=scale,
                            buy_quantities=stock_qty, opening_nav_override=opening_nav)
        etf_fills.extend(_etf_fill_buys(etf_order_rows, etf_qty, account, date, etfs))
        fills.extend([{**t, "asset_class": "STOCK", "asset_id": t["symbol"], "units": t["shares"],
                       "side": t["side"], "stamp_tax": t.get("stamp_tax", 0.0)}
                      for t in account.trades[trade_start:]])
        fills.extend(etf_fills)
        stock_rejects += account.rejections[reject_start + len(account.rejections[reject_start:]):]
        # Ex-date claims use the account's own registered-date units; payment at
        # today's close becomes spendable only next common open.
        for e in events_today:
            if e["event_type"] == "cash_distribution":
                event_id = str(e["event_id"])
                if event_id not in account.fund_entitlements:
                    raise GateError(f"missing ETF record-date entitlement snapshot for event {event_id}")
                amount = etfs.half_up_cents(Decimal(account.fund_entitlements[event_id]) * Decimal(str(e["cash_per_old_unit"])))
                account.fund_receivables[event_id] = amount
                etf_events_log.append({"date": date, "event_id": event_id, "asset_id": e["asset_id"],
                                       "type": "DISTRIBUTION_RECEIVABLE", "units": account.fund_entitlements[event_id],
                                       "cash": amount, "verification_status": e.get("verification_status")})
        # Set closing marks and cash credits at the official payment close.
        for asset in etfs.ASSET_ORDER:
            row = lookup.get((asset, date))
            if row is None or not np.isfinite(row.close_cny) or row.close_cny <= 0:
                raise GateError(f"missing close price for {asset} on {date.date()}")
            account.fund_prices[asset] = float(row.close_cny)
        for event_id in list(account.fund_receivables):
            if pay_dates[event_id] == date:
                amount = account.fund_receivables.pop(event_id)
                account.cash += amount
                etf_events_log.append({"date": date, "event_id": event_id, "type": "DISTRIBUTION_PAID_AT_CLOSE", "cash": amount})
        # The stock close adapter also marks stock positions, pays stock
        # receivables, and computes the stock dividend-tax liability.
        account.close(date, quotes)
        drow = account.daily[-1]
        drow.update({"policy": policy, "cost_mode": mode,
                     "outer_candidate": day_picks[date],
                     "stock_family_target_fraction": (1.0 if policy == "stage" and day_picks[date] == "STOCK_S01" else 0.0 if policy == "stage" else .5),
                     "etf_family_target_fraction": (1.0 if policy == "stage" and day_picks[date] == "ETF_R60_TOP2" else 0.0 if policy == "stage" else .5),
                     "shared_buy_ration_scale": scale})
        drow["fund_units"] = sum(account.fund_units.values())
        daily_rows.append(dict(drow))
        expected_nav = (account.cash + sum(account.quantity(s) * account.prices.get(s, 0) for s in account.lots)
                        + sum(account.fund_units[a] * account.fund_prices[a] for a in etfs.ASSET_ORDER)
                        + sum(x["cash"] for x in account.receivables) + sum(account.fund_receivables.values())
                        - account.tax_liability(date))
        error = float(account.daily[-1]["nav"] - expected_nav)
        if abs(error) > 1e-6 or account.cash < -1e-6:
            raise GateError(f"daily shared cash/NAV reconciliation failed {date.date()}: {error}")
        for symbol, lots in account.lots.items():
            inventory.append({"date": date, "asset_class": "STOCK", "symbol": symbol,
                              "units": account.quantity(symbol), "close": account.prices.get(symbol, np.nan),
                              "market_value": account.quantity(symbol) * account.prices.get(symbol, 0),
                              "lot_count": len(lots), "dividend_tax_base": sum(l.quantity * l.dividend_base for l in lots)})
        for asset in etfs.ASSET_ORDER:
            if account.fund_units.get(asset, 0):
                inventory.append({"date": date, "asset_class": "ETF", "symbol": asset,
                                  "units": account.fund_units[asset], "close": account.fund_prices[asset],
                                  "market_value": account.fund_units[asset] * account.fund_prices[asset],
                                  "lot_count": 1, "dividend_tax_base": 0.0})
        for event in by_record.get(date, []):
            account.fund_entitlements[str(event["event_id"])] = int(account.fund_units.get(str(event["asset_id"]), 0))
        # Freeze the causal score and form ETF targets at 20-session nodes.
        if i % 20 == 0:
            scores = score_by_day[date]
            family_fraction = (1.0 if policy == "stage" and day_picks[date] == "ETF_R60_TOP2"
                               else 0.0 if policy == "stage" else .5)
            targets, score_detail = _etf_targets(date, account.daily[-1]["nav"], family_fraction,
                                                  scores, {(a, pd.Timestamp(date)): float(lookup[(a, date)].close_cny)
                                                           for a in etfs.ASSET_ORDER}, contract, etfs)
            if not all(v is not None and v["Q"] is not None for v in scores.values()) and i == 0:
                raise GateError("first ETF score node requires valid four-ETF 60-session histories")
            etf_order = {"signal_date": date, "target_map": targets, "selected": score_detail["selected"]}
            # A signal-close target is executable only next common session.
            if i + 1 < len(calendar):
                pending_etf_targets[pd.Timestamp(calendar[i + 1])] = targets
            score_detail.update({"date": date.strftime("%Y-%m-%d"), "account_id": account_id,
                                 "family_fraction": family_fraction, "target_nav": account.daily[-1]["nav"]})
            # Keep the account-specific decision/audit row with the shared score table.
            outer_logs.append({"type": "ETF_TARGET", **score_detail})
        # Update stock targets on the five-session clock; inactive family uses a
        # real zero target, producing a five-session residual-exit retry cadence.
        if i % 5 == 0 and i + 1 < len(calendar):
            stock_fraction = (1.0 if policy == "stage" and day_picks[date] == "STOCK_S01"
                              else 0.0 if policy == "stage" else .5)
            rank = rankings["S01"].get(date)
            if rank is None:
                raise GateError(f"missing frozen S01 ranking for {date.date()}")
            account.plan(date, rank, quotes, stock_fraction=stock_fraction)
        # For stage-only policy, an inactive ETF family exits and retries on the
        # global five-session cadence; no unscheduled constituent changes.
        etf_fraction = (1.0 if policy == "stage" and day_picks[date] == "ETF_R60_TOP2"
                        else 0.0 if policy == "stage" else .5)
        if policy == "stage" and etf_fraction == 0 and i % 5 == 0 and i + 1 < len(calendar):
            pending_etf_targets[pd.Timestamp(calendar[i + 1])] = {a: 0 for a in etfs.ASSET_ORDER}
        # Record the stock target attempt as typed orders at the time it executes.
        if trade_start < len(account.trades):
            for trade in account.trades[trade_start:]:
                stock_orders.append({"date": date, "asset_class": "STOCK", "symbol": trade["symbol"],
                                     "side": trade["side"], "requested_units": trade["shares"],
                                     "filled_units": trade["shares"], "status": "FILLED_OR_PARTIAL_BY_LOT_RULE"})
        for rejection in account.rejections[reject_start:]:
            stock_orders.append({"date": date, "asset_class": "STOCK", "symbol": rejection["symbol"],
                                 "side": rejection["side"], "requested_units": np.nan,
                                 "filled_units": 0, "status": rejection["reason"]})
        etf_orders.extend([{k: v for k, v in row.items() if k != "fee_spec"} for row in etf_order_rows])
    daily = pd.DataFrame(daily_rows)
    trades = pd.DataFrame(fills)
    account_dir = output_dir / account_id.lower()
    account_dir.mkdir(parents=True, exist_ok=False)
    daily.to_csv(account_dir / "daily.csv", index=False)
    trades.to_csv(account_dir / "fills.csv", index=False)
    pd.DataFrame(stock_orders + etf_orders).to_csv(account_dir / "typed_orders.csv", index=False)
    pd.DataFrame(account.rejections).to_csv(account_dir / "stock_rejections.csv", index=False)
    pd.DataFrame(etf_events_log).to_csv(account_dir / "etf_events.csv", index=False)
    pd.DataFrame(account.events).to_csv(account_dir / "stock_events.csv", index=False)
    pd.DataFrame(inventory).to_csv(account_dir / "daily_inventory.csv", index=False)
    closing = []
    for symbol, lots in account.lots.items():
        for lot in lots:
            closing.append({"asset_class": "STOCK", "symbol": symbol, "units": lot.quantity,
                            "acquired": lot.acquired, "available": lot.available,
                            "dividend_tax_base_per_share": lot.dividend_base})
    for asset, units in account.fund_units.items():
        if units:
            closing.append({"asset_class": "ETF", "symbol": asset, "units": units,
                            "acquired": account.fund_buy_date[asset], "available": None,
                            "dividend_tax_base_per_share": 0.0})
    pd.DataFrame(closing).to_csv(account_dir / "closing_typed_holdings.csv", index=False)
    pd.DataFrame([{"event_id": k, "receivable_cny": v, "payment_date": pay_dates[k]}
                  for k, v in account.fund_receivables.items()]).to_csv(account_dir / "closing_etf_receivables.csv", index=False)
    metrics = _metrics(daily, trades)
    metrics.update({"account_id": account_id, "policy": policy, "cost_mode": mode,
                    "closing_cash": float(account.cash), "closing_stock_symbols": len(account.lots),
                    "closing_etf_units": {a: int(account.fund_units[a]) for a in etfs.ASSET_ORDER},
                    "issues": len(account.issues), "stock_rejections": len(account.rejections),
                    "etf_orders": len(etf_orders), "etf_fill_count": sum(t["asset_class"] == "ETF" for t in fills),
                    "max_daily_reconciliation_abs_cny": float(np.max(np.abs(
                        daily.nav - (daily.cash + daily.market_value + daily.dividend_receivable - daily.estimated_dividend_tax))))})
    return {"daily": daily, "trades": trades, "metrics": metrics,
            "stock_orders": pd.DataFrame(stock_orders), "etf_orders": pd.DataFrame(etf_orders),
            "scores": pd.DataFrame(), "inventory": pd.DataFrame(inventory)}


def build_causal_score_table(calendar, factors, contract, etfs):
    score_by_day, rows, missing_nodes = {}, [], []
    node_idx = list(range(0, len(calendar), 20))
    for i in node_idx:
        date = pd.Timestamp(calendar[i])
        scores = _etf_scores(date, factors, contract)
        score_by_day[date] = scores
        missing = [a for a, value in scores.items() if value is None or value["Q"] is None]
        rows.append({"date": date, "index": i,
                     **{f"R60_{a}": v["R60"] if v else np.nan for a, v in scores.items()},
                     **{f"s60_{a}": v["s60"] if v else np.nan for a, v in scores.items()},
                     **{f"Q_{a}": v["Q"] if v else np.nan for a, v in scores.items()},
                     "invalid_assets": "|".join(missing)})
        if missing:
            missing_nodes.append(date)
    if node_idx and any(score_by_day[pd.Timestamp(calendar[node_idx[0]])][a] is None or
                        score_by_day[pd.Timestamp(calendar[node_idx[0]])][a]["Q"] is None for a in etfs.ASSET_ORDER):
        raise GateError("first ETF decision node lacks a complete 60-return score for all four assets")
    invalid_fraction = len(missing_nodes) / max(1, len(node_idx))
    if invalid_fraction > .10:
        raise GateError(f"invalid ETF score at {invalid_fraction:.1%} of ETF nodes (>10%)")
    return score_by_day, pd.DataFrame(rows), invalid_fraction


def summarize_stage_returns(daily, outer_logs):
    rows = []
    nav_by_date = daily.set_index(pd.to_datetime(daily.date)).nav
    initial = 1_000_000.0
    for row in outer_logs:
        if row.get("type") == "ETF_TARGET":
            continue
        start = int(row["index"])
        end = int(row["stage_end_exclusive"])
        start_nav = initial if start == 0 else float(nav_by_date.iloc[start - 1])
        end_nav = float(nav_by_date.iloc[end - 1])
        rows.append({"selected_candidate": row["selected_candidate"], "start_index": start,
                     "end_index_exclusive": end, "sessions": end - start,
                     "full_40_session_stage": bool(row["full_40_session_stage"]),
                     "start_date": row["decision_date"],
                     "end_date": daily.iloc[end - 1].date.strftime("%Y-%m-%d"),
                     "start_nav": start_nav, "end_nav": end_nav,
                     "return": end_nav / start_nav - 1.0})
    return pd.DataFrame(rows)


def _accepted_reference_metrics():
    refs = {}
    files = [
        ("stock_S01_standard", ROOT / "results" / "strategy_switch_trial_v1" / "comparators" / "results.json", "S01_standard"),
        ("stock_S01_stress", ROOT / "results" / "strategy_switch_trial_v1" / "comparators" / "results.json", "S01_stress"),
        ("etf_static_standard", ROOT / "results" / "multi_asset_etf_trial_v1" / "coordinator_acceptance.json", "static_standard"),
        ("etf_static_stress", ROOT / "results" / "multi_asset_etf_trial_v1" / "coordinator_acceptance.json", "static_stress"),
    ]
    for key, path, rowkey in files:
        if not path.is_file():
            refs[key] = {"status": "MISSING"}
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        account = data.get("accounts", {}).get(rowkey) or data.get("metrics", {}).get(rowkey)
        refs[key] = {"status": "AVAILABLE_DESCRIPTIVE_ONLY", "source": str(path.relative_to(ROOT)).replace("\\", "/"),
                     "source_sha256": sha256_file(path), "metrics": account}
    return refs


def run_all(binding_path: Path):
    binding, contract = load_binding(binding_path)
    if RUN_OUT.exists():
        raise GateError(f"refusing to overwrite existing economic results: {RUN_OUT}")
    inputs = _load_inputs(contract)
    stocks, etfs, engine, market, calendar, stock_events, unsupported, mergers, rankings, panel, states, prices, events, factors = inputs
    if len(calendar) != int(contract["period"]["expected_sessions"]):
        raise GateError(f"expected {contract['period']['expected_sessions']} dates, got {len(calendar)}")
    if any(set(ranking.keys()) != set(calendar) for ranking in rankings.values()):
        raise GateError("frozen stock ranking dates do not exactly cover the common account calendar")
    day_picks, outer_df = _stock_state_schedule(calendar, states, contract)
    score_by_day, score_df, invalid_etf_fraction = build_causal_score_table(calendar, factors, contract, etfs)
    RUN_OUT.mkdir(parents=True, exist_ok=False)
    outer_df.to_csv(RUN_OUT / "outer_selection_log.csv", index=False)
    score_df.to_csv(RUN_OUT / "causal_etf_score_table.csv", index=False)
    (RUN_OUT / "causal_etf_score_table.json").write_text(
        json.dumps({str(k.date()): v for k, v in score_by_day.items()}, ensure_ascii=False, indent=2,
                   default=lambda x: None) + "\n", encoding="utf-8")
    stage_counts = outer_df.loc[outer_df.full_40_session_stage].selected_candidate.value_counts().to_dict()
    required_min = int(contract["evaluation"]["selection_increment_pass_requires_all"]["each_candidate_complete_selected_40_session_stages_min"])
    if any(stage_counts.get(name, 0) < required_min for name in ("STOCK_S01", "ETF_R60_TOP2")):
        raise GateError(f"insufficient complete outer-stage coverage: {stage_counts}")
    results, target_logs = {}, []
    for account_id, policy, mode in (
        ("STAGE_STANDARD", "stage", "standard"), ("FIXED_STANDARD", "fixed", "standard"),
        ("STAGE_STRESS", "stage", "stress"), ("FIXED_STRESS", "fixed", "stress")):
        results[account_id] = run_hybrid_account(account_id, policy, mode, inputs, contract,
                                                  day_picks, score_by_day, target_logs, RUN_OUT)
    # Save the global selector and account-specific ETF target decisions.
    target_frame = pd.DataFrame([r for r in target_logs if r.get("type") == "ETF_TARGET"])
    target_frame.to_csv(RUN_OUT / "etf_target_decisions.csv", index=False)
    metrics = {k: v["metrics"] for k, v in results.items()}
    comparisons = {}
    for mode in ("standard", "stress"):
        stage_key = f"STAGE_{mode.upper()}"
        fixed_key = f"FIXED_{mode.upper()}"
        stage_daily, fixed_daily = results[stage_key]["daily"], results[fixed_key]["daily"]
        stage_rets = stage_daily.nav.to_numpy(float) / np.r_[1_000_000.0, stage_daily.nav.to_numpy(float)[:-1]] - 1
        fixed_rets = fixed_daily.nav.to_numpy(float) / np.r_[1_000_000.0, fixed_daily.nav.to_numpy(float)[:-1]] - 1
        bootstrap = etfs._bootstrap_sharpe_delta(pd.Series(stage_rets), pd.Series(fixed_rets),
                                                  reps=2000, block=20, seed=20261010)
        comparisons[mode] = {
            "sharpe_delta_stage_minus_fixed": metrics[stage_key]["sharpe_zero_cash"] - metrics[fixed_key]["sharpe_zero_cash"],
            "paired_moving_block_bootstrap": bootstrap,
        }
    primary = contract["evaluation"]["selection_increment_pass_requires_all"]
    gates = {
        "inputs_and_engineering_valid": all(abs(v["metrics"]["max_daily_reconciliation_abs_cny"]) <= 1e-6 and v["metrics"]["issues"] == 0 for v in results.values()),
        "stage_standard_sharpe_positive": metrics["STAGE_STANDARD"]["sharpe_zero_cash"] is not None and metrics["STAGE_STANDARD"]["sharpe_zero_cash"] > 0,
        "standard_sharpe_delta_at_least_0_1": comparisons["standard"]["sharpe_delta_stage_minus_fixed"] >= .1,
        "stage_stress_sharpe_positive": metrics["STAGE_STRESS"]["sharpe_zero_cash"] is not None and metrics["STAGE_STRESS"]["sharpe_zero_cash"] > 0,
        "stress_sharpe_delta_positive": comparisons["stress"]["sharpe_delta_stage_minus_fixed"] > 0,
        "candidate_full_stage_minimum_met": all(stage_counts.get(name, 0) >= required_min for name in ("STOCK_S01", "ETF_R60_TOP2")),
        "etf_invalid_score_fraction_at_most_10pct": invalid_etf_fraction <= .10,
    }
    stage_frames = {}
    for key, run in results.items():
        stages = summarize_stage_returns(run["daily"], outer_df.to_dict("records"))
        stages.to_csv(RUN_OUT / f"{key.lower()}_stage_returns.csv", index=False)
        stage_frames[key] = stages
        d = run["daily"].copy()
        d["date"] = pd.to_datetime(d.date)
        year_rows = []
        prior = 1_000_000.0
        for year, g in d.groupby(d.date.dt.year, sort=True):
            last = float(g.nav.iloc[-1])
            year_rows.append({"year": int(year), "begin_nav": prior, "end_nav": last, "return": last / prior - 1})
            prior = last
        pd.DataFrame(year_rows).to_csv(RUN_OUT / f"{key.lower()}_year_returns.csv", index=False)
    summary = {
        "status": "FOUR_MIXED_ACCOUNTS_COMPLETED_EXPLORATORY",
        "contract_sha256": binding["contract_sha256"], "runner_sha256": binding["runner_sha256"],
        "binding_sha256": sha256_file(binding_path), "input_sha256": binding["input_sha256"],
        "period": {"start": str(pd.Timestamp(calendar[0]).date()), "end": str(pd.Timestamp(calendar[-1]).date()), "sessions": len(calendar)},
        "model_fits": 0, "downloads": 0, "evaluated_accounts": list(results),
        "accounts": metrics, "comparisons": comparisons, "gates": gates,
        "outer_stage_counts_complete_40": stage_counts,
        "outer_invalid_state_fraction": float((~outer_df.state_valid).mean()),
        "etf_nodes": len(score_df), "etf_nodes_with_any_invalid_score_fraction": invalid_etf_fraction,
        "references_descriptive_only": _accepted_reference_metrics(),
        "limitations": contract["limitations"],
        "interpretation": "Historical exploratory selection increment only; not blind validation or investment suitability.",
    }
    (RUN_OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    acceptance = {
        "status": "PASS" if gates["inputs_and_engineering_valid"] else "ENGINEERING_FAIL",
        "account_reconciliation_abs_max_cny": {k: v["metrics"]["max_daily_reconciliation_abs_cny"] for k, v in results.items()},
        "cash_nonnegative": {k: bool((v["daily"].cash >= -1e-6).all()) for k, v in results.items()},
        "calendar_sessions": len(calendar), "first_signal_date": str(pd.Timestamp(calendar[0]).date()),
        "next_open_first_execution": str(pd.Timestamp(calendar[1]).date()),
        "typed_stock_etf_inventory": True, "nav_sleeve_splicing": False,
        "account_ids": list(results), "summary_gates": gates,
    }
    (RUN_OUT / "independent_acceptance.json").write_text(json.dumps(acceptance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--smoke", action="store_true", help="run synthetic ledger checks only")
    group.add_argument("--bind", action="store_true", help="validate inputs and write a hash binding")
    group.add_argument("--run", action="store_true", help="run exactly the four frozen mixed-account identities")
    parser.add_argument("--binding", type=Path, default=OUT / "execution_binding.json")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        if args.smoke:
            result = smoke_check()
            (OUT / "scaffold_smoke.json").write_text(
                json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(result, ensure_ascii=False))
        elif args.bind:
            print(json.dumps(prepare_binding(), ensure_ascii=False, indent=2))
        else:
            result = run_all(args.binding)
            print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        return 0
    except (GateError, OSError, json.JSONDecodeError) as exc:
        print(f"BLOCKED: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
