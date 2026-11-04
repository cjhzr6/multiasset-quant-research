"""Isolated four-ETF validation helpers and gated account runner.

This runner deliberately refuses an economic run until a coordinator binding and
verified event data are supplied. It never infers a zero-event history from an
empty/missing action file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from decimal import Decimal, ROUND_HALF_UP
from decimal import ROUND_CEILING
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "configs" / "multi_asset_etf_trial_v1.contract.json"
PRICE_PATH = ROOT / "data" / "processed" / "multi_asset_etf_trial_v1" / "four_etf_daily_prices.parquet"
PRODUCTS_PATH = ROOT / "results" / "multi_asset_etf_trial_v1" / "products.json"
EVENTS_READINESS_PATH = ROOT / "results" / "multi_asset_etf_trial_v1" / "event_readiness.json"
CALENDAR_OUT = ROOT / "results" / "multi_asset_etf_trial_v1" / "common_calendar.csv"
RUN_OUT = ROOT / "results" / "multi_asset_etf_trial_v1" / "account_run"
EVENT_COLUMNS = {
    "asset_id", "event_type", "announcement_date", "ex_date", "record_date",
    "payment_date", "cash_per_old_unit", "new_units_per_old_unit", "source_url",
    "source_sha256", "verified",
}
ASSET_ORDER = ["csi300", "sp500_qdii", "physical_gold", "rmb_gov_bond"]
PRODUCT_RULES = {
    # Tick/lot/T+ rules frozen from exchange guidance and product type; the
    # liability for the single soft-source cash event remains separately tagged.
    "csi300": {"buy_lot": 100, "sell_lot": 1, "tick": 0.001, "limit_pct": 0.10, "t_plus": 1},
    "sp500_qdii": {"buy_lot": 100, "sell_lot": 100, "tick": 0.001, "limit_pct": 0.10, "t_plus": 0},
    "physical_gold": {"buy_lot": 100, "sell_lot": 100, "tick": 0.001, "limit_pct": 0.10, "t_plus": 0},
    "rmb_gov_bond": {"buy_lot": 100, "sell_lot": 100, "tick": 0.001, "limit_pct": 0.10, "t_plus": 0},
}


class GateError(ValueError):
    """Raised when evidence or execution inputs do not meet the frozen contract."""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def half_up_cents(value: float | Decimal) -> float:
    return float(Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def convert_holder_units_rounded_up(old_units: int, ratio: float | Decimal) -> int:
    """Apply issuer-directed holder-level upward rounding for a unit consolidation.

    This integer account treatment is deliberately separate from the continuous
    ratio used by the theoretical total-return signal.
    """
    if isinstance(old_units, bool) or int(old_units) != old_units or old_units < 0:
        raise GateError("份额合并前持仓须为非负整数")
    ratio_d = Decimal(str(ratio))
    if ratio_d <= 0:
        raise GateError("份额合并比例须为正")
    return int((Decimal(int(old_units)) * ratio_d).to_integral_value(rounding=ROUND_CEILING))


def distribution_account_effects(
    *, entitled_units: int, cash_per_unit: float, ex_date: str, pay_date: str,
    asof_date: str, next_session_after_pay: str | None,
) -> dict[str, float | str]:
    """Return receivable/cash status without making payment-date cash spendable early."""
    if entitled_units < 0 or int(entitled_units) != entitled_units or cash_per_unit < 0:
        raise GateError("分红权利份额须为非负整数，单位金额须非负")
    ex = pd.Timestamp(ex_date)
    pay = pd.Timestamp(pay_date)
    asof = pd.Timestamp(asof_date)
    if pay < ex:
        raise GateError("分红支付日早于除息日")
    amount = half_up_cents(Decimal(int(entitled_units)) * Decimal(str(cash_per_unit)))
    receivable = amount if ex <= asof and (next_session_after_pay is None or asof < pd.Timestamp(next_session_after_pay)) else 0.0
    cash_received = amount if next_session_after_pay is not None and pd.Timestamp(next_session_after_pay) <= asof else 0.0
    return {"receivable_cny": receivable, "cash_received_cny": cash_received,
            "cash_available_session": next_session_after_pay or "UNRESOLVED"}


def validate_event_frame(events: pd.DataFrame, *, allow_empty_verified: bool = False) -> None:
    missing = EVENT_COLUMNS - set(events.columns)
    if missing:
        raise GateError(f"事件表缺字段: {sorted(missing)}")
    if events.empty:
        if allow_empty_verified:
            return
        raise GateError("事件表为空；空表不能证明历史无分红或份额转换")
    if not events["verified"].map(lambda x: x is True or str(x).lower() == "true").all():
        raise GateError("存在未核实公司行动，禁止经济账户运行")
    for col in ("source_url", "source_sha256"):
        if events[col].isna().any() or (events[col].astype(str).str.len() == 0).any():
            raise GateError(f"公司行动缺少来源证据: {col}")
    dates = ["announcement_date", "ex_date", "record_date", "payment_date"]
    for col in dates:
        parsed = pd.to_datetime(events[col], errors="coerce")
        if parsed.isna().any():
            raise GateError(f"公司行动日期无效/缺失: {col}")
        events.loc[:, col] = parsed.dt.strftime("%Y-%m-%d")
    if (pd.to_datetime(events["announcement_date"]) > pd.to_datetime(events["ex_date"])).any():
        raise GateError("发现公告日期晚于除息/转换日的事件，须先核验")
    types = set(events["event_type"].astype(str))
    if not types <= {"cash_distribution", "unit_conversion"}:
        raise GateError(f"未知事件类型: {sorted(types)}")
    cash = events[events.event_type == "cash_distribution"]
    conv = events[events.event_type == "unit_conversion"]
    if not cash.empty and (cash.cash_per_old_unit.isna().any() or (cash.cash_per_old_unit < 0).any()):
        raise GateError("现金分配金额缺失或为负")
    if not conv.empty and (conv.new_units_per_old_unit.isna().any() or (conv.new_units_per_old_unit <= 0).any()):
        raise GateError("份额转换比例缺失或非正")


def total_return_gross_factors(close: pd.Series, events: pd.DataFrame, *, events_complete: bool = False) -> pd.Series:
    """Daily gross factors from raw closes and verified events, indexed by date.

    Each event amount/ratio is defined per old unit. Simultaneous actions are
    aggregated; distributions add to the ex-date economic value and conversions
    multiply the old-unit price into new units.
    """
    close = close.astype(float).sort_index()
    if close.index.has_duplicates or close.isna().any() or (close <= 0).any():
        raise GateError("收盘价序列须日期唯一、无缺失且严格为正")
    validate_event_frame(events, allow_empty_verified=events_complete)
    ev = events.copy()
    ev["ex_date"] = pd.to_datetime(ev["ex_date"]).dt.strftime("%Y-%m-%d")
    cash_by_date = ev[ev.event_type == "cash_distribution"].groupby("ex_date").cash_per_old_unit.sum().to_dict()
    ratio_by_date = ev[ev.event_type == "unit_conversion"].groupby("ex_date").new_units_per_old_unit.prod().to_dict()
    dates = close.index.astype(str)
    factors = pd.Series(index=close.index, dtype=float)
    for i in range(1, len(close)):
        d = dates[i]
        prior = float(close.iloc[i - 1])
        ratio = float(ratio_by_date.get(d, 1.0))
        cash = float(cash_by_date.get(d, 0.0))
        factors.iloc[i] = (ratio * float(close.iloc[i]) + cash) / prior
    return factors.iloc[1:]


def trailing_total_return(close: pd.Series, events: pd.DataFrame, signal_date: str, intervals: int = 252,
                          *, events_complete: bool = False) -> float:
    signal_stamp = pd.Timestamp(signal_date)
    known_events = events[pd.to_datetime(events["announcement_date"], errors="coerce") <= signal_stamp].copy()
    price_slice = close.loc[:signal_stamp]
    factors = total_return_gross_factors(price_slice, known_events, events_complete=events_complete)
    if len(factors) < intervals:
        raise GateError(f"{signal_date}: 仅{len(factors)}个有效收益区间，需{intervals}")
    window = factors.iloc[-intervals:]
    if not np.isfinite(window.to_numpy()).all() or (window <= 0).any():
        raise GateError(f"{signal_date}: 总回报区间含无效因子")
    return float(np.prod(window.to_numpy(dtype=float)) - 1.0)


def target_weights(returns: dict[str, float | None], strategy: str) -> dict[str, float]:
    if set(returns) != set(ASSET_ORDER):
        raise GateError("四类资产必须齐全")
    if strategy == "static":
        return {a: 0.25 for a in ASSET_ORDER}
    if strategy != "trend":
        raise GateError(f"未知策略: {strategy}")
    return {a: (0.25 if returns[a] is not None and returns[a] > 0 else 0.0) for a in ASSET_ORDER}


def commission(notional: float, rate: float, minimum: float) -> float:
    return half_up_cents(max(minimum, notional * rate))


def validate_binding(binding: dict[str, Any], contract_sha: str, price_sha: str, products_sha: str, events_sha: str) -> None:
    expected = {
        "contract_sha256": contract_sha,
        "price_sha256": price_sha,
        "products_sha256": products_sha,
        "events_sha256": events_sha,
        "runner_sha256": sha256_file(Path(__file__)),
    }
    if binding.get("status") != "BOUND_FOR_RUN":
        raise GateError("协调者绑定状态不是 BOUND_FOR_RUN")
    mismatches = {k: (binding.get(k), v) for k, v in expected.items() if binding.get(k) != v}
    if mismatches:
        raise GateError(f"协调者绑定哈希不匹配: {mismatches}")


def normalize_event_ledger(raw_events: pd.DataFrame) -> pd.DataFrame:
    """Adapt the canonical project event ledger to the runner's event schema."""
    if raw_events.empty:
        raise GateError("公司行为事件账本为空")
    out = raw_events.copy()
    out["event_type"] = out["action_type"].map({
        "cash_distribution": "cash_distribution", "share_consolidation": "unit_conversion"
    })
    if out.event_type.isna().any():
        raise GateError("公司行为账本含未知类型")
    out["asset_id"] = out["asset_id"].astype(str)
    out["announcement_date"] = out["announcement_date"].astype(str)
    out["ex_date"] = out["ex_date"].astype(str)
    out["record_date"] = out["record_date"].astype(str)
    out["payment_date"] = out["pay_date"].fillna(out["effective_date"]).fillna(out["ex_date"]).astype(str)
    out["cash_per_old_unit"] = pd.to_numeric(out["cash_cny_per_unit"], errors="coerce")
    out["new_units_per_old_unit"] = pd.to_numeric(out["ratio_old_to_new"], errors="coerce")
    out["source_url"] = out["source_url"].astype(str)
    out["source_sha256"] = out["source_file"].astype(str)
    out["verified"] = out["verification_status"].isin(["official_verified", "secondary_verified_primary_pdf_missing"])
    validate_event_frame(out)
    return out


def load_and_validate_inputs(events_path: Path, binding_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    products = json.loads(PRODUCTS_PATH.read_text(encoding="utf-8"))
    prices = pd.read_parquet(PRICE_PATH)
    raw_events = pd.read_csv(events_path, dtype={"asset_id": str})
    events = normalize_event_ledger(raw_events)
    readiness = json.loads(EVENTS_READINESS_PATH.read_text(encoding="utf-8"))
    if readiness.get("status") != "PARTIAL_EVENTS_READY_PRIMARY_PDF_MISSING_511010_2026_09":
        raise GateError("公司行为核验状态与已审阅交付不符")
    if int(readiness.get("ledger_rows", -1)) != len(events):
        raise GateError("公司行为账本行数与验收记录不符")
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    validate_binding(binding, sha256_file(CONTRACT_PATH), sha256_file(PRICE_PATH),
                     sha256_file(PRODUCTS_PATH), sha256_file(events_path))
    if set(prices.asset_id.unique()) != set(ASSET_ORDER):
        raise GateError("价格输入不含且仅含固定四产品")
    if set(events.asset_id.unique()) - set(ASSET_ORDER):
        raise GateError("公司行动输入含固定篮子以外产品")
    if contract["period"]["initial_capital_cny"] != 1_000_000:
        raise GateError("初始资金与冻结合同不符")
    # Coordinator permits one clearly downgraded secondary-source event. It
    # remains tagged and is not upgraded to primary verification in outputs.
    if not (events.verification_status == "secondary_verified_primary_pdf_missing").sum() == 1:
        raise GateError("降级来源事件数量与当前事件审计不一致")
    events.attrs["readiness"] = readiness
    return prices, events, contract, products


def round_tick(price: float, tick: float, side: str) -> float:
    q = Decimal(str(price)) / Decimal(str(tick))
    rounding = ROUND_CEILING if side == "buy" else __import__("decimal").ROUND_FLOOR
    return float(q.to_integral_value(rounding=rounding) * Decimal(str(tick)))


def fund_fee(notional: float, fee_spec: dict) -> float:
    return commission(notional, fee_spec["commission_rate"], fee_spec["commission_min_cny_per_fill"])


def price_limit(prev_close: float, tick: float, pct: float, side: str) -> float:
    raw = prev_close * (1 + pct if side == "up" else 1 - pct)
    # Exchange price-limit quotation is rounded to the security tick.
    q = Decimal(str(raw)) / Decimal(str(tick))
    return float(q.quantize(Decimal("1"), rounding=ROUND_HALF_UP) * Decimal(str(tick)))


def reference_price_after_actions(prev_close: float, tick: float, ratio: float, cash_per_old_unit: float) -> float:
    raw = (prev_close - cash_per_old_unit) / ratio
    return float((Decimal(str(raw)) / Decimal(str(tick))).quantize(Decimal("1"), rounding=ROUND_HALF_UP) * Decimal(str(tick)))


def _round_sell_qty(holding: int, requested: int, lot: int = 100) -> int:
    q = min(max(0, int(requested)), int(holding))
    if q % lot == 0 or q == holding:
        return q
    odd = holding % lot
    if odd and q >= odd and (q - odd) % lot == 0:
        return q
    return q - (q % lot)


def _gross_factors_for_asset(prices: pd.DataFrame, events: pd.DataFrame, asset: str) -> pd.Series:
    g = prices[prices.asset_id == asset].copy()
    g["trade_date"] = pd.to_datetime(g["trade_date"])
    g = g.sort_values("trade_date").set_index("trade_date")
    close = g.close_cny.astype(float)
    ev = events[events.asset_id == asset].copy()
    ev["event_type"] = ev.event_type.replace({"share_consolidation": "unit_conversion"})
    ev["cash_per_old_unit"] = ev["cash_cny_per_unit"]
    ev["new_units_per_old_unit"] = ev["ratio_old_to_new"]
    # Cash rows have no conversion ratio; unit conversions have no payout.
    ev["cash_per_old_unit"] = pd.to_numeric(ev.cash_per_old_unit, errors="coerce").fillna(0.0)
    ev["new_units_per_old_unit"] = pd.to_numeric(ev.new_units_per_old_unit, errors="coerce").fillna(1.0)
    factors = pd.Series(index=close.index, dtype=float)
    for i in range(1, len(close)):
        date = close.index[i]
        known = ev[(pd.to_datetime(ev.announcement_date) <= date) & (pd.to_datetime(ev.ex_date) == date)]
        ratio = float(known.loc[known.event_type == "unit_conversion", "new_units_per_old_unit"].prod())
        cash = float(known.loc[known.event_type == "cash_distribution", "cash_per_old_unit"].sum())
        factors.iloc[i] = (ratio * close.iloc[i] + cash) / close.iloc[i - 1]
    return factors.iloc[1:]


def _metrics(nav: pd.Series) -> dict[str, float]:
    daily = nav.pct_change().fillna(0.0)
    if len(daily) == 0:
        raise GateError("NAV序列过短")
    vol = float(daily.std(ddof=1))
    sharpe = float(np.sqrt(252) * daily.mean() / vol) if vol > 0 else float("nan")
    years = len(daily) / 252
    total = float(nav.iloc[-1] / nav.iloc[0] - 1)
    ann = float((nav.iloc[-1] / nav.iloc[0]) ** (1 / years) - 1) if nav.iloc[-1] > 0 and years > 0 else float("nan")
    draw = nav / nav.cummax() - 1
    return {"total_return": total, "annualized_return": ann, "volatility": vol,
            "sharpe_zero_cash": sharpe, "max_drawdown": float(draw.min())}


def _bootstrap_sharpe_delta(a: pd.Series, b: pd.Series, reps: int = 2000, block: int = 20,
                            seed: int = 20261009) -> dict[str, float]:
    x = np.column_stack([a.to_numpy(dtype=float), b.to_numpy(dtype=float)])
    n = len(x)
    rng = np.random.default_rng(seed)
    out = []
    blocks = math.ceil(n / block)
    for _ in range(reps):
        starts = rng.integers(0, n, size=blocks)
        idx = np.concatenate([(np.arange(s, s + block) % n) for s in starts])[:n]
        sample = x[idx]
        ss = []
        for j in range(2):
            sd = sample[:, j].std(ddof=1)
            ss.append(np.sqrt(252) * sample[:, j].mean() / sd if sd > 0 else np.nan)
        out.append(ss[0] - ss[1])
    vals = np.asarray(out, dtype=float)
    vals = vals[np.isfinite(vals)]
    return {"mean": float(vals.mean()), "ci_2_5": float(np.quantile(vals, .025)),
            "ci_97_5": float(np.quantile(vals, .975)), "repetitions": int(len(vals)),
            "block_sessions": block, "seed": seed}


def simulate_account(prices: pd.DataFrame, events: pd.DataFrame, calendar: pd.DatetimeIndex,
                     contract: dict, mode: str, strategy: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fee_spec = contract["fees_and_tax"][mode]
    p = prices.copy()
    p["trade_date"] = pd.to_datetime(p["trade_date"])
    lookup = {(r.asset_id, r.trade_date): r for r in p.itertuples(index=False)}
    closes = {a: p[p.asset_id == a].set_index("trade_date").close_cny.astype(float) for a in ASSET_ORDER}
    full_calendar = pd.DatetimeIndex(p.pivot(index="trade_date", columns="asset_id", values="close_cny")
                                     .dropna(subset=ASSET_ORDER).index).sort_values()
    factors = {a: _gross_factors_for_asset(p, events, a) for a in ASSET_ORDER}
    unit = {a: 0 for a in ASSET_ORDER}
    buy_date: dict[str, pd.Timestamp | None] = {a: None for a in ASSET_ORDER}
    cash = 1_000_000.0
    receivables: dict[str, float] = {}
    entitlements: dict[str, int] = {}
    ledger, orders, fills = [], [], []
    pending_targets: dict[str, dict[str, int]] = {}
    event_ids = events.reset_index(drop=True)
    event_ids["event_id"] = event_ids.index.astype(str)
    eval_idx = {d: i for i, d in enumerate(calendar)}
    for i, date in enumerate(calendar):
        # Conversion applies before marking/trading on the official ex-date.
        conversion_today: dict[str, float] = {}
        action_ratios_today: dict[str, float] = {}
        action_cash_today: dict[str, float] = {}
        for e in event_ids[(event_ids.event_type == "unit_conversion") & (pd.to_datetime(event_ids.ex_date) == date)].itertuples():
            asset = e.asset_id
            ratio = float(e.ratio_old_to_new)
            conversion_today[asset] = conversion_today.get(asset, 1.0) * ratio
            action_ratios_today[asset] = action_ratios_today.get(asset, 1.0) * ratio
            unit[asset] = convert_holder_units_rounded_up(unit[asset], e.ratio_old_to_new)
            if date in pending_targets:
                old_target = pending_targets[date][asset]
                pending_targets[date][asset] = (int(old_target * ratio) // 100) * 100
        for e in event_ids[(event_ids.event_type == "cash_distribution") & (pd.to_datetime(event_ids.ex_date) == date)].itertuples():
            action_cash_today[e.asset_id] = action_cash_today.get(e.asset_id, 0.0) + float(e.cash_cny_per_unit)
        # Execute only prior-close decisions; same-day sales cash may fund buys.
        if date in pending_targets:
            target_map = pending_targets.pop(date)
            for side in ("sell", "buy"):
                for asset in ASSET_ORDER:
                    row = lookup.get((asset, date))
                    if row is None:
                        orders.append({"date": date, "asset_id": asset, "side": side, "requested_units": 0,
                                       "filled_units": 0, "status": "NO_ROW"})
                        continue
                    desired = target_map[asset]
                    diff = unit[asset] - desired if side == "sell" else desired - unit[asset]
                    requested = max(0, int(diff))
                    if side == "sell":
                        # 159919 follows SZSE odd-lot rule; SSE ETFs use the same
                        # 100-unit minimum with final residual sell permitted.
                        requested = _round_sell_qty(unit[asset], requested, PRODUCT_RULES[asset]["sell_lot"])
                    else:
                        requested = (requested // PRODUCT_RULES[asset]["buy_lot"]) * PRODUCT_RULES[asset]["buy_lot"]
                    prior_dates = full_calendar[full_calendar < date][-20:]
                    prior_vol = [lookup[(asset, d)].volume_lots * 100 for d in prior_dates if (asset, d) in lookup]
                    if asset in conversion_today:
                        prior_vol = [v * conversion_today[asset] for v in prior_vol]
                    buy_lot = PRODUCT_RULES[asset]["buy_lot"]
                    cap = int(np.floor(np.mean(prior_vol) * 0.01 / buy_lot) * buy_lot) if prior_vol else 0
                    qty = min(requested, cap)
                    if side == "sell":
                        qty = _round_sell_qty(unit[asset], qty, PRODUCT_RULES[asset]["sell_lot"])
                    else:
                        qty = (qty // buy_lot) * buy_lot
                    tick, limit_pct = PRODUCT_RULES[asset]["tick"], PRODUCT_RULES[asset]["limit_pct"]
                    prev_close = float(closes[asset].loc[calendar[i - 1]]) if i > 0 and calendar[i - 1] in closes[asset].index else None
                    if prev_close is not None and (asset in action_ratios_today or asset in action_cash_today):
                        ratio = action_ratios_today.get(asset, 1.0)
                        cash_dist = action_cash_today.get(asset, 0.0)
                        prev_close = reference_price_after_actions(prev_close, tick, ratio, cash_dist)
                    raw_open = float(row.open_cny)
                    if qty <= 0 or row.volume_lots <= 0:
                        status = "NO_LIQUIDITY_OR_LOT"
                    elif prev_close is None:
                        status = "NO_PREVIOUS_CLOSE"
                    else:
                        up = price_limit(prev_close, tick, limit_pct, "up")
                        down = price_limit(prev_close, tick, limit_pct, "down")
                        if (side == "buy" and raw_open >= up - 1e-9) or (side == "sell" and raw_open <= down + 1e-9):
                            status = "OPEN_AT_PRICE_LIMIT"
                            qty = 0
                        else:
                            slip = contract["fees_and_tax"][mode]["slippage_rate_each_side"]
                            fill_price = round_tick(raw_open * (1 + slip if side == "buy" else 1 - slip), tick, side)
                            if fill_price > row.high_cny + 1e-9 or fill_price < row.low_cny - 1e-9:
                                status, qty = "SLIPPAGE_OUTSIDE_DAILY_RANGE", 0
                            else:
                                status = "READY"
                    # Buys are scaled together after all feasible sell proceeds are known below.
                    orders.append({"date": date, "asset_id": asset, "side": side, "requested_units": requested,
                                   "capacity_units": cap, "planned_units": qty, "status": status,
                                   "raw_open": raw_open if row else np.nan,
                                   "fill_price": fill_price if status == "READY" else np.nan})
            day_orders = orders[-8:]
            sell_cash = cash
            for o in day_orders:
                if o["side"] != "sell" or o.get("status") != "READY" or o["planned_units"] <= 0:
                    continue
                asset, qty, fp = o["asset_id"], int(o["planned_units"]), float(o["fill_price"])
                if PRODUCT_RULES[asset]["t_plus"] == 1 and buy_date[asset] is not None and date <= buy_date[asset]:
                    o["status"] = "T1_NOT_SELLABLE"
                    continue
                qty = min(qty, unit[asset])
                notional = qty * fp
                fee = fund_fee(notional, fee_spec)
                cash += notional - fee
                unit[asset] -= qty
                slip_cost = qty * abs(fp - float(o["raw_open"]))
                fills.append({"date": date, "asset_id": asset, "side": "sell", "units": qty,
                              "raw_open": o["raw_open"], "fill_price": fp, "notional": notional,
                              "fee": fee, "slippage_cost": slip_cost})
                o["filled_units"] = qty
                o["status"] = "FILLED" if qty == o["planned_units"] else "PARTIAL_CAPACITY"
            buy_orders = [o for o in day_orders if o["side"] == "buy" and o.get("status") == "READY" and o["planned_units"] > 0]
            prelim = []
            for o in buy_orders:
                prelim.append((o, int(o["planned_units"]), float(o["fill_price"])))
            total_need = sum(q * fp + fund_fee(q * fp, fee_spec) for _, q, fp in prelim)
            if total_need > cash and total_need > 0:
                lo, hi = 0.0, 1.0
                for _ in range(48):
                    mid = (lo + hi) / 2
                    need = 0.0
                    for o, q, fp in prelim:
                        lot = PRODUCT_RULES[o["asset_id"]]["buy_lot"]
                        scaled = int((q * mid) // lot) * lot
                        if scaled:
                            need += scaled * fp + fund_fee(scaled * fp, fee_spec)
                    if need <= cash:
                        lo = mid
                    else:
                        hi = mid
                prelim = [(o, int((q * lo) // PRODUCT_RULES[o["asset_id"]]["buy_lot"]) * PRODUCT_RULES[o["asset_id"]]["buy_lot"], fp)
                          for o, q, fp in prelim]
            for o, qty, fp in prelim:
                qty = int(qty)
                if qty <= 0:
                    o["status"] = "CASH_SCALED_TO_ZERO"
                    o["filled_units"] = 0
                    continue
                notional = qty * fp
                fee = fund_fee(notional, fee_spec)
                if notional + fee > cash + 1e-6:
                    lot = PRODUCT_RULES[o["asset_id"]]["buy_lot"]
                    qty = max(0, int((cash / fp) // lot) * lot)
                    notional = qty * fp
                    fee = fund_fee(notional, fee_spec) if qty else 0.0
                if qty <= 0 or notional + fee > cash + 1e-6:
                    o["status"] = "INSUFFICIENT_CASH"
                    o["filled_units"] = 0
                    continue
                cash -= notional + fee
                unit[o["asset_id"]] += qty
                buy_date[o["asset_id"]] = date
                fills.append({"date": date, "asset_id": o["asset_id"], "side": "buy", "units": qty,
                              "raw_open": o["raw_open"], "fill_price": fp, "notional": notional,
                              "fee": fee, "slippage_cost": qty * abs(fp - float(o["raw_open"]))})
                o["filled_units"] = qty
                o["status"] = "FILLED" if qty == o["planned_units"] else "PARTIAL_OR_CASH_SCALED"
        # Ex-date entitlements arise from units held at the preceding record-date close.
        for e in event_ids[(event_ids.event_type == "cash_distribution") & (pd.to_datetime(event_ids.ex_date) == date)].itertuples():
            q = entitlements.get(e.event_id)
            if q is None:
                raise GateError(f"缺少登记日持仓快照: event={e.event_id}")
            amount = half_up_cents(Decimal(q) * Decimal(str(e.cash_cny_per_unit)))
            receivables[e.event_id] = amount
        # Payouts become cash only at end of actual payment date.
        paid_events = event_ids[(event_ids.event_type == "cash_distribution") &
                                (pd.to_datetime(event_ids.payment_date) <= date)]
        for e in paid_events.itertuples():
            if e.event_id in receivables:
                cash += receivables.pop(e.event_id)
        nav_positions, gross_returns = 0.0, {}
        for asset in ASSET_ORDER:
            row = lookup.get((asset, date))
            if row is None:
                raise GateError(f"共同日历中缺少交易价格: {asset} {date.date()}")
            nav_positions += unit[asset] * float(row.close_cny)
        nav = cash + sum(receivables.values()) + nav_positions
        for e in event_ids[pd.to_datetime(event_ids.record_date) == date].itertuples():
            entitlements[e.event_id] = unit[e.asset_id]
        decision_index = i - 0
        if i % contract["strategy"]["decision_cadence_sessions"] == 0:
            signal_returns = {}
            for asset in ASSET_ORDER:
                f = factors[asset].loc[:date]
                signal_returns[asset] = float(np.prod(f.tail(252).to_numpy()) - 1) if len(f) >= 252 else None
            weights = target_weights(signal_returns, strategy)
            targets = {}
            for asset in ASSET_ORDER:
                close = float(lookup[(asset, date)].close_cny)
                lot = PRODUCT_RULES[asset]["buy_lot"]
                raw_target = int((weights[asset] * nav / close) // lot) * lot if weights[asset] > 0 else 0
                targets[asset] = raw_target
            if i + 1 < len(calendar):
                pending_targets[calendar[i + 1]] = targets
        ledger.append({"date": date, "cash_cny": cash, "receivable_cny": sum(receivables.values()),
                       "nav_cny": nav, "position_value_cny": nav_positions,
                       **{f"units_{a}": unit[a] for a in ASSET_ORDER},
                       **{f"signal_{a}": signal_returns.get(a, np.nan) if i % contract["strategy"]["decision_cadence_sessions"] == 0 else np.nan for a in ASSET_ORDER},
                       "strategy": strategy, "cost_mode": mode})
    return pd.DataFrame(ledger), pd.DataFrame(orders), pd.DataFrame(fills)


def run_economic_accounts(prices: pd.DataFrame, events: pd.DataFrame, contract: dict,
                          binding: dict) -> dict:
    prices = prices.copy()
    prices["trade_date"] = pd.to_datetime(prices["trade_date"])
    wide = prices.pivot(index="trade_date", columns="asset_id", values="close_cny")
    calendar = wide.dropna(subset=ASSET_ORDER).index
    calendar = calendar[(calendar >= contract["period"]["evaluation_start"]) &
                        (calendar <= contract["period"]["evaluation_end"])]
    if len(calendar) != 418:
        raise GateError(f"共同评价日历应为绑定的418日，实际{len(calendar)}")
    # preserve an auditable calendar rather than infer a market-wide calendar from a proxy index
    pd.DataFrame({"session": calendar.strftime("%Y-%m-%d"), "source": "intersection_of_four_actual_etf_rows"}).to_csv(CALENDAR_OUT, index=False)
    RUN_OUT.mkdir(parents=True, exist_ok=True)
    events.to_csv(RUN_OUT / "event_ledger_used.csv", index=False)
    accounts = {}
    for mode in ("standard", "stress"):
        for strategy in ("trend", "static"):
            name = f"{strategy}_{mode}"
            ledger, orders, fills = simulate_account(prices, events, calendar, contract, mode, strategy)
            nav_bridge = ledger.cash_cny + ledger.receivable_cny + ledger.position_value_cny
            recon_error = float((nav_bridge - ledger.nav_cny).abs().max())
            if recon_error > 1e-8 or (ledger.cash_cny < -1e-7).any() or (ledger.nav_cny <= 0).any():
                raise GateError(f"{name}: 现金/NAV约束失败 max_error={recon_error}")
            ledger.to_csv(RUN_OUT / f"{name}_ledger.csv", index=False)
            orders.to_csv(RUN_OUT / f"{name}_orders.csv", index=False)
            fills.to_csv(RUN_OUT / f"{name}_fills.csv", index=False)
            metrics = _metrics(ledger.nav_cny)
            total_nav = ledger.nav_cny.replace(0, np.nan)
            fills_notional = float(fills.notional.sum()) if not fills.empty else 0.0
            metrics.update({
                "total_fees_cny": float(fills.fee.sum()) if not fills.empty else 0.0,
                "total_slippage_cost_cny": float(fills.slippage_cost.sum()) if not fills.empty else 0.0,
                "total_traded_notional_cny": fills_notional,
                "turnover_over_average_nav": fills_notional / float(ledger.nav_cny.mean()),
                "fill_count": int(len(fills)),
                "unfilled_or_partial_order_count": int((~orders.status.isin(["FILLED"])).sum()) if not orders.empty else 0,
                "cash_weight_mean": float((ledger.cash_cny / total_nav).mean()),
                "receivable_peak_cny": float(ledger.receivable_cny.max()),
                "nav_reconciliation_max_abs_cny": recon_error,
            })
            values = ledger[["date"]].copy()
            for asset in ASSET_ORDER:
                close_map = prices[prices.asset_id == asset].assign(trade_date=lambda d: pd.to_datetime(d.trade_date)).set_index("trade_date").close_cny
                values[f"weight_{asset}"] = ledger[f"units_{asset}"].to_numpy() * close_map.reindex(pd.to_datetime(ledger.date)).to_numpy() / total_nav.to_numpy()
            metrics["mean_class_weights"] = {a: float(values[f"weight_{a}"].mean()) for a in ASSET_ORDER}
            ledger_dates = pd.to_datetime(ledger.date)
            month_last = ledger.assign(_date=ledger_dates).set_index("_date").nav_cny.resample("ME").last()
            month_ret = month_last.pct_change()
            if len(month_ret):
                month_ret.iloc[0] = month_last.iloc[0] / contract["period"]["initial_capital_cny"] - 1
            month_ret.rename("return").to_csv(RUN_OUT / f"{name}_monthly_returns.csv")
            yearly = ledger.assign(_year=ledger_dates.dt.year).groupby("_year").nav_cny.agg(["first", "last"])
            yearly["return"] = yearly["last"] / yearly["first"] - 1
            yearly.to_csv(RUN_OUT / f"{name}_year_segments.csv")
            accounts[name] = {"metrics": metrics, "ledger": ledger, "orders": orders, "fills": fills}
    comparisons = {}
    for mode in ("standard", "stress"):
        trend, static = accounts[f"trend_{mode}"], accounts[f"static_{mode}"]
        comparisons[mode] = {"sharpe_delta_trend_minus_static": trend["metrics"]["sharpe_zero_cash"] - static["metrics"]["sharpe_zero_cash"],
                             "paired_bootstrap": _bootstrap_sharpe_delta(
                                 trend["ledger"].nav_cny.pct_change().fillna(0),
                                 static["ledger"].nav_cny.pct_change().fillna(0))}
    summary = {"status": "FOUR_ACCOUNTS_COMPLETED_EXPLORATORY", "sessions": len(calendar),
               "start": calendar[0].strftime("%Y-%m-%d"), "end": calendar[-1].strftime("%Y-%m-%d"),
               "accounts": {k: v["metrics"] for k, v in accounts.items()}, "comparisons": comparisons,
               "binding_sha256": sha256_file(Path(binding["binding_file"])) if binding.get("binding_file") else None,
               "limitations": ["Daily open/slippage simulation is not auction replay.",
                               "511010 2026-09 distribution uses secondary-source event terms; official PDF not located.",
                               "2022-03-29 QDII price row gap is outside 2023+ signal window and evaluation; all four prices align on 418 evaluation sessions.",
                               "Market data comes from Eastmoney public quote API rather than exchange-native feed.",
                               "All four asset classes used 10% daily price-limit bands based on product/exchange guidance; no separate daily trading-status feed was present."]}
    summary["input_sha256"] = binding.get("input_sha256", {})
    summary["product_rules"] = PRODUCT_RULES
    (RUN_OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def smoke_test() -> None:
    dates = pd.date_range("2024-01-01", periods=254, freq="B")
    close = pd.Series(np.linspace(1.0, 1.2, len(dates)), index=dates)
    event = pd.DataFrame([{
        "asset_id": "csi300", "event_type": "cash_distribution", "announcement_date": "2024-06-01",
        "ex_date": dates[120].strftime("%Y-%m-%d"), "record_date": dates[121].strftime("%Y-%m-%d"),
        "payment_date": dates[130].strftime("%Y-%m-%d"), "cash_per_old_unit": 0.01,
        "new_units_per_old_unit": np.nan, "source_url": "https://example.invalid/event",
        "source_sha256": "a" * 64, "verified": True,
    }])
    factors = total_return_gross_factors(close, event, events_complete=True)
    assert len(factors) == len(dates) - 1
    assert math.isfinite(trailing_total_return(close, event, dates[-1].strftime("%Y-%m-%d"), events_complete=True))
    assert target_weights({a: (0.01 if i % 2 == 0 else -0.01) for i, a in enumerate(ASSET_ORDER)}, "trend") == {
        a: (0.25 if i % 2 == 0 else 0.0) for i, a in enumerate(ASSET_ORDER)
    }
    assert target_weights({a: None for a in ASSET_ORDER}, "static") == {a: 0.25 for a in ASSET_ORDER}
    assert commission(10_000, 0.0003, 5) == 5.0
    assert commission(100_000, 0.0003, 5) == 30.0
    merge_ratio = Decimal("0.94231553")
    assert convert_holder_units_rounded_up(3, merge_ratio) == 3
    assert convert_holder_units_rounded_up(100, merge_ratio) == 95
    # With a reciprocal price jump, the signal factor stays at 1; integer
    # holder-level ceiling is a separate account entitlement convention.
    new_close = 100.0 / float(merge_ratio)
    signal_factor = (float(merge_ratio) * new_close) / 100.0
    assert math.isclose(signal_factor, 1.0, rel_tol=1e-12)
    assert not math.isclose(convert_holder_units_rounded_up(100, merge_ratio) * new_close / 100.0, 1.0)
    cutoff_event = distribution_account_effects(
        entitled_units=400, cash_per_unit=0.6204, ex_date="2026-09-18", pay_date="2026-09-23",
        asof_date="2026-09-21", next_session_after_pay="2026-09-24")
    assert cutoff_event == {"receivable_cny": 248.16, "cash_received_cny": 0.0,
                            "cash_available_session": "2026-09-24"}
    paid_event = distribution_account_effects(
        entitled_units=400, cash_per_unit=0.6204, ex_date="2026-09-18", pay_date="2026-09-23",
        asof_date="2026-09-24", next_session_after_pay="2026-09-24")
    assert paid_event["receivable_cny"] == 0.0 and paid_event["cash_received_cny"] == 248.16
    future_event = event.copy()
    future_event.loc[0, "ex_date"] = dates[-1].strftime("%Y-%m-%d")
    future_event.loc[0, "record_date"] = dates[-1].strftime("%Y-%m-%d")
    future_event.loc[0, "payment_date"] = dates[-1].strftime("%Y-%m-%d")
    try:
        trailing_total_return(close, future_event, dates[150].strftime("%Y-%m-%d"), events_complete=True)
    except GateError:
        pass
    else:
        raise AssertionError("future event must not enter an earlier signal")
    try:
        validate_event_frame(pd.DataFrame(columns=sorted(EVENT_COLUMNS)))
    except GateError:
        pass
    else:
        raise AssertionError("empty event table must block")
    print("SMOKE_PASS: event-normalized total return, 252-interval window, target rules, fees, empty-event gate")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="run synthetic engineering checks only")
    parser.add_argument("--events", type=Path)
    parser.add_argument("--binding", type=Path)
    args = parser.parse_args()
    if args.smoke:
        smoke_test()
        return
    if not args.events or not args.binding:
        raise SystemExit("Economic run requires --events and --binding; use --smoke for engineering checks.")
    prices, events, contract, products = load_and_validate_inputs(args.events, args.binding)
    binding = json.loads(args.binding.read_text(encoding="utf-8"))
    summary = run_economic_accounts(prices, events, contract, binding)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
