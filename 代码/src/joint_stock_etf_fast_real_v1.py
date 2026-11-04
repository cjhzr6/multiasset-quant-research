# -*- coding: utf-8 -*-
"""N04：股票候选纳入原四 ETF 多资产对照（共享现金混合账户）。

任务（一次性交接补充第13行）：
    用「预先固定股票候选」替代四 ETF 篮子中的「沪深300 ETF 模块」，不重复中国股票权重；
    其余三模块（标普500 QDII / 黄金 / 国债）不变，各 25%。
    在「同池固定配置（四 ETF 各 25%）」之外，增加一个「价格选择」对照，避免
    把「模型更换」和「资产扩展」混为一因。

设计：
    - 主方案 stock_e00_3etf：E00 股票候选 25% + 标普500 25% + 黄金 25% + 国债 25%。
    - 价格选择对照 stock_t01_3etf：纯量价特征（ret60/ret120/momentum20_skip5 百分位均值，T01）
      股票候选 25% + 三 ETF 各 25%。用于区分「增量来自 E00 模型」还是「增量来自引入股票资产」。
    - 同池固定配置 static_4etf：四 ETF 各 25%（直接引用 N03 的 static_25pct 结果）。

账户引擎：
    - 股票侧复用 stock_etf_stage_account_engine_v1.Account（混合账本：股票 lots + 基金 units + 共享现金，
      T+1/整手/涨跌停/3%单股上限/股息税/公司行为/并购），fee 覆盖为 audit_account_inputs(2020-2025)。
    - ETF 侧复用 multi_asset_etf_trial_v1 的 helper + stock_etf_stage_trial_v1 的 ETF 交易函数
      （整手/T+0,T+1/涨跌停/滑点/容量/份额转换/分红）。
    - 股票 5 日调仓、ETF 20 日调仓，先卖后买、共享现金缩放。

评价期：2024-01-01 ~ 2025-12-31（485 交易日）。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import stock_etf_stage_account_engine_v1 as engine
import multi_asset_etf_trial_v1 as etfs
from stock_etf_stage_trial_v1 import (
    _etf_order_templates, _etf_fill_sells, _etf_fill_buys,
    _buy_options_stock, _buy_options_etf, _ration_buys, _metrics,
    GateError, sha256_file,
)
from audit_account_inputs import fee_components as audit_fee
from run_account_backtest import prepare_events

# 覆盖股票费用为 2020-2025 口径（N04 评价期 2024-2025）
engine.fee_components = audit_fee

PRICE_PATH = ROOT / "data" / "processed" / "multi_asset_etf_trial_v1" / "four_etf_daily_prices.parquet"
EVENT_PATH = ROOT / "data" / "processed" / "multi_asset_etf_trial_v1" / "corporate_actions.csv"
MARKET_PATH = ROOT / "data" / "processed" / "account_market_2020_2025.parquet"
PRED_PATH = ROOT / "results" / "rolling_signal_v1" / "predictions_2020_2025.parquet"
PANEL_PATH = ROOT / "data" / "processed" / "training_panel_v1.parquet"
OUT = ROOT / "results" / "joint_stock_etf_fast_v1" / "real_accounts"

START = pd.Timestamp("2024-01-01")
END = pd.Timestamp("2025-12-31")
FEATURES = ["ret60", "ret120", "momentum20_skip5"]
STOCK_FRACTION = 0.25
ETF_3_MODULES = ["sp500_qdii", "physical_gold", "rmb_gov_bond"]  # 去掉 csi300（被股票替代）
DECISION_CADENCE = 20


def load_stock_market():
    market = pd.read_parquet(MARKET_PATH)
    market["trade_date"] = pd.to_datetime(market.trade_date).dt.normalize()
    market = market.loc[market.trade_date.between(START, END)].copy()
    market["symbol"] = market.symbol.astype(str)
    calendar = pd.DatetimeIndex(sorted(market.trade_date.unique()))
    return market, calendar


def build_stock_rankings():
    """E00 排名 + T01 排名（2024-2025），返回 {day: [symbol...]}。"""
    preds = pd.read_parquet(PRED_PATH, columns=["signal_date", "symbol", "model", "score"])
    preds["signal_date"] = pd.to_datetime(preds.signal_date).dt.normalize()
    e00 = preds[preds.model.eq("E00")]
    e00 = e00[e00.signal_date.between(START, END)]
    e00_rank = {d: g.sort_values(["score", "symbol"], ascending=[False, True]).symbol.tolist()
                for d, g in e00.groupby("signal_date", observed=True)}

    panel = pd.read_parquet(PANEL_PATH, columns=["signal_date", "symbol", "prediction_ready", *FEATURES])
    panel["signal_date"] = pd.to_datetime(panel.signal_date).dt.normalize()
    panel["symbol"] = panel.symbol.astype(str)
    panel = panel[(panel.signal_date.between(START, END)) & (panel.prediction_ready.eq(True))].copy()
    # 只对三特征均有限的行做截面 rank
    finite = panel[FEATURES].notna().all(axis=1)
    panel = panel[finite].copy()
    ranks = panel.groupby("signal_date", observed=True)[FEATURES].rank(method="average", pct=True)
    panel["T01_score"] = ranks.mean(axis=1)
    t01_rank = {d: g.sort_values(["T01_score", "symbol"], ascending=[False, True]).symbol.tolist()
                for d, g in panel.groupby("signal_date", observed=True)}
    # 对齐：E00 与 T01 的日期/股票集合应一致
    assert e00_rank.keys() == t01_rank.keys(), "E00/T01 日期集合不一致"
    for d in e00_rank:
        assert set(e00_rank[d]) == set(t01_rank[d]), f"E00/T01 股票集合不一致 {d}"
    return {"E00": e00_rank, "T01": t01_rank}


def load_etf():
    prices = pd.read_parquet(PRICE_PATH)
    prices["trade_date"] = pd.to_datetime(prices.trade_date).dt.normalize()
    raw_events = pd.read_csv(EVENT_PATH, dtype={"asset_id": str})
    events = etfs.normalize_event_ledger(raw_events)
    lookup = {(r.asset_id, pd.Timestamp(r.trade_date)): r for r in prices.itertuples(index=False)}
    return prices, events, lookup


def run_mixed_account(stock_rank, mode, calendar, market, prices, events, lookup,
                      policy_id, predictions, market_returns, fixed_target_weights=None,
                      target_weight_schedule=None):
    """Frozen E00 picks plus three ETFs under the N04 shared-cash ledger."""
    fee_spec = json.loads(ROOT.joinpath("configs", "multi_asset_etf_trial_v1.contract.json").read_text(encoding="utf-8"))["fees_and_tax"][mode]
    fee_mult = 1.0 if mode == "standard" else 2.0
    stock_slip = 0.0005 if mode == "standard" else 0.001

    account = engine.Account(capital=1_000_000.0, slippage=stock_slip, commission_multiplier=fee_mult)
    account.fund_units = {a: 0 for a in etfs.ASSET_ORDER}
    account.fund_prices = {a: 0.0 for a in etfs.ASSET_ORDER}
    account.fund_receivables = {}
    account.fund_entitlements = {}
    account.fund_buy_date = {a: None for a in etfs.ASSET_ORDER}

    # 股票事件
    stock_events, unsupported = prepare_events(calendar)
    by_stock_event = {d: g.to_dict("records") for d, g in stock_events.groupby("ex_date", observed=True)}
    by_unsupported = {d: g.symbol.astype(str).tolist() for d, g in unsupported.groupby("trade_date", observed=True)}
    mergers = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((ROOT / "results" / "corporate_dates").glob("merger_*_2025.json"))]
    merger_dates = {pd.Timestamp(m["conversion_date"]): m for m in mergers}

    # ETF 事件
    p_events = events.copy().reset_index(drop=True)
    p_events["event_id"] = p_events.index.astype(str)
    for col in ("ex_date", "record_date", "payment_date", "announcement_date"):
        p_events[col] = pd.to_datetime(p_events[col], errors="coerce").dt.normalize()
    by_ex = {d: g.to_dict("records") for d, g in p_events.groupby("ex_date", dropna=False)}
    by_record = {d: g.to_dict("records") for d, g in p_events.groupby("record_date", dropna=False)}
    pay_dates = {str(r.event_id): pd.Timestamp(r.payment_date) for r in p_events.itertuples(index=False)}

    pending_etf_targets = {}
    fills, etf_events_log, stock_rejects, etf_order_log = [], [], [], []
    daily_rows = []
    decision_rows = []
    active_stock_fraction = 0.0
    last_weights = np.zeros(4)

    for i, date in enumerate(calendar):
        date = pd.Timestamp(date)
        next_day = pd.Timestamp(calendar[i + 1]) if i + 1 < len(calendar) else date + pd.Timedelta(days=10)
        quotes = market.loc[market.trade_date == date].set_index("symbol").to_dict("index")

        # 非交易日到账的基金分红现金，下一开盘可用
        for event_id in list(account.fund_receivables):
            if pay_dates[event_id] < date:
                account.cash += account.fund_receivables.pop(event_id)

        # ETF 份额转换（除息日开盘前）
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
                ot = int(pending_etf_targets[date].get(asset, 0))
                pending_etf_targets[date][asset] = int(ot * ratio) // etfs.PRODUCT_RULES[asset]["buy_lot"] * etfs.PRODUCT_RULES[asset]["buy_lot"]

        # 开盘价标记
        for asset in etfs.ASSET_ORDER:
            row = lookup.get((asset, date))
            if row is not None and np.isfinite(row.open_cny) and row.open_cny > 0:
                account.fund_prices[asset] = float(row.open_cny)

        if date in merger_dates:
            account.merger(merger_dates[date], date, quotes)
        for symbol in by_unsupported.get(date, []):
            if account.quantity(symbol):
                account.issue(date, symbol, "UNSUPPORTED_FACTOR_CHANGE_WHILE_HELD")

        # 股票公司行为
        for event in by_stock_event.get(date, []):
            if event.get("event_type") == "REFERENCE_ADJUSTMENT_ONLY_NO_OLD_HOLDER_SHARES":
                continue
            account.corporate_action(event, date, pd.Timestamp(calendar[i - 1]) if i else date - pd.Timedelta(days=1))
        for symbol in account.lots:
            row = quotes.get(symbol, {})
            if np.isfinite(row.get("open", np.nan)) and row["open"] > 0:
                account.prices[symbol] = float(row["open"])
        opening_nav = account.nav(date)

        trade_start, reject_start = len(account.trades), len(account.rejections)
        # 先卖：股票 SELL + ETF SELL
        if account.orders:
            account.execute(date, next_day, quotes, side_filter="SELL", opening_nav_override=opening_nav)
        if date in pending_etf_targets:
            due_target = pending_etf_targets.pop(date)
            etf_order_rows = _etf_order_templates(date, due_target, account, lookup, calendar, i,
                                                  events_today, p_events.to_dict("records"), mode, {}, etfs)
        else:
            etf_order_rows = []
        etf_sell_fills = _etf_fill_sells(etf_order_rows, account, date, etfs)
        fills.extend(etf_sell_fills)

        # 再买：股票 + ETF 共享现金缩放
        stock_opts = _buy_options_stock(account, quotes, date, opening_nav, engine, fee_mult)
        etf_opts = _buy_options_etf(etf_order_rows)
        scale, stock_qty, etf_qty = _ration_buys(stock_opts, etf_opts, account.cash, date, fee_mult, engine, etfs)
        if account.orders:
            account.execute(date, next_day, quotes, side_filter="BUY", buy_scale=scale,
                            buy_quantities=stock_qty, opening_nav_override=opening_nav)
        fills.extend([{**t, "asset_class": "STOCK", "asset_id": t["symbol"], "units": t["shares"],
                       "side": t["side"], "stamp_tax": t.get("stamp_tax", 0.0)}
                      for t in account.trades[trade_start:]])
        fills.extend(_etf_fill_buys(etf_order_rows, etf_qty, account, date, etfs))

        # 除息日分红应收
        for e in events_today:
            if e["event_type"] == "cash_distribution":
                event_id = str(e["event_id"])
                if event_id not in account.fund_entitlements:
                    raise GateError(f"missing ETF record-date entitlement for {event_id}")
                amount = etfs.half_up_cents(
                    __import__("decimal").Decimal(account.fund_entitlements[event_id]) *
                    __import__("decimal").Decimal(str(e["cash_per_old_unit"])))
                account.fund_receivables[event_id] = amount

        # 收盘标记 + 到账
        for asset in etfs.ASSET_ORDER:
            row = lookup.get((asset, date))
            if row is None or not np.isfinite(row.close_cny) or row.close_cny <= 0:
                raise GateError(f"missing close price for {asset} on {date.date()}")
            account.fund_prices[asset] = float(row.close_cny)
        for event_id in list(account.fund_receivables):
            if pay_dates[event_id] == date:
                account.cash += account.fund_receivables.pop(event_id)

        account.close(date, quotes)
        drow = dict(account.daily[-1])
        drow["fund_units"] = sum(account.fund_units.values())
        drow["target_stock_fraction"] = active_stock_fraction
        for asset in etfs.ASSET_ORDER:
            drow[f"fund_units_{asset}"] = int(account.fund_units.get(asset, 0))
            drow[f"fund_mv_{asset}"] = float(account.fund_units.get(asset, 0) * account.fund_prices.get(asset, 0.0))
            drow[f"fund_weight_{asset}"] = float(drow[f"fund_mv_{asset}"] / drow["nav"]) if drow["nav"] else 0.0
        daily_rows.append(drow)

        # NAV 恒等式核对
        expected_nav = (account.cash
                        + sum(account.quantity(s) * account.prices.get(s, 0) for s in account.lots)
                        + sum(account.fund_units[a] * account.fund_prices[a] for a in etfs.ASSET_ORDER)
                        + sum(x["cash"] for x in account.receivables) + sum(account.fund_receivables.values())
                        - account.tax_liability(date))
        if abs(account.daily[-1]["nav"] - expected_nav) > 1e-6 or account.cash < -1e-6:
            raise GateError(f"NAV reconciliation failed {date.date()}")

        # 登记日持仓快照
        for e in by_record.get(date, []):
            account.fund_entitlements[str(e["event_id"])] = int(account.fund_units.get(str(e["asset_id"]), 0))

        # Frozen model outputs decide the common portfolio weights after close;
        # the shared-cash engine executes them at the next session open.
        if i % DECISION_CADENCE == 0:
            if target_weight_schedule is not None:
                target_weights = np.asarray(target_weight_schedule[date], dtype=float)
                solver_status = "PRECOMPUTED_DYNAMIC_TARGET"
            elif fixed_target_weights is not None:
                target_weights = np.asarray(fixed_target_weights, dtype=float)
                solver_status = "FROZEN_FIXED_WEIGHTS"
            elif policy_id == "C1_fixed25":
                target_weights = np.full(4, .25)
                solver_status = "FIXED_25_PERCENT"
            else:
                mu = np.asarray(predictions.loc[date, [f"{policy_id}_{a}" for a in ["csi500", "sp500", "gold", "gov_bond"]]], dtype=float)
                hist = market_returns.loc[:date].tail(60).to_numpy(dtype=float)
                if len(hist) < 40 or not np.isfinite(hist).all():
                    target_weights = last_weights.copy()
                    solver_status = "FALLBACK_PREVIOUS_INSUFFICIENT_COV"
                else:
                    nav_now = float(account.daily[-1]["nav"])
                    held_stock = sum(account.quantity(s) * account.prices.get(s, 0.0) for s in account.lots) / nav_now
                    held_etf = [account.fund_units[a] * account.fund_prices[a] / nav_now for a in ("sp500_qdii", "physical_gold", "rmb_gov_bond")]
                    current = np.asarray([held_stock, *held_etf], dtype=float)
                    cov = np.cov(hist, rowvar=False, ddof=1)
                    from run_joint_stock_etf_fast_v1 import weights as _solve_weights
                    target_weights, solver_status = _solve_weights(mu, cov, current)
                last_weights = np.asarray(target_weights, dtype=float)
            active_stock_fraction = float(target_weights[0])
            targets = {a: 0 for a in etfs.ASSET_ORDER}
            for asset, w in zip(ETF_3_MODULES, target_weights[1:]):
                close = float(lookup[(asset, date)].close_cny)
                lot = etfs.PRODUCT_RULES[asset]["buy_lot"]
                targets[asset] = int((float(w) * account.daily[-1]["nav"] / close) // lot) * lot
            if i + 1 < len(calendar):
                pending_etf_targets[pd.Timestamp(calendar[i + 1])] = targets
            decision_rows.append({"signal_date":date,"execute_not_before":next_day,"policy_id":policy_id,
                "w_stock":float(target_weights[0]),"w_sp500":float(target_weights[1]),
                "w_gold":float(target_weights[2]),"w_bond":float(target_weights[3]),
                "w_cash":float(max(0.,1.-np.sum(target_weights))),"optimizer_status":solver_status,
                "prediction_source":"TRAIN_SELECTED_FIXED_WEIGHT" if fixed_target_weights is not None else ("FIXED" if policy_id == "C1_fixed25" else "STOCK_TARGET_PROXY_CSI500")})

        # E00 stock selection is preserved; only its total budget is dynamic.
        if i % 5 == 0 and i + 1 < len(calendar):
            rank = stock_rank.get(date)
            if rank is None:
                raise GateError(f"missing stock ranking for {date.date()}")
            account.plan(date, rank, quotes, stock_fraction=active_stock_fraction)

        if trade_start < len(account.trades):
            for t in account.trades[trade_start:]:
                fills[-1] if fills else None  # no-op guard
        stock_rejects += account.rejections[reject_start:]
        for row in etf_order_rows:
            etf_order_log.append({k: v for k, v in row.items() if k != "fee_spec"})

    daily = pd.DataFrame(daily_rows)
    trades = pd.DataFrame(fills)
    metrics = _metrics(daily, trades)
    metrics["mean_cash_fraction"] = float((daily.cash / daily.nav).mean())
    metrics["mean_stock_fraction"] = float((daily.stock_market_value / daily.nav).mean())
    metrics["mean_etf_fraction"] = float((daily.fund_market_value / daily.nav).mean())
    metrics["closing_cash"] = float(account.cash)
    metrics["issues"] = len(account.issues)
    metrics["stock_rejections"] = len(stock_rejects)
    metrics["etf_orders"] = len(etf_order_log)
    metrics["nav_recon_max_abs"] = float(np.max(np.abs(
        daily.nav - (daily.cash + daily.market_value + daily.dividend_receivable - daily.estimated_dividend_tax))))
    metrics["optimizer_fallbacks"] = sum(str(x["optimizer_status"]).startswith("FALLBACK") for x in decision_rows)
    metrics["mean_target_weights"] = {k:float(pd.DataFrame(decision_rows)[k].mean()) for k in ("w_stock","w_sp500","w_gold","w_bond","w_cash")}
    daily.attrs["decisions"] = decision_rows
    daily.attrs["etf_events"] = etf_events_log
    daily.attrs["stock_rejections"] = stock_rejects
    daily.attrs["etf_orders"] = etf_order_log
    daily.attrs["account_issues"] = account.issues
    return daily, trades, metrics


def main():
    import argparse, hashlib, time
    ap=argparse.ArgumentParser(); ap.add_argument('--mode',choices=['standard','stress'],default='standard'); mode=ap.parse_args().mode
    from run_joint_stock_etf_fast_v1 import load_panel, ASSETS
    out=OUT; out.mkdir(parents=True,exist_ok=True)
    t0=time.perf_counter()
    market, calendar = load_stock_market()
    rankings = build_stock_rankings()
    prices, events, lookup = load_etf()
    _, fullcal, _, _, _, returns, _, _, _ = load_panel()
    returns.index=fullcal
    pred_path=ROOT/"results"/"joint_stock_etf_fast_v1"/"predictions.csv"
    pred=pd.read_csv(pred_path,parse_dates=["trade_date"]).set_index("trade_date")
    pred.index=pred.index.normalize()
    if not calendar.isin(pred.index).all(): raise GateError("model prediction dates do not cover real account calendar")
    if not calendar.isin(returns.index).all(): raise GateError("proxy covariance dates do not cover real account calendar")
    print(f"真实E00+三ETF账户 {calendar[0].date()} ~ {calendar[-1].date()}，{len(calendar)}日")
    result={}
    for policy in ["P","MLP","C1_fixed25"]:
        daily,trades,metrics=run_mixed_account(rankings["E00"],mode,calendar,market,prices,events,lookup,
            policy,pred,returns)
        sub=out/policy/mode; sub.mkdir(parents=True,exist_ok=True)
        daily.to_csv(sub/"daily.csv",index=False); trades.to_csv(sub/"fills.csv",index=False)
        pd.DataFrame(daily.attrs["decisions"]).to_csv(sub/"target_decisions.csv",index=False)
        pd.DataFrame(daily.attrs["etf_events"]).to_csv(sub/"etf_events.csv",index=False)
        pd.DataFrame(daily.attrs["stock_rejections"]).to_csv(sub/"stock_rejections.csv",index=False)
        pd.DataFrame(daily.attrs["etf_orders"]).to_csv(sub/"etf_orders.csv",index=False)
        result[policy]=metrics
        print(f"{policy}: 累计={metrics['total_return']:+.2%} 年化={metrics['annualized_return']:+.2%} "
              f"Sharpe={metrics['sharpe_zero_cash']:.3f} MDD={metrics['max_drawdown']:.2%} "
              f"股票实仓={metrics['mean_stock_fraction']:.1%} ETF={metrics['mean_etf_fraction']:.1%} 现金={metrics['mean_cash_fraction']:.1%}",flush=True)
    sources=[PRICE_PATH,EVENT_PATH,MARKET_PATH,PRED_PATH,PANEL_PATH,
        ROOT/"data"/"processed"/"multi_asset_handoff_v2"/"daily_prices.parquet"]
    hashes={str(p.relative_to(ROOT)).replace("\\","/"):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    summary={"status":"COMPLETE_REAL_SHARED_CASH_N04_EXECUTION_REQUIRES_INDEPENDENT_REVIEW",
        "period":[calendar[0].strftime("%Y-%m-%d"),calendar[-1].strftime("%Y-%m-%d")],"sessions":len(calendar),
        "policies":{"P":"Ridge predicted 20d returns + gamma5, E00 actual stock ranking", "MLP":"single32-hidden MLP return predictor seed11 + same optimizer/E00 ranking", "C1_fixed25":"E00 stocks25% + each of SP500/gold/bond25% targets"},
        "stock_prediction_target_proxy":"CSI500 ETF return; actual account stock positions use frozen E00 ranks, no CSI500 proxy NAV is spliced into account",
        "accounts":result,"input_sha256":hashes,"elapsed_seconds":time.perf_counter()-t0,"cost_mode":mode,
        "stress_accounts":"NOT_RUN","period_classification":"2024-2025 previously viewed exploratory backtest",
        "limitations":["stock prediction target and covariance use CSI500 proxy, not E00 module gross return; E00 actual ranked equities are traded in the account",
        "one MLP seed and 40 epochs only; 2023 report-only, no tuning; three standard accounts only",
        "account engine inherits source review limitations recorded by N04; needs independent inventory/cash/fees/event audit"]}
    summary_path=out/("real_account_summary.json" if mode=="standard" else "stress_account_summary.json")
    summary_path.write_text(json.dumps(summary,ensure_ascii=False,indent=2,default=str),encoding="utf-8")
    print(json.dumps({"status":summary["status"],"metrics":result,"elapsed_seconds":summary["elapsed_seconds"]},ensure_ascii=False,indent=2,default=str))


if __name__ == "__main__":
    main()
