"""Raw-price long-only ledger; no broker connection and no future label access."""
from dataclasses import dataclass
import math
import numpy as np
import pandas as pd
from audit_account_inputs import buy_quantity, fee_components


@dataclass
class Lot:
    quantity: int
    acquired: pd.Timestamp
    available: pd.Timestamp
    dividend_base: float = 0.


def dividend_tax_rate(acquired, sold):
    """Natural-month/year boundaries, not an arbitrary 30/365-day approximation."""
    acquired, sold = pd.Timestamp(acquired), pd.Timestamp(sold)
    if sold < acquired + pd.DateOffset(months=1):
        return .2
    if sold < acquired + pd.DateOffset(years=1):
        return .1
    return 0.


def sale_quantity(symbol, requested, available, total):
    quantity = min(int(requested), int(available))
    if quantity == total:
        return quantity
    if symbol.startswith('SHSE.688'):
        return quantity if quantity >= 200 else 0
    return quantity // 100 * 100


class Account:
    def __init__(self, capital=1_000_000., slippage=.0005, commission_multiplier=1.):
        self.cash = capital
        self.capital = capital
        self.slippage = slippage
        self.commission_multiplier = commission_multiplier
        self.lots = {}
        self.prices = {}
        self.receivables = []
        self.orders = None
        self.trades = []
        self.rejections = []
        self.issues = []
        self.events = []
        self.daily = []

    def quantity(self, symbol):
        return sum(lot.quantity for lot in self.lots.get(symbol, []))

    def issue(self, day, symbol, reason):
        self.issues.append(dict(date=day, symbol=symbol, reason=reason))

    def tax_liability(self, day):
        return sum(lot.quantity * lot.dividend_base * dividend_tax_rate(lot.acquired, day)
                   for lots in self.lots.values() for lot in lots)

    def nav(self, day):
        return (self.cash + sum(self.quantity(s) * self.prices.get(s, 0) for s in self.lots)
                + sum(r['cash'] for r in self.receivables) - self.tax_liability(day))

    def corporate_action(self, event, day, previous_day):
        symbol = event['symbol']
        original = list(self.lots.get(symbol, []))
        shares = sum(l.quantity for l in original)
        if not shares:
            return
        if event.get('record_date') != previous_day:
            self.issue(day, symbol, 'REGISTRATION_NOT_PREVIOUS_SESSION_NEEDS_POSITION_SNAPSHOT')
        if event.get('unresolved', False):
            self.issue(day, symbol, 'UNRESOLVED_CORPORATE_EVENT')
        cash_ps = event['cash_ps']
        bonus, transfer = event['bonus_ratio'], event['transfer_ratio']
        ratio = bonus + transfer
        if cash_ps:
            self.receivables.append(dict(symbol=symbol, cash=shares * cash_ps, pay_date=event.get('pay_date')))
            if pd.isna(event.get('pay_date')):
                self.issue(day, symbol, 'MISSING_CASH_PAYMENT_DATE_RECEIVABLE_FROZEN')
        credit = event.get('share_date')
        if ratio and pd.isna(credit):
            self.issue(day, symbol, 'MISSING_SHARE_LISTING_DATE_NEW_SHARES_FROZEN')
            credit = pd.Timestamp('2099-12-31')
        bonus_total = int(math.floor(shares * ratio + 1e-7))
        allocated = 0
        for i, lot in enumerate(original):
            # Cash dividends and taxable bonus face value belong to original shares.
            lot.dividend_base += cash_ps + bonus
            extra = int(math.floor(lot.quantity * ratio + 1e-7)) if i + 1 < len(original) else bonus_total - allocated
            allocated += extra
            if extra:
                self.lots[symbol].append(Lot(extra, lot.acquired, credit))
        self.lots[symbol].sort(key=lambda lot: lot.acquired)
        if ratio or cash_ps:
            # Adjust stale reference and overnight target after an actual share event.
            self.prices[symbol] = (self.prices[symbol] - cash_ps) / (1 + ratio)
            if ratio and self.orders and symbol in self.orders['targets']:
                old = self.orders['targets'][symbol]
                self.orders['targets'][symbol] = int(math.floor(old * (1 + ratio)))
        self.events.append(dict(date=day, symbol=symbol, type='ENTITLEMENT', original_shares=shares,
                                cash_receivable=shares * cash_ps, new_shares=bonus_total))

    def reject(self, day, symbol, side, reason):
        self.rejections.append(dict(date=day, symbol=symbol, side=side, reason=reason))

    def merger(self, specification, day, market):
        old, new = specification['old_symbol'], specification['new_symbol']
        old_lots = self.lots.get(old, [])
        if not old_lots:
            return
        ratio = specification['ratio']
        total_old = self.quantity(old)
        total_new = int(math.floor(total_old * ratio + 1e-7))
        allocated = 0
        for i, lot in enumerate(old_lots):
            quantity = int(math.floor(lot.quantity * ratio + 1e-7)) if i + 1 < len(old_lots) else total_new - allocated
            allocated += quantity
            if quantity:
                # Pending tax bases stay obligations; acquisition holding period carried.
                self.lots.setdefault(new, []).append(Lot(quantity, lot.acquired, pd.Timestamp(specification['available_date']),
                                                         lot.dividend_base * lot.quantity / quantity))
        del self.lots[old]
        self.lots[new].sort(key=lambda lot: lot.acquired)
        reference = market.get(new, {}).get('open', np.nan)
        if not np.isfinite(reference) or reference <= 0:
            reference = market.get(new, {}).get('pre_close', np.nan)
        if not np.isfinite(reference) or reference <= 0:
            self.issue(day, new, 'MERGER_RECEIVABLE_MISSING_MARK')
            reference = self.prices[old] / ratio
        self.prices[new] = reference
        self.events.append(dict(date=day, symbol=old, type='MERGER_CONVERSION', original_shares=total_old,
                                new_symbol=new, new_shares=total_new, available=specification['available_date']))

    def execute(self, day, next_day, market):
        if not self.orders:
            return
        order = self.orders
        assert order['signal_date'] < day
        targets = order['targets']
        symbols = list(dict.fromkeys(list(self.lots) + order['ranking']))
        # Opening marks are used for execution constraints, never for stock ranking.
        for symbol in self.lots:
            row = market.get(symbol, {})
            if np.isfinite(row.get('open', np.nan)) and row['open'] > 0:
                self.prices[symbol] = row['open']
        opening_nav = self.nav(day)
        # Sell before buy, while yesterday's lots and credited shares remain distinct.
        for side in ('SELL', 'BUY'):
            sequence = symbols if side == 'SELL' else order['ranking']
            for symbol in sequence:
                held = self.quantity(symbol)
                difference = targets.get(symbol, 0) - held
                if (side == 'BUY' and difference <= 0) or (side == 'SELL' and difference >= 0):
                    continue
                row = market.get(symbol, {})
                opening = row.get('open', np.nan)
                if row.get('is_suspended', np.nan) != 0 or not np.isfinite(opening) or opening <= 0:
                    self.reject(day, symbol, side, 'SUSPENDED_OR_MISSING_OPEN')
                    continue
                upper, lower = row.get('upper_limit', np.nan), row.get('lower_limit', np.nan)
                if not np.isfinite(upper) or not np.isfinite(lower):
                    self.reject(day, symbol, side, 'UNKNOWN_LIMITS')
                    continue
                if (side == 'BUY' and upper > 0 and opening >= upper - 1e-5) or (side == 'SELL' and lower > 0 and opening <= lower + 1e-5):
                    self.reject(day, symbol, side, 'OPEN_AT_LIMIT_CONSERVATIVE_NO_FILL')
                    continue
                price = opening * (1 + self.slippage if side == 'BUY' else 1 - self.slippage)
                if (upper > 0 and price > upper + 1e-5) or (lower > 0 and price < lower - 1e-5):
                    self.reject(day, symbol, side, 'SLIPPAGE_OUTSIDE_PRICE_LIMIT')
                    continue
                tax = 0.
                if side == 'SELL':
                    available = sum(l.quantity for l in self.lots.get(symbol, []) if l.available <= day)
                    quantity = sale_quantity(symbol, -difference, available, held)
                    if not quantity:
                        self.reject(day, symbol, side, 'T_PLUS_ONE_OR_FROZEN_ODD_LOT')
                        continue
                    remaining = quantity
                    for lot in self.lots[symbol]:
                        if lot.available > day:
                            continue
                        sold = min(remaining, lot.quantity)
                        tax += sold * lot.dividend_base * dividend_tax_rate(lot.acquired, day)
                        remaining -= sold
                        if not remaining:
                            break
                    fees = fee_components(quantity * price, side, day)
                    fee = fees['total'] * self.commission_multiplier
                    if self.cash + quantity * price - fee - tax < -1e-6:
                        self.reject(day, symbol, side, 'INSUFFICIENT_CASH_FOR_SALE_FEES')
                        continue
                    remaining = quantity
                    for lot in self.lots[symbol]:
                        if lot.available <= day:
                            sold = min(remaining, lot.quantity)
                            lot.quantity -= sold
                            remaining -= sold
                    self.lots[symbol] = [lot for lot in self.lots[symbol] if lot.quantity]
                    if not self.lots[symbol]:
                        del self.lots[symbol]
                    self.cash += quantity * price - fee - tax
                else:
                    # Signal-date budget remains a ceiling, even if today's price falls.
                    cap = max(0., opening_nav * .03 - held * opening)
                    budget = min(cap, order['budgets'].get(symbol, 0), max(0., self.cash - 6 * self.commission_multiplier))
                    quantity = min(int(difference), buy_quantity(symbol, budget, price))
                    quantity = buy_quantity(symbol, quantity * price + 1e-6, price)
                    while quantity:
                        fees = fee_components(quantity * price, side, day)
                        fee = fees['total'] * self.commission_multiplier
                        if quantity * price + fee <= self.cash + 1e-6:
                            break
                        quantity = buy_quantity(symbol, (quantity - 1) * price, price)
                    if not quantity:
                        self.reject(day, symbol, side, 'LOT_CAP_OR_CASH_BUDGET')
                        continue
                    self.cash -= quantity * price + fee
                    self.lots.setdefault(symbol, []).append(Lot(quantity, day, next_day))
                    self.lots[symbol].sort(key=lambda lot: lot.acquired)
                self.trades.append(dict(date=day, signal_date=order['signal_date'], symbol=symbol, side=side,
                                        shares=quantity, open_reference=opening, price=price, notional=quantity * price,
                                        fee=fee, commission=fees['commission_including_exchange_regulatory'] * self.commission_multiplier,
                                        transfer_fee=fees['transfer'] * self.commission_multiplier, stamp_tax=fees['stamp'] * self.commission_multiplier,
                                        dividend_tax=tax, slippage_cost=quantity * abs(price - opening)))
                assert self.cash >= -1e-6
        self.orders = None

    def close(self, day, market):
        for symbol in self.lots:
            row = market.get(symbol, {})
            close = row.get('close', np.nan)
            if np.isfinite(close) and close > 0:
                self.prices[symbol] = close
            elif row.get('is_suspended', np.nan) != 1:
                self.issue(day, symbol, 'UNEXPLAINED_MISSING_CLOSE_LAST_MARK_USED')
        paid = 0.
        pending = []
        for item in self.receivables:
            if pd.notna(item['pay_date']) and item['pay_date'] <= day:
                paid += item['cash']
                self.events.append(dict(date=day, symbol=item['symbol'], type='DIVIDEND_PAYMENT', cash_paid=item['cash']))
            else:
                pending.append(item)
        self.cash += paid
        self.receivables = pending
        market_value = sum(self.quantity(s) * self.prices[s] for s in self.lots)
        nav = self.nav(day)
        self.daily.append(dict(date=day, nav=nav, cash=self.cash, market_value=market_value,
                               dividend_receivable=sum(r['cash'] for r in self.receivables),
                               estimated_dividend_tax=self.tax_liability(day), holdings=len(self.lots), cash_paid=paid))
        assert abs(nav - (self.cash + market_value + self.daily[-1]['dividend_receivable'] - self.daily[-1]['estimated_dividend_tax'])) < 1e-6
        assert all(l.quantity > 0 and isinstance(l.quantity, int) for lots in self.lots.values() for l in lots)

    def plan(self, day, ranking, market, benchmark=False):
        if benchmark:
            selected = ranking
        else:
            retain = [s for s in ranking[:120] if self.quantity(s)]
            selected_set = set(retain[:80])
            for symbol in ranking:
                if len(selected_set) >= 80:
                    break
                selected_set.add(symbol)
            selected = [s for s in ranking if s in selected_set]
        nav = self.nav(day)
        budget = nav / (len(selected) if benchmark and selected else 80)
        targets, budgets = {}, {}
        for symbol in selected:
            close = market.get(symbol, {}).get('close', np.nan)
            if not np.isfinite(close) or close <= 0:
                continue
            targets[symbol] = buy_quantity(symbol, max(0, budget - 6), close)
            budgets[symbol] = budget
        self.orders = dict(signal_date=day, targets=targets, budgets=budgets, ranking=selected)
