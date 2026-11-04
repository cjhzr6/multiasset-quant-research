"""Date-limited 2026 fee adapter for sealed-input quality work.

Commission is a project assumption of 0.03% with a 5 CNY minimum and includes
exchange/regulatory fees. Transfer fee and seller stamp tax are separate.
This does not claim any broker-specific commission schedule.
"""
import math
import pandas as pd

START = pd.Timestamp('2026-01-01')
END_EXCLUSIVE = pd.Timestamp('2026-09-22')  # sealed observation cutoff is 2026-09-21
COMMISSION_RATE = 0.0003
MINIMUM_COMMISSION = 5.0
TRANSFER_RATE = 0.00001
SELL_STAMP_RATE = 0.0005

def fee_components(amount, side, day):
    """Return fee components for supported 2026 dates (through 2026-09-21)."""
    try:
        amount = float(amount)
        day = pd.Timestamp(day)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('Unsupported amount, side or 2026 fee date') from exc
    if not math.isfinite(amount) or amount <= 0 or side not in ('BUY', 'SELL'):
        raise ValueError('Unsupported amount, side or 2026 fee date')
    if pd.isna(day) or not START <= day < END_EXCLUSIVE:
        raise ValueError('Unsupported amount, side or 2026 fee date')
    commission = max(MINIMUM_COMMISSION, amount * COMMISSION_RATE)
    transfer = amount * TRANSFER_RATE
    stamp = amount * SELL_STAMP_RATE if side == 'SELL' else 0.0
    return {
        'commission_including_exchange_regulatory': commission,
        'transfer': transfer,
        'stamp': stamp,
        'total': commission + transfer + stamp,
    }
