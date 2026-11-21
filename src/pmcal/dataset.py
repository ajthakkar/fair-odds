"""Turn price histories into one row per (market, horizon): the YES price some days before close."""
from __future__ import annotations

import bisect
from datetime import datetime, timedelta, timezone

HORIZONS = (1, 7, 30)  # days before the market closed
MAX_STALENESS_HOURS = 48  # the last trade before the horizon must be at most this old


def parse_time(s: str | None) -> datetime | None:
    if not s:
        return None
    s = s.strip().replace(" ", "T")
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    if s.endswith("+00"):
        s += ":00"
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def price_at(history: list[list[float]], when: datetime, max_staleness_hours: float = MAX_STALENESS_HOURS) -> float | None:
    """The last recorded price at or before `when`, if it isn't too stale. History is [[unix_seconds, price], ...]."""
    if not history:
        return None
    times = [t for t, _ in history]
    ts = when.timestamp()
    i = bisect.bisect_right(times, ts) - 1
    if i < 0:
        return None  # the market hadn't traded yet
    if ts - times[i] > max_staleness_hours * 3600:
        return None
    return float(history[i][1])


def anchor_time(market: dict) -> datetime | None:
    """When the outcome stopped being uncertain: the earlier of the scheduled end and the recorded close.
    The recorded close can come hours or days after the result was known (the market is formally
    resolved later), and measuring from it would let prices that already know the answer count as
    forecasts. A market that resolves early (the event happened) closes before its scheduled end."""
    times = [t for t in (parse_time(market.get("closed")), parse_time(market.get("end"))) if t]
    return min(times) if times else None


def horizon_rows(market: dict, history: list[list[float]], horizons=HORIZONS) -> list[dict]:
    close = anchor_time(market)
    if close is None:
        return []
    history = sorted(history)
    rows = []
    for h in horizons:
        p = price_at(history, close - timedelta(days=h))
        if p is None:
            continue
        rows.append(
            {
                "market_id": market["market_id"],
                "event_id": market["event_id"],
                "category": market["category"],
                "volume": float(market["volume"]),
                "close": close.date().isoformat(),
                "horizon": h,
                "price": p,
                "yes_won": int(market["yes_won"]),
            }
        )
    return rows
