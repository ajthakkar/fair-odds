"""Step 2: download each market's price history for its last 35 days of trading.

Uses Polymarket's public CLOB price-history endpoint: daily prices of the YES share. (For closed
markets the endpoint only answers with interval=max at daily fidelity; hourly or date-range
requests come back empty.) Only the last 35 days before close are kept.
Each market is saved to data/prices/<market_id>.json, so the script can be stopped and
restarted without downloading anything twice.

Run: python scripts/fetch_prices.py [--share 0.4]   (0.4 = a stable 40% sample of markets)
"""
import argparse
import csv
import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
CLOB = "https://clob.polymarket.com/prices-history"
DIR = ROOT / "data" / "prices"
WINDOW_DAYS = 35


def parse_time(s):
    s = s.strip().replace(" ", "T").replace("+00", "+00:00") if s else ""
    if s.endswith("+00:00:00"):
        s = s[:-3]
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def close_time(m):
    """When trading stopped: the recorded close time, or the scheduled end date as a fallback."""
    return parse_time(m["closed"]) or parse_time(m["end"])


def fetch(m, session):
    path = DIR / f"{m['market_id']}.json"
    if path.exists():
        return "cached"
    t1 = close_time(m)
    if t1 is None:
        return "no close time"
    t0 = (t1 - timedelta(days=WINDOW_DAYS)).timestamp()
    params = {"market": m["yes_token"], "interval": "max", "fidelity": 1440}
    for attempt in range(6):
        try:
            r = session.get(CLOB, params=params, timeout=60)
            if r.status_code == 429:
                time.sleep(4 * (attempt + 1))
                continue
            r.raise_for_status()
            hist = [[p["t"], p["p"]] for p in r.json().get("history", []) if p["t"] >= t0 - 86400]
            path.write_text(json.dumps(hist))
            return "ok" if hist else "empty"
        except requests.RequestException:
            time.sleep(2 * (attempt + 1))
    return "failed"


def in_sample(market_id: str, share: float) -> bool:
    """A stable random sample: a market is in or out based on a hash of its ID, so a later run
    that adds new markets keeps every earlier choice."""
    h = int(hashlib.sha1(market_id.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return h < share


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", type=float, default=1.0, help="fraction of markets to include (stable hash sample)")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    DIR.mkdir(parents=True, exist_ok=True)
    with (ROOT / "data" / "markets.csv").open(encoding="utf-8") as f:
        markets = list(csv.DictReader(f))
    markets = [m for m in markets if in_sample(m["market_id"], args.share)]
    counts = {}
    session = requests.Session()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(fetch, m, session) for m in markets]
        for i, fut in enumerate(as_completed(futures), 1):
            status = fut.result()
            counts[status] = counts.get(status, 0) + 1
            if i % 200 == 0 or i == len(markets):
                print(f"\r{i:,}/{len(markets):,} {counts}", end="", file=sys.stderr)
    print(file=sys.stderr)
    print(counts)


if __name__ == "__main__":
    main()
