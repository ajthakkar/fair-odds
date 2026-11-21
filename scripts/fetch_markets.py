"""Step 1: list resolved Polymarket markets, with their event, category tags, volume, and outcome.

Polymarket's Gamma API is public (no account or key). Events group related markets
("Who will win the election?" holds one yes/no market per candidate) and carry the
category tags, so we page through closed events and pull out their markets.

Run: python scripts/fetch_markets.py   ->  data/markets.csv   (about 45 minutes; 2026 has ~15,000 events a month)
"""
import csv
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
GAMMA = "https://gamma-api.polymarket.com"
START, END = "2024-01-01", "2026-11-15"
MIN_EVENT_VOLUME = 50_000  # dollars traded across the event
MIN_MARKET_VOLUME = 50_000  # dollars traded in the market itself

# Map Polymarket's tags to a few broad categories, checked in this order.
CATEGORIES = [
    ("Sports", {"sports", "nfl", "nba", "mlb", "nhl", "soccer", "football", "basketball", "tennis", "golf", "ufc", "mma",
                "boxing", "f1", "formula 1", "cricket", "esports", "baseball", "hockey", "epl", "champions league",
                "world cup", "olympics", "ncaa", "cfb", "college football", "games"}),
    ("Crypto", {"crypto", "bitcoin", "ethereum", "solana", "crypto prices", "xrp", "dogecoin", "memecoins", "airdrops"}),
    ("Politics", {"politics", "elections", "us election", "global elections", "trump", "geopolitics", "world",
                  "congress", "senate", "us politics", "middle east", "ukraine", "foreign policy", "courts"}),
    ("Economy", {"economy", "fed", "fed rates", "inflation", "business", "finance", "stocks", "economic policy",
                 "interest rates", "recession", "jobs", "gdp", "tariffs", "earnings", "ipo", "commodities"}),
    ("Tech & culture", {"tech", "ai", "science", "culture", "pop culture", "movies", "music", "awards", "celebrities",
                        "tweet markets", "mentions", "entertainment", "space", "weather", "climate"}),
]


def category(tags):
    labels = {t.lower() for t in tags}
    for name, keys in CATEGORIES:
        if labels & keys:
            return name
    return "Other"


def get(url, params, tries=6):
    for attempt in range(tries):
        try:
            r = requests.get(url, params=params, timeout=120)
            if r.status_code == 429:
                time.sleep(5 * (attempt + 1))
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException:
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"gave up on {url} {params}")


PAGE = 100  # the API returns at most 100 events per request
MAX_OFFSET = 2000  # and refuses large offsets, so long windows are split in half


def events_between(start, end):
    """All closed events ending in [start, end), splitting the window when it holds too many."""
    found, offset = [], 0
    while True:
        batch = get(f"{GAMMA}/events", {
            "closed": "true", "limit": PAGE, "offset": offset, "volume_min": MIN_EVENT_VOLUME,
            "end_date_min": start.isoformat(), "end_date_max": end.isoformat(), "order": "endDate", "ascending": "true",
        })
        if not isinstance(batch, list):
            batch = []
        found += batch
        if len(batch) < PAGE:
            return found
        offset += PAGE
        if offset >= MAX_OFFSET:
            mid = start + (end - start) / 2
            if (end - start).total_seconds() < 3600:
                return found  # an hour with this many events: keep what we have
            return events_between(start, mid) + events_between(mid, end)


def months(start, end):
    cur = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
    stop = datetime.fromisoformat(end).replace(tzinfo=timezone.utc)
    while cur < stop:
        nxt = (cur.replace(day=1) + timedelta(days=32)).replace(day=1)
        yield cur, min(nxt, stop)
        cur = nxt


FIELDS = ["market_id", "event_id", "question", "event_title", "category", "tags", "neg_risk", "volume",
          "start", "end", "closed", "yes_won", "yes_token"]


def main():
    """Writes each month as it finishes, so a crash late in the ~45-minute run keeps everything before it."""
    out = ROOT / "data" / "markets.csv"
    f = out.open("w", newline="", encoding="utf-8")
    w = csv.DictWriter(f, fieldnames=FIELDS)
    w.writeheader()
    kept, n_events, seen = 0, 0, set()
    for a, b in months(START, END):
        rows = []
        batch = [e for e in events_between(a, b) if e["id"] not in seen]
        seen.update(e["id"] for e in batch)
        print(f"{a:%Y-%m}: {len(batch):,} events", file=sys.stderr, flush=True)
        for e in batch:
            n_events += 1
            tags = [t.get("label", "") for t in e.get("tags") or []]
            for m in e.get("markets") or []:
                prices = m.get("outcomePrices")
                tokens = m.get("clobTokenIds")
                if prices not in ('["1", "0"]', '["0", "1"]') or not tokens:
                    continue  # unresolved, voided, or split 50/50
                if json.loads(m.get("outcomes") or "[]") != ["Yes", "No"]:
                    continue  # keep plain yes/no questions so "price" always means P(yes)
                vol = float(m.get("volumeNum") or m.get("volume") or 0)
                if vol < MIN_MARKET_VOLUME:
                    continue
                rows.append({
                    "market_id": m["id"],
                    "event_id": e["id"],
                    "question": (m.get("question") or "").replace("\n", " "),
                    "event_title": (e.get("title") or "").replace("\n", " "),
                    "category": category(tags),
                    "tags": "|".join(tags),
                    "neg_risk": bool(m.get("negRisk")),
                    "volume": round(vol),
                    "start": m.get("startDate") or e.get("startDate") or "",
                    "end": m.get("endDate") or "",
                    "closed": m.get("closedTime") or "",
                    "yes_won": 1 if prices == '["1", "0"]' else 0,
                    "yes_token": json.loads(tokens)[0],
                })
        w.writerows(rows)
        f.flush()
        kept += len(rows)
        print(f"   {n_events:,} events so far, {kept:,} markets kept", file=sys.stderr, flush=True)
    f.close()
    print(f"wrote {out} ({kept:,} markets from {n_events:,} events)")


if __name__ == "__main__":
    main()
