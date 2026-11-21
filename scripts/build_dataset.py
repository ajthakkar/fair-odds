"""Step 3: one row per (market, horizon) with the YES price 1, 7, and 30 days before close.

Run: python scripts/build_dataset.py   ->  data/dataset.csv
"""
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from pmcal.dataset import horizon_rows  # noqa: E402


def main():
    with (ROOT / "data" / "markets.csv").open(encoding="utf-8") as f:
        markets = list(csv.DictReader(f))
    rows, used = [], 0
    for m in markets:
        path = ROOT / "data" / "prices" / f"{m['market_id']}.json"
        if not path.exists():
            continue
        r = horizon_rows(m, json.loads(path.read_text()))
        if r:
            used += 1
        rows += r
    out = ROOT / "data" / "dataset.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out}: {len(rows):,} rows from {used:,} markets")


if __name__ == "__main__":
    main()
