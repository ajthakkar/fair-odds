"""Step 6: a compact JSON of the results for the interactive chart on my site.

Run: python scripts/export_web.py [--out ../portfolio/public/data/prediction-markets.json]
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def bins(cal):
    return [
        {k: c[k] for k in ("lo", "hi", "n", "mean_price", "freq", "lo_ci", "hi_ci")}
        for c in cal
        if c["n"] >= 30
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / "web.json"))
    args = ap.parse_args()
    R = json.loads((ROOT / "results" / "metrics.json").read_text())
    web = {
        "markets": R["markets"],
        "events": R["events"],
        "horizons": {
            h: {"n": d["n"], "brier": d["brier"], "slope": d["slope"], "bins": bins(d["calibration"]), "ends": d["ends"]}
            for h, d in R["by_horizon"].items()
        },
        "categories": {
            c: {"n": d["n"], "brier": d["brier"], "slope": d["slope"], "bins": bins(d["calibration"])}
            for c, d in R["by_category"].items()
        },
        "out_of_sample": R["out_of_sample"],
    }
    Path(args.out).write_text(json.dumps(web, separators=(",", ":")))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
