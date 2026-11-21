"""Step 4: measure calibration, the longshot bias, and whether a simple correction beats the market.

Run: python scripts/analyze.py   ->  results/metrics.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from pmcal.stats import (  # noqa: E402
    apply_recalibration,
    brier,
    calibration_table,
    cluster_bootstrap,
    fit_recalibration,
    log_loss,
    murphy,
)

N_BOOT = 1000
SPLIT_DATE = "2026-01-01"  # fit on markets that closed before this, test on the rest
LONGSHOT, FAVORITE = 0.10, 0.90


def round_dict(d, k=4):
    return {key: (round(v, k) if isinstance(v, float) else v) for key, v in d.items()}


def calibration_with_ci(df):
    p, y, ev = df.price.to_numpy(), df.yes_won.to_numpy(), df.event_id.to_numpy()
    rows = calibration_table(p, y)
    for r in rows:
        if r["n"] < 30:
            r["lo_ci"] = r["hi_ci"] = None
            continue
        in_bin = (p >= r["lo"]) & ((p < r["hi"]) | (r["hi"] >= 1.0))
        sub_idx = np.flatnonzero(in_bin)
        r["lo_ci"], r["hi_ci"] = cluster_bootstrap(ev[sub_idx], lambda i: y[sub_idx][i].mean(), N_BOOT, seed=1)
    return [round_dict(r) for r in rows]


def slope_with_ci(df):
    p, y, ev = df.price.to_numpy(), df.yes_won.to_numpy(), df.event_id.to_numpy()
    a, b = fit_recalibration(p, y)
    lo, hi = cluster_bootstrap(ev, lambda i: fit_recalibration(p[i], y[i])[1], 300, seed=2)
    return {"intercept": round(a, 3), "slope": round(b, 3), "slope_lo": round(lo, 3), "slope_hi": round(hi, 3), "n": len(df)}


def ends(df):
    """How the cheapest and the most expensive contracts actually did."""
    out = {}
    for name, mask in (("longshots", df.price <= LONGSHOT), ("favorites", df.price >= FAVORITE)):
        sub = df[mask]
        p, y, ev = sub.price.to_numpy(), sub.yes_won.to_numpy(), sub.event_id.to_numpy()
        gap_ci = cluster_bootstrap(ev, lambda i: y[i].mean() - p[i].mean(), N_BOOT, seed=3)
        # Return per $1 spent buying YES at the quoted price, before fees and spread.
        ret_ci = cluster_bootstrap(ev, lambda i: (y[i] / p[i]).mean() - 1, N_BOOT, seed=4)
        out[name] = {
            "n": len(sub), "mean_price": round(float(p.mean()), 4), "won": round(float(y.mean()), 4),
            "gap": round(float(y.mean() - p.mean()), 4), "gap_lo": round(gap_ci[0], 4), "gap_hi": round(gap_ci[1], 4),
            "return": round(float((y / p).mean() - 1), 4), "return_lo": round(ret_ci[0], 4), "return_hi": round(ret_ci[1], 4),
        }
    return out


def mid_gap(df):
    """Contracts priced 5-40%: how often they happened minus their price, in probability units."""
    sub = df[(df.price >= 0.05) & (df.price < 0.40)]
    p, y, ev = sub.price.to_numpy(), sub.yes_won.to_numpy(), sub.event_id.to_numpy()
    lo, hi = cluster_bootstrap(ev, lambda i: y[i].mean() - p[i].mean(), N_BOOT, seed=6)
    return {"n": len(sub), "gap": round(float(y.mean() - p.mean()), 4), "gap_lo": round(lo, 4), "gap_hi": round(hi, 4)}


def timing_check(df):
    """Robustness: does the result hold on markets whose snapshot time couldn't depend on the outcome?"""
    out = {"share_fixed_time": round(float(df.drop_duplicates("market_id").fixed_time.mean()), 3)}
    for h, d in df.groupby("horizon"):
        fixed = d[d.fixed_time == 1]
        out[int(h)] = {
            "all": {"n": len(d), "slope": slope_with_ci(d), "mid_gap": mid_gap(d), "ends": ends(d)},
            "fixed_time": {"n": len(fixed), "slope": slope_with_ci(fixed), "mid_gap": mid_gap(fixed), "ends": ends(fixed),
                           "calibration": calibration_with_ci(fixed)},
        }
    return out


def same_markets(df):
    """Accuracy at each horizon on the same markets (those with a price 30, 7, and 1 day out)."""
    have = df.groupby("market_id").horizon.nunique()
    ids = have[have == 3].index
    sub = df[df.market_id.isin(ids)]
    return {"markets": len(ids), **{int(h): round(brier(d.price, d.yes_won), 5) for h, d in sub.groupby("horizon")}}


def out_of_sample(df):
    train, test = df[df.close < SPLIT_DATE], df[df.close >= SPLIT_DATE]
    if len(train) < 100 or len(test) < 100:
        return {"train_n": len(train), "test_n": len(test), "split": SPLIT_DATE, "note": "not enough data on one side"}
    a, b = fit_recalibration(train.price, train.yes_won)
    p, y, ev = test.price.to_numpy(), test.yes_won.to_numpy(), test.event_id.to_numpy()
    q = apply_recalibration(p, a, b)
    diff = lambda i: brier(p[i], y[i]) - brier(q[i], y[i])  # positive = the correction helped
    lo, hi = cluster_bootstrap(ev, diff, N_BOOT, seed=5)
    return {
        "train_n": len(train), "test_n": len(test), "split": SPLIT_DATE,
        "fit_intercept": round(a, 3), "fit_slope": round(b, 3),
        "brier_market": round(brier(p, y), 5), "brier_corrected": round(brier(q, y), 5),
        "brier_gain": round(brier(p, y) - brier(q, y), 5), "gain_lo": round(lo, 5), "gain_hi": round(hi, 5),
        "logloss_market": round(log_loss(p, y), 4), "logloss_corrected": round(log_loss(q, y), 4),
    }


def main():
    df = pd.read_csv(ROOT / "data" / "dataset.csv", dtype={"market_id": str, "event_id": str})
    results = {"markets": int(df.market_id.nunique()), "events": int(df.event_id.nunique()), "by_horizon": {}}
    for h, d in df.groupby("horizon"):
        base = d.yes_won.mean()
        results["by_horizon"][int(h)] = {
            "n": len(d), "events": int(d.event_id.nunique()), "yes_rate": round(float(base), 4),
            **round_dict(murphy(d.price, d.yes_won), 5),
            "brier_base_rate": round(brier(np.full(len(d), base), d.yes_won), 5),
            "log_loss": round(log_loss(d.price, d.yes_won), 4),
            "calibration": calibration_with_ci(d),
            "slope": slope_with_ci(d),
            "ends": ends(d),
        }
        print(f"horizon {h}d: n={len(d):,}  brier={brier(d.price, d.yes_won):.4f}  slope={results['by_horizon'][int(h)]['slope']}")

    week = df[df.horizon == 7]
    results["by_category"] = {}
    for cat, d in week.groupby("category"):
        if len(d) < 300:
            continue
        results["by_category"][cat] = {
            "n": len(d), "events": int(d.event_id.nunique()), "brier": round(brier(d.price, d.yes_won), 5),
            "slope": slope_with_ci(d), "calibration": calibration_with_ci(d), "ends": ends(d),
        }
    terciles = pd.qcut(week.volume, 3, labels=["low", "mid", "high"])
    results["by_volume"] = {}
    for name, d in week.groupby(terciles, observed=True):
        results["by_volume"][str(name)] = {
            "n": len(d), "min_volume": int(d.volume.min()), "max_volume": int(d.volume.max()),
            "brier": round(brier(d.price, d.yes_won), 5), "slope": slope_with_ci(d),
        }
    results["out_of_sample"] = out_of_sample(week)
    results["timing_check"] = timing_check(df)
    results["same_markets_brier"] = same_markets(df)
    for h in (30, 7, 1):
        t = results["timing_check"][h]
        print(f"timing check {h}d: all mid-gap {t['all']['mid_gap']} slope {t['all']['slope']['slope']} | fixed-time n={t['fixed_time']['n']} mid-gap {t['fixed_time']['mid_gap']} slope {t['fixed_time']['slope']}")
    print("same markets brier:", results["same_markets_brier"])
    print("out of sample (7d):", results["out_of_sample"])
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "metrics.json").write_text(json.dumps(results, indent=1))
    print("wrote results/metrics.json")


if __name__ == "__main__":
    main()
