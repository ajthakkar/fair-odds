"""Step 5: charts for the README, from results/metrics.json.

Run: python scripts/charts.py   ->  results/*.png
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
R = json.loads((ROOT / "results" / "metrics.json").read_text())
INK, MUTED, GRID, ACCENT, ACCENT2 = "#1f1f1f", "#6b6b6b", "#e6e6e6", "#2a6f97", "#c2571a"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.bbox": "tight",
})


def reliability(ax, cal, title):
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls=(0, (4, 3)), zorder=1)
    pts = [c for c in cal if c["n"] >= 30]
    x = [c["mean_price"] for c in pts]
    y = [c["freq"] for c in pts]
    lo = [c["freq"] - c["lo_ci"] for c in pts]
    hi = [c["hi_ci"] - c["freq"] for c in pts]
    ax.errorbar(x, y, yerr=[lo, hi], fmt="o", ms=5, color=ACCENT, ecolor=ACCENT, elinewidth=1.4, capsize=0, zorder=3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.grid(color=GRID, lw=0.8)
    ax.set_title(title, loc="left", fontsize=10, color=INK)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])


def main():
    hs = R["by_horizon"]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.2))
    fig.subplots_adjust(wspace=0.32)
    for ax, h in zip(axes, ["30", "7", "1"]):
        d = hs[h]
        reliability(ax, d["calibration"], f"{h} day{'s' if h != '1' else ''} before close  (n = {d['n']:,})")
        ax.set_xlabel("Market price of YES")
    axes[0].set_ylabel("Share that resolved YES")
    fig.suptitle("When the market says X%, it happens about X% of the time (dashed line = perfect)", x=0.06, ha="left", fontsize=11, color=INK)
    fig.savefig(ROOT / "results" / "calibration.png")
    plt.close(fig)

    # The two ends of the price range, as a gap: actual win rate minus price, in percentage points.
    fig, ax = plt.subplots(figsize=(8, 3.6))
    labels, rows = [], []
    for side, name in (("longshots", "Longshots, priced ≤10%"), ("favorites", "Favorites, priced ≥90%")):
        for h in ["30", "7", "1"]:
            rows.append((side, hs[h]["ends"][side]))
            labels.append(f"{name}, {h} day{'s' if h != '1' else ''} out")
    for i, (side, r) in enumerate(rows):
        g, lo, hi = 100 * r["gap"], 100 * r["gap_lo"], 100 * r["gap_hi"]
        color = ACCENT2 if side == "longshots" else ACCENT
        ax.errorbar(g, i, xerr=[[g - lo], [hi - g]], fmt="o", ms=7, color=color, elinewidth=1.6, capsize=0)
        ax.annotate(f"{g:+.1f} pts  (n = {r['n']:,})", (max(hi, g) + 0.15, i), va="center", fontsize=9, color=MUTED)
    ax.axvline(0, color=MUTED, lw=1, ls=(0, (4, 3)))
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_xlabel("Actual win rate minus price (percentage points); 0 = priced right")
    ax.margins(x=0.25)
    ax.set_title("How the cheapest and most expensive contracts actually did", loc="left", fontsize=11, color=INK)
    fig.savefig(ROOT / "results" / "ends.png")
    plt.close(fig)

    # Calibration slope by category (7 days out): 1 = calibrated, > 1 = prices too timid
    cats = sorted(R["by_category"].items(), key=lambda kv: kv[1]["slope"]["slope"])
    fig, ax = plt.subplots(figsize=(7, 0.55 * len(cats) + 1.2))
    for i, (name, d) in enumerate(cats):
        s = d["slope"]
        ax.errorbar(s["slope"], i, xerr=[[s["slope"] - s["slope_lo"]], [s["slope_hi"] - s["slope"]]], fmt="o", ms=6,
                    color=ACCENT, elinewidth=1.4)
    ax.axvline(1, color=MUTED, lw=1, ls=(0, (4, 3)))
    ax.set_yticks(range(len(cats)))
    ax.set_yticklabels([f"{n}  (n = {d['n']:,})" for n, d in cats])
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_xlabel("Calibration slope (1 = calibrated; above 1 = prices too close to 50%)")
    ax.set_title("By category, 7 days before close", loc="left", fontsize=11, color=INK)
    fig.savefig(ROOT / "results" / "categories.png")
    plt.close(fig)
    print("wrote results/calibration.png, ends.png, categories.png")


if __name__ == "__main__":
    main()
