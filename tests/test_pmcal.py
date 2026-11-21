from datetime import datetime, timedelta, timezone

import numpy as np

from pmcal.dataset import horizon_rows, parse_time, price_at
from pmcal.stats import apply_recalibration, brier, calibration_table, cluster_bootstrap, fit_recalibration, murphy


def ts(dt):
    return dt.timestamp()


def test_parse_time_handles_api_formats():
    assert parse_time("2026-11-09 15:17:41+00") == datetime(2026, 11, 9, 15, 17, 41, tzinfo=timezone.utc)
    assert parse_time("2026-11-09T12:00:00Z") == datetime(2026, 11, 9, 12, tzinfo=timezone.utc)
    assert parse_time("") is None


def test_price_at_takes_last_price_before_and_rejects_stale_or_early():
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    hist = [[ts(t0), 0.2], [ts(t0 + timedelta(hours=10)), 0.3], [ts(t0 + timedelta(hours=20)), 0.4]]
    assert price_at(hist, t0 + timedelta(hours=15)) == 0.3
    assert price_at(hist, t0 - timedelta(hours=1)) is None  # before trading started
    assert price_at(hist, t0 + timedelta(hours=20 + 49)) is None  # last trade too old


def test_horizon_rows_uses_close_time_and_skips_missing_horizons():
    close = datetime(2026, 3, 1, tzinfo=timezone.utc)
    hist = [[ts(close - timedelta(days=d)), 0.1 * (10 - d)] for d in range(9, 0, -1)]  # 9 days of daily prices
    m = {"market_id": "m", "event_id": "e", "category": "Sports", "volume": "1e5",
         "closed": "2026-03-01 00:00:00+00", "end": "", "yes_won": "1"}
    rows = {r["horizon"]: r for r in horizon_rows(m, hist)}
    assert set(rows) == {1, 7}  # no price 30 days out
    assert abs(rows[7]["price"] - 0.3) < 1e-9


def test_anchor_is_the_earlier_of_scheduled_end_and_recorded_close():
    from pmcal.dataset import anchor_time

    late_resolution = {"end": "2026-03-01T00:00:00Z", "closed": "2026-03-03 12:00:00+00"}
    early_event = {"end": "2026-12-31T00:00:00Z", "closed": "2026-06-10 09:00:00+00"}
    assert anchor_time(late_resolution) == datetime(2026, 3, 1, tzinfo=timezone.utc)
    assert anchor_time(early_event) == datetime(2026, 6, 10, 9, tzinfo=timezone.utc)


def test_murphy_decomposition_adds_up_for_binned_forecasts():
    rng = np.random.default_rng(1)
    p = rng.choice([0.03, 0.15, 0.5, 0.85, 0.97], 5000)  # one value per bin, so the identity is exact
    y = (rng.random(5000) < p).astype(int)
    d = murphy(p, y)
    assert abs(d["brier"] - (d["reliability"] - d["resolution"] + d["uncertainty"])) < 1e-9
    assert d["reliability"] < 0.002  # these forecasts are calibrated by construction


def test_recalibration_recovers_a_known_bias():
    rng = np.random.default_rng(2)
    true_p = rng.uniform(0.02, 0.98, 40000)
    y = (rng.random(40000) < true_p).astype(int)
    # Market prices that are too timid: pulled halfway toward 50% in log-odds.
    price = 1 / (1 + np.exp(-0.5 * np.log(true_p / (1 - true_p))))
    a, b = fit_recalibration(price, y)
    assert abs(a) < 0.05 and abs(b - 2.0) < 0.1  # undoing a 0.5 squeeze takes a slope of 2
    assert brier(apply_recalibration(price, a, b), y) < brier(price, y)


def test_calibration_table_counts_everything_once():
    p = np.array([0.01, 0.07, 0.5, 0.93, 0.99, 1.0])
    rows = calibration_table(p, [0, 0, 1, 1, 1, 1])
    assert sum(r["n"] for r in rows) == len(p)


def test_cluster_bootstrap_is_reproducible_and_covers_the_truth():
    rng = np.random.default_rng(3)
    clusters = np.repeat(np.arange(300), 5)
    x = rng.normal(0, 1, 300)[clusters] + rng.normal(0, 1, 1500)
    lo, hi = cluster_bootstrap(clusters, lambda i: x[i].mean(), n_boot=400, seed=9)
    assert (lo, hi) == cluster_bootstrap(clusters, lambda i: x[i].mean(), n_boot=400, seed=9)
    assert lo < 0 < hi
