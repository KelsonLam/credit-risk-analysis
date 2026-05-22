"""Tests for the data generator, model, metrics, and scorecard.

All synthetic and offline. The signal in the generated data is real, so the
model is expected to clear a meaningful AUC bar.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from credit_risk import data, model, metrics, scorecard


def test_synthetic_shape_and_rate():
    df = data.generate_synthetic(n_samples=5000, default_rate=0.2, seed=0)
    assert len(df) == 5000
    assert "default" in df.columns
    assert set(df["default"].unique()) <= {0, 1}
    # The realized default rate should be near the target.
    assert abs(df["default"].mean() - 0.2) < 0.03


def test_model_recovers_signal():
    df = data.generate_synthetic(n_samples=8000, seed=1)
    res = model.train_model(df, seed=1)
    # With genuine signal in the data, AUC should be comfortably above chance.
    assert metrics.auc(res.y_test, res.pd_test) > 0.7
    assert ((res.pd_test >= 0) & (res.pd_test <= 1)).all()


def test_ks_and_gini_ranges():
    df = data.generate_synthetic(n_samples=6000, seed=2)
    res = model.train_model(df, seed=2)
    ks = metrics.ks_statistic(res.y_test, res.pd_test)
    g = metrics.gini(res.y_test, res.pd_test)
    assert 0.0 <= ks <= 1.0
    assert 0.0 <= g <= 1.0


def test_score_decreases_with_pd():
    # Higher probability of default must map to a lower score.
    pds = np.array([0.01, 0.1, 0.3, 0.6, 0.9])
    scores = scorecard.pd_to_score(pds)
    assert np.all(np.diff(scores) < 0)


def test_double_odds_adds_fixed_points():
    # Doubling the odds of being good should add exactly `pdo` points.
    pdo = 20.0
    # PD of 0.5 -> odds 1; PD where good-odds is 2 -> 1/3.
    s1 = scorecard.pd_to_score([0.5], points_to_double_odds=pdo)[0]
    s2 = scorecard.pd_to_score([1 / 3], points_to_double_odds=pdo)[0]
    assert (s2 - s1) == pytest.approx(pdo, abs=1e-6)


def test_bands_assigned_in_order():
    scores = [800, 700, 630, 560, 400]
    bands = scorecard.assign_band(scores)
    assert bands[0].startswith("A")
    assert bands[-1].startswith("E")


def test_load_data_falls_back_to_synthetic(tmp_path):
    df = data.load_data(csv_path=str(tmp_path / "missing.csv"), n_samples=1000, seed=3)
    assert len(df) == 1000


def test_csv_missing_target_raises(tmp_path):
    p = tmp_path / "loans.csv"
    pd.DataFrame({"x": [1, 2, 3]}).to_csv(p, index=False)
    with pytest.raises(ValueError):
        data.load_data(csv_path=str(p), target_column="default")
