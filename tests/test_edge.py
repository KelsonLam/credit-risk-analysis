"""Edge-case and validation tests for the credit model."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from credit_risk import data, model, scorecard


def test_single_class_target_raises():
    df = data.generate_synthetic(n_samples=500, seed=0)
    df["default"] = 0          # remove all defaults
    with pytest.raises(ValueError):
        model.train_model(df)


def test_missing_target_column_raises():
    df = pd.DataFrame({"x": [1, 2, 3]})
    with pytest.raises(ValueError):
        model.train_model(df, target_column="default")


def test_extreme_pd_scores_are_clipped_and_finite():
    scores = scorecard.pd_to_score([0.0, 1.0, 0.5])
    assert np.isfinite(scores).all()
    assert scores[0] > scores[2] > scores[1]   # PD 0 best, PD 1 worst


def test_band_summary_covers_all_rows():
    df = data.generate_synthetic(n_samples=4000, seed=1)
    res = model.train_model(df, seed=1)
    table = scorecard.scorecard_table(res.pd_test)
    summary = scorecard.band_summary(table)
    assert summary["count"].sum() == len(table)
