"""Tests for Weight of Evidence and Information Value."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from credit_risk import data
from credit_risk.woe import woe_iv, rank_features


def test_predictive_feature_beats_noise():
    rng = np.random.default_rng(0)
    n = 8000
    signal = rng.normal(0, 1, n)
    # Target driven by the signal feature; noise is independent.
    prob = 1 / (1 + np.exp(-1.5 * signal))
    y = pd.Series(rng.binomial(1, prob))
    noise = pd.Series(rng.normal(0, 1, n))
    _, iv_signal = woe_iv(pd.Series(signal), y)
    _, iv_noise = woe_iv(noise, y)
    assert iv_signal > iv_noise
    assert iv_signal > 0.1     # a real, medium-or-better predictor


def test_iv_is_non_negative():
    rng = np.random.default_rng(1)
    x = pd.Series(rng.normal(0, 1, 3000))
    y = pd.Series(rng.binomial(1, 0.2, 3000))
    _, iv = woe_iv(x, y)
    assert iv >= 0.0


def test_rank_features_orders_by_iv():
    df = data.generate_synthetic(n_samples=6000, seed=2)
    ranked = rank_features(df, "default")
    assert list(ranked.columns) == ["feature", "information_value"]
    ivs = ranked["information_value"].tolist()
    assert ivs == sorted(ivs, reverse=True)
