"""Turn a probability of default into a points-based credit score.

Lenders rarely act on a raw probability. They translate it onto a points scale
where a fixed number of points always doubles the odds of being a good payer,
which is what makes scores from different models comparable. This is the same
"points to double the odds" construction behind scores like FICO.

    odds  = (1 - PD) / PD                      odds of being a good payer
    score = offset + factor * ln(odds)
    factor = pdo / ln(2)
    offset = base_score - factor * ln(base_odds)

Lower default probability means higher odds of being good, so a lower PD maps to
a higher score, as you would expect.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Score bands from safest to riskiest, as (minimum score, label) pairs.
DEFAULT_BANDS = [
    (720, "A (very low risk)"),
    (660, "B (low risk)"),
    (600, "C (medium risk)"),
    (540, "D (high risk)"),
    (0,   "E (very high risk)"),
]


def pd_to_score(
    prob_default,
    base_score: float = 600.0,
    base_odds: float = 50.0,
    points_to_double_odds: float = 20.0,
) -> np.ndarray:
    """Map probability of default to a credit score."""
    pd_arr = np.clip(np.asarray(prob_default, dtype=float), 1e-6, 1 - 1e-6)
    odds_good = (1.0 - pd_arr) / pd_arr
    factor = points_to_double_odds / np.log(2.0)
    offset = base_score - factor * np.log(base_odds)
    return offset + factor * np.log(odds_good)


def assign_band(scores, bands=DEFAULT_BANDS) -> list[str]:
    """Label each score with its risk band."""
    ordered = sorted(bands, key=lambda b: b[0], reverse=True)
    labels = []
    for s in np.asarray(scores, dtype=float):
        for minimum, label in ordered:
            if s >= minimum:
                labels.append(label)
                break
        else:
            labels.append(ordered[-1][1])
    return labels


def scorecard_table(
    prob_default,
    base_score: float = 600.0,
    base_odds: float = 50.0,
    points_to_double_odds: float = 20.0,
    bands=DEFAULT_BANDS,
) -> pd.DataFrame:
    """A per-applicant table of PD, score, and risk band."""
    scores = pd_to_score(prob_default, base_score, base_odds, points_to_double_odds)
    return pd.DataFrame({
        "prob_default": np.asarray(prob_default, dtype=float),
        "score": scores,
        "band": assign_band(scores, bands),
    })


def band_summary(table: pd.DataFrame) -> pd.DataFrame:
    """Count and average PD per risk band, ordered safest to riskiest."""
    order = [label for _, label in sorted(DEFAULT_BANDS, key=lambda b: b[0], reverse=True)]
    summary = (
        table.groupby("band", observed=True)
        .agg(count=("score", "size"),
             avg_prob_default=("prob_default", "mean"),
             avg_score=("score", "mean"))
    )
    return summary.reindex([b for b in order if b in summary.index])
