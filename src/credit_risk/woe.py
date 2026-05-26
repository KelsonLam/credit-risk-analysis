"""Weight of Evidence and Information Value: which features actually predict?

Before fitting a credit model, analysts screen features by Information Value, a
standard measure of how well a single variable separates defaulters from payers.
It is built on Weight of Evidence, which is itself a useful, monotonic, and
interpretable transform of a feature.

For each bin of a feature:

    WoE = ln( share of all goods in this bin / share of all bads in this bin )
    IV contribution = (share of goods - share of bads) * WoE
    IV = sum of contributions across bins

A rough industry reading of IV: under 0.02 is useless, 0.1 to 0.3 is medium, and
above 0.5 is suspiciously strong (often a sign of leakage).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_EPS = 1e-6   # guards the logarithm when a bin has no goods or no bads


def woe_iv(feature: pd.Series, target: pd.Series, bins: int = 10):
    """Return a per-bin WoE/IV table and the total Information Value.

    ``target`` is 1 for a default (bad) and 0 for a paid loan (good). Numeric
    features are cut into quantile bins; low-cardinality or non-numeric features
    are grouped by their values.
    """
    df = pd.DataFrame({"x": feature.values, "y": np.asarray(target)})
    if pd.api.types.is_numeric_dtype(df["x"]) and df["x"].nunique() > bins:
        df["bin"] = pd.qcut(df["x"], q=bins, duplicates="drop")
    else:
        df["bin"] = df["x"]

    total_good = (df["y"] == 0).sum()
    total_bad = (df["y"] == 1).sum()

    rows = []
    for b, g in df.groupby("bin", observed=True):
        good = (g["y"] == 0).sum()
        bad = (g["y"] == 1).sum()
        share_good = max(good / total_good, _EPS)
        share_bad = max(bad / total_bad, _EPS)
        woe = np.log(share_good / share_bad)
        iv_part = (share_good - share_bad) * woe
        rows.append({
            "bin": b, "count": len(g), "goods": good, "bads": bad,
            "woe": woe, "iv": iv_part,
        })

    table = pd.DataFrame(rows)
    return table, float(table["iv"].sum())


def rank_features(df: pd.DataFrame, target_column: str, bins: int = 10) -> pd.DataFrame:
    """Information Value for every feature, sorted strongest first."""
    y = df[target_column]
    out = []
    for col in df.columns:
        if col == target_column:
            continue
        _, iv = woe_iv(df[col], y, bins=bins)
        out.append({"feature": col, "information_value": iv})
    return (pd.DataFrame(out)
            .sort_values("information_value", ascending=False)
            .reset_index(drop=True))
