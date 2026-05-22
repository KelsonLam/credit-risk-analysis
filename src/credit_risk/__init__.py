"""Credit risk modelling: estimate the probability a borrower defaults.

The project builds a logistic-regression scorecard, the workhorse of consumer
credit, because it is interpretable: every feature maps to a transparent
contribution to the odds of default, which matters when a decision has to be
explained or defended.

Modules:

    data        a realistic synthetic loan dataset, or load your own CSV
    model       preprocessing and the logistic-regression model
    metrics     AUC, the KS statistic, Gini, confusion matrix, calibration
    scorecard   turn a default probability into a points-based credit score
    plotting    ROC, KS, score distribution, and calibration charts
"""

__version__ = "0.1.0"
