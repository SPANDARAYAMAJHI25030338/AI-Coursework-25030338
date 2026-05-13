"""Evaluation helpers shared by every model.

We compute the same metric set for every model so the comparison is fair.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class MetricBundle:
    """One row of metrics for a single model run."""
    model: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    macro_f1: float
    weighted_f1: float
    roc_auc: float | None
    confusion: list[list[int]]
    per_class_report: str

    def to_dict(self) -> dict:
        return asdict(self)


def compute_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    y_proba: np.ndarray | None,
    model_name: str,
    labels: Sequence[str] | None = None,
) -> MetricBundle:
    """Compute the full metric bundle for one model."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # ROC-AUC only defined for binary or probabilistic inputs
    roc = None
    if y_proba is not None:
        try:
            if y_proba.ndim == 2 and y_proba.shape[1] == 2:
                roc = float(roc_auc_score(y_true, y_proba[:, 1]))
            elif y_proba.ndim == 1:
                roc = float(roc_auc_score(y_true, y_proba))
        except ValueError:
            roc = None

    return MetricBundle(
        model=model_name,
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, average="binary",
                                        zero_division=0)),
        recall=float(recall_score(y_true, y_pred, average="binary",
                                  zero_division=0)),
        f1=float(f1_score(y_true, y_pred, average="binary", zero_division=0)),
        macro_f1=float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        weighted_f1=float(f1_score(y_true, y_pred, average="weighted",
                                   zero_division=0)),
        roc_auc=roc,
        confusion=confusion_matrix(y_true, y_pred).tolist(),
        per_class_report=classification_report(y_true, y_pred,
                                                target_names=labels,
                                                zero_division=0),
    )
