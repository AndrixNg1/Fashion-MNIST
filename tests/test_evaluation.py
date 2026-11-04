"""Tests for metric calculations."""

import numpy as np

from fashion_cnn.evaluation import classification_metrics


def test_metrics_for_perfect_predictions() -> None:
    """Perfect predictions have perfect aggregate scores."""
    labels = np.arange(10)
    metrics = classification_metrics(labels, labels)
    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert len(metrics["per_class"]) == 10
