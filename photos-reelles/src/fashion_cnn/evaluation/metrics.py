"""Metrics and publication-ready plots for multiclass classification."""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from fashion_cnn.datasets.fashion_mnist import CLASS_NAMES


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    """Compute accuracy and macro scores, treating each class equally."""
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=range(10), zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
        "per_class": {
            CLASS_NAMES[index]: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index in range(10)
        },
        "report": classification_report(
            y_true,
            y_pred,
            labels=range(10),
            target_names=CLASS_NAMES,
            zero_division=0,
            output_dict=True,
        ),
    }


def plot_learning_curves(history: Any, output_path: Path, title: str) -> None:
    """Save side-by-side loss and accuracy trajectories."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history.loss) + 1)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    for axis, train, validation, label in (
        (axes[0], history.loss, history.val_loss, "Loss"),
        (axes[1], history.accuracy, history.val_accuracy, "Accuracy"),
    ):
        axis.plot(epochs, train, label="Training")
        axis.plot(epochs, validation, label="Validation")
        axis.set(xlabel="Epoch", ylabel=label, title=f"{title} — {label}")
        axis.legend()
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def plot_framework_comparison(
    metrics_by_framework: dict[str, dict[str, Any]], output_path: Path
) -> None:
    """Save an accuracy and macro-F1 comparison based on actual saved metrics."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    names = list(metrics_by_framework)
    scores = np.array(
        [
            [
                metrics_by_framework[name]["accuracy"],
                metrics_by_framework[name]["macro_f1"],
            ]
            for name in names
        ]
    )
    positions = np.arange(len(names))
    width = 0.35
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.bar(positions - width / 2, scores[:, 0], width, label="Accuracy")
    axis.bar(positions + width / 2, scores[:, 1], width, label="Macro F1")
    axis.set(
        xticks=positions,
        xticklabels=[name.title() for name in names],
        ylim=(0, 1),
        ylabel="Score",
        title="Test-set framework comparison",
    )
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def plot_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, output_path: Path, title: str
) -> None:
    """Save a labelled confusion matrix for the test set."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    matrix = confusion_matrix(y_true, y_pred, labels=range(10))
    figure, axis = plt.subplots(figsize=(9, 8))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(
        xticks=range(10),
        yticks=range(10),
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        xlabel="Predicted",
        ylabel="True",
        title=title,
    )
    plt.setp(axis.get_xticklabels(), rotation=45, ha="right")
    for row in range(10):
        for column in range(10):
            axis.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
                fontsize=7,
            )
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def plot_prediction_examples(
    images: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path,
    correct: bool,
    count: int = 10,
) -> None:
    """Save correct or incorrect prediction examples from held-out test images."""
    selected = np.flatnonzero(y_true == y_pred if correct else y_true != y_pred)[:count]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 5, figsize=(11, 5))
    for axis, index in zip(axes.flat, selected, strict=False):
        axis.imshow(
            (
                (np.squeeze(images[index]) + 1) / 2
                if np.min(images[index]) < 0
                else np.squeeze(images[index])
            ),
            cmap="gray",
        )
        axis.set_title(
            f"T: {CLASS_NAMES[y_true[index]]}\nP: {CLASS_NAMES[y_pred[index]]}",
            fontsize=8,
        )
        axis.axis("off")
    for axis in axes.flat[len(selected) :]:
        axis.axis("off")
    figure.suptitle("Correct predictions" if correct else "Incorrect predictions")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
