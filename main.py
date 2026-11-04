"""Command-line entry point for Fashion-MNIST CNN experiments."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Support direct execution (``python main.py``) with the project's ``src`` layout.
SOURCE_DIRECTORY = Path(__file__).resolve().parent / "src"
if str(SOURCE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIRECTORY))

from fashion_cnn.config import ExperimentConfig
from fashion_cnn.datasets.fashion_mnist import (
    load_pytorch_data,
    load_tensorflow_data,
    plot_dataset_examples,
)
from fashion_cnn.evaluation import (
    classification_metrics,
    plot_confusion_matrix,
    plot_framework_comparison,
    plot_learning_curves,
    plot_prediction_examples,
)
from fashion_cnn.models import PyTorchCNN
from fashion_cnn.training import predict_pytorch, train_pytorch, train_tensorflow
from fashion_cnn.utils import set_global_seed


def _save_evaluation(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    images: np.ndarray,
    framework: str,
    results_dir: Path,
) -> None:
    """Persist metrics, confusion matrix, and qualitative test predictions."""
    metrics = classification_metrics(y_true, y_pred)
    (results_dir / f"{framework}_test_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    plot_confusion_matrix(
        y_true,
        y_pred,
        results_dir / f"{framework}_confusion_matrix.png",
        f"{framework.title()} test confusion matrix",
    )
    plot_prediction_examples(
        images,
        y_true,
        y_pred,
        results_dir / f"{framework}_correct_predictions.png",
        True,
    )
    plot_prediction_examples(
        images,
        y_true,
        y_pred,
        results_dir / f"{framework}_incorrect_predictions.png",
        False,
    )
    print(
        f"{framework} test accuracy: {metrics['accuracy']:.4f}; macro F1: {metrics['macro_f1']:.4f}"
    )


def run_pytorch(mode: str, config: ExperimentConfig) -> None:
    """Execute a PyTorch train or evaluate workflow."""
    import torch

    train_loader, validation_loader, test_loader = load_pytorch_data(
        config.data_dir, config.batch_size, config.validation_size, config.seed
    )
    checkpoint = config.results_dir / "pytorch_best.pt"
    if mode == "train":
        model, history, _ = train_pytorch(train_loader, validation_loader, config)
        plot_learning_curves(
            history, config.results_dir / "pytorch_learning_curves.png", "PyTorch"
        )
    else:
        if not checkpoint.exists():
            raise FileNotFoundError(f"No checkpoint at {checkpoint}. Run train first.")
        model = PyTorchCNN()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.load_state_dict(
            torch.load(checkpoint, map_location=device, weights_only=False)[
                "model_state"
            ]
        )
        model.to(device)
    y_true, y_pred = predict_pytorch(model, test_loader)
    images = np.concatenate([batch[0].numpy() for batch in test_loader])
    _save_evaluation(y_true, y_pred, images, "pytorch", config.results_dir)


def run_tensorflow(mode: str, config: ExperimentConfig) -> None:
    """Execute a TensorFlow train or evaluate workflow."""
    import tensorflow as tf

    data = load_tensorflow_data(config.data_dir, config.validation_size, config.seed)
    checkpoint = config.results_dir / "tensorflow_best.keras"
    if mode == "train":
        model, history, _ = train_tensorflow(data, config)
        plot_learning_curves(
            history, config.results_dir / "tensorflow_learning_curves.png", "TensorFlow"
        )
    else:
        if not checkpoint.exists():
            raise FileNotFoundError(f"No checkpoint at {checkpoint}. Run train first.")
        model = tf.keras.models.load_model(checkpoint)
    y_pred = model.predict(data.x_test, batch_size=config.batch_size, verbose=0).argmax(
        axis=1
    )
    _save_evaluation(data.y_test, y_pred, data.x_test, "tensorflow", config.results_dir)


def explore(config: ExperimentConfig) -> None:
    """Download once via Keras and generate a labelled dataset preview."""
    data = load_tensorflow_data(config.data_dir, config.validation_size, config.seed)
    plot_dataset_examples(
        data.x_train, data.y_train, config.results_dir / "dataset_examples.png"
    )
    print(
        f"Training: {data.x_train.shape}; validation: {data.x_validation.shape}; test: {data.x_test.shape}"
    )
    print("Saved results/dataset_examples.png")


def parse_arguments() -> argparse.Namespace:
    """Parse the intentionally small experiment CLI."""
    parser = argparse.ArgumentParser(
        description="Train and evaluate custom Fashion-MNIST CNNs."
    )
    parser.add_argument(
        "--framework", choices=("pytorch", "tensorflow", "both"), default="pytorch"
    )
    parser.add_argument(
        "--mode", choices=("explore", "train", "evaluate"), default="explore"
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--validation-size", type=int, default=6_000)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    return parser.parse_args()


def main() -> None:
    """Run the requested command."""
    args = parse_arguments()
    config = ExperimentConfig(
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        validation_size=args.validation_size,
        data_dir=args.data_dir,
        results_dir=args.results_dir,
    )
    set_global_seed(config.seed)
    config.results_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "explore":
        explore(config)
        return
    frameworks = (
        ("pytorch", "tensorflow") if args.framework == "both" else (args.framework,)
    )
    for framework in frameworks:
        (run_pytorch if framework == "pytorch" else run_tensorflow)(args.mode, config)
    if len(frameworks) == 2:
        metrics = {
            name: json.loads(
                (config.results_dir / f"{name}_test_metrics.json").read_text(
                    encoding="utf-8"
                )
            )
            for name in frameworks
        }
        plot_framework_comparison(metrics, config.results_dir / "framework_comparison.png")


if __name__ == "__main__":
    main()
