"""Framework-specific training loops with equivalent experimental settings."""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from fashion_cnn.config import ExperimentConfig
from fashion_cnn.models import PyTorchCNN, build_tensorflow_cnn
from fashion_cnn.utils import set_global_seed


@dataclass(slots=True)
class TrainingHistory:
    """Epoch-level loss and accuracy values for later plotting or comparison."""

    loss: list[float] = field(default_factory=list)
    accuracy: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_accuracy: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, list[float]]:
        """Provide a JSON-serialisable representation."""
        return asdict(self)


def _pytorch_epoch(
    model: Any, loader: Any, criterion: Any, device: Any, optimizer: Any | None = None
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    correct = total = 0
    import torch

    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if training:
                optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
    return total_loss / total, correct / total


def train_pytorch(
    train_loader: Any, validation_loader: Any, config: ExperimentConfig
) -> tuple[Any, TrainingHistory, Path]:
    """Train PyTorchCNN, retaining the checkpoint with lowest validation loss."""
    import torch

    set_global_seed(config.seed)
    config.results_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PyTorchCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    history, best_loss = TrainingHistory(), float("inf")
    checkpoint = config.results_dir / "pytorch_best.pt"
    for epoch in range(config.epochs):
        loss, accuracy = _pytorch_epoch(
            model, train_loader, criterion, device, optimizer
        )
        val_loss, val_accuracy = _pytorch_epoch(
            model, validation_loader, criterion, device
        )
        history.loss.append(loss)
        history.accuracy.append(accuracy)
        history.val_loss.append(val_loss)
        history.val_accuracy.append(val_accuracy)
        print(
            f"PyTorch epoch {epoch + 1}/{config.epochs}: loss={loss:.4f}, acc={accuracy:.4f}, val_loss={val_loss:.4f}, val_acc={val_accuracy:.4f}"
        )
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(
                {"model_state": model.state_dict(), "config": asdict(config)},
                checkpoint,
            )
    model.load_state_dict(
        torch.load(checkpoint, map_location=device, weights_only=False)["model_state"]
    )
    return model, history, checkpoint


def train_tensorflow(
    data: Any, config: ExperimentConfig
) -> tuple[Any, TrainingHistory, Path]:
    """Train the equivalent Keras CNN and save its best validation-loss checkpoint."""
    import tensorflow as tf

    set_global_seed(config.seed)
    config.results_dir.mkdir(parents=True, exist_ok=True)
    model = build_tensorflow_cnn()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(config.learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    checkpoint = config.results_dir / "tensorflow_best.keras"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint, monitor="val_loss", save_best_only=True, verbose=1
        )
    ]
    fitted = model.fit(
        data.x_train,
        data.y_train,
        validation_data=(data.x_validation, data.y_validation),
        batch_size=config.batch_size,
        epochs=config.epochs,
        verbose=2,
        callbacks=callbacks,
    )
    model = tf.keras.models.load_model(checkpoint)
    return (
        model,
        TrainingHistory(
            **{key: [float(v) for v in value] for key, value in fitted.history.items()}
        ),
        checkpoint,
    )


def predict_pytorch(model: Any, loader: Any) -> tuple[np.ndarray, np.ndarray]:
    """Collect test labels and predicted classes from a PyTorch loader."""
    import torch

    device = next(model.parameters()).device
    model.eval()
    labels, predictions = [], []
    with torch.no_grad():
        for images, batch_labels in loader:
            predictions.extend(model(images.to(device)).argmax(1).cpu().numpy())
            labels.extend(batch_labels.numpy())
    return np.asarray(labels), np.asarray(predictions)
