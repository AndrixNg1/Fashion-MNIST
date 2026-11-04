"""Loading, splitting, and visualising Fashion-MNIST without data leakage."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

CLASS_NAMES: tuple[str, ...] = (
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
)


@dataclass(frozen=True, slots=True)
class SplitIndices:
    """Indices defining the training and validation subsets."""

    train: np.ndarray
    validation: np.ndarray


@dataclass(frozen=True, slots=True)
class TensorFlowData:
    """Data arrays in NHWC format, standardized to the [-1, 1] interval."""

    x_train: np.ndarray
    y_train: np.ndarray
    x_validation: np.ndarray
    y_validation: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray


def normalise_images(images: np.ndarray) -> np.ndarray:
    """Convert uint8 images to float32 values standardized between -1 and 1."""
    return images.astype(np.float32) / 127.5 - 1.0


def load_external_image(image_path: Path, invert: bool = False) -> np.ndarray:
    """Prepare one external image as a normalized 28x28 NHWC batch.

    Fashion-MNIST images conventionally show a light item on a dark background.
    Set ``invert`` for a dark item photographed on a light background.
    """
    try:
        from PIL import Image, ImageOps
    except ImportError as exc:
        raise RuntimeError("Pillow is required for image prediction.") from exc
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    with Image.open(image_path) as source:
        image = ImageOps.grayscale(source).resize((28, 28), Image.Resampling.LANCZOS)
        if invert:
            image = ImageOps.invert(image)
        pixels = np.asarray(image)
    return normalise_images(pixels)[np.newaxis, ..., np.newaxis]


def create_split_indices(
    size: int = 60_000, validation_size: int = 6_000, seed: int = 42
) -> SplitIndices:
    """Return a reproducible, disjoint train/validation partition."""
    if not 0 < validation_size < size:
        raise ValueError("validation_size must be strictly between 0 and dataset size.")
    indices = np.random.default_rng(seed).permutation(size)
    return SplitIndices(
        train=indices[validation_size:], validation=indices[:validation_size]
    )


def load_tensorflow_data(
    data_dir: Path, validation_size: int = 6_000, seed: int = 42
) -> TensorFlowData:
    """Download/load Fashion-MNIST and return a fixed split in TensorFlow format."""
    try:
        from tensorflow import keras
    except ImportError as exc:
        raise RuntimeError(
            "TensorFlow is required. Install dependencies from requirements.txt."
        ) from exc
    data_dir.mkdir(parents=True, exist_ok=True)
    # TensorFlow 2.21 exposes no ``path`` parameter and downloads its current
    # IDX source files into Keras' cache.  Using its public loader keeps this
    # compatible with both the installed Keras version and future mirrors.
    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
    split = create_split_indices(len(x_train), validation_size, seed)
    x_train = normalise_images(x_train)[..., np.newaxis]
    x_test = normalise_images(x_test)[..., np.newaxis]
    return TensorFlowData(
        x_train[split.train],
        y_train[split.train],
        x_train[split.validation],
        y_train[split.validation],
        x_test,
        y_test,
    )


def load_pytorch_data(
    data_dir: Path, batch_size: int = 64, validation_size: int = 6_000, seed: int = 42
) -> tuple[Any, Any, Any]:
    """Return train, validation, and test PyTorch data loaders in NCHW format."""
    try:
        import torch
        from torch.utils.data import DataLoader, Subset
        from torchvision import datasets, transforms
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch and TorchVision are required. Install requirements.txt."
        ) from exc
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    train_dataset = datasets.FashionMNIST(
        root=data_dir, train=True, download=True, transform=transform
    )
    test_dataset = datasets.FashionMNIST(
        root=data_dir, train=False, download=True, transform=transform
    )
    split = create_split_indices(len(train_dataset), validation_size, seed)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        Subset(train_dataset, split.train.tolist()),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    validation_loader = DataLoader(
        Subset(train_dataset, split.validation.tolist()),
        batch_size=batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, validation_loader, test_loader


def plot_dataset_examples(
    images: np.ndarray, labels: np.ndarray, output_path: Path, count: int = 10
) -> None:
    """Save labeled examples; accepts NHW, NHWC, or NCHW image batches."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, (count + 1) // 2, figsize=(12, 5))
    for axis, image, label in zip(
        axes.flat, images[:count], labels[:count], strict=False
    ):
        image = np.squeeze(image)
        axis.imshow((image + 1.0) / 2.0 if image.min() < 0 else image, cmap="gray")
        axis.set_title(CLASS_NAMES[int(label)])
        axis.axis("off")
    for axis in axes.flat[count:]:
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
