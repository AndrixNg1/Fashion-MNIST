"""Comparable custom CNN implementations."""

from .pytorch_cnn import PyTorchCNN
from .tensorflow_cnn import build_tensorflow_cnn

__all__ = ["PyTorchCNN", "build_tensorflow_cnn"]
