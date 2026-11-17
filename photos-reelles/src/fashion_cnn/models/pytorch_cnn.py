"""Custom PyTorch CNN for Fashion-MNIST."""

import torch
from torch import nn


class PyTorchCNN(nn.Module):
    """Two-block CNN with dropout and ten unnormalised class logits."""

    def __init__(self, dropout: float = 0.5) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(64 * 7 * 7, 128), nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, 10))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return one logit vector for every NCHW input image."""
        return self.classifier(self.features(inputs))
