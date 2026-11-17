"""Central configuration objects for repeatable experiments."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Hyperparameters and paths shared by both framework implementations."""

    seed: int = 42
    batch_size: int = 64
    epochs: int = 10
    learning_rate: float = 1e-3
    validation_size: int = 6_000
    data_dir: Path = Path("data")
    results_dir: Path = Path("results")
