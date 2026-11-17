"""Utilities for deterministic experiment setup."""

import os
import random
from typing import Literal

import numpy as np


def set_global_seed(
    seed: int = 42, framework: Literal["pytorch", "tensorflow", "both"] = "both"
) -> None:
    """Set seeds without importing an unrelated deep-learning runtime.

    Importing TensorFlow and CUDA-enabled PyTorch in one CPU process can cause
    native-library conflicts.  CLI commands therefore select one framework.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    random.seed(seed)
    np.random.seed(seed)
    if framework in ("pytorch", "both"):
        try:
            import torch

            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except ImportError:
            pass
    if framework in ("tensorflow", "both"):
        try:
            import tensorflow as tf

            tf.keras.utils.set_random_seed(seed)
            tf.config.experimental.enable_op_determinism()
        except ImportError:
            pass
