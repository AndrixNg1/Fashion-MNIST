"""Fast tests for framework-independent data rules."""

import numpy as np
import pytest

from fashion_cnn.datasets.fashion_mnist import (
    CLASS_NAMES,
    create_split_indices,
    normalise_images,
)


def test_split_is_reproducible_and_disjoint() -> None:
    """A validation partition never overlaps training and repeats with seed 42."""
    first = create_split_indices(100, 20, 42)
    second = create_split_indices(100, 20, 42)
    assert len(first.train) == 80 and len(first.validation) == 20
    assert not set(first.train).intersection(first.validation)
    np.testing.assert_array_equal(first.train, second.train)


def test_normalisation_and_labels() -> None:
    """Pixels become float32 values in the expected standardized interval."""
    values = normalise_images(np.array([0, 127, 255], dtype=np.uint8))
    assert values.dtype == np.float32
    assert values[0] == -1.0 and values[-1] == 1.0
    assert len(CLASS_NAMES) == 10 and len(set(CLASS_NAMES)) == 10


def test_invalid_validation_size_is_rejected() -> None:
    """A split must leave at least one training sample."""
    with pytest.raises(ValueError):
        create_split_indices(10, 10)
