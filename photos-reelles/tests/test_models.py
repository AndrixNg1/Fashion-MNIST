"""CNN shape tests, skipped only when an optional framework is unavailable."""

import pytest


def test_pytorch_forward_shape() -> None:
    """PyTorch CNN maps an image batch to ten logits."""
    torch = pytest.importorskip("torch")
    from fashion_cnn.models import PyTorchCNN

    assert PyTorchCNN()(torch.randn(3, 1, 28, 28)).shape == (3, 10)


def test_pytorch_checkpoint_roundtrip(tmp_path) -> None:
    """A saved state dictionary can be loaded into an equivalent model."""
    torch = pytest.importorskip("torch")
    from fashion_cnn.models import PyTorchCNN

    source = PyTorchCNN()
    path = tmp_path / "model.pt"
    torch.save(source.state_dict(), path)
    restored = PyTorchCNN()
    restored.load_state_dict(torch.load(path, weights_only=True))
    for original, loaded in zip(
        source.parameters(), restored.parameters(), strict=True
    ):
        assert torch.equal(original, loaded)


def test_tensorflow_forward_shape() -> None:
    """TensorFlow CNN maps an image batch to ten logits."""
    tf = pytest.importorskip("tensorflow")
    from fashion_cnn.models import build_tensorflow_cnn

    assert tuple(build_tensorflow_cnn()(tf.zeros((3, 28, 28, 1))).shape) == (3, 10)


def test_tensorflow_checkpoint_roundtrip(tmp_path) -> None:
    """A Keras checkpoint can be restored for later evaluation."""
    tf = pytest.importorskip("tensorflow")
    from fashion_cnn.models import build_tensorflow_cnn

    path = tmp_path / "model.keras"
    model = build_tensorflow_cnn()
    model.save(path)
    assert tuple(tf.keras.models.load_model(path)(tf.zeros((1, 28, 28, 1))).shape) == (
        1,
        10,
    )
