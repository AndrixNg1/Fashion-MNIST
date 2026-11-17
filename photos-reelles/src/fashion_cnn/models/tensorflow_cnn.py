"""Custom TensorFlow/Keras CNN for Fashion-MNIST."""


def build_tensorflow_cnn(dropout: float = 0.5):
    """Build the PyTorch-equivalent NHWC CNN which returns logits, not probabilities."""
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError("TensorFlow is required. Install dependencies from requirements.txt.") from exc
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Dense(10),
    ], name="tensorflow_cnn")
