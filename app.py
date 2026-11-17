"""Streamlit interface for comparing Fashion-MNIST predictions."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, ImageOps

SOURCE_DIRECTORY = Path(__file__).resolve().parent / "src"
if str(SOURCE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIRECTORY))

from fashion_cnn.datasets.fashion_mnist import CLASS_NAMES, normalise_images


ROOT = Path(__file__).resolve().parent
RESULTS_DIRECTORY = ROOT / "results"
PYTORCH_CHECKPOINT = RESULTS_DIRECTORY / "pytorch_best.pt"
TENSORFLOW_CHECKPOINT = RESULTS_DIRECTORY / "tensorflow_best.keras"


def prepare_image(source: Image.Image, invert: bool) -> tuple[Image.Image, np.ndarray]:
    """Return the displayed grayscale image and a normalized NHWC batch."""
    grayscale = ImageOps.grayscale(source)
    if invert:
        grayscale = ImageOps.invert(grayscale)
    resized = grayscale.resize((28, 28), Image.Resampling.LANCZOS)
    batch = normalise_images(np.asarray(resized))[np.newaxis, ..., np.newaxis]
    return resized, batch


def softmax(logits: np.ndarray) -> np.ndarray:
    """Convert one model output vector into stable class probabilities."""
    values = np.asarray(logits, dtype=np.float64).reshape(-1)
    exponentials = np.exp(values - values.max())
    return exponentials / exponentials.sum()


@st.cache_resource(show_spinner=False)
def load_pytorch_model(checkpoint_path: str):
    import torch
    from fashion_cnn.models.pytorch_cnn import PyTorchCNN

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PyTorchCNN().to(device)
    saved = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = saved["model_state"] if "model_state" in saved else saved
    model.load_state_dict(state)
    model.eval()
    return model, device


@st.cache_resource(show_spinner=False)
def load_tensorflow_model(checkpoint_path: str):
    import tensorflow as tf

    return tf.keras.models.load_model(checkpoint_path)


def predict_pytorch(batch: np.ndarray) -> tuple[np.ndarray, float, str]:
    import torch

    model, device = load_pytorch_model(str(PYTORCH_CHECKPOINT))
    inputs = torch.from_numpy(batch.transpose(0, 3, 1, 2)).to(device)
    started = time.perf_counter()
    with torch.no_grad():
        logits = model(inputs).cpu().numpy()[0]
    elapsed_ms = (time.perf_counter() - started) * 1_000
    return softmax(logits), elapsed_ms, str(device).upper()


def predict_tensorflow(batch: np.ndarray) -> tuple[np.ndarray, float, str]:
    model = load_tensorflow_model(str(TENSORFLOW_CHECKPOINT))
    started = time.perf_counter()
    logits = model(batch, training=False).numpy()[0]
    elapsed_ms = (time.perf_counter() - started) * 1_000
    return softmax(logits), elapsed_ms, "CPU/GPU TensorFlow"


def render_result(title: str, result: tuple[np.ndarray, float, str]) -> None:
    probabilities, elapsed_ms, device = result
    prediction = int(probabilities.argmax())
    st.subheader(title)
    st.metric("Classe prédite", CLASS_NAMES[prediction])
    first, second = st.columns(2)
    first.metric("Confiance", f"{probabilities[prediction] * 100:.2f} %")
    second.metric("Inférence", f"{elapsed_ms:.2f} ms")
    st.caption(f"Appareil : {device}")
    st.markdown("#### Probabilités des dix classes")
    chart_data = {
        "Classe": list(CLASS_NAMES),
        "Probabilité (%)": (probabilities * 100).tolist(),
    }
    st.bar_chart(
        chart_data,
        x="Classe",
        y="Probabilité (%)",
        horizontal=True,
        color="#6C63FF" if title == "PyTorch" else "#FF6B6B",
        use_container_width=True,
    )
    with st.expander("Voir les 10 probabilités"):
        for name, probability in sorted(
            zip(CLASS_NAMES, probabilities, strict=True),
            key=lambda item: item[1],
            reverse=True,
        ):
            st.write(f"{name}: **{probability * 100:.2f} %**")


st.set_page_config(page_title="Fashion-MNIST • PyTorch vs TensorFlow", page_icon="👕", layout="wide")
st.title("Reconnaissance Fashion-MNIST")
st.caption("Comparez les prédictions des mêmes CNN entraînés avec PyTorch et TensorFlow.")

with st.sidebar:
    st.header("Modèles")
    st.write(f"{'✅' if PYTORCH_CHECKPOINT.is_file() else '❌'} PyTorch")
    st.write(f"{'✅' if TENSORFLOW_CHECKPOINT.is_file() else '❌'} TensorFlow")
    invert = st.toggle(
        "Inverser les couleurs",
        value=False,
        help="Activez cette option pour un vêtement sombre sur un fond clair.",
    )
    st.info("Une image proche du format Fashion-MNIST (objet centré, fond noir) donne de meilleurs résultats.")

uploaded = st.file_uploader(
    "Choisissez une image",
    type=(
        "png",
        "jpg",
        "jpeg",
        "jfif",
        "webp",
        "bmp",
        "gif",
        "tif",
        "tiff",
        "ico",
        "ppm",
        "pgm",
    ),
    help="Formats acceptés : PNG, JPG, JPEG, JFIF, WebP, BMP, GIF, TIFF, ICO, PPM et PGM.",
)

if uploaded is None:
    st.info("Importez une image de vêtement pour lancer la comparaison.")
else:
    try:
        with Image.open(uploaded) as opened:
            original = opened.copy()
        processed, batch = prepare_image(original, invert)
        preview, transformed = st.columns(2)
        preview.image(original, caption="Image originale", width=280)
        transformed.image(processed.resize((280, 280)), caption="Entrée du CNN (28 × 28)", width=280)

        if st.button("Comparer les deux modèles", type="primary", use_container_width=True):
            outputs: dict[str, tuple[np.ndarray, float, str]] = {}
            errors: dict[str, str] = {}
            for name, checkpoint, predictor in (
                ("PyTorch", PYTORCH_CHECKPOINT, predict_pytorch),
                ("TensorFlow", TENSORFLOW_CHECKPOINT, predict_tensorflow),
            ):
                if not checkpoint.is_file():
                    errors[name] = f"Checkpoint absent : {checkpoint.name}. Entraînez d'abord ce modèle."
                    continue
                try:
                    with st.spinner(f"Prédiction {name}…"):
                        outputs[name] = predictor(batch)
                except Exception as exc:  # Present a useful UI error without hiding the other result.
                    errors[name] = str(exc)

            st.markdown("### Accord entre les modèles")
            if len(outputs) == 2:
                pt_class = int(outputs["PyTorch"][0].argmax())
                tf_class = int(outputs["TensorFlow"][0].argmax())
                if pt_class == tf_class:
                    st.success(
                        f"✅ **ACCORD** — Les deux modèles prédisent "
                        f"**{CLASS_NAMES[pt_class]}**."
                    )
                else:
                    st.warning(
                        f"⚠️ **DÉSACCORD** — PyTorch prédit "
                        f"**{CLASS_NAMES[pt_class]}**, tandis que TensorFlow prédit "
                        f"**{CLASS_NAMES[tf_class]}**."
                    )
            else:
                st.info("Comparaison impossible tant que les deux modèles ne sont pas disponibles.")

            st.markdown("### Comparaison détaillée")
            columns = st.columns(2)
            for column, name in zip(columns, ("PyTorch", "TensorFlow"), strict=True):
                with column:
                    if name in outputs:
                        render_result(name, outputs[name])
                    else:
                        st.subheader(name)
                        st.error(errors.get(name, "Modèle indisponible."))

    except Exception as exc:
        st.error(f"Impossible de lire cette image : {exc}")
