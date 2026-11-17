"""Streamlit interface for comparing Fashion-MNIST predictions."""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
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


@dataclass(frozen=True)
class InputDiagnostics:
    """Quality checks used to reject an image that is unlike Fashion-MNIST."""

    foreground_fraction: float
    contrast: float
    usable: bool
    reason: str


def _largest_component(mask: np.ndarray) -> np.ndarray:
    """Keep the largest connected foreground region without adding OpenCV."""
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    best: list[tuple[int, int]] = []
    for row, column in zip(*np.nonzero(mask), strict=True):
        if visited[row, column]:
            continue
        stack = [(int(row), int(column))]
        visited[row, column] = True
        component: list[tuple[int, int]] = []
        while stack:
            current_row, current_column = stack.pop()
            component.append((current_row, current_column))
            for next_row, next_column in (
                (current_row - 1, current_column),
                (current_row + 1, current_column),
                (current_row, current_column - 1),
                (current_row, current_column + 1),
            ):
                if (
                    0 <= next_row < height
                    and 0 <= next_column < width
                    and mask[next_row, next_column]
                    and not visited[next_row, next_column]
                ):
                    visited[next_row, next_column] = True
                    stack.append((next_row, next_column))
        if len(component) > len(best):
            best = component
    largest = np.zeros_like(mask, dtype=bool)
    if best:
        rows, columns = zip(*best, strict=True)
        largest[list(rows), list(columns)] = True
    return largest


def prepare_image(source: Image.Image) -> tuple[Image.Image, np.ndarray, InputDiagnostics]:
    """Crop the main contrasting object and produce a Fashion-MNIST-like input.

    A real photo is first reduced for fast processing.  Its border estimates the
    background.  The largest region that contrasts with that background is
    centered on a black 28x28 canvas; contrast magnitude makes the object light
    regardless of whether the original object was dark or light.
    """
    grayscale = ImageOps.grayscale(source)
    grayscale.thumbnail((512, 512), Image.Resampling.LANCZOS)
    pixels = np.asarray(grayscale, dtype=np.float32)
    height, width = pixels.shape
    border = np.concatenate((pixels[0], pixels[-1], pixels[:, 0], pixels[:, -1]))
    background = float(np.median(border))
    difference = np.abs(pixels - background)
    border_noise = float(np.median(np.abs(border - background)))
    threshold = max(18.0, border_noise * 3.0)
    foreground = _largest_component(difference > threshold)
    foreground_fraction = float(foreground.mean())
    contrast = float(np.percentile(difference, 95))

    if foreground_fraction < 0.01 or contrast < 20:
        diagnostics = InputDiagnostics(
            foreground_fraction, contrast, False,
            "Aucun article principal suffisamment contrasté n'a été détecté.",
        )
        canvas = Image.new("L", (28, 28), color=0)
    elif foreground_fraction > 0.70:
        diagnostics = InputDiagnostics(
            foreground_fraction, contrast, False,
            "L'image contient trop de décor ou ne permet pas d'isoler un seul article.",
        )
        canvas = Image.new("L", (28, 28), color=0)
    else:
        rows, columns = np.nonzero(foreground)
        top, bottom = int(rows.min()), int(rows.max())
        left, right = int(columns.min()), int(columns.max())
        padding = max(4, round(max(bottom - top + 1, right - left + 1) * 0.12))
        top, bottom = max(0, top - padding), min(height, bottom + padding + 1)
        left, right = max(0, left - padding), min(width, right + padding + 1)

        object_contrast = difference[top:bottom, left:right]
        object_mask = foreground[top:bottom, left:right]
        scale = max(float(np.percentile(object_contrast[object_mask], 99)), 1.0)
        object_pixels = np.where(
            object_mask, np.clip(object_contrast * 255.0 / scale, 0, 255), 0
        ).astype(np.uint8)
        cropped = Image.fromarray(object_pixels, mode="L")
        ratio = min(22 / cropped.width, 22 / cropped.height)
        size = (
            max(1, round(cropped.width * ratio)),
            max(1, round(cropped.height * ratio)),
        )
        resized = cropped.resize(size, Image.Resampling.LANCZOS)
        canvas = Image.new("L", (28, 28), color=0)
        canvas.paste(resized, ((28 - size[0]) // 2, (28 - size[1]) // 2))
        diagnostics = InputDiagnostics(foreground_fraction, contrast, True, "")

    batch = normalise_images(np.asarray(canvas))[np.newaxis, ..., np.newaxis]
    return canvas, batch, diagnostics


def softmax(logits: np.ndarray) -> np.ndarray:
    """Convert one model output vector into stable class probabilities."""
    values = np.asarray(logits, dtype=np.float64).reshape(-1)
    exponentials = np.exp(values - values.max())
    return exponentials / exponentials.sum()


@st.cache_resource(show_spinner=False)
def load_pytorch_model(checkpoint_path: str):
    import torch
    from fashion_cnn.models.pytorch_cnn import PyTorchCNN

    # The local Quadro P620 is unsupported by the installed CUDA build.  A
    # single-image inference is fast on CPU and avoids a hardware-only failure.
    device = torch.device("cpu")
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


def unknown_reasons(
    diagnostics: InputDiagnostics,
    outputs: dict[str, tuple[np.ndarray, float, str]],
) -> list[str]:
    """Return conservative reasons for declining to classify an external image.

    A closed-set CNN cannot prove that an image belongs to no class.  This rule
    therefore combines input quality, confidence, and cross-framework agreement
    instead of presenting a forced class as a reliable answer.
    """
    reasons: list[str] = []
    if not diagnostics.usable:
        reasons.append(diagnostics.reason)
    if len(outputs) != 2:
        reasons.append("Les deux modèles doivent être disponibles pour valider l'image.")
        return reasons
    pytorch_scores = outputs["PyTorch"][0]
    tensorflow_scores = outputs["TensorFlow"][0]
    if int(pytorch_scores.argmax()) != int(tensorflow_scores.argmax()):
        reasons.append("PyTorch et TensorFlow ne sont pas d'accord sur la classe.")
    if min(float(pytorch_scores.max()), float(tensorflow_scores.max())) < 0.75:
        reasons.append("La confiance d'au moins un modèle est inférieure à 75 %.")
    return reasons


def explain_pytorch(batch: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray, int]:
    """Extract intermediate feature maps and a Grad-CAM from the trained CNN."""
    import torch

    model, device = load_pytorch_model(str(PYTORCH_CHECKPOINT))
    inputs = torch.from_numpy(batch.transpose(0, 3, 1, 2)).to(device)
    activations: dict[str, torch.Tensor] = {}

    def capture(name: str, keep_gradient: bool = False):
        def hook(_module, _inputs, output):
            activations[name] = output
            if keep_gradient:
                output.retain_grad()

        return hook

    handles = (
        model.features[0].register_forward_hook(capture("Convolution 1")),
        model.features[2].register_forward_hook(capture("Max Pooling 1")),
        model.features[3].register_forward_hook(capture("Convolution 2", True)),
        model.features[5].register_forward_hook(capture("Max Pooling 2")),
    )
    try:
        model.zero_grad(set_to_none=True)
        logits = model(inputs)
        prediction = int(logits.argmax(dim=1).item())
        logits[0, prediction].backward()

        final_activation = activations["Convolution 2"]
        gradients = final_activation.grad[0]
        weights = gradients.mean(dim=(1, 2), keepdim=True)
        heatmap = torch.relu((weights * final_activation[0]).sum(dim=0))
        heatmap -= heatmap.min()
        maximum = heatmap.max()
        if maximum > 0:
            heatmap /= maximum

        feature_maps = {
            name: tensor[0].detach().cpu().numpy()
            for name, tensor in activations.items()
        }
        return feature_maps, heatmap.detach().cpu().numpy(), prediction
    finally:
        for handle in handles:
            handle.remove()


def render_architecture() -> None:
    """Show the spatial journey through the CNN used by both frameworks."""
    stages = (
        ("Image", "28 × 28 × 1"),
        ("Convolution 1", "28 × 28 × 32"),
        ("Max Pooling 1", "14 × 14 × 32"),
        ("Convolution 2", "14 × 14 × 64"),
        ("Max Pooling 2", "7 × 7 × 64"),
        ("Flatten", "3 136 valeurs"),
        ("Dense", "128 → 10"),
        ("Sortie", "Classe prédite"),
    )
    cards = "".join(
        f"<div style='min-width:125px;flex:1;padding:12px;margin:4px;"
        f"border:1px solid #6c63ff;border-radius:10px;text-align:center'>"
        f"<strong>{name}</strong><br><small>{shape}</small></div>"
        + ("<div style='align-self:center;font-size:22px'>→</div>" if index < len(stages) - 1 else "")
        for index, (name, shape) in enumerate(stages)
    )
    st.markdown(f"<div style='display:flex;flex-wrap:wrap;align-items:stretch'>{cards}</div>", unsafe_allow_html=True)


def render_feature_maps(feature_maps: dict[str, np.ndarray]) -> None:
    """Render a representative sample of channels for every CNN feature stage."""
    import matplotlib.pyplot as plt

    tabs = st.tabs(list(feature_maps))
    for tab, (name, maps) in zip(tabs, feature_maps.items(), strict=True):
        with tab:
            shown = min(8, maps.shape[0])
            figure, axes = plt.subplots(2, 4, figsize=(10, 5))
            for index, axis in enumerate(axes.flat):
                if index < shown:
                    axis.imshow(maps[index], cmap="viridis")
                    axis.set_title(f"Canal {index + 1}")
                axis.axis("off")
            figure.suptitle(f"{name} — {maps.shape[1]} × {maps.shape[2]} × {maps.shape[0]}")
            figure.tight_layout()
            st.pyplot(figure, use_container_width=True)
            plt.close(figure)


def gradcam_overlay(image: Image.Image, heatmap: np.ndarray) -> Image.Image:
    """Blend the Grad-CAM heatmap over the exact grayscale CNN input."""
    import matplotlib.pyplot as plt

    resized = Image.fromarray(heatmap.astype(np.float32), mode="F").resize(
        image.size, Image.Resampling.BILINEAR
    )
    normalized = np.asarray(resized, dtype=np.float32)
    colors = plt.get_cmap("jet")(normalized)[..., :3]
    colored = Image.fromarray(np.uint8(colors * 255), mode="RGB")
    return Image.blend(image.convert("RGB"), colored, alpha=0.48)


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
    st.success("Prétraitement automatique activé")
    st.info(
        "L'interface isole l'objet principal, le centre sur fond noir et adapte "
        "automatiquement le contraste au format Fashion-MNIST."
    )

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
        processed, batch, diagnostics = prepare_image(original)
        preview, transformed = st.columns(2)
        preview.image(original, caption="Image originale", width=280)
        transformed.image(
            processed.resize((280, 280)),
            caption="Entrée automatique du CNN (28 × 28)",
            width=280,
        )
        st.caption(
            f"Objet détecté : {diagnostics.foreground_fraction * 100:.1f} % de l'image ; "
            f"contraste : {diagnostics.contrast:.1f}."
        )
        if not diagnostics.usable:
            st.warning(
                "L'image semble difficile à isoler. L'application pourra répondre "
                "« Je ne sais pas » au lieu de donner une classe trompeuse."
            )

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
            reasons = unknown_reasons(diagnostics, outputs)
            if reasons:
                st.error("🤷 **Je ne sais pas** — cette image semble hors domaine ou ambiguë.")
                for reason in reasons:
                    st.write(f"- {reason}")
                st.caption(
                    "Essayez une image avec un seul article, mieux cadré et un fond plus uniforme."
                )
            elif len(outputs) == 2:
                pt_class = int(outputs["PyTorch"][0].argmax())
                tf_class = int(outputs["TensorFlow"][0].argmax())
                st.success(
                    f"✅ **ACCORD** — Les deux modèles prédisent "
                    f"**{CLASS_NAMES[pt_class]}**."
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

            st.divider()
            st.header("Bonus 2 — À l’intérieur du CNN")
            st.write(
                "Suivez l’image dans le réseau et observez les caractéristiques "
                "détectées par les convolutions et conservées par le max pooling."
            )
            render_architecture()
            if "PyTorch" in outputs:
                try:
                    with st.spinner("Calcul des activations et de Grad-CAM…"):
                        feature_maps, heatmap, explained_class = explain_pytorch(batch)
                    st.subheader("Visualisation des convolutions et du max pooling")
                    st.caption(
                        "Huit canaux représentatifs sont affichés à chaque étape. "
                        "Les zones claires correspondent aux réponses les plus fortes."
                    )
                    render_feature_maps(feature_maps)

                    st.divider()
                    st.header("Bonus 3 — Explication Grad-CAM")
                    st.write(
                        f"La carte de chaleur montre les régions utilisées par PyTorch "
                        f"pour prédire **{CLASS_NAMES[explained_class]}**."
                    )
                    source_col, map_col, overlay_col = st.columns(3)
                    source_col.image(
                        processed.resize((280, 280)),
                        caption="Entrée du CNN",
                        use_container_width=True,
                    )
                    map_col.image(
                        Image.fromarray(np.uint8(heatmap * 255)).resize((280, 280)),
                        caption="Carte Grad-CAM",
                        use_container_width=True,
                    )
                    overlay_col.image(
                        gradcam_overlay(processed, heatmap).resize((280, 280)),
                        caption="Zones importantes superposées",
                        use_container_width=True,
                    )
                    st.caption(
                        "Rouge/jaune : influence forte sur la classe prédite. "
                        "Bleu : influence faible."
                    )
                except Exception as exc:
                    st.error(f"Impossible de produire les explications PyTorch : {exc}")
            else:
                st.info(
                    "Entraînez le modèle PyTorch pour afficher les cartes "
                    "d’activation et Grad-CAM."
                )

    except Exception as exc:
        st.error(f"Impossible de lire cette image : {exc}")
