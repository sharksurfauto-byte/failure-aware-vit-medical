"""
Streamlit Application for Failure-Aware Vision Transformer Malaria Diagnosis.

Features:
- Safety-aware prediction with Monte Carlo Dropout (T=20 forward passes)
- Predictive entropy estimation with threshold-based selective prediction:
    - Entropy <= 0.2015: AUTO (routine, safe to auto-predict, ~99.2% accuracy)
    - Entropy > 0.2015: REVIEW (flagged for expert clinician review)
- Supports local checkpoints, Hugging Face Hub downloads, or direct download URLs.
"""

import io
import os
import sys
from pathlib import Path

import numpy as np
import requests
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image

# Set page config
st.set_page_config(
    page_title="Failure-Aware ViT | Malaria Diagnosis",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.vit_baseline import ViTBaseline

# ==============================================================================
# Constants & Configuration
# ==============================================================================
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
DEFAULT_CHECKPOINT_PATH = CHECKPOINT_DIR / "vit_best.pt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
T_PASSES = 20
ENTROPY_THRESHOLD = 0.2015

MEAN = np.array([0.5644536614418029, 0.4508252143859863, 0.48160773515701294], dtype=np.float32)
STD = np.array([0.3182373344898224, 0.25678306818008423, 0.2707959711551666], dtype=np.float32)
CLASS_NAMES = ["Uninfected", "Parasitized"]


# ==============================================================================
# Checkpoint Resolution & Downloader
# ==============================================================================
def download_file_with_progress(url: str, dest_path: Path):
    """Download large checkpoint file with a Streamlit progress indicator."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, stream=True, timeout=120)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    progress_bar = st.progress(0)
    status_text = st.empty()

    downloaded = 0
    chunk_size = 1024 * 1024  # 1 MB

    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    pct = min(1.0, downloaded / total_size)
                    progress_bar.progress(pct)
                    status_text.text(f"Downloading checkpoint: {downloaded / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB")
                else:
                    status_text.text(f"Downloading checkpoint: {downloaded / (1024*1024):.1f} MB")

    progress_bar.empty()
    status_text.empty()


def get_checkpoint_path() -> Path:
    """Finds or downloads the model weights file."""
    # 1. Local checkpoint check
    if DEFAULT_CHECKPOINT_PATH.exists() and DEFAULT_CHECKPOINT_PATH.stat().st_size > 10_000_000:
        return DEFAULT_CHECKPOINT_PATH

    # 2. Check for configured download URL (via Streamlit secrets or environment variables)
    download_url = None
    if hasattr(st, "secrets") and "MODEL_URL" in st.secrets:
        download_url = st.secrets["MODEL_URL"]
    elif "MODEL_URL" in os.environ:
        download_url = os.environ["MODEL_URL"]

    # 3. Check for Hugging Face Hub identifier
    hf_repo = None
    if hasattr(st, "secrets") and "HF_REPO_ID" in st.secrets:
        hf_repo = st.secrets["HF_REPO_ID"]
    elif "HF_REPO_ID" in os.environ:
        hf_repo = os.environ["HF_REPO_ID"]

    if download_url:
        st.info("Model weights not found locally. Downloading from configured URL...")
        download_file_with_progress(download_url, DEFAULT_CHECKPOINT_PATH)
        return DEFAULT_CHECKPOINT_PATH

    if hf_repo:
        st.info(f"Downloading weights from Hugging Face repository '{hf_repo}'...")
        hf_url = f"https://huggingface.co/{hf_repo}/resolve/main/vit_best.pt"
        download_file_with_progress(hf_url, DEFAULT_CHECKPOINT_PATH)
        return DEFAULT_CHECKPOINT_PATH

    return DEFAULT_CHECKPOINT_PATH


# ==============================================================================
# Model Loading & Inference
# ==============================================================================
def enable_mc_dropout(model: torch.nn.Module):
    """Enable dropout during inference while keeping other layers in eval mode."""
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.train()


def predictive_entropy(probs: torch.Tensor) -> torch.Tensor:
    """H(p) = -sum(p * log(p))"""
    eps = 1e-10
    return -torch.sum(probs * torch.log(probs + eps), dim=1)


@st.cache_resource(show_spinner=False)
def load_cached_model(checkpoint_path_str: str):
    """Initializes and caches the ViT model."""
    model = ViTBaseline(
        num_classes=2,
        embed_dim=384,
        depth=6,
        num_heads=6,
        patch_size=16,
        dropout=0.1
    )
    checkpoint = torch.load(checkpoint_path_str, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()
    enable_mc_dropout(model)
    return model, checkpoint.get("epoch", "N/A")


def preprocess_image(pil_img: Image.Image) -> torch.Tensor:
    """Preprocess PIL image using pure Pillow + NumPy to match training distribution."""
    img = pil_img.convert("RGB").resize((224, 224), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0  # [224, 224, 3]
    arr = (arr - MEAN) / STD
    arr = np.transpose(arr, (2, 0, 1))  # [3, 224, 224]
    return torch.tensor(arr, dtype=torch.float32)


@torch.no_grad()
def run_inference(model: torch.nn.Module, pil_img: Image.Image, t_passes: int = T_PASSES):
    """Runs MC Dropout inference and computes predictive uncertainty."""
    x = preprocess_image(pil_img).unsqueeze(0).to(DEVICE)

    probs_list = []
    for _ in range(t_passes):
        logits = model(x)
        probs = F.softmax(logits, dim=1)
        probs_list.append(probs)

    probs_all = torch.stack(probs_list, dim=0)  # [T, 1, num_classes]
    mean_probs = probs_all.mean(dim=0).squeeze(0)  # [num_classes]
    variance_probs = probs_all.squeeze(1).var(dim=0)  # [num_classes]

    confidence = float(mean_probs.max().item())
    pred_idx = int(mean_probs.argmax().item())
    entropy = float(predictive_entropy(mean_probs.unsqueeze(0)).item())

    decision = "AUTO" if entropy <= ENTROPY_THRESHOLD else "REVIEW"
    prediction = CLASS_NAMES[pred_idx]

    return {
        "prediction": prediction,
        "pred_idx": pred_idx,
        "confidence": confidence,
        "entropy": entropy,
        "decision": decision,
        "threshold": ENTROPY_THRESHOLD,
        "probs": [float(p) for p in mean_probs],
        "variance": [float(v) for v in variance_probs],
        "raw_runs": probs_all.squeeze(1).cpu().numpy()
    }


# ==============================================================================
# UI Layout & Application
# ==============================================================================
st.title("🔬 Failure-Aware ViT for Malaria Diagnosis")
st.markdown(
    "A human-in-the-loop clinical decision support system that estimates predictive uncertainty "
    "using **Monte Carlo Dropout** to defer risky cases to medical experts."
)

# Sidebar
with st.sidebar:
    st.header("⚙️ System Configuration")
    st.markdown(
        """
        - **Model**: Vision Transformer (6 Layers, 384 Dim, 6 Heads)
        - **Parameters**: ~5 Million
        - **Dataset**: NIH Malaria Cell Images
        - **Device in use**: `{}`
        """.format(str(DEVICE).upper())
    )

    st.subheader("Decision Thresholds")
    st.markdown(
        f"""
        - **Entropy Threshold ($\\tau$)**: `{ENTROPY_THRESHOLD:.4f}`
        - **MC Forward Passes ($T$)**: `{T_PASSES}`
        - **Operating Policy**:
            - Entropy $\\le \\tau$: **AUTO** (~85% coverage, **99.20% accuracy**)
            - Entropy $> \\tau$: **REVIEW** (~15% deferred, catches **82.1% failures**)
        """
    )

    st.divider()
    st.subheader("Weight Status")
    ckpt_path = get_checkpoint_path()
    if ckpt_path.exists() and ckpt_path.stat().st_size > 10_000_000:
        st.success(f"✓ Model weights available ({ckpt_path.stat().st_size / (1024*1024):.1f} MB)")
    else:
        st.warning("⚠️ No weights found at `checkpoints/vit_best.pt`.")
        st.markdown(
            """
            To provide weights on Streamlit Cloud:
            1. Upload `vit_best.pt` to Hugging Face or public storage.
            2. In Streamlit Cloud settings $\\rightarrow$ Secrets, add:
            ```toml
            HF_REPO_ID = "your-username/failure-aware-vit"
            # or
            MODEL_URL = "https://your-direct-download-url"
            ```
            """
        )

# Main workflow
input_col, result_col = st.columns([1, 1], gap="large")

with input_col:
    st.subheader("1. Upload Thin Blood Smear Cell")
    uploaded_file = st.file_uploader(
        "Choose a microscopic cell image (PNG / JPG / JPEG)",
        type=["png", "jpg", "jpeg"]
    )

    # Sample demo images helper
    st.markdown("Or test with built-in samples:")
    sample_col1, sample_col2 = st.columns(2)
    demo_image = None

    sample_parasitized = PROJECT_ROOT / "data" / "raw" / "cell_images" / "Parasitized"
    sample_uninfected = PROJECT_ROOT / "data" / "raw" / "cell_images" / "Uninfected"

    if sample_parasitized.exists() and sample_col1.button("Load Parasitized Sample"):
        parasitized_files = list(sample_parasitized.glob("*.png"))
        if parasitized_files:
            demo_image = Image.open(parasitized_files[0])
            st.session_state["sample_img"] = demo_image

    if sample_uninfected.exists() and sample_col2.button("Load Uninfected Sample"):
        uninfected_files = list(sample_uninfected.glob("*.png"))
        if uninfected_files:
            demo_image = Image.open(uninfected_files[0])
            st.session_state["sample_img"] = demo_image

    active_image = None
    if uploaded_file is not None:
        active_image = Image.open(uploaded_file)
        if "sample_img" in st.session_state:
            del st.session_state["sample_img"]
    elif "sample_img" in st.session_state:
        active_image = st.session_state["sample_img"]

    if active_image is not None:
        st.image(active_image, caption="Cell Specimen", use_container_width=True)

with result_col:
    st.subheader("2. Safety & Uncertainty Analysis")

    if active_image is None:
        st.info("👈 Upload an image or select a sample on the left to run inference.")
    else:
        if not ckpt_path.exists() or ckpt_path.stat().st_size < 10_000_000:
            st.error(
                "Model checkpoint file `checkpoints/vit_best.pt` is missing or incomplete. "
                "Please configure `HF_REPO_ID` or `MODEL_URL` in Streamlit secrets."
            )
        else:
            with st.spinner("Executing 20 Monte Carlo forward passes..."):
                try:
                    model, epoch = load_cached_model(str(ckpt_path))
                    results = run_inference(model, active_image, T_PASSES)

                    pred = results["prediction"]
                    conf = results["confidence"]
                    entropy = results["entropy"]
                    decision = results["decision"]

                    # Decision banner
                    if decision == "AUTO":
                        st.success(
                            f"### ✅ DECISION: AUTO-PREDICT (Routine Case)\n"
                            f"**Diagnosis: {pred}** ({conf*100:.1f}% confidence)\n\n"
                            f"Predictive entropy is within safe bounds ({entropy:.4f} $\\le$ {ENTROPY_THRESHOLD:.4f}). "
                            f"Model is certain—safe for autonomous reporting."
                        )
                    else:
                        st.error(
                            f"### ⚠️ DECISION: CLINICIAN REVIEW (High Uncertainty)\n"
                            f"**Tentative Diagnosis: {pred}** ({conf*100:.1f}% confidence)\n\n"
                            f"Predictive entropy is elevated ({entropy:.4f} > {ENTROPY_THRESHOLD:.4f}). "
                            f"This case has been **deferred for human microscopic verification** to prevent a potential misdiagnosis."
                        )

                    # Diagnostic Metrics
                    st.divider()
                    st.markdown("#### Diagnostic Telemetry")
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Diagnosis", pred)
                    m2.metric("Mean Confidence", f"{conf * 100:.2f}%")
                    m3.metric("Predictive Entropy", f"{entropy:.4f}")

                    # Class Probabilities Breakdown
                    st.markdown("#### Class Probabilities (Mean across 20 passes)")
                    p_uninfected, p_parasitized = results["probs"][0], results["probs"][1]
                    var_uninfected, var_parasitized = results["variance"][0], results["variance"][1]

                    st.write(f"**Uninfected:** {p_uninfected*100:.2f}% (variance: {var_uninfected:.5f})")
                    st.progress(float(p_uninfected))

                    st.write(f"**Parasitized:** {p_parasitized*100:.2f}% (variance: {var_parasitized:.5f})")
                    st.progress(float(p_parasitized))

                except Exception as e:
                    st.exception(e)

# Clinical disclaimer
st.markdown("---")
st.caption(
    "⚠️ **Disclaimer**: This tool is an academic demonstration of failure-aware machine learning "
    "and selective prediction under the human-in-the-loop paradigm. It is not an FDA-cleared "
    "medical diagnostic device and must not replace professional clinical laboratory assessment."
)
