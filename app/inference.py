"""
Inference module for failure-aware malaria diagnosis.

Implements MC Dropout uncertainty estimation and decision logic:
- AUTO: Low uncertainty (entropy ≤ 0.2015)
- REVIEW: High uncertainty (entropy > 0.2015)
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.vit_baseline import ViTBaseline

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
T = 20  # MC Dropout forward passes
ENTROPY_THRESHOLD = 0.2015  # Corresponds to ~15% rejection (from Stage 3A)

# Normalization stats (from training)
MEAN = [0.5644536614418029, 0.4508252143859863, 0.48160773515701294]
STD = [0.3182373344898224, 0.25678306818008423, 0.2707959711551666]

# Model singleton (loaded once at startup)
_model = None


def enable_mc_dropout(model):
    """Enable dropout layers during inference while keeping other layers in eval mode."""
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.train()


def predictive_entropy(probs):
    """
    Compute predictive entropy: H(p) = -sum(p * log(p))
    
    Args:
        probs: Tensor of shape [N, C] with predicted probabilities
    
    Returns:
        Tensor of shape [N] with entropy values
    """
    eps = 1e-10
    return -torch.sum(probs * torch.log(probs + eps), dim=1)


def load_model(checkpoint_path: str):
    """Load ViT model from checkpoint."""
    global _model
    
    if _model is not None:
        return _model
    
    print(f"Loading model from {checkpoint_path}...")
    
    # Initialize model
    model = ViTBaseline(
        num_classes=2,
        embed_dim=384,
        depth=6,
        num_heads=6,
        patch_size=16,
        dropout=0.1
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()
    
    # Enable MC Dropout
    enable_mc_dropout(model)
    
    _model = model
    print(f"Model loaded successfully (epoch {checkpoint['epoch']})")
    
    return _model


def preprocess_image(pil_img: Image.Image) -> torch.Tensor:
    """
    Preprocess PIL image for ViT inference.
    
    Args:
        pil_img: PIL Image (RGB)
    
    Returns:
        Preprocessed tensor of shape [1, 3, 224, 224]
    """
    # Resize with aspect ratio preservation + padding
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])
    
    return transform(pil_img)


@torch.no_grad()
def analyze_image(pil_img: Image.Image, model) -> dict:
    """
    Analyze image using MC Dropout and return prediction with decision.
    
    Args:
        pil_img: PIL Image to analyze
        model: Loaded ViT model
    
    Returns:
        Dictionary with prediction, confidence, entropy, and decision
    """
    # Preprocess
    x = preprocess_image(pil_img).unsqueeze(0).to(DEVICE)  # [1, 3, 224, 224]
    
    # MC Dropout: T forward passes
    probs_T = []
    for _ in range(T):
        logits = model(x)
        probs = F.softmax(logits, dim=1)
        probs_T.append(probs)
    
    # Stack and compute mean prediction
    probs_T = torch.stack(probs_T, dim=0)  # [T, 1, C]
    p_mean = probs_T.mean(dim=0).squeeze(0)  # [C]
    
    # Metrics
    confidence = float(p_mean.max().item())
    pred_idx = int(p_mean.argmax().item())
    entropy = float(predictive_entropy(p_mean.unsqueeze(0)).item())
    
    # Decision: AUTO if low uncertainty, REVIEW if high
    decision = "AUTO" if entropy <= ENTROPY_THRESHOLD else "REVIEW"
    
    # Class names
    class_names = ["Parasitized", "Uninfected"]
    prediction = class_names[pred_idx]
    
    return {
        "prediction": prediction,
        "confidence": round(confidence, 4),
        "entropy": round(entropy, 4),
        "decision": decision,
        "threshold": ENTROPY_THRESHOLD
    }
