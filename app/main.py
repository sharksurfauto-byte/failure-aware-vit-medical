"""
FastAPI service for failure-aware malaria diagnosis.

Exposes human-in-the-loop prediction endpoint:
- POST /analyze: Upload image, get prediction + AUTO/REVIEW decision
- GET /health: Health check endpoint
"""

import io
import os
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

from app.inference import analyze_image, load_model

# Configuration
CHECKPOINT_PATH = os.getenv(
    "CHECKPOINT_PATH",
    str(Path(__file__).parent.parent / "checkpoints" / "vit_best.pt")
)

# Initialize FastAPI
app = FastAPI(
    title="Failure-Aware Malaria Diagnosis API",
    description="Human-in-the-loop medical AI: auto-predicts low-risk cases, flags uncertain ones for expert review",
    version="1.1.0"
)

# Load model at startup
model = None


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    global model
    
    # Check if checkpoint exists
    if not Path(CHECKPOINT_PATH).exists():
        print(f"WARNING: Checkpoint not found at {CHECKPOINT_PATH}")
        print("Model will be loaded on first request if checkpoint is available")
    else:
        model = load_model(CHECKPOINT_PATH)


@app.get("/")
def root():
    """Root endpoint - redirect to docs."""
    return {
        "message": "Failure-Aware Malaria Diagnosis API",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health")
def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "checkpoint": CHECKPOINT_PATH
    }


@app.post("/analyze")
async def analyze(image: UploadFile = File(...)):
    """
    Analyze malaria cell image using uncertainty-aware prediction.
    
    - **image**: Medical cell image (PNG/JPEG)
    
    Returns:
    - **prediction**: Parasitized or Uninfected
    - **confidence**: Model confidence (0-1)
    - **entropy**: Predictive uncertainty (higher = more uncertain)
    - **decision**: AUTO (safe to auto-predict) or REVIEW (flag for expert)
    """
    global model
    
    # Lazy load model if not loaded at startup
    if model is None:
        if not Path(CHECKPOINT_PATH).exists():
            raise HTTPException(
                status_code=503,
                detail=f"Model checkpoint not found at {CHECKPOINT_PATH}"
            )
        model = load_model(CHECKPOINT_PATH)
    
    # Validate image type
    if image.content_type not in ["image/png", "image/jpeg", "image/jpg"]:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid image type: {image.content_type}. Must be PNG or JPEG."
        )
    
    try:
        # Read and convert image
        img_bytes = await image.read()
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        
        # Analyze
        result = analyze_image(pil_img, model)
        
        return JSONResponse(content=result)
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image: {str(e)}"
        )


@app.get("/info")
def info():
    """Get API information and decision logic."""
    return {
        "model": "Vision Transformer (ViT)",
        "uncertainty_method": "Monte Carlo Dropout",
        "forward_passes": 20,
        "entropy_threshold": 0.2015,
        "decision_logic": {
            "AUTO": "entropy ≤ 0.2015 (low uncertainty, safe to auto-predict)",
            "REVIEW": "entropy > 0.2015 (high uncertainty, flag for expert review)"
        },
        "performance": {
            "coverage": "85% (AUTO predictions)",
            "accuracy_auto": "99.20%",
            "failures_prevented": "82.1%"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
