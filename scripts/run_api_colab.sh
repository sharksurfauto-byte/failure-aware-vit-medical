#!/bin/bash
# Run FastAPI service on Colab

echo "============================================================"
echo "STARTING FAILURE-AWARE API ON COLAB"
echo "============================================================"

# Set checkpoint path from Drive
export CHECKPOINT_PATH="/content/drive/MyDrive/failure-aware-vit-medical/checkpoints/vit_best.pt"

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "ERROR: Checkpoint not found at $CHECKPOINT_PATH"
    exit 1
fi

echo "✓ Checkpoint found"
echo ""

# Install pyngrok for public URL
pip install -q pyngrok

# Start API in background
echo "Starting API server..."
uvicorn app.main:app --host 0.0.0.0 --port 8000 &
API_PID=$!

# Wait for server to start
sleep 3

echo ""
echo "============================================================"
echo "API RUNNING"
echo "============================================================"
echo "Local URL: http://localhost:8000/docs"
echo ""
echo "To create public URL, run in another cell:"
echo "  from pyngrok import ngrok"
echo "  public_url = ngrok.connect(8000)"
echo "  print(f'Public URL: {public_url}')"
echo "============================================================"

# Keep running
wait $API_PID
