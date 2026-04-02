"""
backend/main.py
---------------
FastAPI application entry point.

Endpoints:
  WS  /ws/predict    — WebSocket: receives frames, returns predictions
  GET /api/health    — Health check
  GET /api/model-info — Model metadata
  GET /api/signs     — List of all 100 supported signs

Run:
  uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from backend.inference import InferencePipeline
from backend.websocket_handler import websocket_endpoint

load_dotenv()

# ─── App Setup ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="ASL Sign Language Translator API",
    description="Real-time ASL recognition via WebSocket",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Startup: load model once ─────────────────────────────────────────────────

pipeline: InferencePipeline = None

@app.on_event("startup")
async def startup_event():
    global pipeline
    checkpoint = os.getenv("MODEL_CHECKPOINT", "models/checkpoints/best_model.pth")
    num_classes = int(os.getenv("NUM_CLASSES", "100"))
    pipeline = InferencePipeline(
        checkpoint_path=checkpoint,
        num_classes=num_classes,
        conf_threshold=float(os.getenv("CONFIDENCE_THRESHOLD", "0.6")),
    )
    print("[startup] Inference pipeline ready")


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    return {"status": "ok", "model_loaded": pipeline is not None}


@app.get("/api/model-info")
async def model_info():
    if pipeline is None:
        return {"error": "model not loaded"}
    return {
        "num_classes":   pipeline.num_classes,
        "window_size":   32,
        "feature_dim":   399,
        "architecture":  "ASLTransformer (encoder-only, 4 layers, 8 heads, d=256)",
        "parameters":    "~2.5M",
    }


@app.get("/api/signs")
async def get_signs():
    word_list_path = Path("data/word_list.json")
    if not word_list_path.exists():
        return {"error": "word_list.json not found"}
    with open(word_list_path) as f:
        data = json.load(f)
    return {"words": data["words"], "total": data["total_words"]}


@app.websocket("/ws/predict")
async def ws_predict(websocket: WebSocket):
    await websocket_endpoint(websocket, pipeline)
