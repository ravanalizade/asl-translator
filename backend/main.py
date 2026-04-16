"""
backend/main.py
---------------
FastAPI application entry point.

Supports two models via MODEL_TYPE env variable:
  MODEL_TYPE=rtmpose  → uses InferencePipeline (RTMPose)
  MODEL_TYPE=mediapipe → uses InferencePipelineMP (MediaPipe)  ← default

Endpoints:
  WS  /ws/predict      — WebSocket: receives frames, returns predictions
  GET /api/health      — Health check
  GET /api/model-info  — Model metadata
  GET /api/signs       — List of all 100 supported signs

Run:
  uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

from backend.websocket_handler import websocket_endpoint

load_dotenv()

app = FastAPI(
    title="ASL Sign Language Translator API",
    description="Real-time ASL recognition via WebSocket",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Startup ──────────────────────────────────────────────────────────────────

pipeline = None

@app.on_event("startup")
async def startup_event():
    global pipeline

    model_type  = os.getenv("MODEL_TYPE", "mediapipe").lower()
    num_classes = int(os.getenv("NUM_CLASSES", "100"))
    conf        = float(os.getenv("CONFIDENCE_THRESHOLD", "0.6"))

    if model_type == "rtmpose":
        from backend.inference import InferencePipeline
        checkpoint = os.getenv("MODEL_CHECKPOINT", "models/checkpoints/best_model.pth")
        pipeline   = InferencePipeline(
            checkpoint_path=checkpoint,
            num_classes=num_classes,
            conf_threshold=conf,
        )
        print("[startup] RTMPose pipeline ready")
    else:
        from backend.inference_mp import InferencePipelineMP
        checkpoint = os.getenv("MODEL_CHECKPOINT_MP", "models/checkpoints_mp/best_model_mp.pth")
        pipeline   = InferencePipelineMP(
            checkpoint_path=checkpoint,
            num_classes=num_classes,
            conf_threshold=conf,
        )
        print("[startup] MediaPipe pipeline ready")


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    return {
        "status":       "ok",
        "model_loaded": pipeline is not None,
        "model_type":   os.getenv("MODEL_TYPE", "mediapipe"),
    }


@app.get("/api/model-info")
async def model_info():
    if pipeline is None:
        return {"error": "model not loaded"}
    model_type = os.getenv("MODEL_TYPE", "mediapipe")
    return {
        "model_type":   model_type,
        "num_classes":  pipeline.num_classes,
        "window_size":  32,
        "feature_dim":  126 if model_type == "mediapipe" else 399,
        "architecture": "ASLTransformer (4 layers, 8 heads, d=256)",
        "parameters":   "~2.5M",
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
