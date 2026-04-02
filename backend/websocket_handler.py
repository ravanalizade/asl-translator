"""
backend/websocket_handler.py
-----------------------------
Handles the WebSocket lifecycle for one client connection.

Protocol:
  Client → Server:  JSON { "frame": "<base64 JPEG>" }
  Server → Client:  JSON {
      "current_word":    str | null,
      "confidence":      float | null,
      "confirmed_word":  str | null,
      "word_buffer":     [str, ...],
      "sentence":        str | null,       # Gemini output
      "skeleton_points": [...],
  }
"""

import asyncio
import base64
import json
import os
import time
from typing import Optional

import cv2
import numpy as np
from fastapi import WebSocket, WebSocketDisconnect

from backend.inference import InferencePipeline
from backend.gemini_client import GeminiClient


GEMINI_TRIGGER_WORDS = int(os.getenv("GEMINI_TRIGGER_WORDS", "3"))
GEMINI_TIMEOUT_SEC   = float(os.getenv("GEMINI_TIMEOUT_SEC", "5"))


async def websocket_endpoint(websocket: WebSocket, pipeline: InferencePipeline):
    """Main WebSocket handler — one coroutine per connected client."""
    await websocket.accept()
    print("[ws] Client connected")

    # Each connection gets its own pipeline state and Gemini client
    # (pipeline object is shared but state is reset per connection)
    pipeline.reset()

    try:
        gemini = GeminiClient()
    except ValueError as e:
        await websocket.send_json({"error": str(e)})
        await websocket.close()
        return

    # Load word→index mapping
    try:
        import json as _json
        with open("data/raw/manifest.json") as f:
            manifest = _json.load(f)
        word_to_idx = manifest["word_to_idx"]
    except FileNotFoundError:
        # Fallback: empty mapping (model will still run but words will be "UNKNOWN")
        word_to_idx = {}
        print("[ws] Warning: manifest.json not found — word labels unavailable")

    last_gemini_call = time.time()
    last_gemini_sentence: Optional[str] = None

    try:
        while True:
            raw = await websocket.receive_text()
            msg = json.loads(raw)

            # ── Decode frame ──────────────────────────────────────────────────
            if "frame" not in msg:
                continue

            frame_b64 = msg["frame"]
            frame_bytes = base64.b64decode(frame_b64)
            frame_arr = np.frombuffer(frame_bytes, dtype=np.uint8)
            frame_bgr = cv2.imdecode(frame_arr, cv2.IMREAD_COLOR)

            if frame_bgr is None:
                continue

            # ── Run inference ─────────────────────────────────────────────────
            result = pipeline.process_frame(frame_bgr, word_to_idx)

            # ── Gemini trigger ────────────────────────────────────────────────
            should_call_gemini = (
                len(result["word_buffer"]) >= GEMINI_TRIGGER_WORDS
                or (
                    len(result["word_buffer"]) > 0
                    and (time.time() - last_gemini_call) >= GEMINI_TIMEOUT_SEC
                )
            )

            sentence = last_gemini_sentence
            if should_call_gemini and result["word_buffer"]:
                words_to_translate = pipeline.pop_word_buffer()
                sentence = await gemini.translate(words_to_translate)
                last_gemini_sentence = sentence
                last_gemini_call = time.time()
                print(f"[gemini] {words_to_translate} → '{sentence}'")

            # ── Send response ─────────────────────────────────────────────────
            response = {
                "current_word":   result["current_word"],
                "confidence":     result["confidence"],
                "confirmed_word": result["confirmed_word"],
                "word_buffer":    result["word_buffer"],
                "sentence":       sentence,
                "skeleton_points": result["skeleton_points"],
            }
            await websocket.send_json(response)

    except WebSocketDisconnect:
        print("[ws] Client disconnected")
        pipeline.reset()
    except Exception as e:
        print(f"[ws] Error: {e}")
        pipeline.reset()
        try:
            await websocket.send_json({"error": str(e)})
        except Exception:
            pass
