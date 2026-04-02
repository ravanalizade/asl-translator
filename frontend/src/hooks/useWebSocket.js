// frontend/src/hooks/useWebSocket.js
// Manages WebSocket connection, frame sending, and message parsing.

import { useState, useRef, useEffect, useCallback } from "react";

const FRAME_INTERVAL_MS = 100; // send a frame every 100ms = ~10fps to backend

export function useWebSocket(wsUrl, stream, onMessage) {
  const [prediction, setPrediction]     = useState(null);
  const [sentence, setSentence]         = useState(null);
  const [skeletonPoints, setSkeleton]   = useState([]);

  const wsRef          = useRef(null);
  const canvasRef      = useRef(document.createElement("canvas"));
  const intervalRef    = useRef(null);

  // ── Send frames from stream over WebSocket ──────────────────────────────────
  const startSendingFrames = useCallback(() => {
    if (!stream) return;

    const video = document.createElement("video");
    video.srcObject = stream;
    video.play();

    const canvas = canvasRef.current;
    canvas.width  = 640;
    canvas.height = 480;
    const ctx = canvas.getContext("2d");

    intervalRef.current = setInterval(() => {
      if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;
      ctx.drawImage(video, 0, 0, 640, 480);
      const dataUrl = canvas.toDataURL("image/jpeg", 0.7);
      const base64  = dataUrl.split(",")[1];
      wsRef.current.send(JSON.stringify({ frame: base64 }));
    }, FRAME_INTERVAL_MS);
  }, [stream]);

  // ── Connect ─────────────────────────────────────────────────────────────────
  const connect = useCallback(() => {
    if (wsRef.current) return;

    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    ws.onopen = () => {
      console.log("[ws] connected");
      startSendingFrames();
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        setPrediction({
          current_word: data.current_word,
          confidence:   data.confidence,
        });
        if (data.sentence)        setSentence(data.sentence);
        if (data.skeleton_points) setSkeleton(data.skeleton_points);
        if (onMessage)            onMessage(data);
      } catch (e) {
        console.error("[ws] parse error", e);
      }
    };

    ws.onerror = (e) => console.error("[ws] error", e);
    ws.onclose = ()  => {
      console.log("[ws] closed");
      wsRef.current = null;
    };
  }, [wsUrl, startSendingFrames, onMessage]);

  // ── Disconnect ──────────────────────────────────────────────────────────────
  const disconnect = useCallback(() => {
    clearInterval(intervalRef.current);
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    setPrediction(null);
    setSkeleton([]);
  }, []);

  // cleanup on unmount
  useEffect(() => () => disconnect(), [disconnect]);

  return { prediction, sentence, skeletonPoints, connect, disconnect };
}
