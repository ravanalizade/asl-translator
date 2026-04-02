// frontend/src/hooks/useCamera.js
// Manages browser webcam access via getUserMedia.

import { useState, useRef, useCallback } from "react";

export function useCamera() {
  const [stream, setStream]   = useState(null);
  const [error, setError]     = useState(null);
  const streamRef             = useRef(null);

  const startCamera = useCallback(async () => {
    try {
      const s = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480, facingMode: "user" },
        audio: false,
      });
      streamRef.current = s;
      setStream(s);
      setError(null);
      return s;
    } catch (err) {
      const msg = err.name === "NotAllowedError"
        ? "Camera permission denied. Please allow camera access and try again."
        : `Camera error: ${err.message}`;
      setError(msg);
      throw new Error(msg);
    }
  }, []);

  const stopCamera = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
      setStream(null);
    }
  }, []);

  return { stream, error, startCamera, stopCamera };
}
