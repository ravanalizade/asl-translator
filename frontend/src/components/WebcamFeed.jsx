// frontend/src/components/WebcamFeed.jsx
// Displays live webcam stream with RTMPose skeleton overlay on a canvas.

import { useRef, useEffect } from "react";
import { Video, VideoOff } from "lucide-react";

const REGION_COLORS = {
  body:  "#22c55e",   // green
  feet:  "#facc15",   // yellow
  face:  "#60a5fa",   // blue
  hands: "#f87171",   // red
};

export default function WebcamFeed({ stream, skeletonPoints, isRunning }) {
  const videoRef  = useRef(null);
  const canvasRef = useRef(null);

  // Attach stream to video element
  useEffect(() => {
    if (videoRef.current && stream) {
      videoRef.current.srcObject = stream;
    }
  }, [stream]);

  // Draw skeleton overlay whenever skeletonPoints update
  useEffect(() => {
    if (!canvasRef.current || !skeletonPoints?.length) return;
    const canvas = canvasRef.current;
    const ctx    = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    skeletonPoints.forEach(pt => {
      if (pt.score < 0.3) return;
      const color = REGION_COLORS[pt.region] || "#ffffff";
      ctx.beginPath();
      ctx.arc(pt.x, pt.y, pt.region === "hands" ? 4 : 3, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.globalAlpha = Math.min(1, pt.score + 0.2);
      ctx.fill();
      ctx.globalAlpha = 1;
    });
  }, [skeletonPoints]);

  return (
    <div className="relative rounded-2xl overflow-hidden bg-gray-900 border border-gray-800 aspect-video">
      {stream ? (
        <>
          <video
            ref={videoRef}
            autoPlay
            playsInline
            muted
            className="w-full h-full object-cover scale-x-[-1]"  /* mirror for selfie view */
          />
          <canvas
            ref={canvasRef}
            width={640}
            height={480}
            className="absolute inset-0 w-full h-full pointer-events-none scale-x-[-1]"
          />
          {/* Status badge */}
          <div className="absolute top-3 left-3 flex items-center gap-2 bg-black/60 rounded-full px-3 py-1">
            <span className="w-2 h-2 rounded-full bg-red-500 animate-pulse" />
            <span className="text-xs text-white font-medium">LIVE</span>
          </div>
          {/* Legend */}
          <div className="absolute bottom-3 right-3 flex flex-col gap-1 bg-black/60 rounded-lg p-2">
            {Object.entries(REGION_COLORS).map(([region, color]) => (
              <div key={region} className="flex items-center gap-2">
                <span className="w-2 h-2 rounded-full" style={{ backgroundColor: color }} />
                <span className="text-[10px] text-gray-300 capitalize">{region}</span>
              </div>
            ))}
          </div>
        </>
      ) : (
        <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 text-gray-500">
          <VideoOff size={48} />
          <p className="text-sm">Camera not started</p>
          <p className="text-xs text-gray-600">Press Start to begin</p>
        </div>
      )}
    </div>
  );
}
