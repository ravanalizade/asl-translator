// frontend/src/App.jsx
// Main application shell — wires all components together.
// Full implementation: Day 8-10

import { useState, useCallback } from "react";
import Header         from "./components/Header";
import WebcamFeed     from "./components/WebcamFeed";
import PredictionPanel from "./components/PredictionPanel";
import SentencePanel  from "./components/SentencePanel";
import WordHistory    from "./components/WordHistory";
import Controls       from "./components/Controls";
import { useWebSocket } from "./hooks/useWebSocket";
import { useCamera }    from "./hooks/useCamera";

const WS_URL = import.meta.env.VITE_WS_URL || "ws://localhost:8000/ws/predict";

export default function App() {
  const [darkMode, setDarkMode] = useState(true);
  const [isRunning, setIsRunning] = useState(false);
  const [wordHistory, setWordHistory] = useState([]);

  const { stream, startCamera, stopCamera } = useCamera();

  const onMessage = useCallback((data) => {
    if (data.confirmed_word) {
      setWordHistory(prev => [
        ...prev,
        { word: data.confirmed_word, timestamp: new Date().toISOString() }
      ]);
    }
  }, []);

  const { prediction, sentence, skeletonPoints, connect, disconnect } =
    useWebSocket(WS_URL, stream, onMessage);

  const handleStart = async () => {
    await startCamera();
    connect();
    setIsRunning(true);
  };

  const handleStop = () => {
    disconnect();
    stopCamera();
    setIsRunning(false);
  };

  const handleClear = () => setWordHistory([]);

  return (
    <div className={darkMode ? "dark" : ""}>
      <div className="min-h-screen bg-gray-950 text-white flex flex-col">
        <Header darkMode={darkMode} onToggleDark={() => setDarkMode(d => !d)} />

        <main className="flex flex-1 gap-4 p-4">
          {/* Left: webcam */}
          <div className="flex flex-col gap-4 flex-1">
            <WebcamFeed
              stream={stream}
              skeletonPoints={skeletonPoints}
              isRunning={isRunning}
            />
            <Controls
              isRunning={isRunning}
              onStart={handleStart}
              onStop={handleStop}
              onClear={handleClear}
            />
          </div>

          {/* Right: prediction panels */}
          <div className="flex flex-col gap-4 w-80">
            <PredictionPanel
              word={prediction?.current_word}
              confidence={prediction?.confidence}
            />
            <SentencePanel sentence={sentence} />
            <WordHistory history={wordHistory} />
          </div>
        </main>
      </div>
    </div>
  );
}
