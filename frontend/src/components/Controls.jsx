// frontend/src/components/Controls.jsx
import { Play, Square, Trash2 } from "lucide-react";

export default function Controls({ isRunning, onStart, onStop, onClear }) {
  return (
    <div className="flex gap-3">
      {!isRunning ? (
        <button
          onClick={onStart}
          className="flex items-center gap-2 px-5 py-3 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white font-semibold transition-colors flex-1 justify-center"
        >
          <Play size={18} />
          Start
        </button>
      ) : (
        <button
          onClick={onStop}
          className="flex items-center gap-2 px-5 py-3 rounded-xl bg-red-600 hover:bg-red-500 text-white font-semibold transition-colors flex-1 justify-center"
        >
          <Square size={18} />
          Stop
        </button>
      )}

      <button
        onClick={onClear}
        className="flex items-center gap-2 px-4 py-3 rounded-xl bg-gray-800 hover:bg-gray-700 text-gray-300 font-semibold transition-colors"
        title="Clear history"
      >
        <Trash2 size={18} />
      </button>
    </div>
  );
}
