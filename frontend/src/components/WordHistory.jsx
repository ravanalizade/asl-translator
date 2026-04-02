// frontend/src/components/WordHistory.jsx
// Scrolling word history with timestamps.

import { useRef, useEffect } from "react";
import { Clock } from "lucide-react";

export default function WordHistory({ history }) {
  const bottomRef = useRef(null);

  // Auto-scroll to newest entry
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [history]);

  return (
    <div className="rounded-2xl bg-gray-900 border border-gray-800 p-5 flex flex-col gap-3 flex-1">
      <div className="flex items-center gap-2">
        <Clock size={14} className="text-gray-500" />
        <p className="text-xs font-semibold text-gray-500 uppercase tracking-widest">
          Word History
        </p>
      </div>

      <div className="flex flex-wrap gap-2 overflow-y-auto max-h-40 pr-1">
        {history.length === 0 ? (
          <p className="text-xs text-gray-700 italic">No words detected yet</p>
        ) : (
          history.map((item, i) => (
            <span
              key={i}
              className="px-2 py-1 rounded-lg bg-gray-800 text-sm font-mono text-gray-300 border border-gray-700"
              title={new Date(item.timestamp).toLocaleTimeString()}
            >
              {item.word}
            </span>
          ))
        )}
        <div ref={bottomRef} />
      </div>
    </div>
  );
}
