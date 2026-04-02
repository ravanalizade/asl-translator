// frontend/src/components/PredictionPanel.jsx
// Shows current word prediction + animated confidence bar.

export default function PredictionPanel({ word, confidence }) {
  const pct     = confidence != null ? Math.round(confidence * 100) : 0;
  const hasWord = !!word;

  // Color the bar based on confidence
  const barColor = pct >= 80 ? "bg-green-500"
    : pct >= 60 ? "bg-yellow-500"
    : "bg-red-500";

  return (
    <div className="rounded-2xl bg-gray-900 border border-gray-800 p-5 flex flex-col gap-3">
      <p className="text-xs font-semibold text-gray-500 uppercase tracking-widest">
        Current Prediction
      </p>

      <div className="text-4xl font-bold tracking-tight text-white min-h-[3rem]">
        {hasWord ? word : <span className="text-gray-700">—</span>}
      </div>

      {/* Confidence bar */}
      <div>
        <div className="flex justify-between text-xs text-gray-500 mb-1">
          <span>Confidence</span>
          <span>{hasWord ? `${pct}%` : "—"}</span>
        </div>
        <div className="h-2 bg-gray-800 rounded-full overflow-hidden">
          <div
            className={`h-full rounded-full transition-all duration-300 ${barColor}`}
            style={{ width: `${pct}%` }}
          />
        </div>
      </div>
    </div>
  );
}
