// frontend/src/components/SentencePanel.jsx
// Displays the polished English sentence from Gemini.

import { Sparkles } from "lucide-react";

export default function SentencePanel({ sentence }) {
  return (
    <div className="rounded-2xl bg-indigo-950/50 border border-indigo-800/40 p-5 flex flex-col gap-3">
      <div className="flex items-center gap-2">
        <Sparkles size={14} className="text-indigo-400" />
        <p className="text-xs font-semibold text-indigo-400 uppercase tracking-widest">
          Translated Sentence
        </p>
      </div>

      <p className="text-lg font-medium text-white leading-relaxed min-h-[3.5rem]">
        {sentence || <span className="text-gray-600 italic">Waiting for signs…</span>}
      </p>

      <p className="text-[10px] text-indigo-600">Powered by Gemini 2.5 Flash</p>
    </div>
  );
}
