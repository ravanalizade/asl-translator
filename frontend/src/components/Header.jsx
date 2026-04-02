// frontend/src/components/Header.jsx
import { Moon, Sun, Hand } from "lucide-react";

export default function Header({ darkMode, onToggleDark }) {
  return (
    <header className="flex items-center justify-between px-6 py-4 border-b border-gray-800 bg-gray-900">
      <div className="flex items-center gap-3">
        <Hand className="text-indigo-400" size={28} />
        <div>
          <h1 className="text-xl font-bold text-white tracking-tight">
            ASL Translator
          </h1>
          <p className="text-xs text-gray-400">Real-time American Sign Language Recognition</p>
        </div>
      </div>

      <button
        onClick={onToggleDark}
        className="p-2 rounded-lg bg-gray-800 hover:bg-gray-700 transition-colors"
        aria-label="Toggle dark mode"
      >
        {darkMode ? <Sun size={18} className="text-yellow-400" /> : <Moon size={18} className="text-gray-300" />}
      </button>
    </header>
  );
}
