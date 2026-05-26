import { useState } from "react"
import { API_URL } from "../api/url"

interface SequenceItem {
  order: number
  concept: string
  id: number
  url: string
  score: number
  description?: string
}

interface JudgeResult {
  score: number
  missing_concepts: string[]
  incorrect_pictograms: { concept: string; reason: string }[]
  ordering_issues: string[]
  suggestions: string[]
}

interface QueryResponse {
  original_text: string
  concepts_extracted: string[]
  sequence: SequenceItem[]
  analysis: {
    negation: boolean
    temporal_markers: string[]
  }
  judge?: JudgeResult
  judge_skipped?: boolean
}

interface SearchBarProps {
  value: string
  onChange: (query: string) => void
  onSend: (data: QueryResponse | null) => void
  onLoadingChange?: (loading: boolean) => void
}

export const SearchBar = ({ value, onChange, onSend, onLoadingChange }: SearchBarProps) => {
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [runJudge, setRunJudge] = useState(true)

  async function handleSubmit() {
    if (!value.trim() || loading) return
    setLoading(true)
    onLoadingChange?.(true)
    setError(null)
    try {
      const res = await fetch(`${API_URL}/query-and-judge`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: value, top_k: 5, run_judge: runJudge })
      })
      if (!res.ok) throw new Error(`Error ${res.status}`)
      const data: QueryResponse = await res.json()
      onSend(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : "Error desconocido")
      onSend(null)
    } finally {
      setLoading(false)
      onLoadingChange?.(false)
    }
  }

  return (
    <div className="flex flex-col gap-3">
      <div className="flex gap-2">
        <input
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && handleSubmit()}
          className="outline-none border border-gray-200 rounded-xl px-4 py-3 w-full focus:border-indigo-400 focus:ring-2 focus:ring-indigo-100 transition-all text-gray-800 placeholder-gray-400 disabled:bg-gray-50"
          type="text"
          placeholder="Escribe una frase en español…"
          value={value}
          disabled={loading}
        />
        <button
          className="px-5 py-3 bg-indigo-500 rounded-xl text-white font-semibold hover:bg-indigo-600 active:scale-95 transition-all disabled:opacity-60 whitespace-nowrap flex items-center gap-2 min-w-[130px] justify-center"
          onClick={handleSubmit}
          disabled={loading || !value.trim()}
        >
          {loading ? (
            <>
              <span className="w-4 h-4 border-2 border-white/60 border-t-white rounded-full animate-spin" />
              Generando
            </>
          ) : "Generar"}
        </button>
      </div>
      {error && (
        <p className="text-red-600 text-sm bg-red-50 border border-red-200 rounded-lg px-3 py-2">{error}</p>
      )}
      <label className="flex items-center gap-2.5 cursor-pointer select-none w-fit">
        <button
          type="button"
          role="switch"
          aria-checked={runJudge}
          onClick={() => setRunJudge(!runJudge)}
          className={`relative w-9 h-5 rounded-full transition-colors focus:outline-none focus:ring-2 focus:ring-indigo-400 focus:ring-offset-1 ${
            runJudge ? "bg-indigo-500" : "bg-gray-300"
          }`}
        >
          <span className={`absolute top-0.5 left-0.5 w-4 h-4 bg-white rounded-full shadow-sm transition-transform duration-200 ${
            runJudge ? "translate-x-4" : "translate-x-0"
          }`} />
        </button>
        <span className="text-sm text-gray-600">Evaluar con LLM</span>
      </label>
    </div>
  )
}
