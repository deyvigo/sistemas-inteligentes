import { useState } from "react"
import { API_URL } from "../api/url"

interface SequenceItem {
  order: number
  concept: string
  id: number
  url: string
  score: number
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
}

interface SearchBarProps {
  value: string
  onChange: (query: string) => void
  onSend: (data: QueryResponse | null) => void
}

export const SearchBar = ({ value, onChange, onSend }: SearchBarProps) => {
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  async function handleSubmit() {
    if (!value) return
    if (loading) return

    setLoading(true)
    setError(null)
    try {
      console.log("[SEARCHBAR DEBUG] API_URL is:", API_URL)
      const url = `${API_URL}/query-and-judge`
      console.log("[SEARCHBAR DEBUG] Fetching from:", url)
      const res = await fetch(url, {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({ query: value, top_k: 5 })
      })
      if (!res.ok) throw new Error(`Error: ${res.status}`)
      const data: QueryResponse = await res.json()
      console.log("[SEARCHBAR DEBUG] SearchBar received:", data)
      onSend(data)
    } catch (err) {
      console.log("[SEARCHBAR DEBUG] Error:", err)
      setError(err instanceof Error ? err.message : "Error desconocido")
      onSend(null)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="flex flex-col gap-2">
      <div className="flex gap-2 justify-between">
        <input
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && handleSubmit()}
          className="outline-none border border-gray-300 rounded-lg indent-2 p-3 w-full"
          type="text"
          placeholder="Escribe una frase en español..."
          value={value}
        />
        <button
          className="p-3 bg-indigo-500 rounded-lg text-white font-medium hover:bg-indigo-600 transition-colors disabled:opacity-50"
          onClick={handleSubmit}
          disabled={loading}
        >
          {loading ? "Generando..." : "Generar"}
        </button>
      </div>
      {error && (
        <p className="text-red-500 text-sm">{error}</p>
      )}
    </div>
  )
}