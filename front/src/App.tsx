import { useState } from "react"
import { SearchBar } from "./components/SearchBar"
import { ConceptsDisplay } from "./components/ConceptsDisplay"
import { ImageList } from "./components/ImageList"
import { JudgeDisplay } from "./components/JudgeDisplay"

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

function App() {
  const [query, setQuery] = useState("")
  const [data, setData] = useState<QueryResponse | null>(null)

  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-500 to-purple-600 p-8">
      <div className="max-w-4xl mx-auto space-y-6">
        <h1 className="text-3xl font-bold text-white text-center">
          Generador de Pictogramas con Auto-Mejora
        </h1>

        <div className="bg-white rounded-2xl p-6 shadow-xl">
          <SearchBar value={query} onChange={setQuery} onSend={setData} />
        </div>

        {data && (
          <>
            <div className="bg-white rounded-2xl p-6 shadow-xl">
              <h2 className="text-lg font-semibold text-gray-700 mb-3">Texto Original</h2>
              <p className="text-xl text-gray-800 bg-gray-50 p-4 rounded-lg border-l-4 border-indigo-500">
                {data.original_text}
              </p>
            </div>

            <div className="bg-white rounded-2xl p-6 shadow-xl">
              <h2 className="text-lg font-semibold text-gray-700 mb-3">Conceptos Extraídos</h2>
              <ConceptsDisplay
                concepts={data.concepts_extracted}
                negation={data.analysis.negation}
                temporal_markers={data.analysis.temporal_markers}
              />
            </div>

            <div className="bg-white rounded-2xl p-6 shadow-xl">
              <h2 className="text-lg font-semibold text-gray-700 mb-3">Pictogramas</h2>
              <ImageList sequence={data.sequence} />
              {data.judge ? (
                <>
                  <div className="mb-2">Judge data exists: score = {data.judge.score}</div>
                  <JudgeDisplay judge={data.judge} />
                </>
              ) : (
                <div className="text-red-500">judge is falsy: {JSON.stringify(data.judge)}</div>
              )}
            </div>
          </>
        )}
      </div>
    </div>
  )
}

export default App