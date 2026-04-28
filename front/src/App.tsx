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
}

function App() {
  const [query, setQuery] = useState("")
  const [data, setData] = useState<QueryResponse | null>(null)
  const [searchVisible, setSearchVisible] = useState(false)
  const [searchResults, setSearchResults] = useState<SequenceItem[]>([])
  const [searchQuery, setSearchQuery] = useState("")

  const handleReorder = (newSequence: SequenceItem[]) => {
    setData(prev => prev ? { ...prev, sequence: newSequence } : null)
  }

  const handleDelete = (id: number) => {
    setData(prev => {
      if (!prev) return null
      
      const newSequence = prev.sequence.filter(item => item.id !== id)
      // Update order values
      newSequence.forEach((item, index) => {
        item.order = index + 1
      })
      
      return { ...prev, sequence: newSequence }
    })
  }

  const handleSearch = async () => {
    if (!searchQuery.trim()) {
      setSearchResults([])
      return
    }
    try {
      const response = await fetch("http://localhost:5000/search-pictograms", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: searchQuery, top_k: 8 })
      })
      if (!response.ok) {
        throw new Error(`Error: ${response.status}`)
      }
      const result = await response.json()
      setSearchResults(result.results || [])
    } catch (error) {
      console.error("Search error:", error)
      setSearchResults([])
    }
  }

  const handleAddPictogram = (pictogram: SequenceItem) => {
    setData(prev => {
      if (!prev) {
        // Initialize data if not present (should not happen but safe)
        return {
          original_text: "",
          concepts_extracted: [],
          sequence: [{ ...pictogram, order: 1 }],
          analysis: { negation: false, temporal_markers: [] },
          gemini_configured: false
        }
      }
      
      const newSequence = [...prev.sequence, { ...pictogram, order: prev.sequence.length + 1 }]
      return { ...prev, sequence: newSequence }
    })
    setSearchVisible(false)
    setSearchQuery("")
    setSearchResults([])
  }

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
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-lg font-semibold text-gray-700">Pictogramas</h2>
                <button
                  onClick={() => setSearchVisible(true)}
                  className="px-4 py-2 bg-indigo-500 text-white rounded-lg hover:bg-indigo-600 transition-colors"
                >
                  + Buscar pictogramas
                </button>
              </div>
              <ImageList 
                sequence={data.sequence} 
                onReorder={handleReorder}
                onDelete={handleDelete}
              />
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

      {/* Search Dialog */}
      {searchVisible && (
        <div className="fixed inset-0 bg-gray-800 bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-xl p-6 w-full max-w-md">
            <h2 className="text-xl font-bold text-gray-800 mb-4">Buscar pictogramas</h2>
            <div className="mb-4">
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                onKeyPress={(e) => { if (e.key === 'Enter') handleSearch() }}
                placeholder="Escribe una palabra para buscar..."
                className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500"
              />
            </div>
            <button
              onClick={handleSearch}
              className="w-full px-4 py-2 bg-indigo-500 text-white rounded-lg hover:bg-indigo-600 transition-colors mb-4"
            >
              Buscar
            </button>
            {searchResults.length > 0 && (
              <div className="space-y-2">
                <h3 className="text-lg font-semibold text-gray-700 mb-2">Resultados:</h3>
                    <div className="space-y-1">
                      {searchResults.map((item) => (
                        <div
                          key={item.id}
                          onClick={() => handleAddPictogram(item)}
                          className="p-3 border border-gray-200 rounded-lg hover:bg-gray-50 cursor-pointer flex items-center"
                        >
                          <img
                            src={item.url}
                            alt={item.concept}
                            className="w-16 h-16 object-contain rounded-lg mr-3"
                          />
                          <div>
                            <p className="font-medium text-gray-800">{item.concept}</p>
                          </div>
                        </div>
                      ))}
                    </div>
              </div>
            )}
            {searchResults.length === 0 && searchQuery && (
              <p className="text-gray-500 italic">No se encontraron pictogramas para "{searchQuery}"</p>
            )}
            <button
              onClick={() => {
                setSearchVisible(false)
                setSearchQuery("")
                setSearchResults([])
              }}
              className="mt-4 w-full px-4 py-2 bg-gray-300 text-gray-700 rounded-lg hover:bg-gray-400 transition-colors"
            >
              Cerrar
            </button>
          </div>
        </div>
      )}
    </div>
  )
}

export default App