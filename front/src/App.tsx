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
  gemini_configured?: boolean
}

function App() {
  const [query, setQuery] = useState("")
  const [data, setData] = useState<QueryResponse | null>(null)
  const [originalData, setOriginalData] = useState<QueryResponse | null>(null) // To track initial generation for feedback
  const [searchVisible, setSearchVisible] = useState(false)
  const [searchResults, setSearchResults] = useState<SequenceItem[]>([])
  const [searchQuery, setSearchQuery] = useState("")
  const [searchOffset, setSearchOffset] = useState(0)
  const [loadingMore, setLoadingMore] = useState(false)
  const [hasMore, setHasMore] = useState(false)
  const [feedbackSent, setFeedbackSent] = useState(false)
  const [feedbackSending, setFeedbackSending] = useState(false)
  const [feedbackHistory, setFeedbackHistory] = useState<Array<any>>([])

   const handleReorder = (newSequence: SequenceItem[]) => {
     if (!data) return;
     setData(prev => {
       if (!prev) return null;
       const newData = { ...prev, sequence: newSequence };
       return newData;
     });
   };

   const handleDataUpdate = (newData: QueryResponse | null) => {
     setData(newData)
     if (newData && !originalData) {
       // Store the original data for feedback purposes (only set once)
       setOriginalData(newData)
     }
   }

  const handleDelete = (id: number) => {
    if (!data) return;
    setData(prev => {
      if (!prev) return null
      
      const newSequence = prev.sequence.filter(item => item.id !== id)
      // Update order values
      newSequence.forEach((item, index) => {
        item.order = index + 1
      })
      
      return {
        ...prev,
        sequence: newSequence
      }
    })
  }

  const handleSearch = async () => {
    if (!searchQuery.trim()) {
      setSearchResults([])
      setSearchOffset(0)
      return
    }
    try {
      const response = await fetch("http://localhost:5000/search-pictograms", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: searchQuery, top_k: 8, offset: 0 })
      })
      if (!response.ok) {
        throw new Error(`Error: ${response.status}`)
      }
      const result = await response.json()
      setSearchResults(result.results || [])
      setHasMore((result.results?.length || 0) >= 8) // Assume more if we got full batch
    } catch (error) {
      console.error("Search error:", error)
      setSearchResults([])
      setHasMore(false)
    }
  }

  const handleLoadMore = async () => {
    if (loadingMore || !hasMore) return
    
    setLoadingMore(true)
    try {
      const response = await fetch("http://localhost:5000/search-pictograms", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ 
          query: searchQuery, 
          top_k: 8, 
          offset: searchOffset + 8 
        })
      })
      if (!response.ok) {
        throw new Error(`Error: ${response.status}`)
      }
      const result = await response.json()
      const newResults = result.results || []
      setSearchResults(prev => [...prev, ...newResults])
      setSearchOffset(prev => prev + 8)
      setHasMore(newResults.length >= 8) // More results if we got a full batch
    } catch (error) {
      console.error("Load more error:", error)
    } finally {
      setLoadingMore(false)
    }
  }

  const handleAddPictogram = (pictogram: SequenceItem) => {
    if (!data) {
      // Initialize data if not present (should not happen but safe)
      handleDataUpdate({
        original_text: "",
        concepts_extracted: [],
        sequence: [{ ...pictogram, order: 1 }],
        analysis: { negation: false, temporal_markers: [] }
      })
    } else {
      const newSequence = [...data.sequence, { ...pictogram, order: data.sequence.length + 1 }]
      handleDataUpdate({
        ...data,
        sequence: newSequence
      })
    }
    setSearchVisible(false)
    setSearchQuery("")
    setSearchResults([])
  }

     const handleSendFeedback = async () => {
     if (!data || !originalData) return;

     setFeedbackSending(true);
     try {
       // Compute user modifications
       const originalIds = new Set(originalData.sequence.map(item => item.id));
       const currentIds = new Set(data.sequence.map(item => item.id));

       const deletedPictogramIds = originalData.sequence
         .filter(item => !currentIds.has(item.id))
         .map(item => item.id);

       const addedPictogramIds = data.sequence
         .filter(item => !originalIds.has(item.id))
         .map(item => item.id);

       // For reorder details, we compare the order of items that exist in both
       const originalOrderMap = new Map();
       originalData.sequence.forEach(item => {
         originalOrderMap.set(item.id, item.order);
       });

       const reorderDetails: {
         pictogram_id: number;
         from_order: number;
         to_order: number;
       }[] = [];

       data.sequence.forEach(item => {
         const originalOrder = originalOrderMap.get(item.id);
         if (originalOrder !== undefined && originalOrder !== item.order) {
           reorderDetails.push({
             pictogram_id: item.id,
             from_order: originalOrder,
             to_order: item.order
           });
         }
       });

       const reordered = reorderDetails.length > 0;

       // Prepare feedback for local storage (following README format)
       const feedbackEntry = {
         texto: data.original_text,
         prediccion: originalData?.sequence || [],
         judge_output: data.judge || {
           score: 0,
           missing_concepts: [],
           incorrect_pictograms: [],
           ordering_issues: [],
           suggestions: []
         },
         correccion_humana: data.sequence,
         acciones: {
           reordered,
           deletedPictogramIds,
           addedPictogramIds,
           reorder_details: reorderDetails
         }
       };

       // Add to local history
       setFeedbackHistory(prev => [...prev, feedbackEntry]);

       const feedbackPayload = {
         session_id: `session_${Date.now()}`,
         timestamp: new Date().toISOString(),
         input: {
           original_text: data.original_text,
           concepts_extracted: data.concepts_extracted
         },
         system_generation: {
           sequence: originalData?.sequence || [],
           analysis: originalData?.analysis || { negation: false, temporal_markers: [] }
         },
         llm_evaluation: data.judge || {
           score: 0,
           missing_concepts: [],
           incorrect_pictograms: [],
           ordering_issues: [],
           suggestions: []
         },
         user_modifications: {
           final_sequence: data.sequence,
           actions_taken: {
             reordered,
             deletedPictogramIds,
             addedPictogramIds,
             reorder_details: reorderDetails
           }
         }
       };

       const response = await fetch("http://localhost:5000/feedback", {
         method: "POST",
         headers: { "Content-Type": "application/json" },
         body: JSON.stringify(feedbackPayload)
       });

       if (!response.ok) {
         throw new Error(`Error: ${response.status}`);
       }

       const result = await response.json();
       setFeedbackSent(true);
       // Optionally, we can reset feedback state after a delay
       setTimeout(() => {
         setFeedbackSent(false);
       }, 3000);
     } catch (error) {
       console.error("Feedback error:", error);
       // We could set an error state, but for simplicity we'll just log and maybe show a message
       // For now, we'll set feedbackSent to false and let the UI show an error if we had a state for it.
       // We'll add an error state if needed, but let's keep it simple for now.
     } finally {
       setFeedbackSending(false);
     }
   };

  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-500 to-purple-600 p-8">
      <div className="max-w-4xl mx-auto space-y-6">
        <h1 className="text-3xl font-bold text-white text-center">
          Generador de Pictogramas con Auto-Mejora
        </h1>

        <div className="bg-white rounded-2xl p-6 shadow-xl">
          <SearchBar value={query} onChange={setQuery} onSend={handleDataUpdate} />
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
                  
                  {/* Feedback button */}
                  {!feedbackSending && (
                    <div className="mt-4">
                      <button
                        onClick={handleSendFeedback}
                        disabled={feedbackSent}
                        className={`w-full px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition-colors ${
                          feedbackSent ? 'bg-indigo-400 cursor-not-allowed' : 'hover:bg-indigo-700'
                        }`}
                      >
                        {feedbackSent ? 'Feedback enviado ✓' : 'Enviar feedback para mejora'}
                      </button>
                    </div>
                  )}
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
            <div className="relative">
              {searchResults.length > 0 && (
                <div
                  className="space-y-2 max-h-[400px] overflow-y-auto"
                  onScroll={(e) => {
                    const div = e.target as HTMLDivElement;
                    if (div.scrollTop + div.clientHeight >= div.scrollHeight - 200) {
                      // Near bottom, load more
                      handleLoadMore();
                    }
                  }}
                >
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
                    {loadingMore && (
                      <div className="flex justify-center py-4">
                        <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-indigo-500"></div>
                        <span className="ml-2 text-sm text-gray-500">Cargando más...</span>
                      </div>
                    )}
                    {(!loadingMore && !hasMore && searchResults.length > 0) && (
                      <p className="text-center text-gray-500 text-sm py-4">
                        No hay más pictogramas para mostrar
                      </p>
                    )}
                  </div>
                </div>
              )}
              {searchResults.length === 0 && searchQuery && (
                <p className="text-gray-500 italic">No se encontraron pictogramas para "{searchQuery}"</p>
              )}
            </div>
            <button
              onClick={() => {
                setSearchVisible(false)
                setSearchQuery("")
                setSearchResults([])
                setSearchOffset(0)
                setLoadingMore(false)
                setHasMore(false)
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