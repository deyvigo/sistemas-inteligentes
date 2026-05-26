import { useState, useRef, useCallback, useEffect } from "react"
import { SearchBar } from "./components/SearchBar"
import { ConceptsDisplay } from "./components/ConceptsDisplay"
import { ImageList } from "./components/ImageList"
import { JudgeDisplay } from "./components/JudgeDisplay"
import { ExperimentsView } from "./components/ExperimentsView"
import { API_URL } from "./api/url"

interface SequenceItem {
  uid: string
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
  error?: boolean
}

interface JudgeRefinement {
  strategy: string
  description?: string
  sequence: SequenceItem[]
  actions: {
    type: string
    concept?: string
    old_id?: number
    new_id?: number
    before?: string
  }[]
  changed: boolean
  error?: string
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
  judge_refinement?: JudgeRefinement
  gemini_configured?: boolean
  judge_skipped?: boolean
}

const scoreColors: Record<number, string> = {
  1: "bg-red-100 text-red-700 border-red-300",
  2: "bg-orange-100 text-orange-700 border-orange-300",
  3: "bg-yellow-100 text-yellow-700 border-yellow-300",
  4: "bg-lime-100 text-lime-700 border-lime-300",
  5: "bg-green-100 text-green-700 border-green-300",
}

const createSequenceUid = (id: number) =>
  `${id}-${Date.now()}-${Math.random().toString(36).slice(2, 9)}`

const normalizeSequence = (sequence: Omit<SequenceItem, "uid">[] | SequenceItem[]): SequenceItem[] =>
  sequence.map((item, index) => ({
    ...item,
    uid: "uid" in item && item.uid ? item.uid : createSequenceUid(item.id),
    order: index + 1,
  }))

const normalizeQueryResponse = (
  newData: Omit<QueryResponse, "sequence" | "judge_refinement"> & {
    sequence: Omit<SequenceItem, "uid">[] | SequenceItem[]
    judge_refinement?: Omit<JudgeRefinement, "sequence"> & {
      sequence: Omit<SequenceItem, "uid">[] | SequenceItem[]
    }
  }
): QueryResponse => ({
  ...newData,
  sequence: normalizeSequence(newData.sequence),
  judge_refinement: newData.judge_refinement
    ? {
        ...newData.judge_refinement,
        sequence: normalizeSequence(newData.judge_refinement.sequence),
      }
    : undefined,
})

const SequencePreview = ({ title, sequence }: { title: string; sequence: SequenceItem[] }) => (
  <div className="border border-gray-200 rounded-xl p-4 bg-gray-50/50">
    <h3 className="text-xs font-semibold uppercase tracking-wider text-gray-500 mb-3">{title}</h3>
    <div className="flex flex-wrap gap-3">
      {sequence.map((item) => (
        <div key={item.uid} className="w-20 text-center">
          <div className="relative bg-white rounded-lg p-2 border border-gray-200">
            <img src={item.url} alt={item.concept} className="w-16 h-16 object-contain mx-auto" />
            <span className="absolute -top-2 -right-2 w-5 h-5 bg-indigo-500 text-white rounded-full flex items-center justify-center text-xs font-bold">
              {item.order}
            </span>
          </div>
          <p className="mt-1 text-xs text-gray-500 truncate" title={item.concept}>{item.concept}</p>
        </div>
      ))}
    </div>
  </div>
)

const GeneratingSkeleton = () => (
  <div className="grid grid-cols-1 xl:grid-cols-[1fr_380px] gap-5 animate-pulse">
    <div className="bg-white rounded-2xl p-6 shadow-xl">
      <div className="h-3 bg-gray-200 rounded w-1/4 mb-5" />
      <div className="flex gap-4 justify-center flex-wrap">
        {[1, 2, 3, 4, 5].map(i => <div key={i} className="w-36 h-44 bg-gray-100 rounded-xl" />)}
      </div>
    </div>
    <div className="space-y-5">
      <div className="bg-white rounded-2xl p-6 shadow-xl">
        <div className="h-3 bg-gray-200 rounded w-1/4 mb-4" />
        <div className="h-14 bg-gray-100 rounded-xl" />
        <div className="h-3 bg-gray-200 rounded w-1/3 mt-5 mb-3" />
        <div className="flex flex-wrap gap-2">
          {[1, 2, 3, 4].map(i => <div key={i} className="h-7 w-20 bg-gray-100 rounded-full" />)}
        </div>
      </div>
      <div className="bg-white rounded-2xl p-6 shadow-xl">
        <div className="h-3 bg-gray-200 rounded w-2/3 mb-4" />
        <div className="h-28 bg-gray-100 rounded-xl" />
      </div>
    </div>
  </div>
)

const FeedbackSection = ({
  userScore,
  onScoreToggle,
  feedbackSent,
  feedbackSending,
  onSendFeedback,
}: {
  userScore: number | null
  onScoreToggle: (score: number) => void
  feedbackSent: boolean
  feedbackSending: boolean
  onSendFeedback: () => void
}) => (
  <div className="bg-white rounded-2xl p-6 shadow-xl space-y-4">
    <div>
      <h3 className="text-sm font-semibold text-gray-700 mb-1">Tu puntuación de la generación</h3>
      <p className="text-xs text-gray-400 mb-3">Evalúa la calidad de los pictogramas generados</p>
      <div className="flex gap-2">
        {[1, 2, 3, 4, 5].map((score) => (
          <button
            key={score}
            onClick={() => onScoreToggle(score)}
            className={`w-10 h-10 rounded-full font-bold text-sm border-2 transition-all ${
              userScore === score
                ? `${scoreColors[score]} scale-110 shadow-md`
                : "bg-gray-50 text-gray-400 border-gray-200 hover:bg-gray-100"
            }`}
          >
            {score}
          </button>
        ))}
      </div>
    </div>
    {feedbackSending ? (
      <div className="flex justify-center py-2">
        <span className="w-5 h-5 border-2 border-indigo-400 border-t-transparent rounded-full animate-spin" />
      </div>
    ) : (
      <button
        onClick={onSendFeedback}
        disabled={feedbackSent}
        className={`w-full px-4 py-2.5 rounded-xl font-medium transition-all text-sm ${
          feedbackSent
            ? "bg-green-50 text-green-700 border border-green-200 cursor-default"
            : "bg-indigo-600 text-white hover:bg-indigo-700 active:scale-[0.99] shadow-sm"
        }`}
      >
        {feedbackSent ? "✓ Feedback enviado correctamente" : "Enviar feedback para mejora del sistema"}
      </button>
    )}
  </div>
)

function App() {
  const [view, setView] = useState<"generador" | "experimentos">("generador")
  const [query, setQuery] = useState("")
  const [data, setData] = useState<QueryResponse | null>(null)
  const [originalData, setOriginalData] = useState<QueryResponse | null>(null)
  const [isGenerating, setIsGenerating] = useState(false)
  const [searchVisible, setSearchVisible] = useState(false)
  const [searchResults, setSearchResults] = useState<SequenceItem[]>([])
  const [searchQuery, setSearchQuery] = useState("")
  const [searchOffset, setSearchOffset] = useState(0)
  const [loadingMore, setLoadingMore] = useState(false)
  const [hasMore, setHasMore] = useState(false)
  const [feedbackSent, setFeedbackSent] = useState(false)
  const [feedbackSending, setFeedbackSending] = useState(false)
  const [judgeLoading, setJudgeLoading] = useState(false)
  const [searchingPictograms, setSearchingPictograms] = useState(false)
  const [userScore, setUserScore] = useState<number | null>(null)
  const [acceptedVariant, setAcceptedVariant] = useState<"original" | "judge_refined" | "human_edited">("original")
  const searchDebounceRef = useRef<ReturnType<typeof setTimeout>>(null)

  const closeSearch = useCallback(() => {
    setSearchVisible(false)
    setSearchQuery("")
    setSearchResults([])
    setSearchOffset(0)
    setLoadingMore(false)
    setHasMore(false)
    setSearchingPictograms(false)
  }, [])

  useEffect(() => {
    if (!searchVisible) return
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") closeSearch()
    }
    window.addEventListener("keydown", handler)
    return () => window.removeEventListener("keydown", handler)
  }, [searchVisible, closeSearch])

  const handleReorder = (newSequence: SequenceItem[]) => {
    if (!data) return
    setData(prev => prev ? { ...prev, sequence: normalizeSequence(newSequence) } : null)
    setAcceptedVariant("human_edited")
  }

  const handleDataUpdate = (
    newData: Parameters<typeof normalizeQueryResponse>[0] | null
  ) => {
    const normalizedData = newData ? normalizeQueryResponse(newData) : null
    setData(normalizedData)
    setOriginalData(normalizedData ? JSON.parse(JSON.stringify(normalizedData)) : null)
    setUserScore(null)
    setAcceptedVariant("original")
    setFeedbackSent(false)
  }

  const handleDelete = (uid: string) => {
    if (!data) return
    setData(prev => {
      if (!prev) return null
      const newSequence = prev.sequence
        .filter(item => item.uid !== uid)
        .map((item, index) => ({ ...item, order: index + 1 }))
      return { ...prev, sequence: newSequence }
    })
    setAcceptedVariant("human_edited")
  }

  const doSearch = useCallback(async (q: string) => {
    if (!q.trim()) {
      setSearchResults([])
      setSearchOffset(0)
      setHasMore(false)
      setSearchingPictograms(false)
      return
    }
    setSearchingPictograms(true)
    try {
      const response = await fetch(`${API_URL}/search-pictograms`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: q, top_k: 9, offset: 0 })
      })
      if (!response.ok) throw new Error(`Error: ${response.status}`)
      const result = await response.json()
      setSearchResults(result.results || [])
      setHasMore((result.results?.length || 0) >= 9)
    } catch (error) {
      console.error("Search error:", error)
      setSearchResults([])
      setHasMore(false)
    } finally {
      setSearchingPictograms(false)
    }
  }, [])

  useEffect(() => {
    if (searchDebounceRef.current) clearTimeout(searchDebounceRef.current)
    if (!searchQuery.trim()) {
      setSearchResults([])
      setSearchOffset(0)
      setHasMore(false)
      setSearchingPictograms(false)
      return
    }
    setSearchingPictograms(true)
    searchDebounceRef.current = setTimeout(() => {
      doSearch(searchQuery)
    }, 350)
    return () => {
      if (searchDebounceRef.current) clearTimeout(searchDebounceRef.current)
    }
  }, [searchQuery, doSearch])

  const handleSearch = () => {
    if (searchDebounceRef.current) clearTimeout(searchDebounceRef.current)
    doSearch(searchQuery)
  }

  const handleLoadMore = async () => {
    if (loadingMore || !hasMore) return
    setLoadingMore(true)
    try {
      const response = await fetch(`${API_URL}/search-pictograms`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: searchQuery, top_k: 9, offset: searchOffset + 9 })
      })
      if (!response.ok) throw new Error(`Error: ${response.status}`)
      const result = await response.json()
      const newResults = result.results || []
      setSearchResults(prev => [...prev, ...newResults])
      setSearchOffset(prev => prev + 9)
      setHasMore(newResults.length >= 9)
    } catch (error) {
      console.error("Load more error:", error)
    } finally {
      setLoadingMore(false)
    }
  }

  const handleAddPictogram = (pictogram: SequenceItem) => {
    const newItem = { ...pictogram, uid: createSequenceUid(pictogram.id) }
    if (!data) {
      setData({
        original_text: "",
        concepts_extracted: [],
        sequence: [{ ...newItem, order: 1 }],
        analysis: { negation: false, temporal_markers: [] }
      })
    } else {
      const newSequence = [...data.sequence, { ...newItem, order: data.sequence.length + 1 }]
      setData({ ...data, sequence: normalizeSequence(newSequence) })
    }
    setAcceptedVariant("human_edited")
    closeSearch()
  }

  const handleAcceptOriginal = () => {
    if (!data || !originalData) return
    setData({ ...data, sequence: normalizeSequence(originalData.sequence) })
    setAcceptedVariant("original")
  }

  const handleAcceptJudgeRefinement = () => {
    if (!data?.judge_refinement) return
    setData({ ...data, sequence: normalizeSequence(data.judge_refinement.sequence) })
    setAcceptedVariant("judge_refined")
  }

  const handleSendFeedback = async () => {
    if (!data || !originalData) return
    setFeedbackSending(true)
    try {
      const originalUids = new Set(originalData.sequence.map(item => item.uid))
      const currentUids = new Set(data.sequence.map(item => item.uid))
      const deletedPictogramIds = originalData.sequence
        .filter(item => !currentUids.has(item.uid))
        .map(item => item.id)
      const addedPictogramIds = data.sequence
        .filter(item => !originalUids.has(item.uid))
        .map(item => item.id)
      const originalOrderMap = new Map<string, number>()
      originalData.sequence.forEach(item => originalOrderMap.set(item.uid, item.order))
      const reorderDetails: { local_uid: string; pictogram_id: number; from_order: number; to_order: number }[] = []
      data.sequence.forEach(item => {
        const originalOrder = originalOrderMap.get(item.uid)
        if (originalOrder !== undefined && originalOrder !== item.order) {
          reorderDetails.push({ local_uid: item.uid, pictogram_id: item.id, from_order: originalOrder, to_order: item.order })
        }
      })
      const feedbackPayload = {
        session_id: `session_${Date.now()}`,
        timestamp: new Date().toISOString(),
        input: { original_text: data.original_text, concepts_extracted: data.concepts_extracted },
        system_generation: {
          sequence: originalData.sequence,
          analysis: originalData.analysis || { negation: false, temporal_markers: [] }
        },
        llm_evaluation: data.judge || { score: 0, missing_concepts: [], incorrect_pictograms: [], ordering_issues: [], suggestions: [] },
        user_score: userScore,
        selected_generation_variant: acceptedVariant,
        judge_refinement: data.judge_refinement || null,
        user_modifications: {
          final_sequence: data.sequence,
          actions_taken: { reordered: reorderDetails.length > 0, deletedPictogramIds, addedPictogramIds, reorder_details: reorderDetails }
        }
      }
      const response = await fetch(`${API_URL}/feedback`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(feedbackPayload)
      })
      if (!response.ok) throw new Error(`Error: ${response.status}`)
      setFeedbackSent(true)
      setTimeout(() => setFeedbackSent(false), 4000)
    } catch (error) {
      console.error("Feedback error:", error)
    } finally {
      setFeedbackSending(false)
    }
  }

  const handleRunJudge = async () => {
    if (!data) return
    setJudgeLoading(true)
    try {
      const response = await fetch(`${API_URL}/judge`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: data.original_text, sequence: data.sequence })
      })
      if (!response.ok) throw new Error(`Error: ${response.status}`)
      const judgeResult = await response.json()
      let refinement: JudgeRefinement | undefined
      if (!judgeResult.error) {
        const refineResponse = await fetch(`${API_URL}/judge-refinement`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ sequence: data.sequence, judge: judgeResult })
        })
        if (refineResponse.ok) {
          const refinementResult = await refineResponse.json()
          refinement = { ...refinementResult, sequence: normalizeSequence(refinementResult.sequence || data.sequence) }
        }
      }
      setData(prev => prev ? { ...prev, judge: judgeResult, judge_refinement: refinement, judge_skipped: false } : null)
    } catch (error) {
      console.error("Judge error:", error)
    } finally {
      setJudgeLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-600 via-indigo-500 to-purple-600">
      {/* Top bar */}
      <div className="px-6 sm:px-10 pt-6 pb-4 w-full max-w-[1600px] mx-auto">
        <div className="flex items-center justify-between mb-5">
          <div>
            <h1 className="text-2xl sm:text-3xl font-bold text-white tracking-tight leading-none">
              Generador de Pictogramas AAC
            </h1>
            <p className="text-white/55 text-sm mt-1">Auto-mejora basada en LLM</p>
          </div>
          <div className="flex gap-1 bg-white/15 rounded-xl p-1 backdrop-blur-sm">
            {(["generador", "experimentos"] as const).map((v) => (
              <button
                key={v}
                onClick={() => setView(v)}
                className={`py-2 px-5 rounded-lg text-sm font-semibold capitalize transition-all ${
                  view === v
                    ? "bg-white text-indigo-700 shadow"
                    : "text-white/80 hover:text-white hover:bg-white/10"
                }`}
              >
                {v === "generador" ? "Generador" : "Experimentos"}
              </button>
            ))}
          </div>
        </div>

        {/* SearchBar always visible */}
        {view === "generador" && (
          <div className="bg-white rounded-2xl px-6 py-5 shadow-xl">
            <SearchBar
              value={query}
              onChange={(v) => { setQuery(v); setData(null) }}
              onSend={handleDataUpdate}
              onLoadingChange={setIsGenerating}
            />
          </div>
        )}
      </div>

      {/* Main content */}
      <div className="px-6 sm:px-10 pb-10 w-full max-w-[1600px] mx-auto">
        {view === "experimentos" ? (
          <ExperimentsView />
        ) : (
          <>
            {isGenerating && !data && <GeneratingSkeleton />}

            {data && (
              <div className="grid grid-cols-1 xl:grid-cols-[1fr_380px] gap-5 items-start">

                {/* LEFT: Pictograms + Refinement */}
                <div className="space-y-5">
                  {/* Pictograms */}
                  <div className="bg-white rounded-2xl p-6 shadow-xl">
                    <div className="flex items-center justify-between mb-5">
                      <h2 className="text-lg font-semibold text-gray-800">Pictogramas generados</h2>
                      <button
                        onClick={() => setSearchVisible(true)}
                        className="px-4 py-2 bg-indigo-50 text-indigo-600 rounded-xl hover:bg-indigo-100 transition-colors text-sm font-medium"
                      >
                        + Agregar
                      </button>
                    </div>
                    <ImageList
                      sequence={data.sequence}
                      onReorder={handleReorder}
                      onDelete={handleDelete}
                    />
                  </div>

                  {/* Judge Refinement */}
                  {originalData && data.judge_refinement && (
                    <div className="bg-white rounded-2xl p-6 shadow-xl">
                      <div className="flex items-center justify-between gap-4 mb-4">
                        <h3 className="text-base font-semibold text-gray-800">Refinamiento por LLM-Judge</h3>
                        <span className={`px-3 py-1 rounded-full text-xs font-medium shrink-0 ${
                          data.judge_refinement.changed
                            ? "bg-green-100 text-green-700"
                            : "bg-gray-100 text-gray-500"
                        }`}>
                          {data.judge_refinement.changed ? "Mejorado" : "Sin cambios"}
                        </span>
                      </div>

                      {data.judge_refinement.changed ? (
                        <>
                          <p className="text-xs text-gray-400 mb-4">
                            El judge detectó problemas y generó una versión mejorada. Compara y elige cuál usar.
                          </p>
                          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                            <SequencePreview title="Original del generador" sequence={originalData.sequence} />
                            <SequencePreview title="Mejorada por LLM-Judge" sequence={data.judge_refinement.sequence} />
                          </div>

                          {data.judge_refinement.actions.length > 0 && (
                            <div className="mt-4 bg-indigo-50 rounded-xl p-4">
                              <h4 className="text-xs font-semibold uppercase tracking-wider text-indigo-400 mb-2">
                                Cambios aplicados
                              </h4>
                              <ul className="space-y-1.5">
                                {data.judge_refinement.actions.map((action, index) => (
                                  <li key={index} className="text-xs text-gray-600 flex items-start gap-1.5">
                                    <span className="text-indigo-400 mt-0.5 shrink-0">→</span>
                                    {action.type === "add_missing"
                                      ? `Se agregó "${action.concept}" (pictograma ID ${action.new_id}).`
                                      : action.type === "replace_incorrect"
                                        ? `Se reemplazó "${action.concept}" por un pictograma más adecuado.`
                                        : `Se reordenó "${action.concept}" antes de "${action.before}".`}
                                  </li>
                                ))}
                              </ul>
                            </div>
                          )}

                          <div className="mt-4 grid grid-cols-1 sm:grid-cols-2 gap-3">
                            <button
                              type="button"
                              onClick={handleAcceptOriginal}
                              className={`px-4 py-2.5 rounded-xl border text-sm font-medium transition-all ${
                                acceptedVariant === "original"
                                  ? "bg-white border-indigo-400 text-indigo-700 shadow-sm"
                                  : "bg-white border-gray-200 text-gray-600 hover:border-indigo-300"
                              }`}
                            >
                              {acceptedVariant === "original" ? "✓ Original seleccionado" : "Usar original"}
                            </button>
                            <button
                              type="button"
                              onClick={handleAcceptJudgeRefinement}
                              className={`px-4 py-2.5 rounded-xl border text-sm font-medium transition-all ${
                                acceptedVariant === "judge_refined"
                                  ? "bg-indigo-600 border-indigo-600 text-white shadow-sm"
                                  : "bg-indigo-50 border-indigo-200 text-indigo-700 hover:bg-indigo-100"
                              }`}
                            >
                              {acceptedVariant === "judge_refined" ? "✓ Versión mejorada seleccionada" : "Usar versión mejorada"}
                            </button>
                          </div>
                        </>
                      ) : (
                        <div className="flex items-center gap-3 bg-green-50 border border-green-200 rounded-xl p-4">
                          <span className="text-green-500 text-xl">✓</span>
                          <div>
                            <p className="text-sm font-medium text-green-700">Secuencia ya es óptima</p>
                            <p className="text-xs text-green-600 mt-0.5">
                              El LLM-Judge no encontró mejoras necesarias para esta secuencia.
                            </p>
                          </div>
                        </div>
                      )}
                    </div>
                  )}
                </div>

                {/* RIGHT: Info + Judge + Feedback */}
                <div className="space-y-5">
                  {/* Text + Concepts */}
                  <div className="bg-white rounded-2xl p-6 shadow-xl space-y-4">
                    <div>
                      <h2 className="text-xs font-semibold uppercase tracking-wider text-gray-400 mb-2">
                        Texto original
                      </h2>
                      <p className="text-base text-gray-800 bg-indigo-50 px-4 py-3 rounded-xl border-l-4 border-indigo-400 leading-snug">
                        {data.original_text}
                      </p>
                    </div>
                    <div>
                      <h2 className="text-xs font-semibold uppercase tracking-wider text-gray-400 mb-2">
                        Conceptos extraídos
                      </h2>
                      <ConceptsDisplay concepts={data.concepts_extracted} />
                    </div>
                  </div>

                  {/* Judge evaluation */}
                  {data.judge ? (
                    <div className="bg-white rounded-2xl p-6 shadow-xl">
                      <h2 className="text-base font-semibold text-gray-700 mb-4">Evaluación LLM-Judge</h2>
                      <JudgeDisplay judge={data.judge} />
                    </div>
                  ) : (
                    <div className="bg-white rounded-2xl p-6 shadow-xl">
                      <div className="bg-amber-50 border border-amber-200 rounded-xl p-4">
                        <div className="flex items-center gap-2 text-amber-700 mb-2">
                          <span>⚡</span>
                          <span className="font-medium text-sm">Evaluación omitida</span>
                        </div>
                        <p className="text-xs text-amber-600 mb-3">
                          Puedes evaluar manualmente esta secuencia con el LLM-Judge.
                        </p>
                        <button
                          onClick={handleRunJudge}
                          disabled={judgeLoading}
                          className="w-full px-4 py-2 bg-amber-500 text-white rounded-lg hover:bg-amber-600 transition-colors disabled:opacity-60 flex items-center justify-center gap-2 text-sm font-medium"
                        >
                          {judgeLoading && (
                            <span className="w-4 h-4 border-2 border-white/60 border-t-white rounded-full animate-spin" />
                          )}
                          {judgeLoading ? "Evaluando…" : "Evaluar con LLM-Judge"}
                        </button>
                      </div>
                    </div>
                  )}

                  {/* Feedback */}
                  <FeedbackSection
                    userScore={userScore}
                    onScoreToggle={(score) => setUserScore(score === userScore ? null : score)}
                    feedbackSent={feedbackSent}
                    feedbackSending={feedbackSending}
                    onSendFeedback={handleSendFeedback}
                  />
                </div>

              </div>
            )}
          </>
        )}
      </div>

      {/* Search Dialog */}
      {searchVisible && (
        <div
          className="fixed inset-0 bg-gray-900/60 backdrop-blur-sm flex items-center justify-center z-50 p-4"
          onClick={(e) => { if (e.target === e.currentTarget) closeSearch() }}
        >
          <div className="bg-white rounded-2xl p-6 w-full max-w-xl shadow-2xl">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-lg font-bold text-gray-800">Buscar pictogramas</h2>
              <button
                onClick={closeSearch}
                className="w-8 h-8 rounded-full bg-gray-100 hover:bg-gray-200 text-gray-500 flex items-center justify-center text-lg transition-colors"
                aria-label="Cerrar"
              >
                ×
              </button>
            </div>

            <div className="flex gap-2 mb-4">
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                onKeyDown={(e) => { if (e.key === "Enter") handleSearch() }}
                placeholder="Escribe una palabra para buscar…"
                className="w-full px-4 py-2.5 border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-indigo-300 focus:border-indigo-400 transition-all"
                autoFocus
              />
              <button
                onClick={handleSearch}
                className="px-4 py-2 bg-indigo-500 text-white rounded-xl hover:bg-indigo-600 transition-colors shrink-0 font-medium"
              >
                Buscar
              </button>
            </div>

            {/* Loading */}
            {searchingPictograms && (
              <div className="flex justify-center py-10">
                <span className="w-7 h-7 border-2 border-indigo-400 border-t-transparent rounded-full animate-spin" />
              </div>
            )}

            {/* Results grid */}
            {!searchingPictograms && searchResults.length > 0 && (
              <div
                className="max-h-[420px] overflow-y-auto -mx-1 px-1"
                onScroll={(e) => {
                  const div = e.target as HTMLDivElement
                  if (div.scrollTop + div.clientHeight >= div.scrollHeight - 200) {
                    handleLoadMore()
                  }
                }}
              >
                <div className="grid grid-cols-3 gap-2.5">
                  {searchResults.map((item, index) => (
                    <button
                      key={`${item.id}-${index}`}
                      onClick={() => handleAddPictogram(item)}
                      className="flex flex-col items-center p-3 border border-gray-200 rounded-xl hover:bg-indigo-50 hover:border-indigo-300 transition-all cursor-pointer group"
                    >
                      <img
                        src={item.url}
                        alt={item.concept}
                        className="w-14 h-14 object-contain rounded-lg mb-2 group-hover:scale-105 transition-transform"
                      />
                      <span className="text-xs font-medium text-gray-700 text-center leading-tight truncate w-full">
                        {item.concept}
                      </span>
                    </button>
                  ))}
                </div>

                {loadingMore && (
                  <div className="flex justify-center py-4">
                    <span className="w-5 h-5 border-2 border-indigo-400 border-t-transparent rounded-full animate-spin" />
                  </div>
                )}
                {!loadingMore && !hasMore && searchResults.length > 0 && (
                  <p className="text-center text-gray-400 text-xs py-4">No hay más resultados</p>
                )}
              </div>
            )}

            {/* Empty state */}
            {!searchingPictograms && searchResults.length === 0 && searchQuery && (
              <div className="text-center py-10">
                <p className="text-gray-400 text-sm">
                  No se encontraron pictogramas para{" "}
                  <span className="font-medium text-gray-500">"{searchQuery}"</span>
                </p>
              </div>
            )}

            {!searchQuery && !searchingPictograms && (
              <div className="text-center py-10 text-gray-300">
                <p className="text-4xl mb-3">🔍</p>
                <p className="text-sm">Escribe para buscar pictogramas</p>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}

export default App
