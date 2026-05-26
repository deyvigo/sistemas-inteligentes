import { useState, useEffect } from "react"
import { API_URL } from "../api/url"

// ── Types ─────────────────────────────────────────────────────────────────────

interface CondSummary {
  score:  number
  errors: number
  recall: number
  bleu:   number
  chrf:   number
}

interface JudgeImpactData {
  metadata: { run_at: string; n: number; seed: number; generator_requested?: string; generator_modes?: Record<string, number> }
  summary: {
    cond_A:       CondSummary
    cond_B:       CondSummary
    improvements: {
      score_improved: number
      score_same: number
      score_worse: number
      errors_reduced: number
      avg_errors_corrected?: number
      total_postprocessing_actions?: number
    }
    human_choice_stats?: {
      total_feedback_with_choice: number
      accepted_original: number
      accepted_judge_refined: number
      human_edited_after_choice: number
      judge_refined_acceptance_rate: number
    }
  }
  results: {
    text:     string
    cond_A:   { score: number; errors: number; recall: number; bleu?: number; chrf?: number }
    cond_B:   { score: number; errors: number; recall: number; bleu?: number; chrf?: number; errors_corrected?: number }
  }[]
}

interface SessionStat {
  session:       number
  text:          string
  judge_score:   number | null
  user_score?:   number | null
  n_corrections: number
  n_missing:     number
  n_incorrect:   number
  annotation_time_seconds?: number | null
}

interface CumulativeStat {
  after_session:         number
  correction_table_size: number
  total_overrides:       number
  total_rejects:         number
}

interface RoundStat {
  label:             string
  feedback_sessions: number
  avg_score:         number
  avg_errors:        number
  avg_recall:        number
  avg_bleu?:         number
  avg_chrf?:         number
  avg_overrides?:    number
  avg_errors_corrected_vs_round0?: number
}

interface IterativeData {
  history: {
    total_sessions: number
    per_session:    SessionStat[]
    cumulative:     CumulativeStat[]
    trends: Record<string, number | null>
  }
  simulation: {
    metadata:  { run_at: string; n: number }
    per_round: RoundStat[]
  } | null
}

// ── Helpers ───────────────────────────────────────────────────────────────────

const deltaClass = (d: number) =>
  d > 0 ? "text-green-600 font-semibold" : d < 0 ? "text-red-600 font-semibold" : "text-gray-500"

// ── Experiment 1 ─────────────────────────────────────────────────────────────

const JudgeImpactSection = () => {
  const [data,    setData]    = useState<JudgeImpactData | null>(null)
  const [loading, setLoading] = useState(true)
  const [running, setRunning] = useState(false)
  const [showRows, setShowRows] = useState(false)

  const fetchData = () => {
    setLoading(true)
    fetch(`${API_URL}/eval/judge-impact`)
      .then(r => r.ok ? r.json() : null)
      .then(d => { setData(d); setLoading(false) })
      .catch(() => { setData(null); setLoading(false) })
  }

  useEffect(() => { fetchData() }, [])

  const handleRun = async () => {
    setRunning(true)
    await fetch(`${API_URL}/eval/judge-impact/run`, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ n: 30, seed: 42, delay: 1.0 }),
    })
    // Poll every 5 s until results appear
    const poll = setInterval(() => {
      fetch(`${API_URL}/eval/judge-impact`)
        .then(r => r.ok ? r.json() : null)
        .then(d => { if (d) { setData(d); setRunning(false); clearInterval(poll) } })
        .catch(() => {})
    }, 5000)
  }

  const metrics: {
    key: "bleu" | "chrf" | "recall" | "errors_corrected"
    label: string
    a: number
    b: number
    higherIsBetter: boolean
  }[] = data ? [
    {
      key: "bleu",
      label: "BLEU-2",
      a: data.summary.cond_A.bleu,
      b: data.summary.cond_B.bleu,
      higherIsBetter: true,
    },
    {
      key: "chrf",
      label: "chrF++",
      a: data.summary.cond_A.chrf,
      b: data.summary.cond_B.chrf,
      higherIsBetter: true,
    },
    {
      key: "recall",
      label: "Recall de conceptos",
      a: data.summary.cond_A.recall,
      b: data.summary.cond_B.recall,
      higherIsBetter: true,
    },
    {
      key: "errors_corrected",
      label: "Número de errores corregidos",
      a: 0,
      b: data.summary.improvements.avg_errors_corrected ?? 0,
      higherIsBetter: true,
    },
  ] : []

  return (
    <section className="bg-white rounded-2xl p-6 shadow-md">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h2 className="text-lg font-bold text-gray-800">
            Experimento 1 — Impacto del LLM-Judge
          </h2>
          <p className="text-sm text-gray-500 mt-0.5">
            Compara la calidad de la generación solo con el generador vs. generador + post-procesado
            guiado por el juez.
          </p>
        </div>
        {!running && (
          <button
            onClick={handleRun}
            className="px-4 py-2 bg-indigo-600 text-white text-sm rounded-lg hover:bg-indigo-700 transition-colors shrink-0"
          >
            {data ? "Re-ejecutar" : "Ejecutar análisis"}
          </button>
        )}
        {running && (
          <span className="text-sm text-indigo-500 animate-pulse shrink-0">
            Ejecutando (puede tardar varios minutos)…
          </span>
        )}
      </div>

      {loading && (
        <div className="flex items-center gap-2 text-gray-400 py-8 justify-center">
          <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-indigo-400" />
          <span>Cargando resultados…</span>
        </div>
      )}

      {!loading && !data && (
        <div className="bg-amber-50 border border-amber-200 rounded-xl p-5 text-center">
          <p className="text-amber-700 font-medium mb-1">Análisis no ejecutado</p>
          <p className="text-sm text-amber-600">
            Pulsa "Ejecutar análisis" para lanzar la evaluación sobre 30 oraciones
            del conjunto de prueba. Requiere la clave Gemini configurada en el servidor.
          </p>
        </div>
      )}

      {!loading && data && (
        <>
          {/* Metadata */}
          <p className="text-xs text-gray-400 mb-4">
            Última ejecución: {new Date(data.metadata.run_at).toLocaleString("es")}
            &ensp;·&ensp;{data.metadata.n} oraciones
            {data.metadata.generator_requested && (
              <>&ensp;·&ensp;generador: {data.metadata.generator_requested}</>
            )}
          </p>

          {/* Requested metrics only */}
          <div className="overflow-x-auto">
            <table className="w-full text-sm border-collapse">
              <thead>
                <tr className="bg-gray-50">
                  <th className="text-left py-2 px-3 text-gray-600 font-semibold rounded-tl-lg">Métrica</th>
                  <th className="text-right py-2 px-3 text-gray-600 font-semibold">Solo Generador (A)</th>
                  <th className="text-right py-2 px-3 text-gray-600 font-semibold">Gen + LLM-Judge (B)</th>
                  <th className="text-right py-2 px-3 text-gray-600 font-semibold rounded-tr-lg">Δ (B − A)</th>
                </tr>
              </thead>
              <tbody>
                {metrics.map((metric, i) => {
                  const a   = metric.a
                  const b   = metric.b
                  const d   = b - a
                  const positive = metric.higherIsBetter ? d > 0 : d < 0
                  return (
                    <tr key={metric.key} className={i % 2 === 0 ? "bg-white" : "bg-gray-50/50"}>
                      <td className="py-2 px-3 text-gray-700">{metric.label}</td>
                      <td className="py-2 px-3 text-right text-gray-600">{a.toFixed(3)}</td>
                      <td className="py-2 px-3 text-right text-gray-600">{b.toFixed(3)}</td>
                      <td className={`py-2 px-3 text-right ${positive ? "text-green-600" : d === 0 ? "text-gray-400" : "text-red-500"}`}>
                        {d >= 0 ? "+" : ""}{d.toFixed(3)}
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>

          {/* Per-sentence toggle */}
          <button
            onClick={() => setShowRows(v => !v)}
            className="mt-4 text-sm text-indigo-500 underline hover:text-indigo-700"
          >
            {showRows ? "Ocultar detalle por oración" : "Ver detalle por oración"}
          </button>

          {showRows && (
            <div className="mt-3 max-h-64 overflow-y-auto rounded-lg border border-gray-200">
              <table className="w-full text-xs">
                <thead className="bg-gray-50 sticky top-0">
                  <tr>
                    <th className="text-left py-1.5 px-3 text-gray-500">Oración</th>
                    <th className="text-right py-1.5 px-2 text-gray-500">BLEU Δ</th>
                    <th className="text-right py-1.5 px-2 text-gray-500">chrF++ Δ</th>
                    <th className="text-right py-1.5 px-2 text-gray-500">Recall Δ</th>
                    <th className="text-right py-1.5 px-2 text-gray-500">Err. ↓</th>
                  </tr>
                </thead>
                <tbody>
                  {data.results.map((r, i) => {
                    const dBleu = (r.cond_B.bleu ?? 0) - (r.cond_A.bleu ?? 0)
                    const dChrf = (r.cond_B.chrf ?? 0) - (r.cond_A.chrf ?? 0)
                    const dRecall = r.cond_B.recall - r.cond_A.recall
                    return (
                      <tr key={i} className={i % 2 === 0 ? "" : "bg-gray-50/50"}>
                        <td className="py-1 px-3 text-gray-600 truncate max-w-xs">{r.text}</td>
                        <td className={`py-1 px-2 text-right ${deltaClass(dBleu)}`}>{dBleu >= 0 ? "+" : ""}{dBleu.toFixed(2)}</td>
                        <td className={`py-1 px-2 text-right ${deltaClass(dChrf)}`}>{dChrf >= 0 ? "+" : ""}{dChrf.toFixed(2)}</td>
                        <td className={`py-1 px-2 text-right ${deltaClass(dRecall)}`}>{dRecall >= 0 ? "+" : ""}{dRecall.toFixed(3)}</td>
                        <td className="py-1 px-2 text-right text-sky-600 font-medium">{r.cond_B.errors_corrected ?? 0}</td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          )}
        </>
      )}
    </section>
  )
}

// ── Experiment 4 ─────────────────────────────────────────────────────────────

const IterativeSection = () => {
  const [data,    setData]    = useState<IterativeData | null>(null)
  const [loading, setLoading] = useState(true)
  const [running, setRunning] = useState(false)
  const [tab,     setTab]     = useState<"history" | "simulation">("history")

  const fetchData = () => {
    setLoading(true)
    fetch(`${API_URL}/eval/iterative`)
      .then(r => r.ok ? r.json() : null)
      .then(d => { setData(d); setLoading(false) })
      .catch(() => { setData(null); setLoading(false) })
  }

  useEffect(() => { fetchData() }, [])

  const handleRunSim = async () => {
    setRunning(true)
    await fetch(`${API_URL}/eval/iterative/run`, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ n: 15, seed: 42, delay: 1.5 }),
    })
    const poll = setInterval(() => {
      fetch(`${API_URL}/eval/iterative`)
        .then(r => r.ok ? r.json() : null)
        .then(d => {
          if (d?.simulation) {
            setData(d)
            setRunning(false)
            setTab("simulation")
            clearInterval(poll)
          }
        })
        .catch(() => {})
    }, 5000)
  }

  return (
    <section className="bg-white rounded-2xl p-6 shadow-md">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h2 className="text-lg font-bold text-gray-800">
            Experimento 4 — Mejora Iterativa
          </h2>
          <p className="text-sm text-gray-500 mt-0.5">
            Ciclo: Generar → Evaluar (LLM) → Corregir (Humano) → Mejorar → Re-generar
          </p>
        </div>
        <button onClick={fetchData} className="text-sm text-indigo-500 hover:text-indigo-700 underline shrink-0">
          Actualizar
        </button>
      </div>

      {loading && (
        <div className="flex items-center gap-2 text-gray-400 py-8 justify-center">
          <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-indigo-400" />
          <span>Cargando…</span>
        </div>
      )}

      {!loading && data && (
        <>
          {/* Sub-tabs */}
          <div className="flex gap-1 mb-4 border-b border-gray-200">
            {(["history", "simulation"] as const).map(t => (
              <button
                key={t}
                onClick={() => setTab(t)}
                className={`px-4 py-2 text-sm font-medium rounded-t-lg transition-colors ${
                  tab === t
                    ? "bg-indigo-50 text-indigo-700 border-b-2 border-indigo-500"
                    : "text-gray-500 hover:text-gray-700"
                }`}
              >
                {t === "history" ? "Historial de feedback" : "Simulación por rondas"}
              </button>
            ))}
          </div>

          {/* History tab */}
          {tab === "history" && (
            <>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 mb-4">
                <div className="bg-gray-50 rounded-xl p-4 border border-gray-100">
                  <p className="text-xs text-gray-500 font-medium">Tiempo de anotación promedio</p>
                  <p className="text-2xl font-bold text-gray-800 mt-1">
                    {typeof data.history.trends.avg_annotation_time_seconds === "number"
                      ? `${data.history.trends.avg_annotation_time_seconds.toFixed(2)}s`
                      : "N/D"}
                  </p>
                </div>
                <div className="bg-gray-50 rounded-xl p-4 border border-gray-100">
                  <p className="text-xs text-gray-500 font-medium">Correlación LLM-humano</p>
                  <p className="text-2xl font-bold text-gray-800 mt-1">
                    {typeof data.history.trends.llm_human_correlation === "number"
                      ? data.history.trends.llm_human_correlation.toFixed(3)
                      : "N/D"}
                  </p>
                  {data.history.trends.llm_human_pairs !== undefined && (
                    <p className="text-xs text-gray-400 mt-1">
                      {data.history.trends.llm_human_pairs} pares evaluados
                    </p>
                  )}
                </div>
              </div>

              <div className="overflow-x-auto">
                <table className="w-full text-sm border-collapse">
                  <thead>
                    <tr className="bg-gray-50">
                      <th className="text-left py-2 px-3 text-gray-500 font-semibold">Sesión</th>
                      <th className="text-left py-2 px-3 text-gray-500 font-semibold">Texto</th>
                      <th className="text-right py-2 px-3 text-gray-500 font-semibold">Tiempo</th>
                    </tr>
                  </thead>
                  <tbody>
                    {data.history.per_session.map((s, i) => (
                      <tr key={s.session} className={i % 2 === 0 ? "bg-white" : "bg-gray-50/40"}>
                        <td className="py-2 px-3 text-gray-500 font-medium">{s.session}</td>
                        <td className="py-2 px-3 text-gray-700 max-w-xs truncate">{s.text}</td>
                        <td className="py-2 px-3 text-right text-gray-500">
                          {s.annotation_time_seconds != null ? `${s.annotation_time_seconds.toFixed(1)}s` : "—"}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </>
          )}

          {/* Simulation tab */}
          {tab === "simulation" && (
            <>
              {!data.simulation ? (
                <div className="bg-amber-50 border border-amber-200 rounded-xl p-5 text-center">
                  <p className="text-amber-700 font-medium mb-1">Simulación no ejecutada</p>
                  <p className="text-sm text-amber-600 mb-3">
                    Evalúa el sistema en 4 rondas sobre 15 oraciones fijas: sin feedback, con 1/3,
                    2/3 y la totalidad del feedback acumulado.
                  </p>
                  {!running ? (
                    <button
                      onClick={handleRunSim}
                      className="px-4 py-2 bg-indigo-600 text-white text-sm rounded-lg hover:bg-indigo-700 transition-colors"
                    >
                      Ejecutar simulación
                    </button>
                  ) : (
                    <span className="text-sm text-indigo-500 animate-pulse">
                      Ejecutando (puede tardar varios minutos)…
                    </span>
                  )}
                </div>
              ) : (
                <>
                  <p className="text-xs text-gray-400 mb-4">
                    Última ejecución: {new Date(data.simulation.metadata.run_at).toLocaleString("es")}
                    &ensp;·&ensp;{data.simulation.metadata.n} oraciones
                  </p>

                  <div className="overflow-x-auto">
                    <table className="w-full text-sm border-collapse">
                      <thead>
                        <tr className="bg-gray-50">
                          <th className="text-left py-2 px-3 text-gray-500 font-semibold">Ronda</th>
                          <th className="text-right py-2 px-3 text-gray-500 font-semibold">BLEU</th>
                          <th className="text-right py-2 px-3 text-gray-500 font-semibold">chrF++</th>
                          <th className="text-right py-2 px-3 text-gray-500 font-semibold">Recall</th>
                          <th className="text-right py-2 px-3 text-gray-500 font-semibold">Err. ↓</th>
                        </tr>
                      </thead>
                      <tbody>
                        {data.simulation.per_round.map((r, i) => (
                          <tr key={i} className={i % 2 === 0 ? "bg-white" : "bg-gray-50/40"}>
                            <td className="py-2 px-3 text-gray-700 font-medium">{r.label}</td>
                            <td className="py-2 px-3 text-right text-gray-600">{(r.avg_bleu ?? 0).toFixed(3)}</td>
                            <td className="py-2 px-3 text-right text-gray-600">{(r.avg_chrf ?? 0).toFixed(3)}</td>
                            <td className="py-2 px-3 text-right text-gray-600">{r.avg_recall.toFixed(3)}</td>
                            <td className="py-2 px-3 text-right text-sky-600 font-medium">
                              {(r.avg_errors_corrected_vs_round0 ?? 0).toFixed(3)}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>

                  {!running && (
                    <button
                      onClick={handleRunSim}
                      className="mt-4 text-sm text-indigo-500 underline hover:text-indigo-700"
                    >
                      Re-ejecutar simulación
                    </button>
                  )}
                </>
              )}
            </>
          )}
        </>
      )}

      {!loading && !data && (
        <div className="text-center py-8 text-red-400">
          Error cargando datos. Verifica que el servidor esté activo.
        </div>
      )}
    </section>
  )
}

// ── Main view ─────────────────────────────────────────────────────────────────

export const ExperimentsView = () => (
  <div className="space-y-6">
    <JudgeImpactSection />
    <IterativeSection />
  </div>
)
