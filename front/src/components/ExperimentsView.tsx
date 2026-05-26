import { useState, useEffect } from "react"
import { API_URL } from "../api/url"

// ── Types ─────────────────────────────────────────────────────────────────────

interface CondSummary {
  score:       number
  errors:      number
  recall:      number
  bleu:        number
  chrf:        number
  seq_length?: number
}

interface OnlineStats {
  total_sessions: number
  avg_score_A:    number
  avg_errors_A:   number
  avg_recall_A:   number
  avg_length_A:   number
  avg_length_B:   number
  avg_actions:    number
  refined_pct:    number
  last_updated:   string
}

interface JudgeImpactData {
  online_only?: boolean
  metadata?: {
    run_at: string
    n: number
    seed: number
    status?: "running" | "complete"
    processed?: number
    generator_requested?: string
    generator_modes?: Record<string, number>
  }
  summary?: {
    cond_A:       CondSummary
    cond_B:       CondSummary
    improvements: {
      score_improved: number
      score_same: number
      score_worse: number
      errors_reduced: number
      avg_errors_corrected?: number
      total_postprocessing_actions?: number
      refined_count?: number
      skipped_errors?: number
    }
    human_choice_stats?: {
      total_feedback_with_choice: number
      accepted_original: number
      accepted_judge_refined: number
      human_edited_after_choice: number
      judge_refined_acceptance_rate: number
    }
  }
  results?: {
    text:     string
    error?:   string
    cond_A:   { score: number; errors: number; recall: number; bleu?: number; chrf?: number; seq_length?: number; ids?: number[] }
    cond_B:   { score: number; errors: number; recall: number; bleu?: number; chrf?: number; seq_length?: number; errors_corrected?: number; postprocessing_actions?: unknown[]; ids?: number[] }
  }[]
  online_stats: OnlineStats | null
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
    metadata:  { run_at: string; n: number; status?: "running" | "complete"; processed?: number; round_cuts?: number[] }
    per_round: RoundStat[]
  } | null
}

// ── Helpers ───────────────────────────────────────────────────────────────────

const deltaClass = (d: number) =>
  d > 0 ? "text-green-600 font-semibold" : d < 0 ? "text-red-600 font-semibold" : "text-gray-500"

const deltaColorPositive = (d: number, higherIsBetter: boolean) => {
  const good = higherIsBetter ? d > 0 : d < 0
  const bad  = higherIsBetter ? d < 0 : d > 0
  return good ? "text-green-600 font-semibold" : bad ? "text-red-500 font-semibold" : "text-gray-400"
}

// ── Experiment 1 ─────────────────────────────────────────────────────────────

const JudgeImpactSection = () => {
  const [data,    setData]    = useState<JudgeImpactData | null>(null)
  const [loading, setLoading] = useState(true)
  const [running, setRunning] = useState(false)
  const [showRows, setShowRows] = useState(false)

  const fetchData = (silent = false) => {
    if (!silent) setLoading(true)
    fetch(`${API_URL}/eval/judge-impact`)
      .then(r => r.ok ? r.json() : null)
      .then(d => { setData(d); setLoading(false) })
      .catch(() => { setData(null); setLoading(false) })
  }

  useEffect(() => { fetchData() }, [])

  // Poll while running — fast interval to show incremental results
  useEffect(() => {
    if (!running) return
    const poll = setInterval(() => {
      fetch(`${API_URL}/eval/judge-impact`)
        .then(r => r.ok ? r.json() : null)
        .then(d => {
          if (!d) return
          setData(d)
          if (d.metadata?.status === "complete") {
            setRunning(false)
            clearInterval(poll)
          }
        })
        .catch(() => {})
    }, 3000)
    return () => clearInterval(poll)
  }, [running])

  const handleRun = async () => {
    setRunning(true)
    await fetch(`${API_URL}/eval/judge-impact/run`, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ n: 15, seed: 42, delay: 0.3 }),
    })
  }

  const meta    = data?.metadata
  const summary = data?.summary
  const imp     = summary?.improvements
  const hcs     = summary?.human_choice_stats
  const online  = data?.online_stats
  const n       = meta?.n ?? 1

  const metrics: {
    key: string
    label: string
    description: string
    a: number
    b: number
    higherIsBetter: boolean
  }[] = summary ? [
    {
      key: "score",
      label: "Puntuación LLM-Judge",
      description: "Calificación 1–5 de coherencia e inteligibilidad",
      a: summary.cond_A.score,
      b: summary.cond_B.score,
      higherIsBetter: true,
    },
    {
      key: "errors",
      label: "Errores detectados",
      description: "Conceptos faltantes + pictogramas incorrectos",
      a: summary.cond_A.errors,
      b: summary.cond_B.errors,
      higherIsBetter: false,
    },
    {
      key: "recall",
      label: "Recall de conceptos",
      description: "% de conceptos clave con pictograma asociado",
      a: summary.cond_A.recall,
      b: summary.cond_B.recall,
      higherIsBetter: true,
    },
    {
      key: "seq_length",
      label: "Longitud de secuencia",
      description: "Número de pictogramas (B puede crecer por conceptos añadidos)",
      a: summary.cond_A.seq_length ?? 0,
      b: summary.cond_B.seq_length ?? 0,
      higherIsBetter: true,
    },
    {
      key: "bleu",
      label: "BLEU-2",
      description: "Precisión de n-gramas vs. referencia dorada",
      a: summary.cond_A.bleu,
      b: summary.cond_B.bleu,
      higherIsBetter: true,
    },
    {
      key: "chrf",
      label: "chrF++",
      description: "F-score de n-gramas, robusto ante variaciones",
      a: summary.cond_A.chrf,
      b: summary.cond_B.chrf,
      higherIsBetter: true,
    },
  ] : []

  return (
    <section className="bg-white rounded-2xl p-6 shadow-md">
      {/* Header */}
      <div className="flex items-start justify-between gap-4 mb-1">
        <div>
          <h2 className="text-lg font-bold text-gray-800">
            Experimento 1 — Impacto del LLM-Judge
          </h2>
          <p className="text-sm text-gray-500 mt-0.5">
            Condición A (Solo Generador) vs. Condición B (Generador + post-procesado guiado por el LLM-Judge)
            sobre oraciones de prueba complejas (≥5 palabras, ≥3 pictogramas de referencia).
          </p>
        </div>
        {!running ? (
          <button
            onClick={handleRun}
            className="px-4 py-2 bg-indigo-600 text-white text-sm rounded-lg hover:bg-indigo-700 transition-colors shrink-0 font-medium"
          >
            {summary ? "Re-ejecutar" : "Ejecutar análisis"}
          </button>
        ) : (
          <div className="text-right shrink-0">
            <div className="flex items-center gap-2 text-sm text-indigo-500 justify-end">
              <span className="w-4 h-4 border-2 border-indigo-400 border-t-transparent rounded-full animate-spin" />
              <span>
                {meta?.processed != null
                  ? `Procesando ${meta.processed}/${meta.n}…`
                  : "Iniciando…"}
              </span>
            </div>
            {meta?.processed != null && meta?.n && (
              <div className="mt-1.5 w-36 h-1.5 bg-gray-200 rounded-full overflow-hidden">
                <div
                  className="h-full bg-indigo-400 rounded-full transition-all duration-500"
                  style={{ width: `${(meta.processed / meta.n) * 100}%` }}
                />
              </div>
            )}
          </div>
        )}
      </div>

      {loading && (
        <div className="flex items-center gap-2 text-gray-400 py-10 justify-center">
          <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-indigo-400" />
          <span>Cargando resultados…</span>
        </div>
      )}

      {!loading && !data && (
        <div className="bg-amber-50 border border-amber-200 rounded-xl p-5 text-center mt-4">
          <p className="text-amber-700 font-medium mb-1">Análisis no ejecutado</p>
          <p className="text-sm text-amber-600">
            Pulsa "Ejecutar análisis" para lanzar la evaluación sobre 15 oraciones complejas
            del conjunto de prueba. Requiere la clave Gemini configurada en el servidor.
          </p>
        </div>
      )}

      {/* ── Estadísticas en línea (uso real) ── */}
      {!loading && online && online.total_sessions > 0 && (
        <>
          <h3 className="text-xs font-semibold uppercase tracking-wider text-gray-400 mt-5 mb-3">
            Estadísticas acumuladas de uso real
            <span className="ml-2 text-gray-300 normal-case font-normal">
              ({online.total_sessions} sesiones · actualizado automáticamente)
            </span>
          </h3>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-2">
            <div className="bg-indigo-50 rounded-xl p-3 border border-indigo-100">
              <p className="text-xs text-indigo-500 font-medium">Score promedio</p>
              <p className="text-xl font-bold text-indigo-700 mt-0.5">{online.avg_score_A.toFixed(2)}</p>
              <p className="text-xs text-indigo-400">generador (cond A)</p>
            </div>
            <div className="bg-indigo-50 rounded-xl p-3 border border-indigo-100">
              <p className="text-xs text-indigo-500 font-medium">Errores promedio</p>
              <p className="text-xl font-bold text-indigo-700 mt-0.5">{online.avg_errors_A.toFixed(2)}</p>
              <p className="text-xs text-indigo-400">antes de refinar</p>
            </div>
            <div className="bg-indigo-50 rounded-xl p-3 border border-indigo-100">
              <p className="text-xs text-indigo-500 font-medium">Longitud A → B</p>
              <p className="text-xl font-bold text-indigo-700 mt-0.5">
                {online.avg_length_A.toFixed(1)}→{online.avg_length_B.toFixed(1)}
              </p>
              <p className="text-xs text-indigo-400">pictogramas prom.</p>
            </div>
            <div className="bg-indigo-50 rounded-xl p-3 border border-indigo-100">
              <p className="text-xs text-indigo-500 font-medium">Secuencias refinadas</p>
              <p className="text-xl font-bold text-indigo-700 mt-0.5">
                {(online.refined_pct * 100).toFixed(0)}%
              </p>
              <p className="text-xs text-indigo-400">modificadas por judge</p>
            </div>
          </div>
          <p className="text-xs text-gray-400 mb-4">
            Última sesión: {new Date(online.last_updated).toLocaleString("es")}
          </p>
        </>
      )}

      {!loading && data && imp && (
        <>
          {/* Metadata + estado */}
          <div className="flex items-center justify-between mt-3 mb-5">
            <p className="text-xs text-gray-400">
              Última ejecución batch: {meta ? new Date(meta.run_at).toLocaleString("es") : "—"}
              &ensp;·&ensp;{meta?.processed ?? meta?.n}/{meta?.n} oraciones
              {meta?.generator_requested && (
                <>&ensp;·&ensp;generador: <span className="font-medium">{meta.generator_requested}</span></>
              )}
              {imp.skipped_errors != null && imp.skipped_errors > 0 && (
                <>&ensp;·&ensp;<span className="text-amber-500">{imp.skipped_errors} omitidas por error</span></>
              )}
            </p>
            {meta?.status === "running" && (
              <span className="text-xs px-2 py-0.5 bg-indigo-100 text-indigo-600 rounded-full animate-pulse">
                En progreso
              </span>
            )}
            {meta?.status === "complete" && (
              <span className="text-xs px-2 py-0.5 bg-green-100 text-green-600 rounded-full">
                Completado
              </span>
            )}
          </div>

          {/* ── Tabla de métricas ── */}
          <h3 className="text-xs font-semibold uppercase tracking-wider text-gray-400 mb-3">
            Comparación de métricas (promedios) — experimento batch
          </h3>
          <div className="overflow-x-auto rounded-xl border border-gray-200">
            <table className="w-full text-sm border-collapse">
              <thead>
                <tr className="bg-gray-50 border-b border-gray-200">
                  <th className="text-left py-3 px-4 text-gray-500 font-semibold">Métrica</th>
                  <th className="text-right py-3 px-4 text-gray-500 font-semibold">Solo Gen (A)</th>
                  <th className="text-right py-3 px-4 text-gray-500 font-semibold">Gen + Judge (B)</th>
                  <th className="text-right py-3 px-4 text-gray-500 font-semibold">Δ (B − A)</th>
                </tr>
              </thead>
              <tbody>
                {metrics.map((m, i) => {
                  const d = m.b - m.a
                  return (
                    <tr key={m.key} className={`border-b border-gray-100 last:border-0 ${i % 2 === 0 ? "bg-white" : "bg-gray-50/40"}`}>
                      <td className="py-3 px-4">
                        <span className="text-gray-800 font-medium block">{m.label}</span>
                        <span className="text-xs text-gray-400">{m.description}</span>
                      </td>
                      <td className="py-3 px-4 text-right font-mono text-gray-600">{m.a.toFixed(3)}</td>
                      <td className="py-3 px-4 text-right font-mono text-gray-600">{m.b.toFixed(3)}</td>
                      <td className={`py-3 px-4 text-right font-mono ${d === 0 ? "text-gray-400" : deltaColorPositive(d, m.higherIsBetter)}`}>
                        {d === 0
                          ? <span className="text-gray-300">sin cambio</span>
                          : <>{d > 0 ? "+" : ""}{d.toFixed(3)} <span className="text-xs">{(m.higherIsBetter ? d > 0 : d < 0) ? "↑" : "↓"}</span></>
                        }
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>

          {/* ── Tarjetas de mejora ── */}
          <h3 className="text-xs font-semibold uppercase tracking-wider text-gray-400 mt-6 mb-3">
            Resumen sobre {n} oraciones de prueba
          </h3>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            <div className="bg-gray-50 rounded-xl p-4 border border-gray-100">
              <p className="text-xs text-gray-500 mb-1">Score mejoró</p>
              <p className="text-2xl font-bold text-green-600">
                {imp.score_improved}
                <span className="text-base font-normal text-gray-400">/{n}</span>
              </p>
              <div className="mt-2 flex gap-1 flex-wrap text-xs text-gray-400">
                <span className="px-1.5 py-0.5 bg-gray-200 rounded-full">{imp.score_same} igual</span>
                <span className="px-1.5 py-0.5 bg-red-100 text-red-500 rounded-full">{imp.score_worse} peor</span>
              </div>
            </div>
            <div className="bg-gray-50 rounded-xl p-4 border border-gray-100">
              <p className="text-xs text-gray-500 mb-1">Errores reducidos</p>
              <p className="text-2xl font-bold text-indigo-600">
                {imp.errors_reduced}
                <span className="text-base font-normal text-gray-400">/{n}</span>
              </p>
              {imp.avg_errors_corrected !== undefined && (
                <p className="mt-2 text-xs text-gray-400">
                  Prom. <span className="font-medium text-gray-600">{imp.avg_errors_corrected.toFixed(2)}</span>/oración
                </p>
              )}
            </div>
            <div className="bg-gray-50 rounded-xl p-4 border border-gray-100">
              <p className="text-xs text-gray-500 mb-1">Secuencias modificadas</p>
              <p className="text-2xl font-bold text-teal-600">
                {imp.refined_count ?? "—"}
                <span className="text-base font-normal text-gray-400">/{n}</span>
              </p>
              <p className="mt-2 text-xs text-gray-400">judge cambió la secuencia</p>
            </div>
            <div className="bg-gray-50 rounded-xl p-4 border border-gray-100">
              <p className="text-xs text-gray-500 mb-1">Acciones de post-procesado</p>
              <p className="text-2xl font-bold text-gray-700">
                {imp.total_postprocessing_actions ?? "—"}
              </p>
              <p className="mt-2 text-xs text-gray-400">
                Prom. <span className="font-medium text-gray-600">
                  {imp.total_postprocessing_actions != null ? (imp.total_postprocessing_actions / n).toFixed(1) : "—"}
                </span>/oración
              </p>
            </div>
          </div>

          {/* ── Detalle por oración ── */}
          <button
            onClick={() => setShowRows(v => !v)}
            className="mt-5 text-sm text-indigo-500 hover:text-indigo-700 flex items-center gap-1"
          >
            <span>{showRows ? "▾" : "▸"}</span>
            {showRows ? "Ocultar detalle por oración" : "Ver detalle por oración"}
          </button>

          {showRows && (
            <div className="mt-3 max-h-72 overflow-y-auto rounded-xl border border-gray-200">
              <table className="w-full text-xs">
                <thead className="bg-gray-50 sticky top-0 border-b border-gray-200">
                  <tr>
                    <th className="text-left py-2 px-3 text-gray-500 font-semibold">Oración</th>
                    <th className="text-right py-2 px-2 text-gray-500 font-semibold">Score A→B</th>
                    <th className="text-right py-2 px-2 text-gray-500 font-semibold">Err. A→B</th>
                    <th className="text-right py-2 px-2 text-gray-500 font-semibold">Long. A→B</th>
                    <th className="text-right py-2 px-2 text-gray-500 font-semibold">Acciones</th>
                    <th className="text-right py-2 px-2 text-gray-500 font-semibold">Recall Δ</th>
                  </tr>
                </thead>
                <tbody>
                  {(data.results ?? []).map((r, i) => {
                    if (r.error) {
                      return (
                        <tr key={i} className="border-b border-gray-100 last:border-0 bg-red-50/40">
                          <td className="py-2 px-3 text-red-400 truncate max-w-[200px]" title={r.text}>{r.text}</td>
                          <td colSpan={5} className="py-2 px-2 text-center text-red-400 text-xs">error al procesar</td>
                        </tr>
                      )
                    }
                    const unchanged = JSON.stringify(r.cond_A.ids) === JSON.stringify(r.cond_B.ids)
                    const dScore    = r.cond_B.score  - r.cond_A.score
                    const dErrors   = r.cond_B.errors - r.cond_A.errors
                    const dRecall   = r.cond_B.recall - r.cond_A.recall
                    const lenA      = r.cond_A.seq_length ?? r.cond_A.ids?.length ?? 0
                    const lenB      = r.cond_B.seq_length ?? r.cond_B.ids?.length ?? 0
                    const actions   = r.cond_B.postprocessing_actions?.length ?? 0
                    return (
                      <tr key={i} className={`border-b border-gray-100 last:border-0 ${unchanged ? "opacity-50" : i % 2 === 0 ? "bg-white" : "bg-gray-50/40"}`}>
                        <td className="py-2 px-3 text-gray-600 truncate max-w-[200px]" title={r.text}>
                          {r.text}
                          {unchanged && <span className="ml-1 text-gray-300 text-[10px]">sin cambio</span>}
                        </td>
                        <td className={`py-2 px-2 text-right font-mono ${deltaClass(dScore)}`}>
                          {r.cond_A.score}→{r.cond_B.score}
                        </td>
                        <td className={`py-2 px-2 text-right font-mono ${deltaClass(-dErrors)}`}>
                          {r.cond_A.errors}→{r.cond_B.errors}
                        </td>
                        <td className={`py-2 px-2 text-right font-mono ${deltaClass(lenB - lenA)}`}>
                          {lenA}→{lenB}
                        </td>
                        <td className="py-2 px-2 text-right font-mono text-indigo-500">
                          {actions > 0 ? actions : <span className="text-gray-300">0</span>}
                        </td>
                        <td className={`py-2 px-2 text-right font-mono ${deltaClass(dRecall)}`}>
                          {dRecall >= 0 ? "+" : ""}{dRecall.toFixed(3)}
                        </td>
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

  const fetchData = (silent = false) => {
    if (!silent) setLoading(true)
    fetch(`${API_URL}/eval/iterative`)
      .then(r => r.ok ? r.json() : null)
      .then(d => { setData(d); setLoading(false) })
      .catch(() => { setData(null); setLoading(false) })
  }

  useEffect(() => { fetchData() }, [])

  // Poll while running — fast interval to show incremental results
  useEffect(() => {
    if (!running) return
    const poll = setInterval(() => {
      fetch(`${API_URL}/eval/iterative`)
        .then(r => r.ok ? r.json() : null)
        .then(d => {
          if (!d) return
          setData(d)
          if (d.simulation?.metadata?.status === "complete") {
            setRunning(false)
            setTab("simulation")
            clearInterval(poll)
          }
        })
        .catch(() => {})
    }, 3000)
    return () => clearInterval(poll)
  }, [running])

  const handleRunSim = async () => {
    setRunning(true)
    setTab("simulation")
    await fetch(`${API_URL}/eval/iterative/run`, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ n: 10, seed: 42, delay: 0.3 }),
    })
  }

  const sim = data?.simulation

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
        <button onClick={() => fetchData()} className="text-sm text-indigo-500 hover:text-indigo-700 underline shrink-0">
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

              <div className="overflow-x-auto rounded-xl border border-gray-200">
                <table className="w-full text-sm border-collapse">
                  <thead>
                    <tr className="bg-gray-50 border-b border-gray-200">
                      <th className="text-left py-2 px-3 text-gray-500 font-semibold">#</th>
                      <th className="text-left py-2 px-3 text-gray-500 font-semibold">Texto</th>
                      <th className="text-right py-2 px-3 text-gray-500 font-semibold">Score LLM</th>
                      <th className="text-right py-2 px-3 text-gray-500 font-semibold">Score usuario</th>
                      <th className="text-right py-2 px-3 text-gray-500 font-semibold">Correcciones</th>
                      <th className="text-right py-2 px-3 text-gray-500 font-semibold">Tiempo</th>
                    </tr>
                  </thead>
                  <tbody>
                    {data.history.per_session.map((s, i) => (
                      <tr key={s.session} className={`border-b border-gray-100 last:border-0 ${i % 2 === 0 ? "bg-white" : "bg-gray-50/40"}`}>
                        <td className="py-2 px-3 text-gray-400 font-medium">{s.session}</td>
                        <td className="py-2 px-3 text-gray-700 max-w-[200px] truncate" title={s.text}>{s.text}</td>
                        <td className="py-2 px-3 text-right font-mono text-gray-600">
                          {s.judge_score != null ? s.judge_score : "—"}
                        </td>
                        <td className="py-2 px-3 text-right font-mono text-gray-600">
                          {s.user_score != null ? s.user_score : "—"}
                        </td>
                        <td className="py-2 px-3 text-right text-gray-600">
                          {s.n_corrections > 0 ? (
                            <span className="text-indigo-600 font-medium">{s.n_corrections}</span>
                          ) : (
                            <span className="text-gray-400">0</span>
                          )}
                        </td>
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
              {!sim ? (
                <div className="bg-amber-50 border border-amber-200 rounded-xl p-5 text-center">
                  <p className="text-amber-700 font-medium mb-1">Simulación no ejecutada</p>
                  <p className="text-sm text-amber-600 mb-3">
                    Evalúa el sistema en múltiples rondas sobre 10 oraciones fijas comparando
                    el rendimiento con diferentes volúmenes de feedback acumulado.
                  </p>
                  {!running ? (
                    <button
                      onClick={handleRunSim}
                      className="px-4 py-2 bg-indigo-600 text-white text-sm rounded-lg hover:bg-indigo-700 transition-colors"
                    >
                      Ejecutar simulación
                    </button>
                  ) : (
                    <div className="flex items-center justify-center gap-2 text-sm text-indigo-500">
                      <span className="w-4 h-4 border-2 border-indigo-400 border-t-transparent rounded-full animate-spin" />
                      <span className="animate-pulse">Iniciando simulación…</span>
                    </div>
                  )}
                </div>
              ) : (
                <>
                  {/* Metadata + progress */}
                  <div className="flex items-center justify-between mb-4">
                    <p className="text-xs text-gray-400">
                      Última ejecución: {new Date(sim.metadata.run_at).toLocaleString("es")}
                      &ensp;·&ensp;{sim.metadata.processed ?? sim.metadata.n}/{sim.metadata.n} oraciones
                    </p>
                    <div className="flex items-center gap-2">
                      {sim.metadata.status === "running" && (
                        <>
                          <span className="w-3.5 h-3.5 border-2 border-indigo-400 border-t-transparent rounded-full animate-spin" />
                          <span className="text-xs text-indigo-500 animate-pulse">En progreso</span>
                        </>
                      )}
                      {sim.metadata.status === "complete" && (
                        <span className="text-xs px-2 py-0.5 bg-green-100 text-green-600 rounded-full">Completado</span>
                      )}
                    </div>
                  </div>

                  {/* Progress bar when running */}
                  {sim.metadata.status === "running" && sim.metadata.processed != null && (
                    <div className="mb-4 h-1.5 bg-gray-200 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-indigo-400 rounded-full transition-all duration-500"
                        style={{ width: `${(sim.metadata.processed / sim.metadata.n) * 100}%` }}
                      />
                    </div>
                  )}

                  {/* Metrics table — now with Score and Errors */}
                  <div className="overflow-x-auto rounded-xl border border-gray-200">
                    <table className="w-full text-sm border-collapse">
                      <thead>
                        <tr className="bg-gray-50 border-b border-gray-200">
                          <th className="text-left py-2.5 px-3 text-gray-500 font-semibold">Ronda</th>
                          <th className="text-right py-2.5 px-3 text-gray-500 font-semibold">Fb</th>
                          <th className="text-right py-2.5 px-3 text-gray-500 font-semibold">Score</th>
                          <th className="text-right py-2.5 px-3 text-gray-500 font-semibold">Errores</th>
                          <th className="text-right py-2.5 px-3 text-gray-500 font-semibold">Overrides</th>
                          <th className="text-right py-2.5 px-3 text-gray-500 font-semibold">Recall</th>
                          <th className="text-right py-2.5 px-3 text-gray-500 font-semibold">BLEU</th>
                          <th className="text-right py-2.5 px-3 text-gray-500 font-semibold">chrF++</th>
                        </tr>
                      </thead>
                      <tbody>
                        {sim.per_round.map((r, i) => {
                          const base = sim.per_round[0]
                          const dScore  = i > 0 ? r.avg_score  - base.avg_score  : 0
                          const dErrors = i > 0 ? r.avg_errors - base.avg_errors : 0
                          return (
                            <tr key={i} className={`border-b border-gray-100 last:border-0 ${i % 2 === 0 ? "bg-white" : "bg-gray-50/40"}`}>
                              <td className="py-2.5 px-3 text-gray-700 font-medium">{r.label}</td>
                              <td className="py-2.5 px-3 text-right text-gray-500">{r.feedback_sessions}</td>
                              <td className="py-2.5 px-3 text-right font-mono">
                                <span className="text-gray-700">{r.avg_score.toFixed(3)}</span>
                                {i > 0 && dScore !== 0 && (
                                  <span className={`ml-1 text-xs ${dScore > 0 ? "text-green-500" : "text-red-500"}`}>
                                    {dScore > 0 ? "↑" : "↓"}
                                  </span>
                                )}
                              </td>
                              <td className="py-2.5 px-3 text-right font-mono">
                                <span className="text-gray-700">{r.avg_errors.toFixed(3)}</span>
                                {i > 0 && dErrors !== 0 && (
                                  <span className={`ml-1 text-xs ${dErrors < 0 ? "text-green-500" : "text-red-500"}`}>
                                    {dErrors < 0 ? "↓" : "↑"}
                                  </span>
                                )}
                              </td>
                              <td className="py-2.5 px-3 text-right font-mono text-indigo-600">
                                {(r.avg_overrides ?? 0).toFixed(2)}
                              </td>
                              <td className="py-2.5 px-3 text-right font-mono text-gray-600">{r.avg_recall.toFixed(3)}</td>
                              <td className="py-2.5 px-3 text-right font-mono text-gray-600">{(r.avg_bleu ?? 0).toFixed(3)}</td>
                              <td className="py-2.5 px-3 text-right font-mono text-gray-600">{(r.avg_chrf ?? 0).toFixed(3)}</td>
                            </tr>
                          )
                        })}
                      </tbody>
                    </table>
                  </div>

                  {!running && (
                    <button
                      onClick={handleRunSim}
                      className="mt-4 text-sm text-indigo-500 hover:text-indigo-700 flex items-center gap-1"
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
