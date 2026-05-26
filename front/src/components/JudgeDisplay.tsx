import { useState } from "react"

interface JudgeResult {
  score: number
  missing_concepts: string[]
  incorrect_pictograms: { concept: string; reason: string }[]
  ordering_issues: string[]
  suggestions: string[]
  error?: boolean
  error_type?: string
  message?: string
  details?: string
}

interface JudgeDisplayProps {
  judge: JudgeResult
}

const scoreColors: Record<number, string> = {
  1: "bg-red-100 text-red-700 border-red-300",
  2: "bg-orange-100 text-orange-700 border-orange-300",
  3: "bg-yellow-100 text-yellow-700 border-yellow-300",
  4: "bg-lime-100 text-lime-700 border-lime-300",
  5: "bg-green-100 text-green-700 border-green-300",
}

const scoreLabels: Record<number, string> = {
  1: "Muy malo",
  2: "Malo",
  3: "Regular",
  4: "Bueno",
  5: "Excelente",
}

const scoreBarColors: Record<number, string> = {
  1: "bg-red-400",
  2: "bg-orange-400",
  3: "bg-yellow-400",
  4: "bg-lime-400",
  5: "bg-green-500",
}

export const JudgeDisplay = ({ judge }: JudgeDisplayProps) => {
  const [showJson, setShowJson] = useState(false)
  const hasError = Boolean(judge.error)

  return (
    <div>
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-3 flex-wrap">
          <span className={`px-3 py-1.5 rounded-full text-sm font-bold border ${
            hasError ? "bg-gray-100 text-gray-600 border-gray-300" : scoreColors[judge.score]
          }`}>
            {hasError ? "Sin evaluación" : `${judge.score}/5 – ${scoreLabels[judge.score]}`}
          </span>
          {!hasError && (
            <div className="h-2 w-28 bg-gray-100 rounded-full overflow-hidden hidden sm:block">
              <div
                className={`h-full rounded-full transition-all ${scoreBarColors[judge.score]}`}
                style={{ width: `${(judge.score / 5) * 100}%` }}
              />
            </div>
          )}
        </div>
        <button
          onClick={() => setShowJson(!showJson)}
          className="px-2.5 py-1 text-xs bg-gray-100 hover:bg-gray-200 text-gray-500 rounded-lg transition-colors shrink-0"
        >
          {showJson ? "← detalles" : "ver JSON"}
        </button>
      </div>

      {showJson ? (
        <pre className="bg-gray-800 text-green-400 p-4 rounded-xl overflow-x-auto text-sm font-mono">
          {JSON.stringify(judge, null, 2)}
        </pre>
      ) : (
        <div className="space-y-4">
          {hasError && (
            <div className="bg-amber-50 border border-amber-200 rounded-xl p-4">
              <h3 className="font-medium text-amber-700 mb-1">Evaluación no disponible</h3>
              <p className="text-sm text-amber-700">
                {judge.message || "El LLM-Judge no pudo evaluar la secuencia por un error técnico."}
              </p>
            </div>
          )}

          {!hasError && judge.missing_concepts.length > 0 && (
            <div>
              <h3 className="text-sm font-semibold text-red-600 mb-2">Conceptos faltantes</h3>
              <div className="flex flex-wrap gap-2">
                {judge.missing_concepts.map((concept, i) => (
                  <span key={i} className="px-3 py-1 bg-red-50 text-red-600 rounded-full text-sm border border-red-200">
                    {concept}
                  </span>
                ))}
              </div>
            </div>
          )}

          {!hasError && judge.incorrect_pictograms.length > 0 && (
            <div>
              <h3 className="text-sm font-semibold text-orange-600 mb-2">Pictogramas incorrectos</h3>
              <ul className="space-y-1">
                {judge.incorrect_pictograms.map((item, i) => (
                  <li key={i} className="text-sm text-gray-600 flex items-start gap-1.5">
                    <span className="text-orange-400 mt-0.5 shrink-0">•</span>
                    <span><span className="font-medium">{item.concept}</span>: {item.reason}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {!hasError && judge.ordering_issues.length > 0 && (
            <div>
              <h3 className="text-sm font-semibold text-amber-600 mb-2">Problemas de orden</h3>
              <ul className="space-y-1">
                {judge.ordering_issues.map((issue, i) => (
                  <li key={i} className="text-sm text-gray-600 flex items-start gap-1.5">
                    <span className="text-amber-400 mt-0.5 shrink-0">•</span>
                    {issue}
                  </li>
                ))}
              </ul>
            </div>
          )}

          {!hasError && judge.suggestions.length > 0 && (
            <div>
              <h3 className="text-sm font-semibold text-indigo-600 mb-2">Sugerencias</h3>
              <ul className="space-y-1">
                {judge.suggestions.map((sug, i) => (
                  <li key={i} className="text-sm text-gray-600 flex items-start gap-1.5">
                    <span className="text-indigo-400 mt-0.5 shrink-0">•</span>
                    {sug}
                  </li>
                ))}
              </ul>
            </div>
          )}

          {!hasError &&
           judge.missing_concepts.length === 0 &&
           judge.incorrect_pictograms.length === 0 &&
           judge.ordering_issues.length === 0 &&
           judge.suggestions.length === 0 && (
            <div className="bg-green-50 border border-green-200 rounded-xl p-4 text-center">
              <p className="text-green-700 font-medium">¡Secuencia perfecta!</p>
              <p className="text-sm text-green-600 mt-1">No hay mejoras necesarias.</p>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
