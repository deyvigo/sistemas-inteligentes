import { useState } from "react"

interface JudgeResult {
  score: number
  missing_concepts: string[]
  incorrect_pictograms: { concept: string; reason: string }[]
  ordering_issues: string[]
  suggestions: string[]
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

export const JudgeDisplay = ({ judge }: JudgeDisplayProps) => {
  const [showJson, setShowJson] = useState(false)

  return (
    <div className="bg-white rounded-2xl p-6 shadow-xl">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg font-semibold text-gray-700">Evaluación del LLM</h2>
        <div className="flex items-center gap-3">
          <span className={`px-4 py-2 rounded-full font-bold border-2 ${scoreColors[judge.score]}`}>
            {judge.score}/5 - {scoreLabels[judge.score]}
          </span>
          <button
            onClick={() => setShowJson(!showJson)}
            className="px-3 py-1 text-sm bg-gray-100 hover:bg-gray-200 text-gray-600 rounded-lg transition-colors"
          >
            {showJson ? "Ver detalles" : "Ver JSON"}
          </button>
        </div>
      </div>

      {showJson ? (
        <pre className="bg-gray-800 text-green-400 p-4 rounded-lg overflow-x-auto text-sm font-mono">
          {JSON.stringify(judge, null, 2)}
        </pre>
      ) : (
        <div className="space-y-4">
          {judge.missing_concepts.length > 0 && (
            <div>
              <h3 className="font-medium text-red-600 mb-2">Conceptos faltantes:</h3>
              <div className="flex flex-wrap gap-2">
                {judge.missing_concepts.map((concept, i) => (
                  <span key={i} className="px-3 py-1 bg-red-50 text-red-600 rounded-full text-sm">
                    {concept}
                  </span>
                ))}
              </div>
            </div>
          )}

          {judge.incorrect_pictograms.length > 0 && (
            <div>
              <h3 className="font-medium text-orange-600 mb-2">Pictogramas incorrectos:</h3>
              <ul className="list-disc list-inside text-sm text-gray-600">
                {judge.incorrect_pictograms.map((item, i) => (
                  <li key={i}>
                    <span className="font-medium">{item.concept}</span>: {item.reason}
                  </li>
                ))}
              </ul>
            </div>
          )}

          {judge.ordering_issues.length > 0 && (
            <div>
              <h3 className="font-medium text-amber-600 mb-2">Problemas de orden:</h3>
              <ul className="list-disc list-inside text-sm text-gray-600">
                {judge.ordering_issues.map((issue, i) => (
                  <li key={i}>{issue}</li>
                ))}
              </ul>
            </div>
          )}

          {judge.suggestions.length > 0 && (
            <div>
              <h3 className="font-medium text-indigo-600 mb-2">Sugerencias:</h3>
              <ul className="list-disc list-inside text-sm text-gray-600">
                {judge.suggestions.map((sug, i) => (
                  <li key={i}>{sug}</li>
                ))}
              </ul>
            </div>
          )}

          {judge.missing_concepts.length === 0 &&
           judge.incorrect_pictograms.length === 0 &&
           judge.ordering_issues.length === 0 &&
           judge.suggestions.length === 0 && (
            <p className="text-gray-500 italic">La secuencia es perfecta, ¡no hay mejoras necessárias!</p>
          )}
        </div>
      )}
    </div>
  )
}