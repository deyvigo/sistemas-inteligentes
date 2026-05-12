interface ConceptTagProps {
  concept: string
  type: "noun" | "negation" | "time"
}

export const ConceptTag = ({ concept, type }: ConceptTagProps) => {
  const baseClasses = "px-3 py-1 rounded-full text-sm font-medium"

  const typeClasses = {
    noun: "bg-indigo-100 text-indigo-700",
    negation: "bg-red-100 text-red-700",
    time: "bg-amber-100 text-amber-700"
  }

  return (
    <span className={`${baseClasses} ${typeClasses[type]}`}>
      {concept}
    </span>
  )
}

interface ConceptsDisplayProps {
  concepts: string[]
}

export const ConceptsDisplay = ({ concepts }: ConceptsDisplayProps) => {
  return (
    <div className="flex flex-wrap gap-2">
      {concepts.map((concept) => (
        <ConceptTag key={concept} concept={concept} type="noun" />
      ))}
    </div>
  )
}