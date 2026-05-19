import { useState, useRef } from "react"
import { Pictogram } from "./Pictogram"

interface SequenceItem {
  order: number
  concept: string
  id: number
  url: string
  score: number
  description?: string
}

interface ImageListProps {
  sequence: SequenceItem[]
  onReorder: (newSequence: SequenceItem[]) => void
  onDelete: (id: number) => void
}

export const ImageList = ({ sequence, onReorder, onDelete }: ImageListProps) => {
  const [draggedId, setDraggedId] = useState<number | null>(null)
  const [dragOverId, setDragOverId] = useState<number | null>(null)
  const lastHoverRef = useRef<number | null>(null)

  const handleDragStart = (e: React.DragEvent, id: number) => {
    setDraggedId(id)
    e.dataTransfer.effectAllowed = "move"
    e.dataTransfer.setData("text/plain", id.toString())
  }

  const handleDragEnd = () => {
    setDraggedId(null)
    setDragOverId(null)
    lastHoverRef.current = null
  }

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault()
    e.dataTransfer.dropEffect = "move"
  }

  const handleDragEnter = (e: React.DragEvent, id: number) => {
    e.preventDefault()
    if (id !== draggedId && id !== lastHoverRef.current) {
      lastHoverRef.current = id
      setDragOverId(id)
    }
  }

  const handleDragLeave = (e: React.DragEvent) => {
    if (e.currentTarget.contains(e.relatedTarget as Node)) return
    lastHoverRef.current = null
    setDragOverId(null)
  }

  const handleDrop = (e: React.DragEvent, targetId: number) => {
    e.preventDefault()
    lastHoverRef.current = null
    setDragOverId(null)
    setDraggedId(null)
    if (draggedId === null || draggedId === targetId) return
    const newSequence = [...sequence]
    const draggedIndex = newSequence.findIndex(item => item.id === draggedId)
    const targetIndex = newSequence.findIndex(item => item.id === targetId)
    if (draggedIndex < 0 || targetIndex < 0) return
    const [draggedItem] = newSequence.splice(draggedIndex, 1)
    newSequence.splice(targetIndex, 0, draggedItem)
    const reordered = newSequence.map((item, index) => ({ ...item, order: index + 1 }))
    onReorder(reordered)
  }

  if (!sequence || sequence.length === 0) {
    return (
      <div className="text-center py-8 text-gray-400">
        No hay pictogramas para mostrar
      </div>
    )
  }

  return (
    <div className="flex flex-wrap justify-center gap-4">
      {sequence.map((item) => (
        <div
          key={item.id}
          draggable={true}
          onDragStart={(e) => handleDragStart(e, item.id)}
          onDragEnd={handleDragEnd}
          onDragOver={handleDragOver}
          onDragEnter={(e) => handleDragEnter(e, item.id)}
          onDragLeave={handleDragLeave}
          onDrop={(e) => handleDrop(e, item.id)}
          className={`relative cursor-grab active:cursor-grabbing ${
            draggedId === item.id ? "opacity-40" : ""
          } ${dragOverId === item.id ? "ring-2 ring-indigo-500 rounded-xl" : ""}`}
        >
          <Pictogram
            id={item.id}
            url={item.url}
            concept={item.concept}
            order={item.order}
            description={item.description}
            onDelete={() => onDelete(item.id)}
          />
          {dragOverId === item.id && (
            <div className="absolute inset-0 border-2 border-indigo-500 bg-indigo-50/50 rounded-xl pointer-events-none" />
          )}
        </div>
      ))}
    </div>
  )
}
