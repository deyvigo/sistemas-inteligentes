import { useState } from "react"
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

  const handleDragStart = (id: number) => {
    setDraggedId(id)
  }

  const handleDragOver = (_e: React.DragEvent) => {
    _e.preventDefault()
  }

  const handleDrop = (e: React.DragEvent, targetId: number) => {
    e.preventDefault()
    if (draggedId !== null && draggedId !== targetId) {
      // Create new sequence with reordered items
      const newSequence = [...sequence]
      const draggedIndex = newSequence.findIndex(item => item.id === draggedId)
      const targetIndex = newSequence.findIndex(item => item.id === targetId)
      
      if (draggedIndex >= 0 && targetIndex >= 0) {
        const [draggedItem] = newSequence.splice(draggedIndex, 1)
        newSequence.splice(targetIndex, 0, draggedItem)
        
        // Update order values
        newSequence.forEach((item, index) => {
          item.order = index + 1
        })
        
        onReorder(newSequence)
      }
    }
    setDraggedId(null)
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
          onDragStart={(_e) => handleDragStart(item.id)}
          onDragOver={(_e) => handleDragOver(_e)}
          onDrop={(_e) => handleDrop(_e, item.id)}
          className="relative"
        >
          <Pictogram
            id={item.id}
            url={item.url}
            concept={item.concept}
            order={item.order}
            onDelete={() => onDelete(item.id)}
          />
          
          {/* Visual feedback for dragged item */}
          {draggedId === item.id && (
            <div className="absolute inset-0 border-2 border-dashed border-blue-500 animate-pulse" />
          )}
        </div>
      ))}
    </div>
  )
}