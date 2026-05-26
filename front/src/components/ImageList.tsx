import { useState, useRef } from "react"
import { Pictogram } from "./Pictogram"

interface SequenceItem {
  uid: string
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
  onDelete: (uid: string) => void
}

export const ImageList = ({ sequence, onReorder, onDelete }: ImageListProps) => {
  const [draggedUid, setDraggedUid] = useState<string | null>(null)
  const [dragOverUid, setDragOverUid] = useState<string | null>(null)
  const lastHoverRef = useRef<string | null>(null)

  const renumber = (items: SequenceItem[]) =>
    items.map((item, index) => ({ ...item, order: index + 1 }))

  const moveItem = (uid: string, direction: -1 | 1) => {
    const currentIndex = sequence.findIndex(item => item.uid === uid)
    const targetIndex = currentIndex + direction
    if (currentIndex < 0 || targetIndex < 0 || targetIndex >= sequence.length) return
    const next = [...sequence]
    const [item] = next.splice(currentIndex, 1)
    next.splice(targetIndex, 0, item)
    onReorder(renumber(next))
  }

  const handleDragStart = (e: React.DragEvent, uid: string) => {
    setDraggedUid(uid)
    e.dataTransfer.effectAllowed = "move"
    e.dataTransfer.setData("text/plain", uid)
  }

  const handleDragEnd = () => {
    setDraggedUid(null)
    setDragOverUid(null)
    lastHoverRef.current = null
  }

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault()
    e.dataTransfer.dropEffect = "move"
  }

  const handleDragEnter = (e: React.DragEvent, uid: string) => {
    e.preventDefault()
    if (uid !== draggedUid && uid !== lastHoverRef.current) {
      lastHoverRef.current = uid
      setDragOverUid(uid)
    }
  }

  const handleDragLeave = (e: React.DragEvent) => {
    if (e.currentTarget.contains(e.relatedTarget as Node)) return
    lastHoverRef.current = null
    setDragOverUid(null)
  }

  const handleDrop = (e: React.DragEvent, targetUid: string) => {
    e.preventDefault()
    lastHoverRef.current = null
    setDragOverUid(null)
    setDraggedUid(null)
    if (draggedUid === null || draggedUid === targetUid) return
    const newSequence = [...sequence]
    const draggedIndex = newSequence.findIndex(item => item.uid === draggedUid)
    const targetIndex = newSequence.findIndex(item => item.uid === targetUid)
    if (draggedIndex < 0 || targetIndex < 0) return
    const [draggedItem] = newSequence.splice(draggedIndex, 1)
    newSequence.splice(targetIndex, 0, draggedItem)
    onReorder(renumber(newSequence))
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
      {sequence.map((item, index) => (
        <div
          key={item.uid}
          draggable={true}
          onDragStart={(e) => handleDragStart(e, item.uid)}
          onDragEnd={handleDragEnd}
          onDragOver={handleDragOver}
          onDragEnter={(e) => handleDragEnter(e, item.uid)}
          onDragLeave={handleDragLeave}
          onDrop={(e) => handleDrop(e, item.uid)}
          className={`relative cursor-grab active:cursor-grabbing ${
            draggedUid === item.uid ? "opacity-40" : ""
          } ${dragOverUid === item.uid ? "ring-2 ring-indigo-500 rounded-xl" : ""}`}
        >
          <Pictogram
            id={item.id}
            url={item.url}
            concept={item.concept}
            order={item.order}
            description={item.description}
            canMoveLeft={index > 0}
            canMoveRight={index < sequence.length - 1}
            onMoveLeft={() => moveItem(item.uid, -1)}
            onMoveRight={() => moveItem(item.uid, 1)}
            onDelete={() => onDelete(item.uid)}
          />
          {dragOverUid === item.uid && (
            <div className="absolute inset-0 border-2 border-indigo-500 bg-indigo-50/50 rounded-xl pointer-events-none" />
          )}
        </div>
      ))}
    </div>
  )
}
