import { Pictogram } from "./Pictogram"

interface SequenceItem {
  order: number
  concept: string
  id: number
  url: string
  score: number
}

export const ImageList = ({ sequence }: { sequence: SequenceItem[] }) => {
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
        <Pictogram
          key={item.id}
          id={item.id}
          url={item.url}
          concept={item.concept}
          order={item.order}
        />
      ))}
    </div>
  )
}