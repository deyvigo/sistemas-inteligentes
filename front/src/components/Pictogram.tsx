interface PictogramProps {
  id: number
  url: string
  concept: string
  order: number
  onDelete: () => void
}

export const Pictogram = ({ url, concept, order, onDelete }: PictogramProps) => {
  return (
    <div className="flex flex-col items-center p-3 bg-gray-50 rounded-xl hover:bg-gray-100 transition-colors">
      <div className="relative group">
        <img
          src={url}
          alt={concept}
          className="w-32 h-32 object-contain rounded-lg"
          loading="lazy"
        />
        <span className="absolute -top-2 -right-2 w-7 h-7 bg-indigo-500 text-white rounded-full flex items-center justify-center text-sm font-bold">
          {order}
        </span>
        {/* Delete button */}
        <button
          onClick={onDelete}
          className="absolute top-0 right-0 -mt-2 -mr-2 w-6 h-6 bg-red-500 text-white rounded-full flex items-center justify-center text-xs hover:bg-red-600 transition-colors group-hover:bg-red-600"
          aria-label="Eliminar pictograma"
        >
          ×
        </button>
      </div>
      <span className="mt-2 text-sm text-gray-600">{concept}</span>
    </div>
  )
}