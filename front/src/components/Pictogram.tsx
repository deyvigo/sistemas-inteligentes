interface PictogramProps {
  id: number
  url: string
  concept: string
  order: number
  onDelete: () => void
  description?: string
}

export const Pictogram = ({ url, concept, order, onDelete, description }: PictogramProps) => {
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
        {/* Description tooltip icon */}
        {description && (
          <div className="absolute top-0 left-0 -mt-2 -ml-2 group/desc">
            <div className="w-6 h-6 bg-blue-500 text-white rounded-full flex items-center justify-center text-xs cursor-help hover:bg-blue-600 transition-colors">
              ℹ
            </div>
            {/* Tooltip on hover */}
            <div className="absolute left-1/2 -translate-x-1/2 bottom-full mb-2 w-64 p-2 bg-gray-800 text-white text-xs rounded-lg shadow-lg opacity-0 invisible group-hover/desc:opacity-100 group-hover/desc:visible transition-all duration-200 z-10">
              <p className="font-semibold mb-1">Descripción:</p>
              <p className="whitespace-pre-wrap">{description}</p>
              <div className="absolute left-1/2 -translate-x-1/2 top-full w-0 h-0 border-l-4 border-r-4 border-t-4 border-transparent border-t-gray-800"></div>
            </div>
          </div>
        )}
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
