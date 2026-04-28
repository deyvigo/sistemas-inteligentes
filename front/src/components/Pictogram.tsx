interface PictogramProps {
  id: number
  url: string
  concept: string
  order: number
}

export const Pictogram = ({ url, concept, order }: PictogramProps) => {
  return (
    <div className="flex flex-col items-center p-3 bg-gray-50 rounded-xl hover:bg-gray-100 transition-colors">
      <div className="relative">
        <img
          src={url}
          alt={concept}
          className="w-32 h-32 object-contain rounded-lg"
          loading="lazy"
        />
        <span className="absolute -top-2 -right-2 w-7 h-7 bg-indigo-500 text-white rounded-full flex items-center justify-center text-sm font-bold">
          {order}
        </span>
      </div>
      <span className="mt-2 text-sm text-gray-600">{concept}</span>
    </div>
  )
}