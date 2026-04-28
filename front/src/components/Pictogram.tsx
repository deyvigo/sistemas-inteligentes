export const Pictogram = ({ path }: { path: string }) => {
  return (
    <div>
      <img
        src={path} alt="pictogram"
        className="w-40 h-40"
      />
    </div>
  )
}