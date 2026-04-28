import { Pictogram } from "./Pictogram"

export const ImageList = ({ paths }: { paths: string[]}) => {
  return (
    <div
      className="flex flex-row justify-center"
    >
      {paths.map((path, i) => (
        <Pictogram key={i} path={path} />
      ))}
    </div>
  )
}