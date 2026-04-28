import { useState } from "react"
import { ImageList } from "./components/ImageList"
import { SearchBar } from "./components/SearchBar"

function App() {
  const [query, setQuery] = useState("")
  const [results, setResults] = useState<string[]>([])

  return (
    <>
      <SearchBar value={query} onChange={setQuery} onSend={setResults} />
      <div className="h-40"></div>
      <ImageList paths={results} />
    </>
  )
}

export default App
