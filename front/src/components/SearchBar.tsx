import { useState } from "react"
import { API_URL } from "../api/url"

interface SearchBarProps {
  value: string
  onChange: (query: string) => void
  onSend: (ids: string[]) => void
}

interface QueryResponse {
  paths: string[]
}

export const SearchBar = ({ value, onChange, onSend }: SearchBarProps) => {
  const [loading, setLoading] = useState(false)

  async function handleSubmit() {
    if (!value) return
    if (loading) return

    setLoading(true)
    try {
      const res = await fetch(`${API_URL}/query`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json"
          },
          body: JSON.stringify({ query: value })
        }
      )
      const data: QueryResponse = await res.json()
      console.log(data)
      onSend(data.paths)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="flex gap-2">
      <input
        onChange={(e) => onChange(e.target.value)}
        className="outline-none border border-gray-300 rounded-lg indent-2 p-2 w-250"
        type="text"
        placeholder="Frase a buscar"
        value={value}
      />
      <button
        className="p-2 bg-red-200 rounded-lg text-white"
        onClick={handleSubmit}
      >
        Buscar
      </button>
    </div>
  )
}