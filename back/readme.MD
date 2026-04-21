Paso 1: Retrieval (RAG)

Tu JSON (tags + descripciones) → embeddings
Buscar los pictogramas más relevantes

Paso 2: Generación

Gemini recibe:
input del usuario
contexto recuperado
Devuelve: lista de IDs + justificación

Paso 3: Evaluación (LLM Judge)

Otro prompt que evalúe:
relevancia
coherencia
faltantes

Paso 4: Feedback loop

Guardas:
errores
sugerencias
Ajustas prompt o reglas (NO solo “mejorar prompt”, también lógica)