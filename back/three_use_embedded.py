import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("intfloat/multilingual-e5-small")

embeddings = np.load("./embeddings/embeddings.npy")
ids = np.load("./embeddings/ids.npy")
texts = np.load("./embeddings/texts.npy")

def search(query, top_k=5):
    query_emb = model.encode([query], normalize_embeddings=True)[0]

    scores = np.dot(embeddings, query_emb)

    top_indices = np.argsort(scores)[-top_k:][::-1]

    results = []
    for i in top_indices:
        results.append({
            "id": ids[i],
            "text": texts[i],
            "score": scores[i]
        })

    return results

def search_sequence(concepts, top_k=3):
    sequence_results = []
    seen_ids = set()

    for concept in concepts:
        concept_results = search(concept, top_k)
        for result in concept_results:
            if result["id"] not in seen_ids:
                seen_ids.add(result["id"])
                sequence_results.append({
                    "concept": concept,
                    "id": result["id"],
                    "text": result["text"],
                    "score": result["score"]
                })
                break

    return sequence_results