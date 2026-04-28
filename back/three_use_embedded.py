import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("intfloat/multilingual-e5-small")

embeddings = np.load("./embeddings/embeddings.npy")
ids = np.load("./embeddings/ids.npy")
texts = np.load("./embeddings/texts.npy")

def search(query, top_k=5, offset=0):
    query_emb = model.encode([query], normalize_embeddings=True)[0]

    scores = np.dot(embeddings, query_emb)

    # Get more results than needed to handle offset properly
    # We need at least offset + top_k results to slice correctly
    num_results_needed = offset + top_k
    # But don't exceed total available results
    num_results_needed = min(num_results_needed, len(scores))
    
    # Get indices of top scores, sorted descending
    top_indices = np.argsort(scores)[-num_results_needed:][::-1]

    results = []
    for i in top_indices:
        results.append({
            "id": ids[i],
            "text": texts[i],
            "score": scores[i]
        })
    
    # Apply offset (skip first 'offset' results)
    return results[offset:offset+top_k]

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