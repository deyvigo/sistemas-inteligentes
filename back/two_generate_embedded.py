import json
from sentence_transformers import SentenceTransformer
import numpy as np

""""
{
  "categories": string[],
  "tags": string[],
  "_id": number
  "keywords": [
    {
      "keyword": string,
      "meaning"?: string,
      "plural": string
    }, ...
  ]

}
"""

model = SentenceTransformer("intfloat/multilingual-e5-small")

def embed_texts(texts):
  return model.encode(texts, normalize_embeddings=True)

def load_categories_and_tags():
  with open("./dictionaries/categories_translated.json") as f:
    categories = json.load(f)
  with open("./dictionaries/tags_translated.json") as f:
    tags = json.load(f)

  categories = {item["original"]: item["translated"] for item in categories}
  tags = {item["original"]: item["translated"] for item in tags}

  return categories, tags

def build_text(item, categories_translated, tags_translated):
  id = item.get("_id", 0)

  keywords = item.get("keywords", [])
  main_keyword = keywords[0].get("keyword") if keywords else "unknown"
  
  description = keywords[0].get("meaning", "") if keywords else ""
  synonyms = [k.get("keyword") for k in keywords[1:]] if len(keywords) > 1 else []
  categories = [categories_translated[c] for c in item.get("categories", [])]
  tags = [tags_translated[t] for t in item.get("tags", [])]

  text = f"""
  query: Concepto: {main_keyword}
  Descripción: {description}
  Sinónimos: {', '.join(synonyms)}
  Categorías: {', '.join(categories)}
  Contexto: {', '.join(tags)}
  """

  return id, text.strip()

if __name__ == "__main__":
  with open("./pictogram-model/pictogramasArasaac.json") as f:
    items = json.load(f)

  categories_translated, tags_translated = load_categories_and_tags()

  ids = []
  texts = []

  for item in items:
    item_id, text = build_text(item, categories_translated, tags_translated)
    ids.append(item_id)
    texts.append(text)

  embeddings = embed_texts(texts)

  import os
  os.makedirs("./embeddings", exist_ok=True)

  np.save("./embeddings/embeddings.npy", embeddings)
  np.save("./embeddings/ids.npy", ids)
  np.save("./embeddings/texts.npy", texts)