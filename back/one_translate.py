import json
from pathlib import Path

from one_translate_cat_and_tags import translate
from pydantic import BaseModel

class TagItem(BaseModel):
  original: str
  translated: str

class CategoryItem(BaseModel):
  original: str
  translated: str

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

def get_unique_tags():
  with open("./pictogram-model/pictogramasArasaac.json") as f:
    items = json.load(f)
    all_tags = set()
    for item in items:
      tags = item.get("tags", [])
      all_tags.update(tags)
  return all_tags

def get_unique_categories():
  with open("./pictogram-model/pictogramasArasaac.json") as f:
    items = json.load(f)
    all_categories = set()
    for item in items:
      categories = item.get("categories", [])
      all_categories.update(categories)
  return all_categories

def chunk_list(lst, n):
  for i in range(0, len(lst), n):
    yield lst[i:i + n]

def translate_tags(all_tags, batch_size=10):
  outh_path = Path("./dictionaries/tags_translated.json")
  outh_path.parent.mkdir(parents=True, exist_ok=True)

  results = []
  total = len(all_tags)

  processed = {r.original for r in results}

  for i, batch in enumerate(chunk_list(all_tags, batch_size)):
    batch = [b for b in batch if b not in processed]

    if not batch:
      continue

    start = i * batch_size
    end = start + len(batch)

    print(f"[{start}-{end}] de {total} | batch {i+1} | tamaño {len(batch)}")

    try:
      translations = translate(batch)

      for o, t in zip(batch, translations):
        results.append(TagItem(original=o, translated=t))

      with open(outh_path, "w", encoding="utf-8") as f:
        json.dump(
          [item.model_dump() for item in results],
          f,
          indent=2,
          ensure_ascii=False
        )
    except Exception as e:
      print(f"Error al traducir: {batch}")
    
  with open(outh_path, "w") as f:
    json.dump([item.model_dump() for item in results], f, indent=2, ensure_ascii=False)


def translate_categories(all_categories, batch_size=10):
  outh_path = Path("./dictionaries/categories_translated.json")
  outh_path.parent.mkdir(parents=True, exist_ok=True)

  if outh_path.exists():
    with open(outh_path, "r", encoding="utf-8") as f:
      results = [CategoryItem(**item) for item in json.load(f)]

  results = []
  total = len(all_categories)

  processed = {r.original for r in results}

  for i, batch in enumerate(chunk_list(all_categories, batch_size)):
    batch = [b for b in batch if b not in processed]

    if not batch:
      continue

    start = i * batch_size
    end = start + len(batch)

    print(f"[{start}-{end}] de {total} | batch {i+1} | tamaño {len(batch)}")

    try:
      translated = translate(batch)

      for o, t in zip(batch, translated):
        results.append(CategoryItem(original=o, translated=t))

      with open(outh_path, "w", encoding="utf-8") as f:
        json.dump(
          [item.model_dump() for item in results],
          f,
          indent=2,
          ensure_ascii=False
        )
    except Exception as e:
      print(f"Error al traducir: {batch}")
      
  with open(outh_path, "w") as f:
    json.dump([item.model_dump() for item in results], f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
  all_tags = list(get_unique_tags())
  all_categories = list(get_unique_categories())

  print(f"Total de tags: {len(all_tags)}")
  print("Empezando traducción de tags...")
  translate_tags(all_tags, 50)
  print(f"Total de categories: {len(all_categories)}")
  print("Empezando traducción de categories...")
  translate_categories(all_categories, 50)
