import json
import os
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).parent
OUTPUT_DIR = PROJECT_ROOT / "llm-generator-outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

try:
    from google import genai
    from google.genai import types
except ImportError:
    import subprocess
    subprocess.run(["python", "-m", "pip", "install", "google-genai"])
    from google import genai
    from google.genai import types

# Load embeddings to lookup full text by ID
try:
    _ids = np.load("./embeddings/ids.npy")
    _texts = np.load("./embeddings/texts.npy")
    
    # Build a mapping from ID to text for quick lookup
    _id_to_text = {}
    for idx, pid in enumerate(_ids):
        _id_to_text[int(pid)] = _texts[idx]
except Exception as e:
    print(f"[WARNING] Could not load embeddings for text lookup: {e}")
    _id_to_text = {}

def get_text_by_id(pictogram_id: int) -> str:
    """Get the full ARASAAC text description for a pictogram ID"""
    return _id_to_text.get(int(pictogram_id), "")

MODEL_NAME = "gemini-3.1-flash-lite-preview"

SYSTEM_PROMPT = """Eres un experto en selecionar pictogramas ARASAAC para AAC.

Tu tarea es seleccionar el mejor pictograma para cada concepto de una lista de candidatos.

INSTRUCCIONES:
- Para cada concepto, selecciona UN SOLO pictograma de la lista de candidatos
- Considera la oración completa para elegir el mejor pictograma
- Responde solo con formato JSON, no con texto adicional
- Usa este formato exacto:

{"selections": [{"query_concept": "corriendo", "selected_id": 123, "selected_concept": "Correr", "reason": "mejor coincidencia"}], "sequence": [{"concept": "Correr", "id": 123, "url": "https://static.arasaac.org/pictograms/123/123_500.png", "score": 0.89}]}

IMPORTANT: El campo de "concept" en "sequence" debe ser el concepto ARASAAC real (por ejemplo, "Correr"), NO el concepto de la consulta (por ejemplo, "corriendo").

No markdown, no explicaciones, solo JSON."""

def build_generator_prompt(text: str, concepts: list, candidates: list) -> str:
    desc = []
    for item in candidates:
        query_concept = item["concept"]  # What user searched (e.g., "tomando")
        cands = item["candidates"]
        desc.append(f"\nQuery concept: {query_concept}")
        desc.append("Candidates (with ARASAAC pictogram concepts and descriptions):")
        for i, cand in enumerate(cands, 1):
            # cand["concept"] is the ARASAAC concept (e.g., "Comer")
            # cand["description"] is the FULL text with description
            cand_desc = cand.get("description", "")
            desc.append(f"  {i}. ID: {cand['id']}, Pictogram concept: {cand['concept']}, Score: {cand['score']:.2f}")
            if cand_desc:
                # Show first 150 chars of description for context
                desc.append(f"     Description: {cand_desc[:150]}...")
    
    candidates_str = "\n".join(desc)
    
    return f"""Oracion original: "{text}"
Conceptos extraidos: {concepts}

Pictogramas candidatos por concepto:{candidates_str}

Selecciona el mejor pictograma para cada concepto de la lista de candidatos considerando la oración completa.
IMPORTANTE: Lee cada candidato cuidadosamente - el mismo concepto (por ejemplo, "saco") puede referirse a cosas diferentes.
Retorna el concepto del pictograma ARASAAC en el campo "concept" de la respuesta.
"""

def parse_response(text: str) -> dict:
    text = text.strip()
    
    # Remove markdown code blocks if present
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    
    text = text.strip()
    
    # Try to extract JSON by finding first { and last }
    try:
        start = text.index('{')
        end = text.rindex('}') + 1
        json_str = text[start:end]
        return json.loads(json_str)
    except:
        print(f"[ERROR] Failed to parse LLM response: {text[:200]}")
        return {
            "selections": [],
            "sequence": [],
            "error": "Failed to parse LLM response"
        }

def save_result(text: str, concepts: list, candidates: list, result: dict) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"generator_{timestamp}.json"
    filepath = OUTPUT_DIR / filename
    
    output = {
        "timestamp": timestamp,
        "text": text,
        "concepts": concepts,
        "candidates": candidates,
        "result": result
    }
    
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    return filepath

def generate_sequence(text: str, concepts: list, candidates: list, api_key: Optional[str] = None) -> dict:
    prompt = build_generator_prompt(text, concepts, candidates)
    
    try:
        if api_key:
            client = genai.Client(api_key=api_key)
        else:
            client = genai.Client()
        
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT
            )
        )
        
        print(f"[DEBUG] LLM raw response: {response.text[:300]}")
        
        result = parse_response(response.text)
        
        # Build sequence with URLs - ensure we use pictogram concept, not query concept
        sequence = []
        if "sequence" in result and result["sequence"]:
            for item in result["sequence"]:
                # Use "concept" from LLM (should be pictogram concept like "Correr")
                # Fallback to "selected_concept" if present
                pictogram_concept = item.get("concept") or item.get("selected_concept", "Unknown")
                pictogram_id = int(item["id"])
                
                # Get FULL ARASAAC text description for Judge evaluation
                pictogram_text = get_text_by_id(pictogram_id)
                
                sequence.append({
                    "concept": pictogram_concept,  # Actual ARASAAC pictogram concept (for display)
                    "id": pictogram_id,
                    "url": f"https://static.arasaac.org/pictograms/{pictogram_id}/{pictogram_id}_500.png",
                    "score": item.get("score", 0.0),
                    "description": pictogram_text  # FULL text for Judge to evaluate correctly
                })
                print(f"[DEBUG] LLM Generator sequence item: concept={pictogram_concept}, id={pictogram_id}")
                if pictogram_text:
                    print(f"[DEBUG] Full text for Judge: '{pictogram_text[:100]}...'")
        else:
            # Fallback: build from selections
            # Map by query_concept (what user searched) to get the selection
            selections_map = {}
            if "selections" in result:
                for sel in result["selections"]:
                    # Use query_concept as key (the user's word like "corriendo")
                    key = sel.get("query_concept") or sel.get("concept", "")
                    selections_map[key] = sel
            
            for concept in concepts:
                if concept in selections_map:
                    sel = selections_map[concept]
                    pictogram_id = int(sel["selected_id"])
                    
                    # Get FULL ARASAAC text description
                    pictogram_text = get_text_by_id(pictogram_id)
                    
                    sequence.append({
                        "concept": sel["selected_concept"],  # ARASAAC concept like "Correr"
                        "id": pictogram_id,
                        "url": f"https://static.arasaac.org/pictograms/{pictogram_id}/{pictogram_id}_500.png",
                        "score": 0.0,
                        "description": pictogram_text  # FULL text for Judge
                    })
                    print(f"[DEBUG] LLM Generator fallback: query={concept}, selected_concept={sel['selected_concept']}, id={pictogram_id}")
        
        save_result(text, concepts, candidates, result)
        
        return {"sequence": sequence, "selections": result.get("selections", [])}
        
    except Exception as e:
        print(f"[ERROR] LLM Generator failed: {e}")
        return {
            "sequence": [],
            "selections": [],
            "error": str(e)
        }

if __name__ == "__main__":
    test_concepts = ["nino", "corriendo", "parque"]
    test_candidates = [
        {
            "concept": "nino",
            "candidates": [
                {"id": 1001, "concept": "Nino", "score": 0.92},
                {"id": 1002, "concept": "Nina", "score": 0.85},
            ]
        },
        {
            "concept": "corriendo",
            "candidates": [
                {"id": 2001, "concept": "Correr", "score": 0.89},
                {"id": 2002, "concept": "Carrera", "score": 0.82},
            ]
        }
    ]
    
    result = generate_sequence("El nino esta corriendo", test_concepts, test_candidates)
    print(json.dumps(result, indent=2, ensure_ascii=False))
