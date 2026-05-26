import json
import os
import re
import time
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

MODEL_NAME = "gemini-3.1-flash-lite"

MAX_RETRIES = 3

def _generate_with_retry(client, contents, config):
    for attempt in range(MAX_RETRIES + 1):
        try:
            return client.models.generate_content(
                model=MODEL_NAME,
                contents=contents,
                config=config
            )
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                if attempt == MAX_RETRIES:
                    raise
                match = re.search(r"retryDelay.*?(\d+)s", error_str)
                delay = int(match.group(1)) if match else (2 ** attempt) * 10
                print(f"[RATE LIMIT] Reintentando en {delay}s (intento {attempt + 1}/{MAX_RETRIES})...")
                time.sleep(delay)
            else:
                raise

SYSTEM_PROMPT = """Eres un experto en seleccionar pictogramas ARASAAC para AAC (Comunicación Aumentativa y Alternativa).

Tu tarea es crear una secuencia de pictogramas que represente fielmente el SIGNIFICADO de una frase en español.

INSTRUCCIONES GENERALES:
- Revisa TODOS los pictogramas candidatos disponibles y sus descripciones antes de elegir
- Selecciona los que mejor representen la FRASE COMPLETA, no cada palabra individualmente
- NO es obligatorio elegir un pictograma por cada concepto extraído
- Un MISMO pictograma puede cubrir VARIOS conceptos (ej: "niño comiendo pan" → 2 pictogramas en vez de 3)
- El ORDEN de los IDs debe reflejar el orden lógico de la comunicación AAC
- Compara la descripción del pictograma candidato con el concepto antes de seleccionarlo

MANEJO DE VERBOS EN AAC (MUY IMPORTANTE):
- VERBOS COPULATIVOS (ser, estar): Suelen OMITIRSE en AAC. Representa solo el estado o cualidad.
  Ejemplo: "Estoy cansado" → busca solo "cansado". "Soy feliz" → busca "feliz".
- VERBOS DE ESTADO FÍSICO con tener/sentir (tener frío, tener calor, tener hambre, tener sed, tener miedo, sentir dolor):
  Usa el ESTADO COMPLETO como concepto unificado. Busca "frío", "calor", "hambre", "sed", "miedo", "dolor" directamente.
  NO uses por separado "tener" + "frío". Busca el pictograma del estado.
- VERBO TENER (posesión simple): Si el objeto ya está claro, prioriza el objeto.
  Ejemplo: "tengo una pelota" → usa "pelota" (o "tener" + "pelota" si hay candidato explícito).
- VERBOS DE ACCIÓN (correr, comer, beber, jugar, saltar, caminar, dormir, etc.):
  Busca el pictograma que muestre la ACCIÓN directamente. Verifica que la descripción del candidato
  corresponda a la acción y no a un concepto relacionado diferente (ej: "corriendo" vs "carrera deportiva").
- VERBOS MODALES (querer, poder, deber, necesitar): Busca el pictograma de la acción principal que los acompaña.
  Ejemplo: "quiero comer" → prioriza "querer" + "comer" o solo "comer" según el contexto.

SELECCIÓN PRECISA DE PICTOGRAMAS:
- Si un candidato tiene descripción, léela para confirmar que coincide con el concepto de la frase.
- Si hay varios candidatos para un concepto, elige el que la descripción indique mayor similitud semántica.
- Evita pictogramas cuya descripción sea solo parcialmente similar al concepto buscado.

- Respuesta únicamente JSON, sin texto adicional
- Formato exacto:

{"selected_ids": [456, 123, 789]}

No incluyas conceptos, URLs, scores ni razones. Solo los IDs en el orden que consideres correcto. No markdown, no explicaciones, solo JSON."""

def build_generator_prompt(text: str, concepts: list, candidates: list, feedback_hints: dict = None) -> str:
    desc = []
    for item in candidates:
        query_concept = item["concept"]
        cands = item["candidates"]
        desc.append(f"\nQuery concept: {query_concept}")
        desc.append("Candidates (with ARASAAC pictogram concepts and descriptions):")
        for i, cand in enumerate(cands, 1):
            cand_desc = cand.get("description", "")
            desc.append(f"  {i}. ID: {cand['id']}, Pictogram concept: {cand['concept']}")
            if cand_desc:
                desc.append(f"     Description: {cand_desc[:60]}...")
    
    candidates_str = "\n".join(desc)
    
    # Add feedback hints if available (Strategy B: rule improvement hints)
    hints_str = ""
    if feedback_hints:
        hints_parts = []
        for concept, hint in feedback_hints.items():
            overrides = hint.get('overrides', {})
            if overrides:
                ids_info = ", ".join(
                    [f"ID {pid} (usado {count} veces)" for pid, count in overrides.items()]
                )
                hints_parts.append(
                    f"- Para '{concept}', usuarios anteriores prefirieron: {ids_info}."
                )
            missing_count = hint.get('missing_count', 0)
            if missing_count >= 2:
                hints_parts.append(
                    f"- El concepto '{concept}' se ha detectado como AUSENTE en generaciones anteriores ({missing_count} veces). "
                    "Considera incluirlo si el contexto lo requiere."
                )
        if hints_parts:
            hints_str = "\n\nFEEDBACK DE USUARIOS ANTERIORES:\n" + "\n".join(hints_parts)
            hints_str += "\nUsa esta información como referencia, pero aplica tu criterio según el contexto de la oración."
    
    return f"""Oracion original: "{text}"
Conceptos extraidos: {concepts}

Pictogramas candidatos disponibles:{candidates_str}{hints_str}

Selecciona los pictogramas que mejor representen la ORACION COMPLETA.
- No es obligatorio elegir uno por cada concepto
- Un pictograma puede cubrir varios conceptos
- El orden debe reflejar el orden logico de la idea
- Responde SOLO con los IDs en el orden que consideres correcto
"""

def parse_response(text: str) -> dict:
    text = text.strip()
    
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    
    text = text.strip()
    
    try:
        start = text.index('{')
        end = text.rindex('}') + 1
        json_str = text[start:end]
        result = json.loads(json_str)
        
        # Normalize: accept multiple formats
        if "selected_ids" in result:
            return result
        elif "selections" in result:
            # Legacy format: extract IDs from selections
            ids = [int(sel["selected_id"]) for sel in result["selections"] if "selected_id" in sel]
            return {"selected_ids": ids}
        elif "sequence" in result:
            # Legacy format: extract IDs from sequence
            ids = [int(item["id"]) for item in result["sequence"] if "id" in item]
            return {"selected_ids": ids}
        else:
            return {"selected_ids": [], "error": "No recognized format"}
    except:
        print(f"[ERROR] Failed to parse LLM response: {text[:200]}")
        return {"selected_ids": [], "error": "Failed to parse LLM response"}

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

def generate_sequence(text: str, concepts: list, candidates: list, api_key: Optional[str] = None, custom_system_prompt: Optional[str] = None, feedback_hints: Optional[dict] = None) -> dict:
    prompt = build_generator_prompt(text, concepts, candidates, feedback_hints)
    system_prompt = custom_system_prompt or SYSTEM_PROMPT
    
    try:
        # Build id → concept + id → query_concept maps from candidates (BEFORE LLM call)
        id_to_concept = {}
        id_to_query_concept = {}
        for group in candidates:
            query_concept = group.get("concept", "")
            for cand in group.get("candidates", []):
                pid = int(cand["id"])
                if pid not in id_to_concept:
                    id_to_concept[pid] = cand.get("concept", "Unknown")
                    id_to_query_concept[pid] = query_concept
        
        if api_key:
            client = genai.Client(api_key=api_key)
        else:
            client = genai.Client()
        
        response = _generate_with_retry(
            client,
            prompt,
            types.GenerateContentConfig(system_instruction=system_prompt)
        )
        
        print(f"[DEBUG] LLM raw response: {response.text[:300]}")
        
        result = parse_response(response.text)
        selected_ids = result.get("selected_ids", [])
        
        # Build sequence looking up concept from the map, NOT from LLM response
        sequence = []
        for selected_id in selected_ids:
            pictogram_id = int(selected_id)
            pictogram_concept = id_to_concept.get(pictogram_id, "Unknown")
            pictogram_text = get_text_by_id(pictogram_id)
            
            sequence.append({
                "concept": pictogram_concept,
                "id": pictogram_id,
                "url": f"https://static.arasaac.org/pictograms/{pictogram_id}/{pictogram_id}_500.png",
                "score": 0.0,
                "description": pictogram_text,
                "extracted_query": id_to_query_concept.get(pictogram_id, "")
            })
            print(f"[DEBUG] LLM Generator sequence item: concept={pictogram_concept}, id={pictogram_id}")
            if pictogram_text:
                print(f"[DEBUG] Full text for Judge: '{pictogram_text[:100]}...'")
        
        save_result(text, concepts, candidates, result)
        
        return {"sequence": sequence, "selections": []}
        
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
    
    # Simulate what parse_response now expects
    print("Test id_to_concept mapping:")
    for group in test_candidates:
        for cand in group["candidates"]:
            print(f"  ID {cand['id']} → {cand['concept']}")
    
    print("\nNew format expects: {'selected_ids': [2001, 1001]}")
    print("(LLM decides order and count freely)")
