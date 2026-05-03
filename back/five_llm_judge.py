import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).parent
OUTPUT_DIR = PROJECT_ROOT / "llm-judge-outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

try:
    from google import genai
    from google.genai import types
except ImportError:
    import subprocess
    subprocess.run(["python", "-m", "pip", "install", "google-genai"])
    from google import genai
    from google.genai import types

MODEL_NAME = "gemini-3.1-flash-lite-preview"

SYSTEM_PROMPT = """Eres un evaluador experto en pictogramas AAC (Comunicación Aumentativa y Alternativa).

Tu tarea es evaluar qué tan bien una secuencia de pictogramas transmite el SIGNIFICADO CENTRAL de una frase en español, NO su gramática exacta.

CRITERIOS DE EVALUACIÓN (en orden de importancia):
1. COBERTURA SEMÁNTICA (50%): ¿Los conceptos clave (sustantivos, verbos, adjetivos importantes) están representados?
2. PRECISIÓN DE SELECCIÓN (30%): ¿Cada pictograma representa el concepto correcto?
3. ORDEN LÓGICO (20%): ¿El orden permite entender la idea general?

INSTRUCCIONES CRÍTICAS SOBRE GRAMÁTICA:
- Los ARTÍCULOS (el, la, un, una, los, las) NO son concepts importantes en AAC. IGNÓRALOS completamente.
- Las PREPOSICIONES (a, hacia, en, con, de) son secundarias. Solo marca como faltante si cambian el significado drásticamente (ej: "a" vs "de" cambia dirección).
- Palabras como "un", "al" (a+el), "del" (de+el) NO deben listarse como faltantes.
- No penalices por falta de conectores gramaticales. En AAC, "Niño corre parque" es aceptable; no necesita "El niño corre al parque".
- Evalúa la INTENCIÓN COMUNICATIVA, no la corrección gramatical.

EJEMPLOS DE LO QUE NO PENALIZAR:
- Falta el artículo "un" o "el"
- Falta la preposición "a" o "hacia" (a menos que sea crítica para el significado)
- Falta de concordancia de género/número en artículos

EJEMPLOS DE LO QUE SÍ PENALIZAR:
- El pictograma no representa el concepto (ej: "corriendo" pero pictograma de "carrera" como evento deportivo)
- Faltan sustantivos o verbos clave
- Orden que invierte el significado (ej: "come niño" en lugar de "niño come")

Responde SOLO con JSON válido:
{
  "score": 1-5,
  "missing_concepts": ["concepto_clave1", "concepto_clave2"] | [],
  "incorrect_pictograms": [{"concept": "X", "reason": "explicación enfocada en significado, no gramática"}] | [],
  "ordering_issues": ["solo si el orden cambia el significado"] | [],
  "suggestions": ["sugerencia concisa"] | []
}

Escala de score (ENFOCADA EN SIGNIFICADO):
- 1: Muy malo - Conceptos clave faltantes o pictogramas totalmente incorrectos
- 2: Malo - Algunos conceptos clave incorrectos o faltantes
- 3: Regular - Mayormente comprensible, errores menores de significado
- 4: Bueno - Transmite bien la idea, quizás un error menor de selección
- 5: Excelente - Representación clara y precisa de la idea central"""

def build_prompt(text: str, sequence: list) -> str:
    # Construir descripción detallada de cada pictograma
    pictograms_details = []
    for item in sequence:
        detail = f"- Concepto: '{item['concept']}'"
        if 'description' in item:
            detail += f", Descripción: '{item['description']}'"
        detail += f", URL: {item['url']}"
        pictograms_details.append(detail)
    
    pictograms_str = "\n    ".join(pictograms_details)

    return f"""Frase original: "{text}"
Pictogramas generados:
{pictograms_str}

Evalúa la secuencia enfocándote ÚNICAMENTE en el significado central:
- Ignora artículos (el, la, un, una) y preposiciones menores
- No penalices por falta de gramática estricta
- Evalúa si la idea general se comunica claramente"""

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
        return json.loads(text)
    except json.JSONDecodeError:
        return {
            "score": 3,
            "missing_concepts": [],
            "incorrect_pictograms": [],
            "ordering_issues": ["Error al parsear respuesta del LLM"],
            "suggestions": []
        }

client = genai.Client()

def save_result(text: str, sequence: list, result: dict) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"judge_{timestamp}.json"
    filepath = OUTPUT_DIR / filename
    
    output = {
        "timestamp": timestamp,
        "text": text,
        "sequence": sequence,
        "result": result
    }
    
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    return filepath

def judge(text: str, sequence: list, api_key: Optional[str] = None) -> dict:
    prompt = build_prompt(text, sequence)

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
        result = parse_response(response.text)
    except Exception as e:
        result = {
            "score": 3,
            "missing_concepts": [],
            "incorrect_pictograms": [],
            "ordering_issues": [f"Error: {str(e)}"],
            "suggestions": []
        }

    save_result(text, sequence, result)
    return result

if __name__ == "__main__":
    test_sequence = [
        {"concept": "niño", "url": "https://static.arasaac.org/pictograms/123/123_500.png"},
        {"concept": "comer", "url": "https://static.arasaac.org/pictograms/456/456_500.png"},
        {"concept": "manzana", "url": "https://static.arasaac.org/pictograms/789/789_500.png"},
    ]

    result = judge("El niño come una manzana", test_sequence)
    print(json.dumps(result, indent=2, ensure_ascii=False))