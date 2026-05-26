import json
import os
import re
import time
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

MODEL_NAME = "gemini-3.1-flash-lite"

MAX_RETRIES = 3
RETRYABLE_ERROR_MARKERS = (
    "429",
    "503",
    "RESOURCE_EXHAUSTED",
    "UNAVAILABLE",
    "high demand",
)

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
            is_retryable = any(marker in error_str for marker in RETRYABLE_ERROR_MARKERS)
            if is_retryable:
                if attempt == MAX_RETRIES:
                    raise
                match = re.search(r"retryDelay.*?(\d+)s", error_str)
                delay = int(match.group(1)) if match else min((2 ** attempt) * 10, 60)
                print(f"[JUDGE RETRY] Servicio ocupado. Reintentando en {delay}s (intento {attempt + 1}/{MAX_RETRIES})...")
                time.sleep(delay)
            else:
                raise

SYSTEM_PROMPT = """Eres un evaluador experto en pictogramas AAC (Comunicación Aumentativa y Alternativa).

Tu tarea es evaluar qué tan bien una secuencia de pictogramas transmite el SIGNIFICADO CENTRAL de una frase en español, NO su gramática exacta.

CRITERIOS DE EVALUACIÓN (en orden de importancia):
1. COBERTURA SEMÁNTICA (50%): ¿Los conceptos clave (sustantivos, verbos de acción, adjetivos importantes) están representados?
2. PRECISIÓN DE SELECCIÓN (30%): ¿Cada pictograma representa el concepto correcto según su descripción?
3. ORDEN LÓGICO (20%): ¿El orden permite entender la idea general?

INSTRUCCIONES CRÍTICAS SOBRE GRAMÁTICA:
- Los ARTÍCULOS (el, la, un, una, los, las) NO son conceptos importantes en AAC. IGNÓRALOS completamente.
- Las PREPOSICIONES (a, hacia, en, con, de) son secundarias. Solo marca como faltante si cambian el significado drásticamente.
- Palabras como "un", "al" (a+el), "del" (de+el) NO deben listarse como faltantes.
- No penalices por falta de conectores gramaticales. En AAC, "Niño corre parque" es aceptable.
- Evalúa la INTENCIÓN COMUNICATIVA, no la corrección gramatical.

REGLAS ESPECIALES PARA VERBOS EN AAC:
- VERBOS COPULATIVOS (ser, estar): Son opcionales en AAC. NO los marques como faltantes si la cualidad o estado sí está representada.
  Ejemplo: "Estoy cansado" → si hay pictograma de "cansado", es suficiente. NO hace falta "estar".
- FRASES DE ESTADO FÍSICO (tener frío, tener calor, tener hambre, tener sed, sentir miedo):
  El concepto correcto en AAC es el ESTADO en sí ("frío", "calor", "hambre", "sed", "miedo").
  Si hay un pictograma del estado, es correcto. NO marques falta de "tener" o "sentir".
- VERBOS DE ACCIÓN (correr, comer, beber, jugar, etc.): SÍ deben estar representados. Verifica que el pictograma
  muestra la acción y no un concepto relacionado diferente (ej: "correr" vs "carrera deportiva").

CÓMO EVALUAR CADA PICTOGRAMA:
- Lee la DESCRIPCIÓN del pictograma (si se provee) y compárala con el concepto que debe representar.
- Un pictograma es INCORRECTO solo si su descripción indica que representa algo diferente al concepto requerido.
- Si la descripción coincide aproximadamente con el concepto, el pictograma es correcto aunque el nombre sea distinto.

CÓMO LLENAR "missing_concepts":
- TODOS los sustantivos, verbos de acción, adjetivos importantes que FALTEN en la secuencia deben ir aquí.
- Ejemplo: "Quiero comer pie de manzana" → si hay [querer, comer, manzana], pero FALTA [pie], entonces missing_concepts: ["pie"]
- Ser muy literal: si dice "pie" o "tarta", ambos son conceptos faltantes posibles.
- Los artículos (el, la, un, una) NUNCA van aquí.
- Las preposiciones menores (de, en, a) NUNCA van aquí (a menos que cambien drásticamente el significado).

REGLAS PARA SUGERENCIAS (MUY IMPORTANTE):
- Las sugerencias deben ser CAMBIOS CONCRETOS y DIFERENTES a lo que ya se muestra.
- NUNCA sugiereas usar el mismo tipo de pictograma que ya está seleccionado.
- Si un pictograma es incorrecto, indica: "Sustituir [concepto_actual] por [concepto_alternativo]" siendo [concepto_alternativo] algo DISTINTO.
- Si falta un concepto (Y TAMBIÉN está en missing_concepts), indica: "Agregar pictograma de [concepto]"
- Si el orden está mal: "Mover [concepto_A] antes de [concepto_B]"
- Si todo es correcto, deja suggestions como lista vacía [].
- Máximo 3 sugerencias. Solo incluye las más importantes.

QUÉ NO PENALIZAR:
- Falta de artículos o preposiciones menores
- Verbos copulativos (ser/estar) ausentes cuando el estado sí está representado
- Falta de "tener"/"sentir" cuando el estado físico sí está representado

QUÉ SÍ PENALIZAR:
- Pictograma cuya descripción no corresponde al concepto de la frase
- Faltan sustantivos o verbos de acción clave
- Orden que invierte el significado

Responde SOLO con JSON válido:
{
  "score": 1-5,
  "missing_concepts": ["concepto_clave1"] | [],
  "incorrect_pictograms": [{"concept": "X", "reason": "descripción exacta del problema y qué pictograma debería usarse en su lugar"}] | [],
  "ordering_issues": ["solo si el orden cambia el significado"] | [],
  "suggestions": ["Sustituir X por Y" | "Agregar pictograma de Z" | "Mover A antes de B"] | []
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
    for i, item in enumerate(sequence):
        detail = f"- Concepto: '{item['concept']}'"
        if 'description' in item:
            detail += f", Descripción: '{item['description'][:60]}'"
        else:
            detail += " [NO DESCRIPTION PROVIDED]"
        if item.get('url'):
            detail += f", URL: {item['url']}"
        pictograms_details.append(detail)
    
    pictograms_str = "\n    ".join(pictograms_details)

    prompt = f"""Frase original: "{text}"
Pictogramas generados:
{pictograms_str}

Evalúa la secuencia enfocándote ÚNICAMENTE en el significado central:
- Ignora artículos (el, la, un, una) y preposiciones menores
- No penalices por falta de gramática estricta
- Evalúa si la idea general se comunica claramente"""

    if os.environ.get("DEBUG_LLM_PROMPTS", "false").lower() == "true":
        print("\n" + "="*80)
        print("[JUDGE DEBUG] FULL PROMPT SENT TO LLM:")
        print("="*80)
        print(prompt)
        print("="*80 + "\n")
    
    return prompt

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
            "score": 0,
            "missing_concepts": [],
            "incorrect_pictograms": [],
            "ordering_issues": [],
            "suggestions": [],
            "error": True,
            "error_type": "parse_error",
            "message": "No se pudo interpretar la respuesta del LLM-Judge."
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

def judge(text: str, sequence: list, api_key: Optional[str] = None, custom_system_prompt: Optional[str] = None) -> dict:
    prompt = build_prompt(text, sequence)
    system_prompt = custom_system_prompt or SYSTEM_PROMPT

    try:
        if api_key:
            client = genai.Client(api_key=api_key)
        else:
            client = genai.Client()

        response = _generate_with_retry(
            client,
            prompt,
            types.GenerateContentConfig(system_instruction=system_prompt)
        )
        result = parse_response(response.text)
    except Exception as e:
        error_str = str(e)
        is_unavailable = "503" in error_str or "UNAVAILABLE" in error_str or "high demand" in error_str
        result = {
            "score": 0,
            "missing_concepts": [],
            "incorrect_pictograms": [],
            "ordering_issues": [],
            "suggestions": [],
            "error": True,
            "error_type": "service_unavailable" if is_unavailable else "api_error",
            "message": (
                "El LLM-Judge no pudo evaluar la secuencia porque el modelo está temporalmente ocupado. "
                "Intenta nuevamente en unos minutos."
                if is_unavailable
                else "El LLM-Judge no pudo evaluar la secuencia por un error técnico."
            ),
            "details": error_str
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
