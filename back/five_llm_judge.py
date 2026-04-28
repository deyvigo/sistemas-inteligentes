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

SYSTEM_PROMPT = """Eres un evaluador experto y estricto de pictogramas AAC (Comunicación Aumentativa y Alternativa).

Tu tarea es evaluar qué tan bien una secuencia de pictogramas representa el significado EXACTO de una frase en español.

Debes ser meticuloso y penalizar fuertemente cualquier discrepancia, incluso menor, entre:
1. El concepto solicitado y lo que realmente representa el pictograma
2. La descripción asociada al pictograma y el concepto esperado

Evalúa rigurosamente según estos criterios:
1. COBERTURA SEMÁNTICA: ¿Todos los conceptos importantes de la frase están representados? Penaliza fuertemente los conceptos faltantes.
2. PRECISIÓN DE SELECCIÓN: ¿Cada pictograma representa EXACTAMENTE el concepto correspondiente? Cualquier mismatch entre el concepto solicitado y lo que representa el pictograma (basándose en su descripción) debe marcarse como incorrecto.
3. ORDEN SINTÁCTICO: ¿Los pictogramas siguen un orden gramatical y lógico coherente para la frase?

INSTRUCCIONES CRÍTICAS:
- Si la descripción de un pictograma no coincide con el concepto solicitado, DEBES marcarlo como incorrecto en "incorrect_pictograms"
- Sé especialmente crítico con verbos, adjetivos y sustantivos específicos - no aceptes aproximaciones genéricas
- La descripción proporcionada es la verdad definitiva sobre qué representa cada pictograma
- Si hay dudas, es mejor marcar como incorrecto que pasar por alto un error

Responde SOLO con JSON válido, sin texto adicional:
{
  "score": 1-5,
  "missing_concepts": ["concepto1", "concepto2"] | [],
  "incorrect_pictograms": [{"concept": "X", "reason": "explicación detallada del por qué es incorrecto"}] | [],
  "ordering_issues": ["explicación del problema de orden"] | [],
  "suggestions": ["sugerencia1 concreta", "sugerencia2 concreta"] | []
}

Escala de score (SE ESTRICTO):
- 1: Muy malo - Errores graves que impiden la comprensión básica
- 2: Malo - Múltiples errores significativos o conceptos faltantes importantes
- 3: Regular - Algunos errores notables que afectan la claridad
- 4: Bueno - Pocos errores menores, generalmente comprensible
- 5: Excelente - Representación perfecta y precisa sin errores detectables

Recuerda: Tu trabajo es proteger al usuario de pictogramas incorrectos que podrían llevar a malentendidos comunicativos graves."""

print(os.getenv("GEMINI_API_KEY"))

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

    Evalúa la calidad de esta secuencia de pictogramas considerando:
    1. Si el concepto representado por cada pictograma corresponde al concepto esperado
    2. Si la descripción asociada al pictograma coincide con el concepto
    3. La adecuación general de la secuencia para representar la frase original"""

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