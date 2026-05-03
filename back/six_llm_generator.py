import json
import os
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

MODEL_NAME = "gemini-3.1-flash-lite-preview"

SYSTEM_PROMPT = """You are an expert in selecting ARASAAC pictograms for AAC.

Your task: Choose the BEST pictogram for each concept from the provided candidates.

INSTRUCTIONS:
- For each concept, choose ONLY ONE pictogram from the candidates list
- Consider the FULL sentence context for meaning
- Respond ONLY with valid JSON, no other text
- Use this exact format:

{"selections": [{"concept": "corriendo", "selected_id": 123, "selected_concept": "Correr", "reason": "best match"}], "sequence": [{"concept": "Correr", "id": 123, "url": "https://static.arasaac.org/pictograms/123/123_500.png", "score": 0.89}]}

No markdown, no explanation, just JSON."""

def build_generator_prompt(text: str, concepts: list, candidates: list) -> str:
    desc = []
    for item in candidates:
        concept = item["concept"]
        cands = item["candidates"]
        desc.append(f"\nConcept: {concept}")
        desc.append("Candidates:")
        for i, cand in enumerate(cands, 1):
            desc.append(f"  {i}. ID: {cand['id']}, Concept: {cand['concept']}, Score: {cand['score']:.2f}")
    
    candidates_str = "\n".join(desc)
    
    return f"""Original sentence: "{text}"
Extracted concepts: {concepts}

Pictogram candidates per concept:{candidates_str}

Select the BEST pictogram for each concept considering the full sentence context."""

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
        
        # Build sequence with URLs
        sequence = []
        if "sequence" in result and result["sequence"]:
            for item in result["sequence"]:
                sequence.append({
                    "concept": item["concept"],
                    "id": item["id"],
                    "url": f"https://static.arasaac.org/pictograms/{item['id']}/{item['id']}_500.png",
                    "score": item.get("score", 0.0)
                })
        else:
            # Fallback: build from selections
            selections_map = {}
            if "selections" in result:
                for sel in result["selections"]:
                    selections_map[sel["concept"]] = sel
            
            for concept in concepts:
                if concept in selections_map:
                    sel = selections_map[concept]
                    sequence.append({
                        "concept": sel["selected_concept"],
                        "id": sel["selected_id"],
                        "url": f"https://static.arasaac.org/pictograms/{sel['selected_id']}/{sel['selected_id']}_500.png",
                        "score": 0.0
                    })
        
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
