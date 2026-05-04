import os
import re
from flask import Flask, jsonify, request
from flask_cors import CORS
from dotenv import load_dotenv
from three_use_embedded import search, search_sequence, search_sequence_candidates
from four_extract_concepts import process_text
from five_llm_judge import judge as llm_judge
import json
from datetime import datetime
from pathlib import Path

def extract_concept(text):
    """Extract only the main keyword from the text field"""
    # Format: "query: Concepto: {keyword}\nDescripción: ..."
    match = re.search(r'Concepto:\s*(.+?)(\n|$)', text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    # Fallback: return first 50 chars
    return text.strip()[:50]

load_dotenv()

# Debug: Print loaded environment variables
print("=== Loaded Environment Variables ===")
print(f"GEMINI_API_KEY set: {(os.environ.get('GEMINI_API_KEY'))}")
print(f"GEMINI_API_KEY_GENERATOR set: {(os.environ.get('GEMINI_API_KEY_GENERATOR'))}")
print(f"GEMINI_API_KEY_JUDGE set: {(os.environ.get('GEMINI_API_KEY_JUDGE'))}")
print(f"USE_LLM_GENERATOR: {os.environ.get('USE_LLM_GENERATOR', 'not set')}")
print("===================================")

app = Flask(__name__)

CORS(app)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_API_KEY_GENERATOR = os.environ.get("GEMINI_API_KEY_GENERATOR", GEMINI_API_KEY)  # Fallback to general key
GEMINI_API_KEY_JUDGE = os.environ.get("GEMINI_API_KEY_JUDGE", GEMINI_API_KEY)      # Fallback to general key

USE_LLM_GENERATOR = os.environ.get("USE_LLM_GENERATOR", "true").lower() == "true"

# Feedback storage
FEEDBACK_DIR = Path("./feedback_logs")
FEEDBACK_DIR.mkdir(exist_ok=True)

@app.route("/helloworld")
def home():
    return jsonify({"message": "Hello World"})

@app.route("/config")
def config():
    return jsonify({
        "gemini_configured": bool(GEMINI_API_KEY)
    })

@app.route("/query", methods=["POST"])
def query():
    body = request.json
    query_text = body["query"]
    top_k = body.get("top_k", 5 )

    processed = process_text(query_text)
    sequence_results = search_sequence(processed["concepts"], top_k)

    pictograms = []
    for i, result in enumerate(sequence_results):
        # Use pictogram concept (ARASAAC concept like "Comer"), not query concept
        pictogram_concept = result.get("concept", "Unknown")
        # Include FULL ARASAAC text description for Judge evaluation
        pictogram_text = result.get("text", "")
        pictograms.append({
            "order": i + 1,
            "concept": pictogram_concept,  # ARASAAC pictogram concept (for display)
            "id": int(result["id"]),
            "url": f"https://static.arasaac.org/pictograms/{result['id']}/{result['id']}_500.png",
            "score": float(result["score"]),
            "description": pictogram_text  # FULL text for Judge to evaluate correctly
        })

    return jsonify({
        "original_text": query_text,
        "concepts_extracted": processed["concepts"],
        "sequence": pictograms,
        "analysis": processed["analysis"]
    })

@app.route("/judge", methods=["POST"])
def judge():
    body = request.json
    text = body["text"]
    sequence = body["sequence"]

    if not GEMINI_API_KEY:
        return jsonify({
            "error": "GEMINI_API_KEY no configurada en el servidor"
        }), 400

    result = llm_judge(text, sequence, GEMINI_API_KEY)
    return jsonify(result)

@app.route("/query-and-judge", methods=["POST"])
def query_and_judge():
    body = request.json
    query_text = body["query"]
    top_k = body.get("top_k", 5)
    # Request body has priority; fallback to env var
    if "use_llm_generator" in body:
        use_llm_generator = body["use_llm_generator"]
    else:
        use_llm_generator = USE_LLM_GENERATOR

    processed = process_text(query_text)

    # Use LLM Generator if enabled and API key is available
    print(f"[DEBUG] use_llm_generator: {use_llm_generator}")
    print(f"[DEBUG] GEMINI_API_KEY_GENERATOR set: {bool(GEMINI_API_KEY_GENERATOR)}")
    print(f"[DEBUG] GEMINI_API_KEY_JUDGE set: {bool(GEMINI_API_KEY_JUDGE)}")
    
    if use_llm_generator and GEMINI_API_KEY_GENERATOR:
        try:
            from six_llm_generator import generate_sequence as llm_generate

            # Get multiple candidates per concept (top 5)
            print(f"[DEBUG] Getting candidates for concepts: {processed['concepts']}")
            candidates = search_sequence_candidates(processed["concepts"], candidate_k=5)
            print(f"[DEBUG] Candidates obtained: {len(candidates)} concepts")

            # LLM selects best pictogram per concept
            print(f"[DEBUG] Calling LLM Generator with separate key...")
            generation_result = llm_generate(query_text, processed["concepts"], candidates, GEMINI_API_KEY_GENERATOR)
            print(f"[DEBUG] LLM Generator result: {generation_result.keys()}")

            sequence_results = generation_result["sequence"]
            llm_selections = generation_result.get("selections", [])
            llm_generator_used = True
            print(f"[DEBUG] LLM Generator used successfully")

        except Exception as e:
            print(f"[ERROR] LLM Generator failed: {e}, falling back to embedding-only")
            sequence_results = search_sequence(processed["concepts"], top_k)
            llm_selections = []
            llm_generator_used = False
    else:
        # Fallback: Original embedding-only selection
        print(f"[DEBUG] LLM Generator NOT used. use_llm_generator={use_llm_generator}, API_KEY_GENERATOR={bool(GEMINI_API_KEY_GENERATOR)}")
        sequence_results = search_sequence(processed["concepts"], top_k)
        llm_selections = []
        llm_generator_used = False

    # Build pictograms list
    pictograms = []
    for i, result in enumerate(sequence_results):
        # Use the pictogram concept from search results
        pictogram_concept = result.get("concept", "Unknown")
        
        # Get FULL ARASAAC text description for Judge to evaluate correctly
        pictogram_text = result.get("description", result.get("text", ""))
        
        # Debug print to verify
        print(f"[DEBUG] Pictogram {i+1}: concept='{pictogram_concept}', id={result.get('id')}, extracted_query={result.get('extracted_query', 'N/A')}")
        if pictogram_text:
            print(f"[DEBUG] Full text for Judge: '{pictogram_text[:100]}...'")
        
        pictograms.append({
            "order": i + 1,
            "concept": pictogram_concept,  # ARASAAC pictogram concept (for display)
            "id": int(result["id"]),
            "url": f"https://static.arasaac.org/pictograms/{result['id']}/{result['id']}_500.png",
            "score": float(result.get("score", 0.0)),
            "description": pictogram_text  # FULL ARASAAC text for Judge evaluation
        })

    # ALWAYS call Judge after Generator (if API key available)
    judge_result = None
    if GEMINI_API_KEY_JUDGE:
        judge_result = llm_judge(query_text, pictograms, GEMINI_API_KEY_JUDGE)
    elif GEMINI_API_KEY:
        # Fallback to general key if Judge-specific key not set
        judge_result = llm_judge(query_text, pictograms, GEMINI_API_KEY)

    response = {
        "original_text": query_text,
        "concepts_extracted": processed["concepts"],
        "sequence": pictograms,
        "analysis": processed["analysis"],
        "gemini_configured": bool(GEMINI_API_KEY),
        "llm_generator_used": llm_generator_used,
        "llm_selections": llm_selections if use_llm_generator else []
    }

    if judge_result:
        response["judge"] = judge_result

    return jsonify(response)

@app.route("/simple-query", methods=["POST"])
def simple_query():
    body = request.json
    query_text = body["query"]
    top_k = body.get("top_k", 3)

    results = search(query_text, top_k)
    ids = [int(r["id"]) for r in results]
    urls = [f"https://static.arasaac.org/pictograms/{id}/{id}_500.png" for id in ids]

    return jsonify({"paths": urls})

@app.route("/search-pictograms", methods=["POST"])
def search_pictograms():
    body = request.json
    query_text = body["query"]
    top_k = body.get("top_k", 8)  # Increased default for better UX
    offset = body.get("offset", 0)

    # For search we want to find pictograms for the query text itself
    results = search(query_text, top_k, offset)

    pictograms = []
    for i, result in enumerate(results):
        # Use the concept from search result (ARASAAC concept like "Comer")
        # NOT the extracted query (like "tomando")
        pictogram_concept = result.get("concept", extract_concept(result["text"]))
        pictograms.append({
            "order": offset + i + 1,  # Adjust order based on offset
            "concept": pictogram_concept,  # ARASAAC pictogram concept
            "id": int(result["id"]),
            "url": f"https://static.arasaac.org/pictograms/{result['id']}/{result['id']}_500.png",
            "score": float(result["score"]),
            "description": result["text"],  # Keep full text for reference
            "query_concept": result.get("extracted_query", "")  # What user searched (optional)
        })

    return jsonify({
        "query": query_text,
        "results": pictograms,
        "offset": offset,
        "limit": top_k
    })

@app.route("/feedback", methods=["POST"])
def receive_feedback():
    """Endpoint para recibir feedback completo del ciclo human-in-the-loop"""
    try:
        feedback_data = request.json
        
        # Añadir timestamp si no viene incluido
        if "timestamp" not in feedback_data:
            feedback_data["timestamp"] = datetime.now().isoformat()
            
        # Generar ID de sesión si no viene
        if "session_id" not in feedback_data:
            feedback_data["session_id"] = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        # Guardar feedback en archivo
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = FEEDBACK_DIR / f"feedback_{timestamp_str}.json"
        
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(feedback_data, f, indent=2, ensure_ascii=False)
        
        return jsonify({
            "status": "success",
            "message": "Feedback received and stored",
            "session_id": feedback_data["session_id"],
            "timestamp": feedback_data["timestamp"]
        }), 200
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Failed to process feedback: {str(e)}"
        }), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)