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
import feedback_analyzer
import prompt_optimizer
import threading

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
        pictogram_concept = result.get("concept", "Unknown")
        pictogram_text = result.get("text", "")
        pictograms.append({
            "order": i + 1,
            "concept": pictogram_concept,
            "id": int(result["id"]),
            "url": f"https://static.arasaac.org/pictograms/{result['id']}/{result['id']}_500.png",
            "score": float(result["score"]),
            "description": pictogram_text,
            "feedback_override": result.get("feedback_override", False)
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
    if "use_llm_generator" in body:
        use_llm_generator = body["use_llm_generator"]
    else:
        use_llm_generator = USE_LLM_GENERATOR

    if "run_judge" in body:
        run_judge = body["run_judge"]
    else:
        run_judge = True

    processed = process_text(query_text)

    # ── Load optimized Generator prompt if available (Strategy C) ──
    prompt_dir = Path("./prompt_versions")
    judge_custom_prompt = None
    generator_custom_prompt = None
    if prompt_dir.exists():
        generator_variants = sorted(prompt_dir.glob("generator_v*.txt"))
        if generator_variants:
            with open(generator_variants[-1], "r", encoding="utf-8") as f:
                generator_custom_prompt = f.read()
            print(f"[DEBUG] Using optimized Generator prompt: {generator_variants[-1].name}")

    # ── Feedback hints + overrides ──
    applied_feedback_info = {"overrides": [], "hints_used": False}

    if use_llm_generator and GEMINI_API_KEY_GENERATOR:
        try:
            from six_llm_generator import generate_sequence as llm_generate

            print(f"[DEBUG] Getting candidates for concepts: {processed['concepts']}")
            candidates, feedback_hints = search_sequence_candidates(processed["concepts"], candidate_k=3)
            print(f"[DEBUG] Candidates obtained: {len(candidates)} concepts")

            if feedback_hints:
                applied_feedback_info["hints_used"] = True
                applied_feedback_info["feedback_hints"] = feedback_hints

            print(f"[DEBUG] Calling LLM Generator with separate key...")
            generation_result = llm_generate(
                query_text, processed["concepts"], candidates,
                GEMINI_API_KEY_GENERATOR,
                custom_system_prompt=generator_custom_prompt,
                feedback_hints=feedback_hints if feedback_hints else None
            )
            print(f"[DEBUG] LLM Generator result keys: {generation_result.keys()}")

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
        print(f"[DEBUG] LLM Generator NOT used. use_llm_generator={use_llm_generator}, API_KEY_GENERATOR={bool(GEMINI_API_KEY_GENERATOR)}")
        sequence_results = search_sequence(processed["concepts"], top_k)
        llm_selections = []
        llm_generator_used = False

    # Build pictograms list
    pictograms = []
    extracted_concepts = processed["concepts"]
    for i, result in enumerate(sequence_results):
        pictogram_concept = result.get("concept", "Unknown")
        pictogram_text = result.get("description", result.get("text", ""))

        extracted_q = result.get("extracted_query", extracted_concepts[i] if i < len(extracted_concepts) else "")

        entry = {
            "order": i + 1,
            "concept": pictogram_concept,
            "id": int(result["id"]),
            "url": f"https://static.arasaac.org/pictograms/{result['id']}/{result['id']}_500.png",
            "score": float(result.get("score", 0.0)),
            "description": pictogram_text,
            "extracted_query": extracted_q
        }
        if result.get("feedback_override"):
            entry["feedback_override"] = True
            applied_feedback_info["overrides"].append({
                "concept": pictogram_concept,
                "pictogram_id": int(result["id"])
            })

        pictograms.append(entry)

    # Only run Judge if requested (can be skipped for speed)
    judge_result = None
    if run_judge and GEMINI_API_KEY_JUDGE:
        judge_result = llm_judge(query_text, pictograms, GEMINI_API_KEY_JUDGE, custom_system_prompt=judge_custom_prompt)
    elif run_judge and GEMINI_API_KEY:
        # Fallback to general key if Judge-specific key not set
        judge_result = llm_judge(query_text, pictograms, GEMINI_API_KEY, custom_system_prompt=judge_custom_prompt)

    response = {
        "original_text": query_text,
        "concepts_extracted": processed["concepts"],
        "sequence": pictograms,
        "analysis": processed["analysis"],
        "gemini_configured": bool(GEMINI_API_KEY),
        "llm_generator_used": llm_generator_used,
        "llm_selections": llm_selections if use_llm_generator else [],
        "applied_feedback": applied_feedback_info,
        "judge_skipped": not run_judge
    }

    if judge_result:
        response["judge"] = judge_result

    return jsonify(response)

@app.route("/simple-query", methods=["POST"])
def simple_query():
    body = request.json
    query_text = body["query"]
    top_k = body.get("top_k", 3)

    results, _ = search(query_text, top_k)
    ids = [int(r["id"]) for r in results]
    urls = [f"https://static.arasaac.org/pictograms/{id}/{id}_500.png" for id in ids]

    return jsonify({"paths": urls})

@app.route("/search-pictograms", methods=["POST"])
def search_pictograms():
    body = request.json
    query_text = body["query"]
    top_k = body.get("top_k", 8)
    offset = body.get("offset", 0)

    results, _ = search(query_text, top_k, offset)

    pictograms = []
    for i, result in enumerate(results):
        pictogram_concept = result.get("concept", extract_concept(result["text"]))
        pictograms.append({
            "order": offset + i + 1,
            "concept": pictogram_concept,
            "id": int(result["id"]),
            "url": f"https://static.arasaac.org/pictograms/{result['id']}/{result['id']}_500.png",
            "score": float(result["score"]),
            "description": result["text"],
            "query_concept": result.get("extracted_query", ""),
            "feedback_override": result.get("feedback_override", False)
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
        
        # Auto-trigger prompt optimization every 5 feedbacks
        total_feedback = len(list(FEEDBACK_DIR.glob("feedback_*.json")))
        if total_feedback > 0 and total_feedback % 5 == 0:
            t = threading.Thread(target=_auto_optimize_prompts, daemon=True)
            t.start()
            print(f"[AUTO] Triggered prompt optimization ({total_feedback} feedback entries)")
        
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

@app.route("/feedback/stats", methods=["GET"])
def feedback_stats():
    """Endpoint: estadísticas generales del feedback acumulado"""
    try:
        stats = feedback_analyzer.get_feedback_stats()
        return jsonify(stats), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/feedback/rules", methods=["GET"])
def feedback_rules():
    """Endpoint: tabla de correcciones aprendidas (overrides)"""
    try:
        history = feedback_analyzer.load_feedback_history()
        correction_table = feedback_analyzer.build_correction_table(history, min_confidence=1)
        return jsonify({
            "total_rules": len(correction_table),
            "rules": correction_table
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/feedback/error-patterns", methods=["GET"])
def feedback_error_patterns():
    """Endpoint: patrones de error recurrentes (desde prompt_optimizer)"""
    try:
        history = prompt_optimizer.load_feedback_history()
        patterns = prompt_optimizer.detect_recurring_errors(history)
        return jsonify({
            "total_patterns": len(patterns),
            "patterns": patterns
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/feedback/applied-suggestions", methods=["GET"])
def feedback_applied_suggestions():
    """Endpoint: sugerencias del LLM más frecuentes"""
    try:
        history = feedback_analyzer.load_feedback_history()
        suggestions = feedback_analyzer.analyze_llm_suggestions(history)
        top_n = request.args.get("top", default=10, type=int)
        top_suggestions = dict(sorted(suggestions.items(), key=lambda x: x[1], reverse=True)[:top_n])
        return jsonify({
            "total_unique_suggestions": len(suggestions),
            "top_suggestions": top_suggestions
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/feedback/optimize-generator", methods=["POST"])
def feedback_optimize_generator():
    """Endpoint: generar nuevo prompt optimizado para el Generator (Strategy C)"""
    try:
        base_prompt = """Eres un experto en selecionar pictogramas ARASAAC para AAC.

Tu tarea es crear una secuencia de pictogramas que represente fielmente el significado de una frase en español.

INSTRUCCIONES:
- Revisa todos los pictogramas candidatos disponibles
- Selecciona los que mejor representen la FRASE COMPLETA, no cada concepto individualmente
- NO es obligatorio elegir un pictograma por cada concepto extraído
- Un MISMO pictograma puede cubrir VARIOS conceptos
- El ORDEN de los IDs debe reflejar el orden logico de la idea, no necesariamente el orden de la lista de conceptos
- Piensa en como se comunicaria esta frase usando pictogramas AAC
- Respuesta unicamente JSON, sin texto adicional
- Formato exacto:

{"selected_ids": [456, 123, 789]}

No incluyas conceptos, URLs, scores ni razones. Solo los IDs en el orden que consideres correcto. No markdown, no explicaciones, solo JSON."""

        history = prompt_optimizer.load_feedback_history()
        if not history:
            return jsonify({"warning": "No hay feedback history para optimizar", "prompt": base_prompt}), 200

        patterns = prompt_optimizer.detect_recurring_errors(history)
        optimized = prompt_optimizer.get_optimized_generator_prompt(base_prompt, patterns)

        prompt_dir = Path("./prompt_versions")
        prompt_dir.mkdir(exist_ok=True)
        existing = sorted(prompt_dir.glob("generator_v*.txt"))
        version = len(existing) + 1
        prompt_optimizer.save_prompt_version("generator", version, optimized)

        return jsonify({
            "status": "success",
            "version": version,
            "optimized_prompt": optimized,
            "patterns_used": patterns
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/feedback/prompt-versions", methods=["GET"])
def feedback_prompt_versions():
    """Endpoint: listar versiones de prompts guardadas"""
    try:
        prompt_dir = Path("./prompt_versions")
        if not prompt_dir.exists():
            return jsonify({"versions": []}), 200

        versions = []
        for meta_file in sorted(prompt_dir.glob("*_meta.json")):
            try:
                with open(meta_file, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                    versions.append(meta)
            except Exception:
                pass

        return jsonify({"versions": versions}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def _auto_optimize_prompts():
    """Background task: optimize Generator prompt based on feedback"""
    try:
        history = prompt_optimizer.load_feedback_history()
        if not history:
            return

        patterns = prompt_optimizer.detect_recurring_errors(history)
        prompt_dir = Path("./prompt_versions")
        prompt_dir.mkdir(exist_ok=True)

        gen_base = """Eres un experto en selecionar pictogramas ARASAAC para AAC.

Tu tarea es crear una secuencia de pictogramas que represente fielmente el significado de una frase en español.

INSTRUCCIONES:
- Revisa todos los pictogramas candidatos disponibles
- Selecciona los que mejor representen la FRASE COMPLETA, no cada concepto individualmente
- NO es obligatorio elegir un pictograma por cada concepto extraído
- Un MISMO pictograma puede cubrir VARIOS conceptos
- El ORDEN de los IDs debe reflejar el orden logico de la idea, no necesariamente el orden de la lista de conceptos
- Piensa en como se comunicaria esta frase usando pictogramas AAC
- Respuesta unicamente JSON, sin texto adicional
- Formato exacto:

{"selected_ids": [456, 123, 789]}

No incluyas conceptos, URLs, scores ni razones. Solo los IDs en el orden que consideres correcto. No markdown, no explicaciones, solo JSON."""
        optimized_gen = prompt_optimizer.get_optimized_generator_prompt(gen_base, patterns)
        existing_gen = sorted(prompt_dir.glob("generator_v*.txt"))
        prompt_optimizer.save_prompt_version("generator", len(existing_gen) + 1, optimized_gen)
        print(f"[AUTO] Generator prompt optimized → version {len(existing_gen) + 1}")

        print(f"[AUTO] Generator prompt optimized successfully after {len(history)} feedback entries")
    except Exception as e:
        print(f"[AUTO] Generator prompt optimization error: {e}")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)