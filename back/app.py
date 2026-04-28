import os
from flask import Flask, jsonify, request
from flask_cors import CORS
from dotenv import load_dotenv
from three_use_embedded import search, search_sequence
from four_extract_concepts import process_text
from five_llm_judge import judge as llm_judge

load_dotenv()

app = Flask(__name__)

CORS(app)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

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
    top_k = body.get("top_k", 5)

    processed = process_text(query_text)
    sequence_results = search_sequence(processed["concepts"], top_k)

    pictograms = []
    for i, result in enumerate(sequence_results):
        pictograms.append({
            "order": i + 1,
            "concept": result["concept"],
            "id": int(result["id"]),
            "url": f"https://static.arasaac.org/pictograms/{result['id']}/{result['id']}_500.png",
            "score": float(result["score"])
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

    processed = process_text(query_text)
    sequence_results = search_sequence(processed["concepts"], top_k)

    pictograms = []
    for i, result in enumerate(sequence_results):
        pictograms.append({
            "order": i + 1,
            "concept": result["concept"],
            "id": int(result["id"]),
            "url": f"https://static.arasaac.org/pictograms/{result['id']}/{result['id']}_500.png",
            "score": float(result["score"]),
            "description": result["text"]
        })

    judge_result = None
    if GEMINI_API_KEY:
        judge_result = llm_judge(query_text, pictograms, GEMINI_API_KEY)

    response = {
        "original_text": query_text,
        "concepts_extracted": processed["concepts"],
        "sequence": pictograms,
        "analysis": processed["analysis"],
        "gemini_configured": bool(GEMINI_API_KEY)
    }

    if judge_result:
        response["judge"] = judge_result

    return jsonify(response)

@app.route("/simple-query", methods=["POST"])
def simple_query():
    body = request.json
    query_text = body["query"]
    top_k = body.get("top_k", 5)

    results = search(query_text, top_k)
    ids = [int(r["id"]) for r in results]
    urls = [f"https://static.arasaac.org/pictograms/{id}/{id}_500.png" for id in ids]

    return jsonify({"paths": urls})

@app.route("/search-pictograms", methods=["POST"])
def search_pictograms():
    body = request.json
    query_text = body["query"]
    top_k = body.get("top_k", 5)

    # For search we want to find pictograms for the query text itself
    results = search(query_text, top_k)

    pictograms = []
    for i, result in enumerate(results):
        pictograms.append({
            "order": i + 1,
            "concept": query_text,  # Use the original query as concept
            "id": int(result["id"]),
            "url": f"https://static.arasaac.org/pictograms/{result['id']}/{result['id']}_500.png",
            "score": float(result["score"]),
            "description": result["text"]  # Keep description for potential use
        })

    return jsonify({
        "query": query_text,
        "results": pictograms
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)