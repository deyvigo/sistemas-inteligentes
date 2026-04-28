from flask import Flask, jsonify, request
from flask_cors import CORS
from three_use_embedded import search, search_sequence
from four_extract_concepts import process_text

app = Flask(__name__)

CORS(app)

@app.route("/helloworld")
def home():
    return jsonify({"message": "Hello World"})

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

@app.route("/simple-query", methods=["POST"])
def simple_query():
    body = request.json
    query_text = body["query"]
    top_k = body.get("top_k", 5)

    results = search(query_text, top_k)
    ids = [int(r["id"]) for r in results]
    urls = [f"https://static.arasaac.org/pictograms/{id}/{id}_500.png" for id in ids]

    return jsonify({"paths": urls})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)