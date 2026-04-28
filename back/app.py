from flask import Flask
from flask import jsonify
from flask import request
from flask_cors import CORS
from three_use_embedded import search

app = Flask(__name__)

CORS(app)

@app.route("/helloworld")
def home():
  return jsonify({"message": "Hello World"})

@app.route("/query", methods=["POST"])
def query():
  body = request.json
  query = body["query"]
  results = search(query, 5)

  id_results = [r["id"] for r in results]
  ids = [f"https://static.arasaac.org/pictograms/{id}/{id}_500.png" for id in id_results]

  return jsonify({"paths": ids})

if __name__ == "__main__":
  app.run(host='0.0.0.0', port=5000)