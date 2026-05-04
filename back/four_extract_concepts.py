import re
import json
from pathlib import Path

NEGATION_WORDS = {"no", "nunca", "nada", "ninguno", "ninguna", "sin", "ni", "tampoco"}
TIME_WORDS = {"ayer", "hoy", "mañana", "ahora", "después", "antes", "luego", "siempre", "pronto", "tarde"}
COMMON_SUBJECTS = {"el", "la", "los", "las", "un", "una", "unos", "unas", "mi", "tu", "su", "nuestro", "nuestra"}
STOPWORDS = {"el", "la", "los", "las", "un", "una", "unos", "unas", "de", "del", "en", "con", "por", "para", "a", "al", "y", "e", "o", "u", "que", "es", "son", "esta", "estan", "tiene", "tienen", "un", "una", "unos", "unas", "este", "esta", "estos", "estas", "ese", "esa", "esos", "esas"}

def tokenize(text):
    text = text.lower()
    words = re.findall(r'\b\w+\b', text)
    return words

def extract_concepts(text):
    words = tokenize(text)
    
    negation = any(word in NEGATION_WORDS for word in words)
    temporal_markers = [word for word in words if word in TIME_WORDS]
    
    important_words = [w for w in words if w not in STOPWORDS and len(w) > 2]
    
    negation_concepts = [{"text": "no", "type": "negation", "role": "negation"}] if negation else []
    
    time_concepts = [{"text": marker, "type": "time", "role": "time"} for marker in temporal_markers]
    
    # Keep original words (no normalization!)
    other_concepts = []
    seen = set()
    for word in important_words:
        if word in NEGATION_WORDS or word in TIME_WORDS:
            continue
            
        if word not in seen:
            seen.add(word)
            other_concepts.append({"text": word, "type": "noun", "role": "concept"})
    
    concepts = negation_concepts + time_concepts + other_concepts
    
    concepts = concepts[:10]
    
    return {
        "concepts": [c["text"] for c in concepts],
        "negation": negation,
        "temporal_markers": temporal_markers,
        "subject": None,
        "action": None,
        "objects": []
    }

def build_sequence(concepts_data):
    sequence = concepts_data["concepts"]
    return sequence

def process_text(text):
    concepts_data = extract_concepts(text)
    sequence = build_sequence(concepts_data)
    
    return {
        "original": text,
        "concepts": concepts_data["concepts"],
        "sequence": sequence,
        "analysis": concepts_data
    }

if __name__ == "__main__":
    test_texts = [
        "El niño come una manzana",
        "La madre no cocina la cena",
        "El perro juega en el parque",
        "Mi amigo lee un libro",
        "La niña dibuja en la pizarra",
        "Hoy no fui a la escuela",
        "El padre trabaja mucho",
        "La gata duerme en el sofá",
    ]

    for text in test_texts:
        result = process_text(text)
        print(f"\nTexto: {text}")
        print(f"Conceptos extraídos: {result['concepts']}")
        print(f"Es negación: {result['analysis']['negation']}")