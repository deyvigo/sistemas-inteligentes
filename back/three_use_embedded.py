import numpy as np
import re
from sentence_transformers import SentenceTransformer
import json
from pathlib import Path
from collections import defaultdict, Counter

# ── Import feedback analysis from central module ──
from feedback_analyzer import (
    load_feedback_history,
    analyze_concept_corrections,
    analyze_corrections_from_feedback,
    apply_rule_improvements,
    analyze_llm_suggestions,
    apply_llm_suggestions_as_postprocessing,
    build_correction_table,
    get_overrides
)

model = SentenceTransformer("intfloat/multilingual-e5-small")

def extract_concept(text):
    """Extract the main concept (keyword) from the text field"""
    lines = text.strip().split('\n')
    for line in lines:
        if 'Concepto:' in line:
            parts = line.split('Concepto:')
            if len(parts) > 1:
                keyword = parts[1].strip()
                return keyword
    for line in lines:
        if line.strip():
            return line.strip()[:50]
    return "unknown"

embeddings = np.load("./embeddings/embeddings.npy")
ids = np.load("./embeddings/ids.npy")
texts = np.load("./embeddings/texts.npy")

# Feedback storage
FEEDBACK_DIR = Path("./feedback_logs")
FEEDBACK_DIR.mkdir(exist_ok=True)

def search(query, top_k=5, offset=0):
    query_emb = model.encode([query], normalize_embeddings=True)[0]

    scores = np.dot(embeddings, query_emb)

    # Apply feedback-based rule improvements if we have history
    applied_overrides = []
    try:
        history = load_feedback_history()
        if history:
            # Re-ranking: boost/suppress known pictograms
            concept_stats = analyze_concept_corrections(history)
            human_stats = analyze_corrections_from_feedback(history)
            scores = apply_rule_improvements(query, concept_stats, human_stats, scores, ids)
    except Exception as e:
        print(f"Warning: Could not apply feedback improvements: {e}")

    num_results_needed = offset + top_k
    num_results_needed = min(num_results_needed, len(scores))
    top_indices = np.argsort(scores)[-num_results_needed:][::-1]

    results = []
    seen_ids = set()
    for i in top_indices:
        full_text = texts[i]
        pictogram_concept = extract_concept(full_text)
        pid = int(ids[i])
        seen_ids.add(pid)
        results.append({
            "id": pid,
            "text": full_text,
            "concept": pictogram_concept,
            "extracted_query": query,
            "score": float(scores[i])
        })
    
    # ── Override & Reject system: force-include / penalize pictograms from feedback ──
    try:
        override_info = get_overrides(query)
        if override_info:
            overrides = override_info.get('overrides', {})  # {id: count} filtered by min_confidence
            rejected = override_info.get('rejected', {})    # {id: count} from LLM Judge

            # Penalize rejected IDs
            for item in results:
                rid = item['id']
                if rid in rejected:
                    penalty = -0.05 * min(rejected[rid], 3)
                    item['score'] += penalty
                    print(f"[JUDGE] Penalized ID {rid} for '{query}': {penalty} (rejected ×{rejected[rid]})")

            id_list = ids.tolist() if hasattr(ids, 'tolist') else list(ids)
            need_sort = False
            for override_id, confidence in overrides.items():
                if override_id not in seen_ids:
                    if override_id in id_list:
                        idx = id_list.index(override_id)
                        full_text = texts[idx]
                        pictogram_concept = extract_concept(full_text)
                        results.append({
                            "id": override_id,
                            "text": full_text,
                            "concept": pictogram_concept,
                            "extracted_query": query,
                            "score": 1.0,
                            "feedback_override": True,
                            "override_confidence": confidence,
                            "override_last_seen": override_info.get('last_seen', '')
                        })
                        seen_ids.add(override_id)
                        applied_overrides.append({
                            "concept": query,
                            "override_id": override_id,
                            "confidence": confidence
                        })
                        need_sort = True
                        print(f"[FEEDBACK] Override applied: '{query}' → ID {override_id} (confidence: {confidence})")
                    else:
                        print(f"[FEEDBACK] Override ID {override_id} for '{query}' not found in dataset")
                else:
                    # Already in results; boost its score
                    for item in results:
                        if item['id'] == override_id:
                            item['score'] = 10.0
                            item['feedback_override'] = True
                            item['override_confidence'] = confidence
                            break
                    applied_overrides.append({
                        "concept": query,
                        "override_id": override_id,
                        "confidence": confidence,
                        "action": "boosted"
                    })
                    need_sort = True
                    print(f"[FEEDBACK] Override boosted: '{query}' → ID {override_id} (confidence: {confidence})")
            if need_sort:
                results.sort(key=lambda x: x['score'], reverse=True)
    except Exception as e:
        print(f"Warning: Could not apply override: {e}")
    
    # Apply LLM suggestion post-processing if we have history
    try:
        history = load_feedback_history()
        if history:
            llm_suggestions = analyze_llm_suggestions(history)
            results = apply_llm_suggestions_as_postprocessing(query, llm_suggestions, results)
    except Exception as e:
        print(f"Warning: Could not apply LLM suggestion improvements: {e}")
    
    return results[offset:offset+top_k], applied_overrides

def search_sequence(concepts, top_k=3):
    sequence_results = []
    all_overrides = []
    seen_ids = set()

    for concept in concepts:
        concept_results, overrides = search(concept, top_k)
        all_overrides.extend(overrides)
        for result in concept_results:
            if result["id"] not in seen_ids:
                seen_ids.add(result["id"])
                sequence_results.append({
                    "concept": result.get("concept", concept),
                    "id": result["id"],
                    "text": result["text"],
                    "score": result["score"],
                    "feedback_override": result.get("feedback_override", False),
                    "extracted_query": result.get("extracted_query", concept)
                })
                break

    return sequence_results


def search_sequence_candidates(concepts, candidate_k=5):
    """
    For each concept, return TOP candidate_k pictograms from embedding search.
    Used by LLM Generator to select the best pictogram considering full text context.
    
    Returns:
        list: [{"concept": "correr" (query concept), "candidates": [{id, concept, text, description, score}, ...]}, ...]
    """
    candidates_per_concept = []
    all_feedback_hints = {}
    
    for concept in concepts:
        # Get top-k candidates for this concept (with overrides)
        candidate_results, overrides = search(concept, top_k=candidate_k)
        
        # Track feedback hints for this concept (used by LLM Generator)
        override_info = get_overrides(concept)
        if override_info:
            hints = {"last_seen": override_info.get('last_seen', '')}
            if 'overrides' in override_info:
                hints['overrides'] = override_info['overrides']
            if 'rejected' in override_info:
                hints['rejected'] = override_info['rejected']
            if 'missing_count' in override_info:
                hints['missing_count'] = override_info['missing_count']
            all_feedback_hints[concept] = hints
        
        candidates_list = []
        for result in candidate_results:
            entry = {
                "id": int(result["id"]),
                "concept": result.get("concept", "Unknown"),
                "query_concept": concept,
                "text": result["text"],
                "description": result["text"],
                "score": float(result["score"])
            }
            if result.get("feedback_override"):
                entry["feedback_override"] = True
            candidates_list.append(entry)
        
        candidates_per_concept.append({
            "concept": concept,
            "candidates": candidates_list
        })
        
        print(f"\n=== Top {candidate_k} candidates for concept: '{concept}' ===")
        for i, cand in enumerate(candidates_list, 1):
            print(f"  {i}. ID: {cand['id']}, Pictogram concept: '{cand['concept']}', Score: {cand['score']:.4f}")
    
    return candidates_per_concept, all_feedback_hints