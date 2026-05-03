import numpy as np
import re
from sentence_transformers import SentenceTransformer
import json
from pathlib import Path
from collections import defaultdict, Counter

model = SentenceTransformer("intfloat/multilingual-e5-small")

def extract_concept(text):
    """Extract the main concept (keyword) from the text field"""
    # The text format is:
    #    query: Concepto: {keyword}
    #    Descripción: {description}
    #    ...
    # We want to extract just the keyword after "Concepto:"
    
    lines = text.strip().split('\n')
    for line in lines:
        if 'Concepto:' in line:
            # Extract everything after "Concepto: "
            parts = line.split('Concepto:')
            if len(parts) > 1:
                keyword = parts[1].strip()
                return keyword
    
    # Fallback: return first non-empty line
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

def load_feedback_history():
    """Load all feedback entries from feedback_logs directory"""
    history = []
    if not FEEDBACK_DIR.exists():
        return history
    
    for feedback_file in FEEDBACK_DIR.glob("feedback_*.json"):
        try:
            with open(feedback_file, "r", encoding="utf-8") as f:
                feedback_data = json.load(f)
                history.append(feedback_data)
        except Exception as e:
            print(f"Error loading {feedback_file}: {e}")
    
    return history

def analyze_concept_corrections(feedback_history):
    """
    Analyze feedback to learn which pictograms humans prefer for each concept
    
    Returns:
        dict: {concept: {preferred_pictogram_ids: [id1, id2], rejected_pictogram_ids: [id3, id4]}}
    """
    concept_stats = defaultdict(lambda: {
        'preferred': Counter(),
        'rejected': Counter(),
        'total_seen': 0
    })
    
    for feedback in feedback_history:
        # Get original prediction and human correction
        original_sequence = feedback.get('system_generation', {}).get('sequence', [])
        corrected_sequence = feedback.get('user_modifications', {}).get('final_sequence', [])
        
        # If no correction was made, skip
        if not corrected_sequence:
            continue
            
        # Create maps for easy lookup
        original_by_concept = {item['concept']: item for item in original_sequence}
        corrected_by_concept = {item['concept']: item for item in corrected_sequence}
        
        # Process each concept that appears in either sequence
        all_concepts = set(list(original_by_concept.keys()) + list(corrected_by_concept.keys()))
        
        for concept in all_concepts:
            original_item = original_by_concept.get(concept)
            corrected_item = corrected_by_concept.get(concept)
            
            # Track that we saw this concept
            concept_stats[concept]['total_seen'] += 1
            
            # If human changed the pictogram for this concept
            if original_item and corrected_item:
                original_id = original_item['id']
                corrected_id = corrected_item['id']
                
                if original_id != corrected_id:
                    # Human rejected original_id, preferred corrected_id
                    concept_stats[concept]['rejected'][original_id] += 1
                    concept_stats[concept]['preferred'][corrected_id] += 1
            elif corrected_item:
                # Concept was added by human (wasn't in original)
                corrected_id = corrected_item['id']
                concept_stats[concept]['preferred'][corrected_id] += 1
            elif original_item:
                # Concept was removed by human (was in original but not corrected)
                original_id = original_item['id']
                concept_stats[concept]['rejected'][original_id] += 1
    
    # Convert to preferred format
    result = {}
    for concept, stats in concept_stats.items():
        if stats['total_seen'] > 0:
            # Get top preferred and rejected pictograms
            preferred_ids = [pid for pid, _ in stats['preferred'].most_common(5)]
            rejected_ids = [pid for pid, _ in stats['rejected'].most_common(5)]
            
            result[concept] = {
                'preferred_pictogram_ids': preferred_ids,
                'rejected_pictogram_ids': rejected_ids,
                'confidence': stats['total_seen']  # More feedback = higher confidence
            }
    
    return result

def apply_rule_improvements(concept, concept_stats, base_scores, id_list):
    """
    Apply learned rules to adjust search scores for a concept
    
    Args:
        concept (str): The concept we're processing
        concept_stats (dict): The analyzed statistics from feedback
        base_scores (np.array): Original similarity scores
        id_list (np.array): Corresponding pictogram IDs
        
    Returns:
        np.array: Modified scores
    """
    if concept not in concept_stats:
        return base_scores  # No learned rules for this concept
    
    stats = concept_stats[concept]
    modified_scores = base_scores.copy()
    
    # Create ID to index mapping for fast lookup
    id_to_index = {id_val: idx for idx, id_val in enumerate(id_list)}
    
    # Boost preferred pictograms
    for pictogram_id in stats['preferred_pictogram_ids']:
        if pictogram_id in id_to_index:
            idx = id_to_index[pictogram_id]
            # Higher confidence = stronger boost
            boost_value = min(stats['confidence'] * 0.1, 2.0)  # Cap at 2.0
            modified_scores[idx] += boost_value
    
    # Suppress rejected pictograms
    for pictogram_id in stats['rejected_pictogram_ids']:
        if pictogram_id in id_to_index:
            idx = id_to_index[pictogram_id]
            # Higher confidence = stronger suppression (negative boost)
            suppress_value = max(-stats['confidence'] * 0.1, -2.0)  # Cap at -2.0
            modified_scores[idx] += suppress_value
    
    return modified_scores

def analyze_llm_suggestions(feedback_history):
    """
    Analyze feedback to learn which suggestions the LLM-Judge makes frequently
    
    Returns:
        dict: {suggestion_pattern: frequency_count}
    """
    suggestion_patterns = Counter()
    
    for feedback in feedback_history:
        suggestions = feedback.get('llm_evaluation', {}).get('suggestions', [])
        
        for suggestion in suggestions:
            # Normalize suggestion text for better pattern matching
            normalized = suggestion.lower().strip()
            
            # Extract key action patterns
            if 'sustituir' in normalized or 'reemplazar' in normalized:
                # Look for patterns like "sustituir X por Y" or "reemplazar X por Y"
                match = re.search(r'(?:sustituir|reemplazar)\s+(?:el\s+)?pictograma\s+de\s+\'([^\']+)\'\s+por\s+(?:un\s+)?pictograma\s+que\s+represe(?:nte|nta)\s+(.+)', normalized)
                if match:
                    concept_from = match.group(1).strip()
                    action_to = match.group(2).strip()
                    pattern = f"replace_{concept_from}_with_{action_to}"
                    suggestion_patterns[pattern] += 1
                else:
                    # Fallback: count general suggestion
                    suggestion_patterns[suggestion[:50]] += 1  # First 50 chars
            elif 'asegurar' in normalized or 'verificar' in normalized:
                # Look for assurance/verification patterns
                suggestion_patterns[f"verify_{normalized[:30]}"] += 1
            else:
                # Generic suggestion counting
                suggestion_patterns[suggestion[:50]] += 1
    
    return dict(suggestion_patterns)

def apply_llm_suggestions_as_postprocessing(concept, llm_suggestions, concept_results):
    """
    Apply learned LLM suggestions as post-processing rules to refine search results
    
    Args:
        concept (str): The concept we're processing
        llm_suggestions (dict): Analyzed LLM suggestions from feedback
        concept_results (list): List of pictogram results for the concept
        
    Returns:
        list: Refined results after applying LLM suggestion rules
    """
    # Create a copy to avoid modifying original
    refined_results = [item.copy() for item in concept_results]
    
    # Look for replacement patterns that match our concept
    for pattern, frequency in llm_suggestions.items():
        if pattern.startswith('replace_') and 'with_' in pattern:
            # Extract what concept we should be looking to replace and what action
            match = re.search(r'replace_(.+)_with_(.+)', pattern)
            if match:
                target_concept = match.group(1)
                suggested_action = match.group(2)
                
                # If this concept matches what the LLM suggests replacing
                if concept == target_concept:
                    # Boost scores for pictograms whose text contains the suggested action
                    for item in refined_results:
                        if suggested_action.lower() in item['text'].lower():
                            # Boost based on frequency of suggestion
                            boost_value = min(frequency * 0.15, 1.5)  # Cap at 1.5
                            item['score'] += boost_value
                            item['llm_suggestion_applied'] = True
                            item['suggestion_source'] = pattern
    
    # Re-sort by score after applying boosts
    refined_results.sort(key=lambda x: x['score'], reverse=True)
    return refined_results

def get_llm_suggestion_modifier(concept):
    """
    Convenience function to get LLM suggestion modifiers for a concept
    """
    history = load_feedback_history()
    llm_suggestions = analyze_llm_suggestions(history)
    return llm_suggestions

def search(query, top_k=5, offset=0):
    query_emb = model.encode([query], normalize_embeddings=True)[0]

    scores = np.dot(embeddings, query_emb)

    # Apply feedback-based rule improvements if we have history
    try:
        # Load feedback history and analyze for this query concept
        history = load_feedback_history()
        if history:
            concept_stats = analyze_concept_corrections(history)
            scores = apply_rule_improvements(query, concept_stats, scores, ids)
    except Exception as e:
        # If feedback processing fails, continue with original scores
        print(f"Warning: Could not apply feedback improvements: {e}")

    # Get more results than needed to handle offset properly
    # We need at least offset + top_k results to slice correctly
    num_results_needed = offset + top_k
    # But don't exceed total available results
    num_results_needed = min(num_results_needed, len(scores))
    
    # Get indices of top scores, sorted descending
    top_indices = np.argsort(scores)[-num_results_needed:][::-1]

    results = []
    for i in top_indices:
        full_text = texts[i]
        concept = extract_concept(full_text)
        results.append({
            "id": ids[i],
            "text": full_text,
            "concept": concept,
            "score": scores[i]
        })
    
    # Apply LLM suggestion post-processing if we have history
    try:
        history = load_feedback_history()
        if history:
            llm_suggestions = analyze_llm_suggestions(history)
            results = apply_llm_suggestions_as_postprocessing(query, llm_suggestions, results)
    except Exception as e:
        # If LLM suggestion processing fails, continue with current results
        print(f"Warning: Could not apply LLM suggestion improvements: {e}")
    
    # Apply offset (skip first 'offset' results)
    return results[offset:offset+top_k]

def search_sequence(concepts, top_k=3):
    sequence_results = []
    seen_ids = set()

    for concept in concepts:
        concept_results = search(concept, top_k)
        for result in concept_results:
            if result["id"] not in seen_ids:
                seen_ids.add(result["id"])
                sequence_results.append({
                    "concept": result.get("concept", concept),
                    "id": result["id"],
                    "text": result["text"],
                    "score": result["score"]
                })
                break

    return sequence_results


def search_sequence_candidates(concepts, candidate_k=5):
    """
    For each concept, return TOP candidate_k pictograms from embedding search.
    Used by LLM Generator to select the best pictogram considering full text context.
    
    Returns:
        list: [{"concept": "correr", "candidates": [{id, concept, text, score}, ...]}, ...]
    """
    candidates_per_concept = []
    
    for concept in concepts:
        # Get top-k candidates for this concept
        candidate_results = search(concept, top_k=candidate_k)
        
        candidates_list = []
        for result in candidate_results:
            candidates_list.append({
                "id": int(result["id"]),
                "concept": result.get("concept", concept),
                "text": result["text"],
                "score": float(result["score"])
            })
        
        candidates_per_concept.append({
            "concept": concept,
            "candidates": candidates_list
        })
        
        # Print top candidates for debugging
        print(f"\n=== Top {candidate_k} candidates for concept: '{concept}' ===")
        for i, cand in enumerate(candidates_list, 1):
            print(f"  {i}. ID: {cand['id']}, Concept: '{cand['concept']}', Score: {cand['score']:.4f}")
    
    return candidates_per_concept