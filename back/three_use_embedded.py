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

def apply_rule_improvements(concept, concept_stats, human_stats, base_scores, id_list):
    """
    Apply learned rules from BOTH LLM suggestions AND human corrections
    
    Args:
        concept (str): The concept we're processing
        concept_stats (dict): LLM suggestion statistics
        human_stats (dict): Human correction statistics
        base_scores (np.array): Original similarity scores
        id_list (np.array): Corresponding pictogram IDs
        
    Returns:
        np.array: Modified scores
    """
    modified_scores = base_scores.copy()
    
    # Create ID to index mapping for fast lookup
    id_to_index = {id_val: idx for idx, id_val in enumerate(id_list)}
    
    # Apply LLM suggestion rules first
    if concept in concept_stats:
        stats = concept_stats[concept]
        # Boost preferred pictograms
        for pictogram_id in stats.get('preferred_pictogram_ids', []):
            if pictogram_id in id_to_index:
                idx = id_to_index[pictogram_id]
                boost_value = min(stats.get('confidence', 1) * 0.1, 2.0)
                modified_scores[idx] += boost_value
        
        # Suppress rejected pictograms
        for pictogram_id in stats.get('rejected_pictogram_ids', []):
            if pictogram_id in id_to_index:
                idx = id_to_index[pictogram_id]
                suppress_value = max(-stats.get('confidence', 1) * 0.1, -2.0)
                modified_scores[idx] += suppress_value
    
    # NEW: Apply human correction rules
    if concept in human_stats:
        stats = human_stats[concept]
        # Boost human-preferred pictograms
        for pid, count in stats.get('preferred_ids', []):
            if pid in id_to_index:
                idx = id_to_index[pid]
                boost_value = min(count * 0.1, 2.0)
                modified_scores[idx] += boost_value
        
        # Suppress human-rejected pictograms
        for pid, count in stats.get('rejected_ids', []):
            if pid in id_to_index:
                idx = id_to_index[pid]
                suppress_value = max(-count * 0.1, -2.0)
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
            
            # NEW: Detect "Añadir preposición X" patterns
            elif 'añadir' in normalized or 'agregar' in normalized:
                match = re.search(r'(?:añadir|agregar)\s+(?:la\s+)?preposición\s+\'([^\']+)\'', normalized)
                if match:
                    preposition = match.group(1)
                    pattern = f"add_preposition_{preposition}"
                    suggestion_patterns[pattern] += 1
            
            # NEW: Detect "Eliminar concepto X" patterns
            elif 'eliminar' in normalized or 'quitar' in normalized:
                match = re.search(r'(?:eliminar|quitar)\s+(?:el\s+)?pictograma\s+de\s+\'([^\']+)\'', normalized)
                if match:
                    concept_to_remove = match.group(1)
                    pattern = f"remove_{concept_to_remove}"
                    suggestion_patterns[pattern] += 1
            
            # NEW: Detect "Cambiar orden: X → Y" patterns
            elif 'orden' in normalized or 'reordenar' in normalized:
                match = re.search(r'(?:cambiar|reordenar)\s+orden:?\s*(.+)', normalized)
                if match:
                    order_desc = match.group(1).strip()[:30]
                    pattern = f"reorder_{order_desc}"
                    suggestion_patterns[pattern] += 1
            
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
        
        # NEW: Handle "add_preposition_X" patterns
        elif pattern.startswith('add_preposition_'):
            preposition = pattern.replace('add_preposition_', '')
            if 'preposición' in concept.lower() or 'preposición' in concept.lower():
                # Boost pictograms that include this preposition
                for item in refined_results:
                    if preposition.lower() in item['text'].lower():
                        boost_value = min(frequency * 0.15, 1.5)
                        item['score'] += boost_value
                        item['llm_suggestion_applied'] = True
        
        # NEW: Handle "remove_X" patterns
        elif pattern.startswith('remove_'):
            concept_to_remove = pattern.replace('remove_', '')
            if concept == concept_to_remove:
                # Suppress all results (they'll be filtered later)
                for item in refined_results:
                    item['score'] -= 5.0  # Strong suppression
                    item['llm_suggestion_applied'] = True
        
        # NEW: Handle "reorder_X" patterns (more complex, might need sequence-level handling)
        elif pattern.startswith('reorder_'):
            # This would need sequence-level post-processing, skip for now
            pass
    
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


def analyze_corrections_from_feedback(feedback_history):
    """
    Analyze human corrections to learn:
    - Which pictogram IDs humans prefer for each concept
    - Which pictogram IDs humans reject
    - Common correction patterns
    
    Returns:
        dict: {concept: {'preferred_ids': [id1, id2], 'rejected_ids': [id3, id4], 'confidence': int}}
    """
    concept_stats = defaultdict(lambda: {
        'preferred': Counter(),
        'rejected': Counter()
    })
    
    for feedback in feedback_history:
        # Get original prediction and human correction
        original_sequence = feedback.get('system_generation', {}).get('sequence', [])
        corrected_sequence = feedback.get('user_modifications', {}).get('final_sequence', [])
        
        if not corrected_sequence:
            continue
        
        # Build ID → item maps
        original_map = {item['id']: item for item in original_sequence}
        corrected_map = {item['id']: item for item in corrected_sequence}
        
        # Process each concept
        all_concepts = set(list(original_map.keys()) + list(corrected_map.keys()))
        
        for pid in all_concepts:
            orig_item = original_map.get(pid)
            corr_item = corrected_map.get(pid)
            
            # Determine concept for this pictogram
            concept = None
            if orig_item:
                concept = orig_item['concept']
            elif corr_item:
                concept = corr_item['concept']
            
            if not concept:
                continue
            
            # If human changed the pictogram for this concept
            if orig_item and corr_item:
                if orig_item['id'] != corr_item['id']:
                    # Human rejected orig_id, preferred corr_id
                    concept_stats[concept]['rejected'][orig_item['id']] += 1
                    concept_stats[concept]['preferred'][corr_item['id']] += 1
            
            elif corr_item and not orig_item:
                # Pictogram was added by human (wasn't in original)
                concept_stats[concept]['preferred'][corr_item['id']] += 1
            
            elif orig_item and not corr_item:
                # Pictogram was removed by human (was in original but not corrected)
                concept_stats[concept]['rejected'][orig_item['id']] += 1
    
    # Convert to serializable format
    result = {}
    for concept, stats in concept_stats.items():
        if stats['preferred'] or stats['rejected']:
            result[concept] = {
                'preferred_ids': [pid for pid, _ in stats['preferred'].most_common(5)],
                'rejected_ids': [pid for pid, _ in stats['rejected'].most_common(5)],
                'confidence': sum(stats['preferred'].values()) + sum(stats['rejected'].values())
            }
    
    return result

def search(query, top_k=5, offset=0):
    query_emb = model.encode([query], normalize_embeddings=True)[0]

    scores = np.dot(embeddings, query_emb)

    # Apply feedback-based rule improvements if we have history
    try:
        history = load_feedback_history()
        if history:
            # Combine both types of feedback
            concept_stats = analyze_concept_corrections(history)
            human_stats = analyze_corrections_from_feedback(history)
            
            # Apply both types of rules in one call
            scores = apply_rule_improvements(query, concept_stats, human_stats, scores, ids)
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
        pictogram_concept = extract_concept(full_text)  
        results.append({
            "id": ids[i],
            "text": full_text,
            "concept": pictogram_concept,
            "extracted_query": query,
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
        list: [{"concept": "correr" (query concept), "candidates": [{id, concept, text, description, score}, ...]}, ...]
    """
    candidates_per_concept = []
    
    for concept in concepts:
        # Get top-k candidates for this concept
        candidate_results = search(concept, top_k=candidate_k)
        
        candidates_list = []
        for result in candidate_results:
            candidates_list.append({
                "id": int(result["id"]),
                "concept": result.get("concept", "Unknown"),  # ARASAAC pictogram concept (e.g., "Comer")
                "query_concept": concept,  # What user searched (e.g., "tomando")
                "text": result["text"],  # Full ARASAAC text
                "description": result["text"],  # Same as text, for clarity in prompt
                "score": float(result["score"])
            })
        
        candidates_per_concept.append({
            "concept": concept,
            "candidates": candidates_list
        })
        
        # Debug print
        print(f"\n=== Top {candidate_k} candidates for concept: '{concept}' ===")
        for i, cand in enumerate(candidates_list, 1):
            print(f"  {i}. ID: {cand['id']}, Pictogram concept: '{cand['concept']}', Score: {cand['score']:.4f}")
    
    return candidates_per_concept