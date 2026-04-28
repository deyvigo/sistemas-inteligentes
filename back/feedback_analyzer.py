import json
import re
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np

FEEDBACK_DIR = Path("./feedback_logs")

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

def apply_rule_improvements(concept, concept_stats):
    """
    Apply learned rules to adjust pictogram selection for a concept
    
    Args:
        concept (str): The concept we're processing
        concept_stats (dict): The analyzed statistics from feedback
        
    Returns:
        dict: Modifiers to apply to search results (boost/suppress certain IDs)
    """
    if concept not in concept_stats:
        return {}  # No learned rules for this concept
    
    stats = concept_stats[concept]
    modifiers = {
        'boost': {},
        'suppress': {}
    }
    
    # Boost preferred pictograms
    for pictogram_id in stats['preferred_pictogram_ids']:
        # Higher confidence = stronger boost
        boost_value = min(stats['confidence'] * 0.1, 2.0)  # Cap at 2.0
        modifiers['boost'][pictogram_id] = boost_value
    
    # Suppress rejected pictograms
    for pictogram_id in stats['rejected_pictogram_ids']:
        # Higher confidence = stronger suppression (negative boost)
        suppress_value = max(-stats['confidence'] * 0.1, -2.0)  # Cap at -2.0
        modifiers['suppress'][pictogram_id] = suppress_value
    
    return modifiers

def get_pictogram_search_modifier(concept):
    """
    Convenience function to get search modifiers for a concept
    Loads feedback history and applies rules
    """
    # In a production system, we'd cache this to avoid reloading every time
    # For now, we'll load on demand
    history = load_feedback_history()
    concept_stats = analyze_concept_corrections(history)
    return apply_rule_improvements(concept, concept_stats)

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

def apply_llm_suggestions_as_postprocessing(concept, llm_suggestions, original_sequence):
    """
    Apply learned LLM suggestions as post-processing rules to refine output
    
    Args:
        concept (str): The concept we're processing
        llm_suggestions (dict): Analyzed LLM suggestions from feedback
        original_sequence (list): Original sequence of pictograms from system
        
    Returns:
        list: Refined sequence after applying LLM suggestion rules
    """
    # Create a copy to avoid modifying original
    refined_sequence = [item.copy() for item in original_sequence]
    
    # Look for replacement patterns that match our concept
    for pattern, frequency in llm_suggestions.items():
        if pattern.startswith('replace_') and f'with_{concept}' in pattern:
            # Extract what concept we should be looking to replace
            match = re.search(r'replace_(.+)_with_.+', pattern)
            if match:
                target_concept = match.group(1)
                
                # Find items in sequence matching target_concept
                for item in refined_sequence:
                    if item['concept'] == target_concept:
                        # Apply the LLM suggestion: boost scores for pictograms that represent the action
                        # In a real implementation, we would look up pictograms that match the suggested action
                        # For now, we'll just mark that this item should be reviewed
                        item['llm_suggestion_applied'] = True
                        item['suggestion_source'] = pattern
                        item['suggestion_frequency'] = frequency
                        
    return refined_sequence

def get_llm_suggestion_modifier(concept):
    """
    Convenience function to get LLM suggestion modifiers for a concept
    """
    history = load_feedback_history()
    llm_suggestions = analyze_llm_suggestions(history)
    # Return suggestions for this concept (would need original sequence to apply fully)
    return llm_suggestions

if __name__ == "__main__":
    # Test the analyzer
    history = load_feedback_history()
    print(f"Loaded {len(history)} feedback entries")
    
    if history:
        stats = analyze_concept_corrections(history)
        print(f"Learned rules for {len(stats)} concepts:")
        for concept, rule in list(stats.items())[:3]:  # Show first 3
            print(f"  {concept}:")
            print(f"    Preferred: {rule['preferred_pictogram_ids']}")
            print(f"    Rejected: {rule['rejected_pictogram_ids']}")
            print(f"    Confidence: {rule['confidence']}")
        
        # Test LLM suggestion analysis
        llm_suggestions = analyze_llm_suggestions(history)
        print(f"\nLLM suggestion patterns:")
        for pattern, count in list(llm_suggestions.items())[:5]:  # Show top 5
            print(f"  {pattern}: {count}")
    else:
        print("No feedback history found")