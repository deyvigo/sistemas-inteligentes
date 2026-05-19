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


# ─── Override system: force-include pictograms learned from human corrections ───

EMBEDDING_SIM_THRESHOLD = 0.7

def _get_embedding_model():
    """Lazy-load embedding model from three_use_embedded to avoid circular imports"""
    from three_use_embedded import model
    return model


def _get_concept_text(item):
    """Extract the best text for embedding matching from a pictogram item"""
    return (item.get('query_concept', '') or
            item.get('extracted_query', '') or
            item.get('concept', '')).lower().strip()


def _find_best_concept_match(corr_text, concepts_list, extracted_embeddings=None):
    """
    Find which extracted concept best matches a corrected item using embedding similarity.
    
    Returns:
        str: The matched concept string, or None if below threshold (new concept)
    """
    if not concepts_list or not corr_text:
        return None

    lower_concepts = [c.lower().strip() for c in concepts_list]

    # Exact match → no embedding needed
    if corr_text in lower_concepts:
        idx = lower_concepts.index(corr_text)
        return concepts_list[idx]

    if len(concepts_list) == 1:
        return concepts_list[0]

    # Use embeddings
    if extracted_embeddings is None:
        model = _get_embedding_model()
        extracted_embeddings = model.encode(concepts_list, normalize_embeddings=True)

    model = _get_embedding_model()
    corr_emb = model.encode([corr_text], normalize_embeddings=True)[0]
    similarities = np.dot(extracted_embeddings, corr_emb)
    best_idx = np.argmax(similarities)
    best_sim = similarities[best_idx]

    # Debug: print all similarity scores
    print(f"[EMBEDDING MATCH] corr_text='{corr_text}' vs {concepts_list}")
    for idx, (concept, sim) in enumerate(zip(concepts_list, similarities)):
        marker = " ← BEST" if idx == best_idx else ""
        print(f"  sim({corr_text}, {concept}) = {sim:.4f}{marker}")
    print(f"  threshold={EMBEDDING_SIM_THRESHOLD}, best_sim={best_sim:.4f}, result={'MATCH' if best_sim >= EMBEDDING_SIM_THRESHOLD else 'NEW_CONCEPT'}")

    if best_sim >= EMBEDDING_SIM_THRESHOLD:
        return concepts_list[best_idx]
    return None  # below threshold → new concept


# ─── Checkpoint system: incremental correction table ───

CHECKPOINT_FILE = FEEDBACK_DIR / "_correction_checkpoint.json"

def _load_checkpoint():
    if CHECKPOINT_FILE.exists():
        try:
            with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[CHECKPOINT] Error loading checkpoint: {e}")
    return None

def _save_checkpoint(processed_files, corrections):
    serializable = {}
    for key, data in corrections.items():
        entry = {}
        if data.get("candidates"):
            entry["candidates"] = {str(k): v for k, v in data["candidates"].items()}
        if data.get("rejected"):
            entry["rejected"] = {str(k): v for k, v in data["rejected"].items()}
        if data.get("missing_count"):
            entry["missing_count"] = data["missing_count"]
        if data.get("last_seen"):
            entry["last_seen"] = data["last_seen"]
        serializable[key] = entry
    try:
        with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
            json.dump({"processed_files": sorted(processed_files), "corrections": serializable}, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"[CHECKPOINT] Error saving checkpoint: {e}")

def _rehydrate_corrections(checkpoint_data):
    corrections = defaultdict(lambda: {"candidates": Counter(), "rejected": Counter(), "missing_count": 0, "last_seen": None})
    for key, data in checkpoint_data.get("corrections", {}).items():
        if data.get("candidates"):
            corrections[key]["candidates"] = Counter({int(k): v for k, v in data["candidates"].items()})
        if data.get("rejected"):
            corrections[key]["rejected"] = Counter({int(k): v for k, v in data["rejected"].items()})
        if data.get("missing_count"):
            corrections[key]["missing_count"] = data["missing_count"]
        if data.get("last_seen"):
            corrections[key]["last_seen"] = data["last_seen"]
    return corrections


def _process_feedback_entries(feedback_history, corrections=None):
    """
    Process feedback entries and accumulate into corrections dict.
    
    Args:
        feedback_history: List of feedback dicts
        corrections: Existing accumulator dict (defaultdict), or None to create new
    
    Returns:
        defaultdict: Accumulated corrections
    """
    if corrections is None:
        corrections = defaultdict(lambda: {"candidates": Counter(), "rejected": Counter(), "missing_count": 0, "last_seen": None})

    for feedback in feedback_history:
        original_sequence = feedback.get('system_generation', {}).get('sequence', [])
        corrected_sequence = feedback.get('user_modifications', {}).get('final_sequence', [])
        concepts_extracted = feedback.get('input', {}).get('concepts_extracted', [])
        timestamp = feedback.get('timestamp', '')

        if not corrected_sequence:
            continue

        # ── Pre-calculate: which extracted concepts lack a pictogram in the original? ──
        concepts_with_pictogram = {_get_concept_text(item).lower().strip() for item in original_sequence}
        concepts_without_pictogram = [
            c for c in concepts_extracted
            if c.lower().strip() not in concepts_with_pictogram
        ]
        target_embeddings = _get_embedding_model().encode(concepts_without_pictogram, normalize_embeddings=True) if concepts_without_pictogram else None
        print(f"[BUILD_CORR_TABLE] concepts_extracted={concepts_extracted}")
        print(f"[BUILD_CORR_TABLE] concepts_with_pictogram={concepts_with_pictogram}")
        print(f"[BUILD_CORR_TABLE] concepts_without_pictogram={concepts_without_pictogram}")
        print(f"[BUILD_CORR_TABLE] original len={len(original_sequence)}, corrected len={len(corrected_sequence)}")

        lower_extracted = [c.lower().strip() for c in concepts_extracted]

        for i, corr_item in enumerate(corrected_sequence):
            corr_id = int(corr_item['id'])
            corr_text = _get_concept_text(corr_item)
            if not corr_text:
                continue

            orig_item = original_sequence[i] if i < len(original_sequence) else None
            if orig_item and orig_item.get('id') == corr_id:
                continue  # no change

            if orig_item:
                # ── Replacement path ──
                # Skip if corrected ID is a displacement (reorder/delete)
                if any(o.get('id') == corr_id and j != i for j, o in enumerate(original_sequence)):
                    print(f"[CORR] Position {i}: DISPLACEMENT (corr_id {corr_id} exists in original), skip")
                    continue

                orig_item_text = _get_concept_text(orig_item)
                # Skip cosmetic changes (same concept, different pictogram)
                if orig_item_text == corr_text:
                    print(f"[CORR] Position {i}: COSMETIC '{orig_item_text}'→'{corr_text}', skip")
                    continue

                # Check insertion: original item's ID exists elsewhere in corrected
                orig_id = original_sequence[i].get('id')
                is_insertion = any(
                    c.get('id') == orig_id and j != i
                    for j, c in enumerate(corrected_sequence)
                )

                if is_insertion:
                    print(f"[CORR] Position {i}: INSERTION id={corr_id} ({corr_text}) — orig {orig_id} moved elsewhere")
                    # Treat as new item (same logic as addition)
                    # 1) Exact match against ALL extracted
                    if corr_text in lower_extracted:
                        idx = lower_extracted.index(corr_text)
                        key = concepts_extracted[idx]
                        print(f"  → EXACT MATCH: '{corr_text}' → '{key}'")
                    # 2) Embedding against concepts without pictogram
                    elif concepts_without_pictogram:
                        key = _find_best_concept_match(corr_text, concepts_without_pictogram, target_embeddings)
                        if key:
                            print(f"  → EMBEDDING: '{corr_text}' → '{key}'")
                        else:
                            key = corr_text
                            print(f"  → NEW CONCEPT: '{corr_text}'")
                    else:
                        key = corr_text
                        print(f"  → NEW CONCEPT: '{corr_text}' (no concepts_without_pictogram)")
                    corrections[key]['candidates'][corr_id] += 1
                    if timestamp:
                        corrections[key]['last_seen'] = timestamp
                    continue

                # Genuine replacement: match original item's text against extracted concepts
                print(f"[CORR] Position {i}: REPLACEMENT orig='{orig_item_text}' → corr id={corr_id} ({corr_text})")
                key = _find_best_concept_match(orig_item_text, concepts_extracted) or orig_item_text
                if key:
                    corrections[key]['candidates'][corr_id] += 1
                    print(f"  → key='{key}'")
                    if timestamp:
                        corrections[key]['last_seen'] = timestamp
                continue

            # ── Addition path (item beyond original length) ──
            # Skip if this ID was displaced from the original (just moved)
            if any(o.get('id') == corr_id for o in original_sequence):
                print(f"[CORR] Position {i}: DISPLACED ID {corr_id} in original, skip")
                continue

            print(f"[CORR] Position {i}: ADDITION id={corr_id} ({corr_text})")

            # 1) Exact match against ALL extracted
            if corr_text in lower_extracted:
                idx = lower_extracted.index(corr_text)
                key = concepts_extracted[idx]
                print(f"  → EXACT MATCH: '{corr_text}' → '{key}'")
            # 2) Embedding against concepts without pictogram
            elif concepts_without_pictogram:
                key = _find_best_concept_match(corr_text, concepts_without_pictogram, target_embeddings)
                if key:
                    print(f"  → EMBEDDING: '{corr_text}' → '{key}'")
                else:
                    key = corr_text
                    print(f"  → NEW CONCEPT: '{corr_text}'")
            else:
                key = corr_text
                print(f"  → NEW CONCEPT: '{corr_text}' (no concepts_without_pictogram)")

            corrections[key]['candidates'][corr_id] += 1
            if timestamp:
                corrections[key]['last_seen'] = timestamp

    # ── LLM Judge feedback: incorrect_pictograms and missing_concepts ──
    for feedback in feedback_history:
        judge = feedback.get('llm_evaluation', {})
        timestamp = feedback.get('timestamp', '')
        orig_seq = feedback.get('system_generation', {}).get('sequence', [])
        concepts_extracted = feedback.get('input', {}).get('concepts_extracted', [])

        # incorrect_pictograms: find the pictogram ID used for each flagged concept
        for item in judge.get('incorrect_pictograms', []):
            concept = item.get('concept', '').strip()
            if not concept:
                continue
            concept_lower = concept.lower()
            # Find which original sequence item matches this concept by ARASAAC concept name
            for seq_item in orig_seq:
                if seq_item.get('concept', '').lower().strip() == concept_lower:
                    rejected_id = int(seq_item['id'])
                    # Use extracted_query as the key (the extracted concept this pictogram was assigned to)
                    item_concept = seq_item.get('extracted_query', '').lower().strip() or concept
                    key = _find_best_concept_match(item_concept, concepts_extracted) or item_concept
                    corrections[key]['rejected'][rejected_id] += 1
                    if timestamp:
                        corrections[key]['last_seen'] = timestamp
                    print(f"[JUDGE] rejected_id={rejected_id} for concept='{concept}' → key='{key}'")
                    break

        # missing_concepts: accumulate count per concept
        for missing in judge.get('missing_concepts', []):
            missing = missing.strip()
            if missing:
                key = _find_best_concept_match(missing, concepts_extracted) or missing
                corrections[key]['missing_count'] += 1
                if timestamp:
                    corrections[key]['last_seen'] = timestamp
                print(f"[JUDGE] missing_count++ for concept='{missing}' → key='{key}' (now {corrections[key]['missing_count']})")

    return corrections


def _filter_corrections(corrections, min_confidence=1):
    """Filter accumulated corrections by min_confidence and return plain dict."""
    result = {}
    for key, data in corrections.items():
        has_candidates = any(c >= min_confidence for c in data['candidates'].values())
        has_rejected = any(c >= min_confidence for c in data['rejected'].values())
        has_missing = data['missing_count'] > 0
        if has_candidates or has_rejected or has_missing:
            entry = {}
            if has_candidates:
                entry['candidates'] = {int(i): c for i, c in data['candidates'].items() if c >= min_confidence}
            if has_rejected:
                entry['rejected'] = {int(i): c for i, c in data['rejected'].items() if c >= min_confidence}
            if has_missing:
                entry['missing_count'] = data['missing_count']
            entry['last_seen'] = data.get('last_seen') or ''
            result[key] = entry
    return result


def build_correction_table(feedback_history, min_confidence=1):
    """
    Build a correction mapping from human + LLM feedback using embedding similarity.
    Delegates to _process_feedback_entries and _filter_corrections.
    
    Returns:
        dict: {
          concept: {
            "candidates": {id: count, ...},
            "rejected": {id: count, ...},
            "missing_count": int,
            "last_seen": str
          }
        }
    """
    corrections = _process_feedback_entries(feedback_history)
    return _filter_corrections(corrections, min_confidence)


def load_and_build_correction_table(min_confidence=1):
    """
    Incrementally load feedback files using a checkpoint on disk.
    
    If checkpoint does not exist → process ALL files from scratch.
    Otherwise → only process new files since last checkpoint.
    
    Returns:
        dict: Same format as build_correction_table
    """
    checkpoint = _load_checkpoint()

    if checkpoint is None:
        print("[CHECKPOINT] No checkpoint found, processing all feedback from scratch")
        history = load_feedback_history()
        corrections = _process_feedback_entries(history)
        processed_files = {f.name for f in FEEDBACK_DIR.glob("feedback_*.json")}
        _save_checkpoint(processed_files, corrections)
        return _filter_corrections(corrections, min_confidence)

    processed_files = set(checkpoint.get("processed_files", []))
    corrections = _rehydrate_corrections(checkpoint)

    all_files = sorted(FEEDBACK_DIR.glob("feedback_*.json"))
    new_files = [f for f in all_files if f.name not in processed_files]

    if new_files:
        print(f"[CHECKPOINT] {len(new_files)} new file(s) to process: {[f.name for f in new_files]}")
        new_history = []
        for f in new_files:
            try:
                with open(f, "r", encoding="utf-8") as fh:
                    new_history.append(json.load(fh))
                processed_files.add(f.name)
            except Exception as e:
                print(f"[CHECKPOINT] Error loading {f}: {e}")
        corrections = _process_feedback_entries(new_history, corrections)
        _save_checkpoint(processed_files, corrections)
    else:
        print("[CHECKPOINT] No new feedback files, using cached table")

    return _filter_corrections(corrections, min_confidence)


def get_overrides(concept, correction_table=None, min_confidence=1):
    """
    Check if there are learned overrides for a concept.
    
    Args:
        concept: The concept to look up
        correction_table: Pre-built table (loads from feedback if None)
        min_confidence: Minimum corrections before override activates
    
    Returns:
        dict with {"overrides": {id: count, ...}, "last_seen": str} or None
    """
    if correction_table is None:
        correction_table = load_and_build_correction_table()

    key = concept.lower().strip()
    if key in correction_table:
        entry = correction_table[key]
        candidates = entry.get('candidates', {})
        overrides = {int(i): c for i, c in candidates.items() if c >= min_confidence}
        rejected = {int(i): c for i, c in entry.get('rejected', {}).items() if c >= min_confidence}
        missing_count = entry.get('missing_count', 0)
        if overrides or rejected or missing_count:
            result = {'last_seen': entry.get('last_seen', '')}
            if overrides:
                result['overrides'] = overrides
            if rejected:
                result['rejected'] = rejected
            if missing_count:
                result['missing_count'] = missing_count
            return result
    return None


# ─── Advanced analysis functions (migrated from three_use_embedded.py) ───

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
        original_sequence = feedback.get('system_generation', {}).get('sequence', [])
        corrected_sequence = feedback.get('user_modifications', {}).get('final_sequence', [])
        
        if not corrected_sequence:
            continue
        
        original_map = {item['id']: item for item in original_sequence}
        corrected_map = {item['id']: item for item in corrected_sequence}
        
        all_concepts = set(list(original_map.keys()) + list(corrected_map.keys()))
        
        for pid in all_concepts:
            orig_item = original_map.get(pid)
            corr_item = corrected_map.get(pid)
            
            concept = None
            if orig_item:
                concept = orig_item.get('concept')
            elif corr_item:
                concept = corr_item.get('concept')
            
            if not concept:
                continue
            
            if orig_item and corr_item:
                if orig_item.get('id') != corr_item.get('id'):
                    concept_stats[concept]['rejected'][orig_item['id']] += 1
                    concept_stats[concept]['preferred'][corr_item['id']] += 1
            elif corr_item and not orig_item:
                concept_stats[concept]['preferred'][corr_item['id']] += 1
            elif orig_item and not corr_item:
                concept_stats[concept]['rejected'][orig_item['id']] += 1
    
    result = {}
    for concept, stats in concept_stats.items():
        if stats['preferred'] or stats['rejected']:
            result[concept] = {
                'preferred_ids': [pid for pid, _ in stats['preferred'].most_common(5)],
                'rejected_ids': [pid for pid, _ in stats['rejected'].most_common(5)],
                'confidence': sum(stats['preferred'].values()) + sum(stats['rejected'].values())
            }
    
    return result


def apply_rule_improvements(concept, concept_stats, human_stats, base_scores, id_list):
    """
    Apply learned rules from BOTH LLM suggestions AND human corrections
    to modify embedding similarity scores.
    
    Args:
        concept: The concept being processed
        concept_stats: LLM suggestion statistics
        human_stats: Human correction statistics
        base_scores: Original similarity scores (np.array)
        id_list: Corresponding pictogram IDs (np.array)
        
    Returns:
        np.array: Modified scores
    """
    modified_scores = base_scores.copy()
    id_to_index = {id_val: idx for idx, id_val in enumerate(id_list)}
    
    # Apply LLM suggestion rules
    if concept in concept_stats:
        stats = concept_stats[concept]
        for pictogram_id in stats.get('preferred_pictogram_ids', []):
            if pictogram_id in id_to_index:
                idx = id_to_index[pictogram_id]
                boost_value = min(stats.get('confidence', 1) * 0.1, 2.0)
                modified_scores[idx] += boost_value
        
        for pictogram_id in stats.get('rejected_pictogram_ids', []):
            if pictogram_id in id_to_index:
                idx = id_to_index[pictogram_id]
                suppress_value = max(-stats.get('confidence', 1) * 0.1, -2.0)
                modified_scores[idx] += suppress_value
    
    # Apply human correction rules
    if concept in human_stats:
        stats = human_stats[concept]
        for pid in stats.get('preferred_ids', []):
            if pid in id_to_index:
                idx = id_to_index[pid]
                boost_value = min(stats.get('confidence', 1) * 0.1, 2.0)
                modified_scores[idx] += boost_value
        
        for pid in stats.get('rejected_ids', []):
            if pid in id_to_index:
                idx = id_to_index[pid]
                suppress_value = max(-stats.get('confidence', 1) * 0.1, -2.0)
                modified_scores[idx] += suppress_value
    
    return modified_scores


def apply_llm_suggestions_as_postprocessing(concept, llm_suggestions, concept_results):
    """
    Apply learned LLM suggestions as post-processing rules to refine search results.
    
    Args:
        concept: The concept being processed
        llm_suggestions: Analyzed LLM suggestions from feedback
        concept_results: List of pictogram results for the concept
    
    Returns:
        list: Refined results after applying LLM suggestion rules
    """
    refined_results = [item.copy() for item in concept_results]
    
    for pattern, frequency in llm_suggestions.items():
        if pattern.startswith('replace_') and 'with_' in pattern:
            match = re.search(r'replace_(.+)_with_(.+)', pattern)
            if match:
                target_concept = match.group(1)
                suggested_action = match.group(2)
                
                if concept == target_concept:
                    for item in refined_results:
                        if suggested_action.lower() in item.get('text', '').lower():
                            boost_value = min(frequency * 0.15, 1.5)
                            item['score'] += boost_value
                            item['llm_suggestion_applied'] = True
                            item['suggestion_source'] = pattern
        
        elif pattern.startswith('add_preposition_'):
            preposition = pattern.replace('add_preposition_', '')
            for item in refined_results:
                if preposition.lower() in item.get('text', '').lower():
                    boost_value = min(frequency * 0.15, 1.5)
                    item['score'] += boost_value
                    item['llm_suggestion_applied'] = True
        
        elif pattern.startswith('remove_'):
            concept_to_remove = pattern.replace('remove_', '')
            if concept == concept_to_remove:
                for item in refined_results:
                    item['score'] -= 5.0
                    item['llm_suggestion_applied'] = True
    
    refined_results.sort(key=lambda x: x['score'], reverse=True)
    return refined_results


# ─── Statistics endpoint helpers ───

def get_feedback_stats():
    """
    Compute aggregate statistics from all feedback entries.
    
    Returns:
        dict: Statistics including total entries, average score, top corrections, etc.
    """
    history = load_feedback_history()
    
    total_entries = len(history)
    if total_entries == 0:
        return {
            "total_feedback_entries": 0,
            "average_judge_score": 0,
            "most_corrected_concepts": [],
            "total_learned_rules": 0,
            "top_llm_suggestions": [],
            "feedback_over_time": []
        }
    
    # Collect judge scores
    scores = []
    concept_correction_count = Counter()
    all_llm_suggestions = Counter()
    
    for feedback in history:
        score = feedback.get('llm_evaluation', {}).get('score', 0)
        if score:
            scores.append(score)
        
        # Track which concepts were corrected
        user_mods = feedback.get('user_modifications', {}).get('actions_taken', {})
        added = user_mods.get('addedPictogramIds', [])
        reorder_details = user_mods.get('reorder_details', [])
        if added or reorder_details:
            # Find which concepts were affected
            original_seq = feedback.get('system_generation', {}).get('sequence', [])
            corrected_seq = feedback.get('user_modifications', {}).get('final_sequence', [])
            for item in original_seq:
                concept_correction_count[item.get('concept', 'unknown')] += 1
            for item in corrected_seq:
                concept_correction_count[item.get('concept', 'unknown')] += 1
        
        # Collect LLM suggestions
        suggestions = feedback.get('llm_evaluation', {}).get('suggestions', [])
        for sug in suggestions:
            all_llm_suggestions[sug[:60]] += 1
    
    avg_score = round(sum(scores) / len(scores), 2) if scores else 0
    
    # Build correction table (uses checkpoint)
    correction_table = load_and_build_correction_table(min_confidence=1)
    
    # Feedback over time (by date)
    date_counts = Counter()
    for feedback in history:
        ts = feedback.get('timestamp', '')
        if ts:
            date = ts[:10]  # YYYY-MM-DD
            date_counts[date] += 1
    
    return {
        "total_feedback_entries": total_entries,
        "average_judge_score": avg_score,
        "most_corrected_concepts": [c for c, _ in concept_correction_count.most_common(10)],
        "total_learned_rules": len(correction_table),
        "learned_rules": correction_table,
        "top_llm_suggestions": [s for s, _ in all_llm_suggestions.most_common(10)],
        "feedback_over_time": [{"date": d, "count": c} for d, c in sorted(date_counts.items())]
    }


if __name__ == "__main__":
    # Test the analyzer
    history = load_feedback_history()
    print(f"Loaded {len(history)} feedback entries")
    
    if history:
        stats = analyze_concept_corrections(history)
        print(f"Learned rules for {len(stats)} concepts:")
        for concept, rule in list(stats.items())[:3]:
            print(f"  {concept}:")
            print(f"    Preferred: {rule['preferred_pictogram_ids']}")
            print(f"    Rejected: {rule['rejected_pictogram_ids']}")
            print(f"    Confidence: {rule['confidence']}")
        
        llm_suggestions = analyze_llm_suggestions(history)
        print(f"\nLLM suggestion patterns:")
        for pattern, count in list(llm_suggestions.items())[:5]:
            print(f"  {pattern}: {count}")
        
        # Test new functions
        print(f"\n--- Override system ---")
        correction_table = build_correction_table(history)
        print(f"Correction table: {len(correction_table)} entries")
        for concept, data in list(correction_table.items())[:5]:
            print(f"  {concept}: candidates={data['candidates']}, last_seen={data['last_seen']}")
        print()
        for concept in ["tomar", "gato", "leche", "yo"]:
            ov = get_overrides(concept)
            print(f"  get_overrides('{concept}'): {ov}")
        
        print(f"\n--- Feedback stats ---")
        fb_stats = get_feedback_stats()
        print(f"Total entries: {fb_stats['total_feedback_entries']}")
        print(f"Average judge score: {fb_stats['average_judge_score']}")
        print(f"Learned rules: {fb_stats['total_learned_rules']}")
    else:
        print("No feedback history found")