import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

def load_feedback_history():
    """Load all feedback entries from feedback_logs directory"""
    FEEDBACK_DIR = Path("./feedback_logs")
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


def detect_recurring_errors(feedback_history) -> Dict[str, int]:
    """
    Find patterns in LLM Judge errors vs human corrections.
    Used for Strategy C: Prompt Refinement.
    
    Returns:
        dict: {error_pattern: frequency_count}
    """
    error_patterns = Counter()
    
    for feedback in feedback_history:
        judge_output = feedback.get('llm_evaluation', {})
        human_corrections = feedback.get('user_modifications', {}).get('final_sequence', [])
        
        # 1. Detect when Judge suggested something but human didn't follow
        judge_suggestions = judge_output.get('suggestions', [])
        for suggestion in judge_suggestions:
            normalized = suggestion.lower().strip()
            
            # Check if human followed this suggestion
            if not was_suggestion_followed(suggestion, human_corrections):
                error_patterns[f"judge_suggestion_not_followed: {normalized[:50]}"] += 1
        
        # 2. Detect recurring incorrect pictogram issues
        incorrect_picts = judge_output.get('incorrect_pictograms', [])
        for item in incorrect_picts:
            concept = item.get('concept', 'unknown')
            error_patterns[f"recurring_incorrect: {concept}"] += 1
        
        # 3. Detect ordering issues that persist
        ordering_issues = judge_output.get('ordering_issues', [])
        for issue in ordering_issues:
            error_patterns[f"recurring_ordering: {issue[:50]}"] += 1
        
        # 4. Detect missing concepts that Judge didn't catch
        missing = judge_output.get('missing_concepts', [])
        for concept in missing:
            error_patterns[f"missing_concept: {concept}"] += 1
    
    return dict(error_patterns)


def was_suggestion_followed(suggestion: str, human_corrections: list) -> bool:
    """Check if a Judge suggestion was followed in human corrections"""
    normalized_suggestion = suggestion.lower()
    
    # Simple check: if suggestion mentions a concept, check if that concept is in corrections
    match = re.search(r'\'(.+?)\'', normalized_suggestion)
    if match:
        suggested_concept = match.group(1)
        for item in human_corrections:
            if suggested_concept in item.get('concept', '').lower():
                return True
    
    return False


def get_optimized_judge_prompt(base_prompt: str, error_patterns: dict, top_n: int = 3) -> str:
    """
    Adapt Judge prompt based on recurring errors.
    
    Args:
        base_prompt: The original SYSTEM_PROMPT for LLM Judge
        error_patterns: Output from detect_recurring_errors()
        top_n: Number of top errors to address
    
    Returns:
        Optimized prompt string
    """
    # Get top recurring errors
    top_errors = sorted(error_patterns.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    if not top_errors:
        return base_prompt  # No optimization needed
    
    # Build additional instructions based on top errors
    additional_instructions = "\n\n# ADDITIONAL INSTRUCTIONS BASED ON FEEDBACK:\n"
    
    for error_pattern, count in top_errors:
        if 'recurring_incorrect' in error_pattern:
            concept = error_pattern.replace('recurring_incorrect: ', '')
            additional_instructions += f"- Pay SPECIAL ATTENTION to selecting '{concept}'. Many users corrected this. Be extra careful with gender, number, and tense.\n"
        
        elif 'missing_concept' in error_pattern:
            concept = error_pattern.replace('missing_concept: ', '')
            additional_instructions += f"- Ensure '{concept}' is NOT marked as missing unless it's truly absent. Check synonyms and related terms.\n"
        
        elif 'ordering' in error_pattern:
            additional_instructions += "- Be more careful with ORDERING. Users frequently correct the sequence order. Consider natural Spanish word order.\n"
        
        elif 'suggestion_not_followed' in error_pattern:
            additional_instructions += "- Some suggestions were ignored by users. Be more precise and actionable in your suggestions.\n"
    
    return base_prompt + additional_instructions


def get_optimized_generator_prompt(base_prompt: str, error_patterns: dict, top_n: int = 3) -> str:
    """
    Adapt Generator prompt based on recurring errors.
    
    Args:
        base_prompt: The original SYSTEM_PROMPT for LLM Generator
        error_patterns: Output from detect_recurring_errors()
        top_n: Number of top errors to address
    
    Returns:
        Optimized prompt string
    """
    # Get top recurring errors (focus on generator-related)
    top_errors = sorted(error_patterns.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    if not top_errors:
        return base_prompt
    
    additional_instructions = "\n\n# ADDITIONAL INSTRUCTIONS BASED ON FEEDBACK:\n"
    
    for error_pattern, count in top_errors:
        if 'recurring_incorrect' in error_pattern:
            concept = error_pattern.replace('recurring_incorrect: ', '')
            additional_instructions += f"- For concept '{concept}', ensure you select the pictogram that matches the EXACT meaning in context. Check verb tense, gender, number.\n"
        
        elif 'ordering' in error_pattern:
            additional_instructions += "- When selecting pictograms, consider the ORIGINAL ORDER of concepts extracted. Maintain that order in your output.\n"
    
    return base_prompt + additional_instructions


def save_prompt_version(prompt_type: str, version: int, prompt_text: str):
    """Save prompt version for A/B testing"""
    PROMPT_DIR = Path("./prompt_versions")
    PROMPT_DIR.mkdir(exist_ok=True)
    
    filename = PROMPT_DIR / f"{prompt_type}_v{version}.txt"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(prompt_text)
    
    # Save metadata
    metadata = {
        "version": version,
        "type": prompt_type,
        "text": prompt_text,
        "timestamp": datetime.now().isoformat()
    }
    
    meta_file = PROMPT_DIR / f"{prompt_type}_v{version}_meta.json"
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    return filename


def ab_test_prompts(prompt_a: str, prompt_b: str, test_phrases: List[str], api_key: Optional[str] = None):
    """
    A/B test two versions of a prompt.
    
    Args:
        prompt_a: First prompt version
        prompt_b: Second prompt version
        test_phrases: List of test phrases
        api_key: Optional API key
    
    Returns:
        dict: Results comparison
    """
    from six_llm_generator import generate_sequence as llm_generate
    from three_use_embedded import search_sequence_candidates
    import random
    
    results_a = []
    results_b = []
    
    for phrase in test_phrases:
        # Randomly assign to A or B
        if random.choice([True, False]):
            # Test prompt A (Generator)
            try:
                from six_llm_generator import MODEL_NAME as model_a
                # ... implementation depends on integration
                results_a.append({"phrase": phrase, "prompt": "A"})
            except:
                pass
        else:
            # Test prompt B
            try:
                # ... implementation depends on integration
                results_b.append({"phrase": phrase, "prompt": "B"})
            except:
                pass
    
    return {
        "prompt_a_results": results_a,
        "prompt_b_results": results_b,
        "total_tested": len(test_phrases)
    }


if __name__ == "__main__":
    # Test the optimizer
    from datetime import datetime
    
    print("Loading feedback history...")
    history = load_feedback_history()
    print(f"Found {len(history)} feedback entries")
    
    if history:
        print("\nDetecting recurring errors...")
        errors = detect_recurring_errors(history)
        print(f"Found {len(errors)} error patterns:")
        for pattern, count in sorted(errors.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  {pattern}: {count} times")
        
        print("\nOptimizing prompts...")
        base_judge_prompt = """You are an expert in selecting ARASAAC pictograms for AAC..."""
        optimized = get_optimized_judge_prompt(base_judge_prompt, errors)
        print(f"Optimized prompt length: {len(optimized)} characters")
        
        print("\nSaving prompt version...")
        save_prompt_version("judge", 1, optimized)
        print("Saved!")
    else:
        print("No feedback history found. Run the system to generate feedback first.")
