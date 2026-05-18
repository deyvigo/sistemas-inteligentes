import json
from collections import Counter
from pathlib import Path
from typing import Dict

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
    Find patterns in human corrections vs LLM Generator errors.
    Used for Strategy C: Generator Prompt Refinement.
    
    Returns:
        dict: {error_pattern: frequency_count}
    """
    error_patterns = Counter()
    
    for feedback in feedback_history:
        human_corrections = feedback.get('user_modifications', {}).get('final_sequence', [])
        orig_sequence = feedback.get('system_generation', {}).get('sequence', [])
        actions = feedback.get('user_modifications', {}).get('actions_taken', {})
        
        # 1. Detect Generator wrong ID selections (replacements made by human)
        for i, corr_item in enumerate(human_corrections):
            if i < len(orig_sequence):
                orig_item = orig_sequence[i]
                if orig_item.get('id') != corr_item.get('id'):
                    concept = corr_item.get('concept', 'unknown')
                    error_patterns[f"generator_wrong_id: {concept}"] += 1
        
        # 2. Detect missing concepts (human added pictograms)
        added_ids = actions.get('addedPictogramIds', [])
        for corr_item in human_corrections:
            if corr_item.get('id') in added_ids:
                concept = corr_item.get('concept', 'unknown')
                if concept:
                    error_patterns[f"generator_missing_concept: {concept}"] += 1
        
        # 3. Detect reordering
        if actions.get('reordered', False):
            error_patterns["reordering"] += 1
    
    return dict(error_patterns)





def get_optimized_generator_prompt(base_prompt: str, error_patterns: dict) -> str:
    """
    Adapt Generator prompt based on recurring human corrections.
    Appends compact per-pattern instructions, max 5 lines.
    
    Args:
        base_prompt: The original SYSTEM_PROMPT for LLM Generator
        error_patterns: Output from detect_recurring_errors()
    
    Returns:
        Optimized prompt string
    """
    selection_errors = []
    missing_concepts = []
    has_reordering = False

    for pattern, count in error_patterns.items():
        if pattern.startswith("generator_wrong_id:"):
            concept = pattern.replace("generator_wrong_id:", "").strip()
            selection_errors.append(f"'{concept}'")
        elif pattern.startswith("generator_missing_concept:"):
            concept = pattern.replace("generator_missing_concept:", "").strip()
            missing_concepts.append(f"'{concept}'")
        elif pattern == "reordering":
            has_reordering = True

    parts = []
    if selection_errors:
        concepts_str = ", ".join(selection_errors[:5])
        parts.append(
            f"- Conceptos con errores frecuentes de selección: {concepts_str}. "
            "Revisa que el pictograma coincida con el significado exacto en el contexto."
        )
    if missing_concepts:
        concepts_str = ", ".join(missing_concepts[:5])
        parts.append(
            f"- Conceptos que suelen faltar: {concepts_str}. "
            "Asegúrate de que cada concepto extraído tenga un pictograma."
        )
    if has_reordering:
        parts.append(
            "- Verifica el orden: manten el orden original de los conceptos extraídos "
            "a menos que el contexto de la frase requiera otro."
        )

    if not parts:
        return base_prompt

    instructions = "\n\n# Instrucciones adicionales basadas en feedback:\n"
    instructions += "\n".join(parts)

    return base_prompt + instructions


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
        
        print("\nOptimizing Generator prompt...")
        base_gen_prompt = """Eres un experto en selecionar pictogramas ARASAAC para AAC.

Tu tarea es crear una secuencia de pictogramas que represente fielmente el significado de una frase en español.

INSTRUCCIONES:
- Revisa todos los pictogramas candidatos disponibles
- Selecciona los que mejor representen la FRASE COMPLETA
- NO es obligatorio elegir un pictograma por cada concepto extraído
- Un MISMO pictograma puede cubrir VARIOS conceptos
- El ORDEN de los IDs debe reflejar el orden logico de la idea"""
        optimized = get_optimized_generator_prompt(base_gen_prompt, errors)
        print(f"Optimized prompt length: {len(optimized)} characters")
        if len(optimized) > len(base_gen_prompt):
            print("Additional instructions appended.")
        
        print("\nSaving prompt version...")
        save_prompt_version("generator", 1, optimized)
        print("Saved!")
    else:
        print("No feedback history found. Run the system to generate feedback first.")
