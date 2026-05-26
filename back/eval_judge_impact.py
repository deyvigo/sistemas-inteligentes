"""
eval_judge_impact.py
====================
Evalúa el impacto del LLM-Judge comparando dos condiciones sobre un conjunto de prueba:

  Condición A — Solo Generador:
    Generador configurado del proyecto → secuencia (sin post-procesado del juez)

  Condición B — Generador + LLM-Judge:
    Generador configurado del proyecto → evaluación del juez → post-procesado
    (añade conceptos faltantes, reemplaza pictogramas incorrectos, basado
    únicamente en embedding — sin llamada adicional al LLM)

Métricas: puntuación juez, recall de conceptos, errores, BLEU-2, chrF++.

Uso como script (desde back/):
    python eval_judge_impact.py [--n 30] [--seed 42] [--delay 1.0]
"""

import json
import os
import re
import math
import time
import argparse
import random
from pathlib import Path
from collections import Counter
from datetime import datetime

os.chdir(Path(__file__).parent)

from dotenv import load_dotenv
load_dotenv()

OUT_DIR    = Path("experimentos")
OUT_DIR.mkdir(exist_ok=True)
LATEST     = OUT_DIR / "exp1_latest.json"
TEST_FILE  = Path("train-prev/test (2).json")
FEEDBACK_DIR = Path("feedback_logs")


# ── Métricas ──────────────────────────────────────────────────────────────────

def extract_ref_ids(traduccion: str) -> list:
    return [int(m) for m in re.findall(r"pict_(\d+)", traduccion)]


def compute_bleu(ref_ids: list, hyp_ids: list, max_n: int = 2) -> float:
    if not hyp_ids or not ref_ids:
        return 0.0
    r = [str(i) for i in ref_ids]
    h = [str(i) for i in hyp_ids]
    clipped, total = Counter(), Counter()
    for n in range(1, max_n + 1):
        rng = Counter(tuple(r[i:i+n]) for i in range(len(r) - n + 1))
        hng = Counter(tuple(h[i:i+n]) for i in range(len(h) - n + 1))
        for ng, c in hng.items():
            clipped[n] += min(c, rng.get(ng, 0))
            total[n]   += c
    prec = [clipped[n] / total[n] if total[n] else 0.0 for n in range(1, max_n + 1)]
    if any(p == 0 for p in prec):
        return 0.0
    bp = math.exp(1 - len(r) / len(h)) if len(h) < len(r) else 1.0
    return round(bp * math.exp(sum(math.log(p) for p in prec) / max_n) * 100, 2)


def compute_chrf(ref_ids: list, hyp_ids: list, beta: float = 2.0) -> float:
    if not hyp_ids or not ref_ids:
        return 0.0
    scores = []
    for n in range(1, 3):
        r = [str(i) for i in ref_ids]
        h = [str(i) for i in hyp_ids]
        rng = Counter(tuple(r[i:i+n]) for i in range(len(r) - n + 1))
        hng = Counter(tuple(h[i:i+n]) for i in range(len(h) - n + 1))
        m = sum(min(c, rng.get(ng, 0)) for ng, c in hng.items())
        p = m / sum(hng.values()) if hng else 0.0
        rc = m / sum(rng.values()) if rng else 0.0
        denom = beta**2 * p + rc
        scores.append((1 + beta**2) * p * rc / denom if denom else 0.0)
    return round(sum(scores) / len(scores) * 100, 2)


def concept_recall(ids: list, concepts: list, id_concept: dict) -> float:
    if not concepts:
        return 1.0
    covered = sum(
        1 for c in concepts
        if any(c.lower() in id_concept.get(pid, "").lower() or
               id_concept.get(pid, "").lower() in c.lower()
               for pid in ids)
    )
    return round(covered / len(concepts), 4)


def error_count(judge_out: dict) -> int:
    return (len(judge_out.get("missing_concepts", [])) +
            len(judge_out.get("incorrect_pictograms", [])))


def clamp_score(value) -> int:
    try:
        return max(1, min(5, int(value)))
    except (ValueError, TypeError):
        return 3

# ── Post-procesado guiado por el juez ─────────────────────────────────────────

# Straight and curly quote chars (via unicode escapes to avoid syntax errors in Py3.12+)
_ALL_QUOTES = "\"'`" + chr(0x201C) + chr(0x201D) + chr(0x2018) + chr(0x2019)
_QUOTE_RE = re.compile(r"^[" + _ALL_QUOTES + r"]+|[" + _ALL_QUOTES + r"]+$")

def _clean_concept(text: str) -> str:
    text = _QUOTE_RE.sub("", text.strip())
    text = re.sub(r"\s+", " ", text)
    return text.strip(" .,;:")


def _item_matches_concept(item: dict, concept: str) -> bool:
    concept_norm = concept.lower().strip()
    if not concept_norm:
        return False
    fields = [
        item.get("concept", ""),
        item.get("extracted_query", ""),
        item.get("query_concept", ""),
    ]
    for value in fields:
        value_norm = str(value).lower().strip()
        if value_norm and (value_norm == concept_norm or concept_norm in value_norm or value_norm in concept_norm):
            return True
    return False


def _find_item_index(sequence: list, concept: str) -> int:
    for idx, item in enumerate(sequence):
        if _item_matches_concept(item, concept):
            return idx
    return -1


def _candidate_to_item(cand: dict, query: str, **flags) -> dict:
    pid = int(cand["id"])
    return {
        "concept": cand.get("concept", query),
        "id": pid,
        "url": f"https://static.arasaac.org/pictograms/{pid}/{pid}_500.png",
        "score": float(cand.get("score", 0)),
        "description": cand.get("text", cand.get("description", "")),
        "extracted_query": query,
        **flags,
    }


def _add_concept_from_search(refined: list, current_ids: set, concept: str, actions: list,
                             source: str) -> bool:
    from three_use_embedded import search

    original_concept = concept
    concept = _clean_concept(concept)
    if not concept:
        return False
    
    strategies = [(concept, "cleaned concept")]
    
    if concept != original_concept:
        strategies.append((original_concept, "original concept"))
    
    # Add singular/plural variants
    if concept.endswith('s') and len(concept) > 2:
        strategies.append((concept[:-1], "singular variant"))
    else:
        strategies.append((concept + 's', "plural variant"))
    
    # Add related terms for certain concepts
    concept_lower = concept.lower()
    if concept_lower in ["pie", "pastel", "torta", "cake"]:
        strategies.extend([
            ("postre", "related: postre"),
            ("tarta", "related: tarta"),
            ("pastel", "related: pastel"),
        ])
    elif concept_lower in ["postre", "tarta"]:
        strategies.extend([
            ("pie", "related: pie"),
            ("pastel", "related: pastel"),
            ("cake", "related: cake"),
        ])
    
    # Try each strategy
    for search_term, strategy_name in strategies:
        if not search_term:
            continue
        try:
            results = list(search(search_term, top_k=8))
            if results:
                for cand, score in results:
                    if cand["id"] not in current_ids:
                        refined.append(_candidate_to_item(cand, search_term, judge_added=True))
                        current_ids.add(cand["id"])
                        actions.append({
                            "type": "add_missing",
                            "concept": concept,
                            "search_term": search_term,
                            "new_id": int(cand["id"]),
                            "source": source,
                            "strategy": strategy_name,
                            "match_score": float(score) if hasattr(score, '__float__') else 0.0,
                        })
                        return True
        except Exception as e:
            # Silently continue to next strategy on search error
            pass
    
    return False


def _replace_concept_from_search(refined: list, current_ids: set, old_concept: str,
                                 new_query: str, actions: list, source: str) -> bool:
    from three_use_embedded import search

    old_concept = _clean_concept(old_concept)
    new_query = _clean_concept(new_query)
    if not old_concept or not new_query:
        return False

    idx = _find_item_index(refined, old_concept)
    if idx < 0:
        return False

    old_item = refined[idx]
    for cand, _ in search(new_query, top_k=8):
        if cand["id"] != old_item["id"] and cand["id"] not in current_ids:
            current_ids.discard(old_item["id"])
            refined[idx] = _candidate_to_item(cand, new_query, judge_replaced=True)
            current_ids.add(cand["id"])
            actions.append({
                "type": "replace_incorrect",
                "concept": old_concept,
                "replacement_query": new_query,
                "old_id": int(old_item["id"]),
                "new_id": int(cand["id"]),
                "source": source,
            })
            return True
    return False


def _extract_replacement(text: str) -> tuple[str, str] | None:
    patterns = [
        r"(?:sustituir|reemplazar|cambiar)\s+(?:el\s+)?(?:pictograma\s+de\s+)?[\"']?(.+?)[\"']?\s+(?:por|con)\s+(?:un\s+)?(?:pictograma\s+(?:de|para|que\s+represente)\s+)?[\"']?(.+?)[\"']?(?:[.;,]|$)",
        r"(?:usar|utilizar)\s+(?:un\s+)?(?:pictograma\s+(?:de|para|que\s+represente)\s+)?[\"']?(.+?)[\"']?\s+(?:en\s+lugar\s+de|en vez de)\s+[\"']?(.+?)[\"']?(?:[.;,]|$)",
    ]
    for idx, pattern in enumerate(patterns):
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            first = _clean_concept(match.group(1))
            second = _clean_concept(match.group(2))
            if idx == 1:
                return second, first
            return first, second
    return None


def _extract_addition(text: str) -> str | None:
    match = re.search(
        r"(?:agregar|añadir|incluir|incorporar)\s+(?:un\s+)?(?:pictograma\s+(?:de|para|que\s+represente)\s+)?[\"']?(.+?)[\"']?(?:[.;,(]|$)",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    concept = _clean_concept(match.group(1))
    concept = re.sub(r"^(?:el|la|los|las|un|una)\s+", "", concept, flags=re.IGNORECASE)
    # Strip parenthetical qualifiers like "(opcional...)"
    concept = concept.split("(")[0].strip(" .,;")
    return concept or None


def _apply_order_suggestion(refined: list, suggestion: str, actions: list) -> bool:
    match = re.search(
        r"(?:mover|colocar|poner|ordenar)\s+[\"']?(.+?)[\"']?\s+antes\s+de\s+[\"']?(.+?)[\"']?(?:[.;,]|$)",
        suggestion,
        flags=re.IGNORECASE,
    )
    if not match:
        return False
    before_concept = _clean_concept(match.group(1))
    after_concept = _clean_concept(match.group(2))
    move_idx = _find_item_index(refined, before_concept)
    target_idx = _find_item_index(refined, after_concept)
    if move_idx < 0 or target_idx < 0 or move_idx == target_idx:
        return False
    item = refined.pop(move_idx)
    if move_idx < target_idx:
        target_idx -= 1
    refined.insert(target_idx, item)
    actions.append({
        "type": "reorder",
        "concept": before_concept,
        "before": after_concept,
        "source": "suggestions",
    })
    return True


def refine_with_judge(sequence: list, judge_out: dict) -> tuple[list, list]:
    """
    Aplica las correcciones del juez de forma determinista (sin llamadas al LLM):
      · Añade pictogramas para conceptos marcados como faltantes.
      · Reemplaza pictogramas marcados como incorrectos usando primero las
        sugerencias "Sustituir X por Y" del juez, luego la razón, y como
        último recurso el mismo concepto (distinto ID por embedding).
    """
    import sys
    
    refined = [item.copy() for item in sequence]
    current_ids = {item["id"] for item in refined}
    actions = []
    
    # Debug logging
    debug = os.environ.get("DEBUG_JUDGE_REFINEMENT", "false").lower() == "true"
    if debug:
        print(f"[REFINE_DEBUG] Starting refinement", file=sys.stderr)
        print(f"[REFINE_DEBUG] Judge output: {json.dumps(judge_out, indent=2)}", file=sys.stderr)
        print(f"[REFINE_DEBUG] Current sequence IDs: {list(current_ids)}", file=sys.stderr)

    # Pre-parsear sugerencias de sustitución para usarlas en incorrect_pictograms
    suggestion_replacements: dict[str, str] = {}
    for sug in judge_out.get("suggestions", []):
        if not isinstance(sug, str):
            continue
        repl = _extract_replacement(sug)
        if repl:
            suggestion_replacements[repl[0].lower()] = repl[1]

    # Process missing concepts
    missing_concepts = judge_out.get("missing_concepts", [])
    if debug:
        print(f"[REFINE_DEBUG] Missing concepts to add: {missing_concepts}", file=sys.stderr)
    
    for missing in missing_concepts:
        if not missing or not isinstance(missing, str):
            continue
        missing = missing.strip()
        
        if debug:
            print(f"[REFINE_DEBUG] Processing missing concept: '{missing}'", file=sys.stderr)
        
        # Manejar variantes separadas por "/" o "|" (ej: "pie/tarta" o "pie | tarta")
        alternatives = re.split(r'[/|]', missing)
        added = False
        
        for alt in alternatives:
            alt = _clean_concept(alt.strip())
            if not alt:
                continue
            if debug:
                print(f"[REFINE_DEBUG]   Trying alternative: '{alt}'", file=sys.stderr)
            if _add_concept_from_search(refined, current_ids, alt, actions, "missing_concepts"):
                added = True
                if debug:
                    print(f"[REFINE_DEBUG]   Successfully added '{alt}'", file=sys.stderr)
                break
        
        # Si no se pudo agregar con splitting, intentar el concepto completo
        if not added:
            if debug:
                print(f"[REFINE_DEBUG]   Trying original concept: '{missing}'", file=sys.stderr)
            _add_concept_from_search(refined, current_ids, missing, actions, "missing_concepts")

    used_suggestion_keys: set[str] = set()

    for flagged in judge_out.get("incorrect_pictograms", []):
        if isinstance(flagged, dict):
            old_concept = flagged.get("concept", "")
            reason = flagged.get("reason", "")

            # 1. Extraer reemplazo del campo reason
            replacement = _extract_replacement(reason) if reason else None
            if replacement:
                ok = _replace_concept_from_search(
                    refined, current_ids, replacement[0], replacement[1], actions, "incorrect_pictograms"
                )
                if ok:
                    used_suggestion_keys.add(old_concept.lower())
                    continue

            # 2. Usar la sugerencia "Sustituir X por Y" que corresponde a este concepto
            new_query_from_suggestion = suggestion_replacements.get(old_concept.lower())
            if new_query_from_suggestion:
                ok = _replace_concept_from_search(
                    refined, current_ids, old_concept, new_query_from_suggestion, actions, "incorrect_pictograms"
                )
                if ok:
                    used_suggestion_keys.add(old_concept.lower())
                    continue

            # 3. Fallback: buscar el mismo concepto pero con distinto ID (segundo mejor match)
            idx = _find_item_index(refined, old_concept)
            if idx >= 0:
                query = refined[idx].get("extracted_query") or old_concept
                _replace_concept_from_search(refined, current_ids, old_concept, query, actions, "incorrect_pictograms")

        elif isinstance(flagged, str):
            new_query_from_suggestion = suggestion_replacements.get(flagged.lower())
            if new_query_from_suggestion:
                ok = _replace_concept_from_search(
                    refined, current_ids, flagged, new_query_from_suggestion, actions, "incorrect_pictograms"
                )
                if ok:
                    used_suggestion_keys.add(flagged.lower())
                    continue
            idx = _find_item_index(refined, flagged)
            if idx >= 0:
                query = refined[idx].get("extracted_query") or flagged
                _replace_concept_from_search(refined, current_ids, flagged, query, actions, "incorrect_pictograms")

    # Procesar sugerencias restantes (las no usadas ya en incorrect_pictograms)
    for suggestion in judge_out.get("suggestions", []):
        if not isinstance(suggestion, str):
            continue

        replacement = _extract_replacement(suggestion)
        if replacement:
            old, new = replacement
            if old.lower() in used_suggestion_keys:
                continue  # ya procesada
            _replace_concept_from_search(refined, current_ids, old, new, actions, "suggestions")
            continue

        addition = _extract_addition(suggestion)
        if addition:
            _add_concept_from_search(refined, current_ids, addition, actions, "suggestions")
            continue

        _apply_order_suggestion(refined, suggestion, actions)

    refined = [{**item, "order": idx + 1} for idx, item in enumerate(refined)]
    return refined, actions


def generate_sequence_for_experiment(text: str, concepts: list, api_key: str,
                                     top_k: int = 3,
                                     use_llm_generator: bool = True) -> tuple[list, dict]:
    """
    Ejecuta el generador del proyecto con fallback determinista.

    El experimento compara el efecto del LLM-Judge sobre la misma salida base.
    Por eso la condición B reutiliza esta secuencia y solo aplica post-procesado.
    """
    from three_use_embedded import search_sequence, search_sequence_candidates

    metadata = {"mode": "embedding", "llm_generator_used": False, "fallback": False}

    if use_llm_generator and api_key:
        try:
            from six_llm_generator import generate_sequence as llm_generate

            candidates, feedback_hints = search_sequence_candidates(concepts, candidate_k=top_k)
            gen = llm_generate(
                text,
                concepts,
                candidates,
                api_key,
                feedback_hints=feedback_hints if feedback_hints else None,
            )
            if gen.get("sequence"):
                metadata.update({
                    "mode": "llm_generator",
                    "llm_generator_used": True,
                    "fallback": False,
                })
                return gen["sequence"], metadata
            metadata["fallback"] = True
            metadata["fallback_reason"] = gen.get("error", "empty_sequence")
        except Exception as exc:
            metadata["fallback"] = True
            metadata["fallback_reason"] = str(exc)

    sequence = search_sequence(concepts, top_k=top_k)
    for item in sequence:
        if "description" not in item and "text" in item:
            item["description"] = item["text"]
        if "url" not in item:
            item["url"] = f"https://static.arasaac.org/pictograms/{item['id']}/{item['id']}_500.png"
    return sequence, metadata


# ── Utilidades de carga ────────────────────────────────────────────────────────

def load_test_sentences(n: int = 30, seed: int = 42) -> list:
    with open(TEST_FILE, encoding="utf-8") as f:
        data = json.load(f)
    valid = [e for e in data if extract_ref_ids(e.get("traduccion", ""))]
    random.seed(seed)
    return random.sample(valid, min(n, len(valid)))


def build_id_concept_map() -> dict:
    from three_use_embedded import ids as emb_ids, texts as emb_texts
    result = {}
    for pid, txt in zip(emb_ids, emb_texts):
        for line in txt.strip().split("\n"):
            if "Concepto:" in line:
                result[int(pid)] = line.split("Concepto:")[-1].strip()
                break
    return result


def get_human_choice_stats() -> dict:
    """
    Resume decisiones humanas tomadas en la UI entre:
      - salida original del generador
      - salida refinada por LLM-Judge
      - edición humana posterior

    Estos datos complementan el Exp. 1 automático con aceptación humana real.
    """
    counts = Counter()
    total_actions = 0
    changed_refinements = 0

    for fpath in sorted(FEEDBACK_DIR.glob("feedback_*.json")):
        try:
            with open(fpath, encoding="utf-8") as f:
                feedback = json.load(f)
        except Exception:
            continue

        selected = feedback.get("selected_generation_variant") or "unknown"
        counts[selected] += 1
        refinement = feedback.get("judge_refinement") or {}
        total_actions += len(refinement.get("actions", []))
        if refinement.get("changed"):
            changed_refinements += 1

    total = sum(counts.values())
    return {
        "total_feedback_with_choice": total,
        "accepted_original": counts.get("original", 0),
        "accepted_judge_refined": counts.get("judge_refined", 0),
        "human_edited_after_choice": counts.get("human_edited", 0),
        "unknown": counts.get("unknown", 0),
        "judge_refined_acceptance_rate": round(counts.get("judge_refined", 0) / total, 4) if total else 0.0,
        "changed_refinements": changed_refinements,
        "total_refinement_actions": total_actions,
    }


# ── Ejecución principal ────────────────────────────────────────────────────────

def run(n: int = 30, seed: int = 42, api_key: str = "", delay: float = 1.0,
        use_llm_generator: bool | None = None) -> dict:
    """
    Ejecuta el análisis de impacto del LLM-Judge.
    Devuelve el dict de resultados y lo guarda en experimentos/exp1_latest.json.
    """
    from five_llm_judge import judge as llm_judge

    if not api_key:
        api_key = os.environ.get("GEMINI_API_KEY_JUDGE") or os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY no configurada")
    generator_api_key = os.environ.get("GEMINI_API_KEY_GENERATOR") or api_key
    if use_llm_generator is None:
        use_llm_generator = os.environ.get("USE_LLM_GENERATOR", "true").lower() == "true"

    test_set   = load_test_sentences(n, seed)
    id_concept = build_id_concept_map()
    results    = []
    generator_modes = Counter()

    for idx, entry in enumerate(test_set, 1):
        text     = entry["oracion"]
        ref_ids  = extract_ref_ids(entry.get("traduccion", ""))
        from four_extract_concepts import process_text
        concepts = process_text(text)["concepts"]

        print(f"[{idx:02d}/{len(test_set)}] {text!r}")

        # Condición A: solo generador del proyecto
        seq_A, generator_meta = generate_sequence_for_experiment(
            text,
            concepts,
            api_key=generator_api_key,
            top_k=3,
            use_llm_generator=use_llm_generator,
        )
        generator_modes[generator_meta["mode"]] += 1
        ids_A = [x["id"] for x in seq_A]
        time.sleep(delay)
        jout_A  = llm_judge(text, seq_A, api_key)

        # Condición B: generador + juez (post-procesado)
        seq_B, actions = refine_with_judge(seq_A, jout_A)
        ids_B   = [x["id"] for x in seq_B]
        time.sleep(delay)
        jout_B  = llm_judge(text, seq_B, api_key)
        errors_A = error_count(jout_A)
        errors_B = error_count(jout_B)

        results.append({
            "text":     text,
            "concepts": concepts,
            "ref_ids":  ref_ids,
            "generator": generator_meta,
            "cond_A": {
                "ids":    ids_A,
                "score":  clamp_score(jout_A.get("score", 0)),
                "errors": errors_A,
                "recall": concept_recall(ids_A, concepts, id_concept),
                "bleu":   compute_bleu(ref_ids, ids_A),
                "chrf":   compute_chrf(ref_ids, ids_A),
                "judge":  jout_A,
            },
            "cond_B": {
                "ids":      ids_B,
                "score":    clamp_score(jout_B.get("score", 0)),
                "errors":   errors_B,
                "recall":   concept_recall(ids_B, concepts, id_concept),
                "bleu":     compute_bleu(ref_ids, ids_B),
                "chrf":     compute_chrf(ref_ids, ids_B),
                "judge":    jout_B,
                "added":    [x["id"] for x in seq_B if x.get("judge_added")],
                "replaced": [x["id"] for x in seq_B if x.get("judge_replaced")],
                "postprocessing_actions": actions,
                "errors_corrected": max(0, errors_A - errors_B),
            },
        })

    def _avg(key, cond):
        vals = [r[cond][key] for r in results]
        return round(sum(vals) / len(vals), 3) if vals else 0.0

    payload = {
        "metadata": {
            "run_at": datetime.now().isoformat(),
            "n": len(results),
            "seed": seed,
            "generator_requested": "llm_generator" if use_llm_generator else "embedding",
            "generator_modes": dict(generator_modes),
        },
        "summary": {
            "cond_A": {k: _avg(k, "cond_A") for k in ("score", "errors", "recall", "bleu", "chrf")},
            "cond_B": {k: _avg(k, "cond_B") for k in ("score", "errors", "recall", "bleu", "chrf")},
            "improvements": {
                "score_improved": sum(1 for r in results if r["cond_B"]["score"] > r["cond_A"]["score"]),
                "score_same":     sum(1 for r in results if r["cond_B"]["score"] == r["cond_A"]["score"]),
                "score_worse":    sum(1 for r in results if r["cond_B"]["score"] < r["cond_A"]["score"]),
                "errors_reduced": sum(1 for r in results if r["cond_B"]["errors"] < r["cond_A"]["errors"]),
                "avg_errors_corrected": round(
                    sum(r["cond_B"]["errors_corrected"] for r in results) / len(results), 3
                ) if results else 0.0,
                "total_postprocessing_actions": sum(
                    len(r["cond_B"]["postprocessing_actions"]) for r in results
                ),
            },
            "human_choice_stats": get_human_choice_stats(),
        },
        "results": results,
    }

    with open(LATEST, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    return payload


def get_latest() -> dict | None:
    """Devuelve los últimos resultados almacenados, o None si no existen."""
    if LATEST.exists():
        with open(LATEST, encoding="utf-8") as f:
            return json.load(f)
    return None


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluación: Impacto del LLM-Judge")
    parser.add_argument("--n",     type=int,   default=30)
    parser.add_argument("--seed",  type=int,   default=42)
    parser.add_argument("--delay", type=float, default=1.0)
    args = parser.parse_args()

    data = run(n=args.n, seed=args.seed, delay=args.delay)
    A = data["summary"]["cond_A"]
    B = data["summary"]["cond_B"]
    print(f"\n{'='*55}")
    print(f"  {'Métrica':<22} {'Solo Gen':>10} {'Gen+Judge':>10} {'Δ':>8}")
    print(f"  {'-'*52}")
    for k in ("score", "errors", "recall", "bleu", "chrf"):
        print(f"  {k:<22} {A[k]:>10.3f} {B[k]:>10.3f} {B[k]-A[k]:>+8.3f}")
    imp = data["summary"]["improvements"]
    print(f"\n  Score mejoró: {imp['score_improved']}/{data['metadata']['n']}  "
          f"| Errores ↓: {imp['errors_reduced']}/{data['metadata']['n']}")
    print(f"  Resultados → {LATEST}")
