"""
eval_iterative.py
=================
Evalúa la mejora iterativa del sistema a lo largo de rondas sucesivas del ciclo:
  Generar → Evaluar (LLM) → Corregir (Humano) → Mejorar → Re-generar

PARTE A — get_history_stats():
  Análisis del historial acumulado en feedback_logs/ (sin llamadas LLM).
  · Puntuación del juez por sesión.
  · Correcciones humanas por sesión.
  · Crecimiento de la tabla de correcciones.

PARTE B — run_simulation():
  Conjunto de prueba fijo evaluado en 4 rondas con tablas de corrección
  progresivamente más ricas (Round 0 = sin overrides).

Uso como script (desde back/):
    python eval_iterative.py --skip-b           # sólo Parte A
    python eval_iterative.py --n 15 --seed 42   # Parte A + B
"""

import os
import json
import re
import time
import argparse
import random
from pathlib import Path
from collections import Counter
from datetime import datetime

os.chdir(Path(__file__).parent)

from dotenv import load_dotenv
load_dotenv()

import numpy as np
from sentence_transformers import SentenceTransformer
from feedback_analyzer import _process_feedback_entries, _filter_corrections

FEEDBACK_DIR = Path("feedback_logs")
TEST_FILE    = Path("train-prev/test (2).json")
OUT_DIR      = Path("experimentos")
OUT_DIR.mkdir(exist_ok=True)
LATEST_SIM   = OUT_DIR / "exp4_latest.json"

# Embeddings (carga única)
_model      = SentenceTransformer("intfloat/multilingual-e5-small")
_embeddings = np.load("./embeddings/embeddings.npy")
_ids        = np.load("./embeddings/ids.npy")
_texts      = np.load("./embeddings/texts.npy")
_id_list    = [int(x) for x in _ids]


def _extract_concept(text: str) -> str:
    for line in text.strip().split("\n"):
        if "Concepto:" in line:
            return line.split("Concepto:")[-1].strip()
    return text.strip()[:50]


# ── Búsqueda con tabla de corrección parametrizable ───────────────────────────

def search_with_table(query: str, top_k: int = 3, table: dict = None) -> list:
    q_emb  = _model.encode([query], normalize_embeddings=True)[0]
    scores = np.dot(_embeddings, q_emb).copy()
    top_idx = np.argsort(scores)[::-1][:top_k + 20]

    results  = []
    seen_ids = set()
    for i in top_idx:
        pid = int(_ids[i])
        seen_ids.add(pid)
        results.append({
            "id":                pid,
            "concept":           _extract_concept(_texts[i]),
            "description":       _texts[i],
            "extracted_query":   query,
            "score":             float(scores[i]),
            "feedback_override": False,
        })

    if table:
        key    = query.lower().strip()
        entry  = table.get(key, {})
        overrides = {int(k): v for k, v in entry.get("candidates", {}).items()}
        rejected  = {int(k): v for k, v in entry.get("rejected",   {}).items()}

        for item in results:
            if item["id"] in rejected:
                item["score"] += -0.05 * min(rejected[item["id"]], 3)

        for oid, _ in overrides.items():
            if oid in seen_ids:
                for item in results:
                    if item["id"] == oid:
                        item["score"] = 10.0
                        item["feedback_override"] = True
                        break
            elif oid in _id_list:
                idx = _id_list.index(oid)
                results.append({
                    "id":                oid,
                    "concept":           _extract_concept(_texts[idx]),
                    "description":       _texts[idx],
                    "extracted_query":   query,
                    "score":             10.0,
                    "feedback_override": True,
                })
                seen_ids.add(oid)

        results.sort(key=lambda x: x["score"], reverse=True)

    return results[:top_k]


def generate_with_table(concepts: list, table: dict = None, top_k: int = 3) -> list:
    sequence = []
    seen_ids = set()
    for concept in concepts:
        for cand in search_with_table(concept, top_k=top_k, table=table):
            if cand["id"] not in seen_ids:
                seen_ids.add(cand["id"])
                sequence.append({
                    "concept":           cand["concept"],
                    "id":                cand["id"],
                    "url":               f"https://static.arasaac.org/pictograms/{cand['id']}/{cand['id']}_500.png",
                    "score":             cand["score"],
                    "description":       cand["description"],
                    "extracted_query":   concept,
                    "feedback_override": cand.get("feedback_override", False),
                })
                break
    return sequence


# ── Métricas ──────────────────────────────────────────────────────────────────

def error_count(judge_out: dict) -> int:
    return (len(judge_out.get("missing_concepts", [])) +
            len(judge_out.get("incorrect_pictograms", [])))


def _extract_ref_ids(traduccion: str) -> list:
    return [int(m) for m in re.findall(r"pict_(\d+)", traduccion)]


def _compute_bleu(ref_ids: list, hyp_ids: list, max_n: int = 2) -> float:
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
            total[n] += c
    prec = [clipped[n] / total[n] if total[n] else 0.0 for n in range(1, max_n + 1)]
    if any(p == 0 for p in prec):
        return 0.0
    bp = np.exp(1 - len(r) / len(h)) if len(h) < len(r) else 1.0
    return round(float(bp * np.exp(sum(np.log(p) for p in prec) / max_n) * 100), 2)


def _compute_chrf(ref_ids: list, hyp_ids: list, beta: float = 2.0) -> float:
    if not hyp_ids or not ref_ids:
        return 0.0
    scores = []
    r = [str(i) for i in ref_ids]
    h = [str(i) for i in hyp_ids]
    for n in range(1, 3):
        rng = Counter(tuple(r[i:i+n]) for i in range(len(r) - n + 1))
        hng = Counter(tuple(h[i:i+n]) for i in range(len(h) - n + 1))
        m = sum(min(c, rng.get(ng, 0)) for ng, c in hng.items())
        p = m / sum(hng.values()) if hng else 0.0
        rc = m / sum(rng.values()) if rng else 0.0
        denom = beta**2 * p + rc
        scores.append((1 + beta**2) * p * rc / denom if denom else 0.0)
    return round(sum(scores) / len(scores) * 100, 2)


def _clamp_score(value) -> int:
    try:
        return max(1, min(5, int(value)))
    except Exception:
        return 0


def concept_recall(seq: list, concepts: list) -> float:
    if not concepts:
        return 1.0
    covered = sum(
        1 for c in concepts
        if any(c.lower() in item.get("concept", "").lower() or
               item.get("extracted_query", "").lower() == c.lower()
               for item in seq)
    )
    return round(covered / len(concepts), 4)


def _annotation_time_seconds(feedback: dict):
    for key in ("annotation_time_seconds", "duration_seconds", "elapsed_seconds"):
        value = feedback.get(key)
        if isinstance(value, (int, float)):
            return round(float(value), 3)
    for section in ("user_modifications", "metadata"):
        nested = feedback.get(section, {})
        if isinstance(nested, dict):
            for key in ("annotation_time_seconds", "duration_seconds", "elapsed_seconds"):
                value = nested.get(key)
                if isinstance(value, (int, float)):
                    return round(float(value), 3)
    return None


def _pearson(xs: list, ys: list):
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den_x = sum((x - mean_x) ** 2 for x in xs)
    den_y = sum((y - mean_y) ** 2 for y in ys)
    den = (den_x * den_y) ** 0.5
    if den == 0:
        return None
    return round(num / den, 4)


# ── Parte A ───────────────────────────────────────────────────────────────────

def _load_sessions() -> list:
    result = []
    for f in sorted(FEEDBACK_DIR.glob("feedback_*.json")):
        try:
            with open(f, encoding="utf-8") as fh:
                result.append((f, json.load(fh)))
        except Exception as e:
            print(f"[WARN] {f.name}: {e}")
    return result


def get_history_stats() -> dict:
    """
    Parte A: analiza el historial de feedback sin llamadas al LLM.
    Devuelve per_session y cumulative (crecimiento de la tabla).
    """
    sessions   = _load_sessions()
    per_session = []
    cumulative  = []

    for idx, (fpath, fb) in enumerate(sessions, 1):
        text    = fb.get("input", {}).get("original_text", "")
        score   = fb.get("llm_evaluation", {}).get("score")
        user_score = fb.get("user_score")
        actions = fb.get("user_modifications", {}).get("actions_taken", {})
        n_add   = len(actions.get("addedPictogramIds",   []))
        n_del   = len(actions.get("deletedPictogramIds", []))
        n_reord = len(actions.get("reorder_details",     []))
        annotation_time = _annotation_time_seconds(fb)

        per_session.append({
            "session":       idx,
            "file":          fpath.name,
            "text":          text,
            "judge_score":   score,
            "user_score":    user_score,
            "n_corrections": n_add + n_del + n_reord,
            "n_adds":        n_add,
            "n_dels":        n_del,
            "n_reorders":    n_reord,
            "n_missing":     len(fb.get("llm_evaluation", {}).get("missing_concepts", [])),
            "n_incorrect":   len(fb.get("llm_evaluation", {}).get("incorrect_pictograms", [])),
            "annotation_time_seconds": annotation_time,
        })

        all_fb = [s for _, s in sessions[:idx]]
        corrs  = _process_feedback_entries(all_fb)
        filt   = _filter_corrections(corrs, min_confidence=1)
        cumulative.append({
            "after_session":         idx,
            "correction_table_size": len(filt),
            "total_overrides":       sum(len(v.get("candidates", {})) for v in filt.values()),
            "total_rejects":         sum(len(v.get("rejected",   {})) for v in filt.values()),
        })

    scores = [s["judge_score"] for s in per_session if s["judge_score"] is not None]
    corrs  = [s["n_corrections"] for s in per_session]
    h      = len(scores) // 2

    trends = {}
    if len(scores) >= 4:
        trends["score_first_half"]  = round(sum(scores[:h]) / h, 3)
        trends["score_second_half"] = round(sum(scores[h:]) / max(1, len(scores) - h), 3)
        trends["score_delta"]       = round(trends["score_second_half"] - trends["score_first_half"], 3)
    if len(corrs) >= 4:
        trends["corrections_first_half"]  = round(sum(corrs[:h]) / h, 3)
        trends["corrections_second_half"] = round(sum(corrs[h:]) / max(1, len(corrs) - h), 3)
        trends["corrections_delta"]       = round(trends["corrections_second_half"] - trends["corrections_first_half"], 3)
    times = [s["annotation_time_seconds"] for s in per_session if s["annotation_time_seconds"] is not None]
    if times:
        trends["avg_annotation_time_seconds"] = round(sum(times) / len(times), 3)
    paired_scores = [
        (s["judge_score"], s["user_score"])
        for s in per_session
        if isinstance(s["judge_score"], (int, float)) and isinstance(s["user_score"], (int, float))
    ]
    if paired_scores:
        judge_scores = [p[0] for p in paired_scores]
        user_scores = [p[1] for p in paired_scores]
        trends["llm_human_correlation"] = _pearson(judge_scores, user_scores)
        trends["llm_human_pairs"] = len(paired_scores)

    return {
        "total_sessions":  len(sessions),
        "per_session":     per_session,
        "cumulative":      cumulative,
        "trends":          trends,
    }


# ── Parte B ───────────────────────────────────────────────────────────────────

def _build_round_tables(sessions: list) -> tuple:
    n = len(sessions)
    if n == 0:
        return [{}], [0]
    cuts = sorted({0, max(1, n // 3), max(2, 2 * n // 3), n})
    tables = []
    for k in cuts:
        if k == 0:
            tables.append({})
        else:
            all_fb = [s for _, s in sessions[:k]]
            corrs  = _process_feedback_entries(all_fb)
            tables.append(_filter_corrections(corrs, min_confidence=1))
    return tables, cuts


def run_simulation(n: int = 15, seed: int = 42,
                   api_key: str = "", delay: float = 1.5) -> dict:
    """
    Parte B: ejecuta la simulación con LLM sobre un conjunto fijo de oraciones.
    Guarda y devuelve los resultados.
    """
    from five_llm_judge import judge as llm_judge
    from four_extract_concepts import process_text

    if not api_key:
        api_key = os.environ.get("GEMINI_API_KEY_JUDGE") or os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY no configurada")

    with open(TEST_FILE, encoding="utf-8") as f:
        all_data = json.load(f)
    valid = [e for e in all_data if _extract_ref_ids(e.get("traduccion", ""))]
    random.seed(seed)
    test_sentences = random.sample(valid, min(n, len(valid)))

    sessions = _load_sessions()
    tables, cuts = _build_round_tables(sessions)
    labels = [f"Round {k} ({c} fb)" for k, c in enumerate(cuts)]

    sentence_results = []
    for s_idx, entry in enumerate(test_sentences, 1):
        text     = entry["oracion"]
        concepts = process_text(text)["concepts"]
        ref_ids  = _extract_ref_ids(entry.get("traduccion", ""))
        print(f"  [{s_idx:02d}/{len(test_sentences)}] {text!r}")
        row = {"text": text, "concepts": concepts, "ref_ids": ref_ids, "rounds": []}
        baseline_errors = None

        for r_idx, (table, label) in enumerate(zip(tables, labels)):
            seq = generate_with_table(concepts, table=table, top_k=3)
            ids = [item["id"] for item in seq]
            time.sleep(delay)
            jout  = llm_judge(text, seq, api_key)
            score = _clamp_score(jout.get("score", 0))
            errors = error_count(jout)
            if baseline_errors is None:
                baseline_errors = errors

            row["rounds"].append({
                "label":             label,
                "feedback_sessions": cuts[r_idx],
                "overrides_applied": sum(1 for x in seq if x.get("feedback_override")),
                "ids":               ids,
                "score":             score,
                "errors":            errors,
                "errors_corrected_vs_round0": max(0, baseline_errors - errors),
                "recall":            concept_recall(seq, concepts),
                "bleu":              _compute_bleu(ref_ids, ids),
                "chrf":              _compute_chrf(ref_ids, ids),
                "judge":             jout,
            })
            print(f"    {label:<22} | score={score}  errores={errors}")
        sentence_results.append(row)

    def _round_avg(r_idx, key):
        vals = [row["rounds"][r_idx][key] for row in sentence_results]
        return round(sum(vals) / len(vals), 3) if vals else 0.0

    per_round = [
        {
            "label":             labels[r],
            "feedback_sessions": cuts[r],
            "avg_score":         _round_avg(r, "score"),
            "avg_errors":        _round_avg(r, "errors"),
            "avg_recall":        _round_avg(r, "recall"),
            "avg_bleu":          _round_avg(r, "bleu"),
            "avg_chrf":          _round_avg(r, "chrf"),
            "avg_overrides":     _round_avg(r, "overrides_applied"),
            "avg_errors_corrected_vs_round0": _round_avg(r, "errors_corrected_vs_round0"),
        }
        for r in range(len(cuts))
    ]

    payload = {
        "metadata": {
            "run_at": datetime.now().isoformat(),
            "n": len(sentence_results),
            "seed": seed,
            "round_cuts": cuts,
        },
        "per_round":        per_round,
        "sentence_results": sentence_results,
    }

    with open(LATEST_SIM, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    return payload


def get_latest_simulation() -> dict | None:
    if LATEST_SIM.exists():
        with open(LATEST_SIM, encoding="utf-8") as f:
            return json.load(f)
    return None


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluación: Mejora Iterativa")
    parser.add_argument("--n",      type=int,   default=15)
    parser.add_argument("--seed",   type=int,   default=42)
    parser.add_argument("--delay",  type=float, default=1.5)
    parser.add_argument("--skip-b", action="store_true",
                        help="Solo Parte A (sin llamadas al LLM)")
    args = parser.parse_args()

    stats = get_history_stats()
    print(f"\nSesiones cargadas: {stats['total_sessions']}")
    print(f"{'Ses':<5} {'Texto':<42} {'Score':<7} {'Correc.':<8} {'Tabla'}")
    print("-" * 65)
    for s, c in zip(stats["per_session"], stats["cumulative"]):
        sc = str(s["judge_score"]) if s["judge_score"] is not None else "N/A"
        print(f"{s['session']:<5} {s['text'][:40]:<42} {sc:<7} {s['n_corrections']:<8} {c['correction_table_size']}")

    if stats["trends"]:
        t = stats["trends"]
        print(f"\nTendencias:")
        if "score_delta" in t:
            print(f"  Score:        1ª mitad={t['score_first_half']}  2ª mitad={t['score_second_half']}  Δ={t['score_delta']:+.3f}")
        if "corrections_delta" in t:
            print(f"  Correcciones: 1ª mitad={t['corrections_first_half']}  2ª mitad={t['corrections_second_half']}  Δ={t['corrections_delta']:+.3f}")

    if not args.skip_b:
        sim = run_simulation(n=args.n, seed=args.seed, delay=args.delay)
        print(f"\nSimulación ({sim['metadata']['n']} oraciones):")
        print(f"{'Ronda':<26} {'Fb':>5} {'Score':>7} {'Errores':>8} {'Recall':>8}")
        print("-" * 55)
        for r in sim["per_round"]:
            print(f"{r['label']:<26} {r['feedback_sessions']:>5} {r['avg_score']:>7.3f} {r['avg_errors']:>8.3f} {r['avg_recall']:>8.3f}")
        print(f"\nResultados → {LATEST_SIM}")
