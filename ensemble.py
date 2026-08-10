"""
SHROOM-visions 2026 — Ensemble Script (v2 - conservative)
Combines zero-shot 72B + finetuned 7B predictions.

Strategy:
- Both models agree → keep (prob from average)
- Only one model   → keep only if prob >= 0.67 (more confident spans)

Usage:
    python ensemble.py
"""

import json

# ── Config ────────────────────────────────────────────────────────────────────
PRED_72B    = "predictions_en_final.jsonl"
PRED_7B     = "predictions_en_finetuned_clean.jsonl"
OUTPUT      = "predictions_en_ensemble.jsonl"
OVERLAP_THR = 0.3   # min overlap to consider spans "same"
SINGLE_MIN_PROB = 0.67  # single-model spans need at least this prob to be kept

# ── Helpers ───────────────────────────────────────────────────────────────────
def span_overlap(s1, e1, s2, e2):
    chars1 = set(range(s1, e1))
    chars2 = set(range(s2, e2))
    if not chars1 or not chars2:
        return 0.0
    return len(chars1 & chars2) / len(chars1 | chars2)

def find_matching_span(span, other_spans, threshold=OVERLAP_THR):
    for other in other_spans:
        if span_overlap(span["start"], span["end"],
                        other["start"], other["end"]) >= threshold:
            return other
    return None

def merge_spans(span1, span2):
    start = min(span1["start"], span2["start"])
    end   = max(span1["end"],   span2["end"])
    label = span1["label"] if span1["prob"] >= span2["prob"] else span2["label"]
    avg_prob = (span1["prob"] + span2["prob"]) / 2
    if avg_prob >= 0.85:
        prob = 1.0
    elif avg_prob >= 0.5:
        prob = 0.67
    else:
        prob = 0.33
    return {"start": start, "end": end, "label": label, "prob": prob}

def load_preds(filepath):
    preds = {}
    with open(filepath) as f:
        for line in f:
            p = json.loads(line)
            preds[p["id"]] = p["labels"]
    return preds

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("Loading predictions...")
    preds_72b = load_preds(PRED_72B)
    preds_7b  = load_preds(PRED_7B)

    all_ids = sorted(set(preds_72b.keys()) | set(preds_7b.keys()))
    print(f"  Total IDs: {len(all_ids)}")

    results = []
    stats = {"both": 0, "only_72b": 0, "only_7b": 0, "dropped": 0}

    for rid in all_ids:
        spans_72b = preds_72b.get(rid, [])
        spans_7b  = preds_7b.get(rid, [])

        ensemble_spans = []
        used_7b = set()

        # For each 72B span check if 7B also found it
        for span in spans_72b:
            match = find_matching_span(span, spans_7b)
            if match:
                # Both agree → always keep
                merged = merge_spans(span, match)
                ensemble_spans.append(merged)
                used_7b.add(id(match))
                stats["both"] += 1
            else:
                # Only 72B → keep only if confident
                if span["prob"] >= SINGLE_MIN_PROB:
                    span_copy = span.copy()
                    span_copy["prob"] = 0.33  # downgrade confidence
                    ensemble_spans.append(span_copy)
                    stats["only_72b"] += 1
                else:
                    stats["dropped"] += 1

        # 7B-only spans
        for match in spans_7b:
            if id(match) not in used_7b:
                # Only 7B → keep only if confident
                if match["prob"] >= SINGLE_MIN_PROB:
                    span_copy = match.copy()
                    span_copy["prob"] = 0.33
                    ensemble_spans.append(span_copy)
                    stats["only_7b"] += 1
                else:
                    stats["dropped"] += 1

        ensemble_spans.sort(key=lambda x: x["start"])
        results.append({"id": rid, "labels": ensemble_spans})

    # Save
    with open(OUTPUT, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    total_spans = sum(len(r["labels"]) for r in results)
    has_hall    = sum(1 for r in results if r["labels"])

    print(f"\n=== Ensemble Results ===")
    print(f"Total records       : {len(results)}")
    print(f"With hallucinations : {has_hall} ({100*has_hall/len(results):.1f}%)")
    print(f"Total spans         : {total_spans}")
    print(f"Avg spans/record    : {total_spans/len(results):.1f}")
    print()
    print(f"Span sources:")
    print(f"  Both agreed       : {stats['both']}  ← highest quality")
    print(f"  Only 72B (kept)   : {stats['only_72b']}")
    print(f"  Only 7B  (kept)   : {stats['only_7b']}")
    print(f"  Dropped (low conf): {stats['dropped']}")
    print()
    print(f"Saved to: {OUTPUT}")

if __name__ == "__main__":
    main()