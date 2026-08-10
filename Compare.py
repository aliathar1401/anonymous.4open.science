import json
from collections import Counter

files = {
    "Zero-shot 72B (baseline)" : "predictions_en_final.jsonl",
    "Finetuned 7B"             : "predictions_en_finetuned_clean.jsonl",
    "Ensemble (72B+7B)"        : "predictions_en_ensemble.jsonl",
}

print(f"{'File':<30} {'Hall%':>6} {'Spans':>6} {'Avg':>5} {'p=0.33':>7} {'p=0.67':>7} {'p=1.0':>7}")
print("-"*75)

for name, filepath in files.items():
    with open(filepath) as f:
        preds = [json.loads(l) for l in f]
    has_hall   = sum(1 for p in preds if p["labels"])
    all_labels = [l for p in preds for l in p["labels"]]
    probs      = Counter(l["prob"] for l in all_labels)
    total      = len(all_labels)
    print(f"{name:<30} {100*has_hall/len(preds):>5.1f}% {total:>6} {total/len(preds):>5.1f} {probs.get(0.33,0):>7} {probs.get(0.67,0):>7} {probs.get(1.0,0):>7}")