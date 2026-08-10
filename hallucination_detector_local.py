"""
SHROOM-visions 2026 — Qwen2.5-VL-72B Local Pipeline (v2 - fixed)
"""

import os
import json
import re
import argparse
from tqdm import tqdm
import torch

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH  = "Qwen/Qwen2.5-VL-72B-Instruct"
TEST_FILE   = "dataset/shroom-visions-data/distrib/english/shroom-vision.test.en.unlabeled.jsonl"
IMAGES_DIR  = "dataset/shroom-visions-images"
OUTPUT_FILE = "predictions_en.jsonl"
RESUME      = True

# ── Prompt ────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a hallucination detection system for vision-language model outputs.

IMPORTANT STATISTICS: About 75% of AI responses contain at least one hallucination. Be thorough.

Your task: Given an image and an AI-generated response, find ALL text spans in the response that are factually wrong based on the image.

Hallucination categories:
- "invention": mentions something NOT visible in the image (wrong objects, colors, attributes, text)
- "mischaracterization": incorrectly describes something that IS visible (wrong color, shape, position, identity)
- "ocr_problem": misreads text that is visible in the image
- "miscounting": reports wrong quantities of visible items
- "other": any other hallucination

IMPORTANT RULES:
- Be generous in flagging — if something seems wrong, flag it
- The text in "text" field must be copied EXACTLY as it appears in the AI Response (same capitalization, same words)
- Use prob=1.0 if certain, prob=0.67 if fairly sure, prob=0.33 if unsure
- If response is fully accurate, return empty hallucinations list
- Respond ONLY with JSON, no explanation

Output format (ONLY this, nothing else):
{"hallucinations": [{"text": "exact text copied from response", "label": "category", "prob": 0.67}]}

If no hallucinations:
{"hallucinations": []}"""

USER_TEMPLATE = """Look carefully at the image.

The AI was asked: {prompt}

The AI responded: "{response}"

Find ALL parts of the AI response that are factually wrong based on what you actually see in the image. Copy the exact text from the response. Output ONLY JSON."""

# ── Model loading ─────────────────────────────────────────────────────────────
def load_model():
    print("Loading model across 3 GPUs...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        device_map="auto",
        attn_implementation="sdpa",
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(MODEL_PATH)

    print("\nGPU memory after model load:")
    for i in range(torch.cuda.device_count()):
        used = torch.cuda.memory_allocated(i) / 1e9
        total = torch.cuda.get_device_properties(i).total_memory / 1e9
        print(f"  GPU {i}: {used:.1f}GB / {total:.0f}GB used")
    return model, processor

# ── Span matching (case-insensitive fallback) ─────────────────────────────────
def find_span(response, span_text):
    """Find span in response, trying exact match first, then case-insensitive."""
    # Try exact match
    idx = response.find(span_text)
    if idx != -1:
        return idx, idx + len(span_text)

    # Try case-insensitive match
    lower_response = response.lower()
    lower_span = span_text.lower()
    idx = lower_response.find(lower_span)
    if idx != -1:
        return idx, idx + len(span_text)

    # Try partial match — find longest matching substring (min 10 chars)
    if len(span_text) >= 10:
        for length in range(len(span_text), 9, -1):
            for start_pos in range(len(span_text) - length + 1):
                substr = span_text[start_pos:start_pos + length]
                idx = response.lower().find(substr.lower())
                if idx != -1:
                    return idx, idx + length

    return -1, -1

# ── Output parsing ────────────────────────────────────────────────────────────
def parse_output(raw_text, response):
    raw_text = re.sub(r"```json\s*", "", raw_text)
    raw_text = re.sub(r"```\s*", "", raw_text)
    raw_text = raw_text.strip()

    # Extract JSON
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError:
        match = re.search(r'\{.*\}', raw_text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
            except Exception:
                print(f"  [WARN] Could not parse: {raw_text[:100]}")
                return []
        else:
            print(f"  [WARN] No JSON found in: {raw_text[:100]}")
            return []

    hallucinations = data.get("hallucinations", [])
    labels = []

    for h in hallucinations:
        span_text = h.get("text", "").strip()
        label     = h.get("label", "other")
        prob_raw  = float(h.get("prob", 0.5))

        # Normalize prob to task-standard values
        if prob_raw >= 0.85:
            prob = 1.0
        elif prob_raw >= 0.5:
            prob = 0.67
        else:
            prob = 0.33

        if not span_text or len(span_text) < 3:
            continue

        start, end = find_span(response, span_text)
        if start != -1:
            labels.append({"start": start, "end": end, "label": label, "prob": prob})
        else:
            print(f"  [WARN] Span not found: '{span_text[:60]}'")

    labels.sort(key=lambda x: x["start"])
    return labels

# ── Inference ─────────────────────────────────────────────────────────────────
def detect_hallucinations(model, processor, record):
    image_name = record["image_name"]
    image_path = os.path.join(IMAGES_DIR, image_name)

    if not os.path.exists(image_path):
        print(f"  [WARN] Image not found: {image_name}")
        return []

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "image", "image": f"file://{os.path.abspath(image_path)}"},
            {"type": "text",  "text": USER_TEMPLATE.format(
                prompt=record["prompt"], response=record["response"]
            )},
        ]},
    ]

    text_input = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text_input], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt"
    ).to("cuda:0")

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=False,
            temperature=None,
            top_p=None,
        )

    generated = output_ids[:, inputs["input_ids"].shape[1]:]
    raw_text = processor.batch_decode(
        generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    return parse_output(raw_text, record["response"])

# ── Resume support ────────────────────────────────────────────────────────────
def load_processed_ids(output_file):
    processed = set()
    if os.path.exists(output_file):
        with open(output_file) as f:
            for line in f:
                try:
                    processed.add(json.loads(line)["id"])
                except Exception:
                    pass
    return processed

# ── Main ──────────────────────────────────────────────────────────────────────
def main(args):
    with open(args.test_file) as f:
        records = [json.loads(l) for l in f]
    print(f"Total test records: {len(records)}")

    processed_ids = load_processed_ids(args.output) if RESUME else set()
    if processed_ids:
        print(f"Resuming — {len(processed_ids)} already done")

    remaining = [r for r in records if r["id"] not in processed_ids]
    print(f"Records to process: {len(remaining)}\n")

    if not remaining:
        print("All done!")
        return

    model, processor = load_model()
    print("\nModel loaded! Starting inference...\n")

    errors = 0
    with open(args.output, "a") as out:
        for record in tqdm(remaining, desc="Processing"):
            rid = record["id"]
            try:
                labels = detect_hallucinations(model, processor, record)
                out.write(json.dumps({"id": rid, "labels": labels}) + "\n")
                out.flush()
                tqdm.write(f"  {rid}: {len(labels)} hallucination(s)")
            except Exception as e:
                tqdm.write(f"  [ERROR] {rid}: {e}")
                out.write(json.dumps({"id": rid, "labels": []}) + "\n")
                out.flush()
                errors += 1

    print(f"\n✅ Done! Saved to: {args.output}")
    print(f"   Errors: {errors}/{len(remaining)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_file",  default=TEST_FILE)
    parser.add_argument("--images_dir", default=IMAGES_DIR)
    parser.add_argument("--output",     default=OUTPUT_FILE)
    args = parser.parse_args()
    IMAGES_DIR = args.images_dir
    main(args)
