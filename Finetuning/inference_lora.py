"""
SHROOM-visions 2026 — Inference with Finetuned 7B LoRA Model
Run this AFTER finetuning to generate predictions on test set.

Usage:
    python inference_lora.py
"""

import os
import json
import re
import argparse
from tqdm import tqdm
import torch
from PIL import Image

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
from qwen_vl_utils import process_vision_info

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH   = "/mnt/ssd2tb/models/Qwen2.5-VL-7B-Instruct"
LORA_DIR     = "./lora_weights_7b/checkpoint-800"   # best checkpoint
TEST_FILE    = "../dataset/shroom-visions-data/distrib/english/shroom-vision.test.en.unlabeled.jsonl"
IMAGES_DIR   = "../dataset/shroom-visions-images"
OUTPUT_FILE  = "../predictions_en_finetuned.jsonl"
IMAGE_SIZE   = 448    # same as training
RESUME       = True

# ── Prompts (same as training) ────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a hallucination detection system for vision-language model outputs.

IMPORTANT STATISTICS: About 75% of AI responses contain at least one hallucination. Be thorough.

Your task: Given an image and an AI-generated response, find ALL text spans in the response that are factually wrong based on the image.

Hallucination categories:
- "invention": mentions something NOT visible in the image
- "mischaracterization": incorrectly describes something that IS visible
- "ocr_problem": misreads text that is visible in the image
- "miscounting": reports wrong quantities of visible items
- "other": any other hallucination

IMPORTANT RULES:
- Be generous in flagging — if something seems wrong, flag it
- The text in "text" field must be copied EXACTLY as it appears in the AI Response
- Use prob=1.0 if certain, prob=0.67 if fairly sure, prob=0.33 if unsure
- If response is fully accurate, return empty hallucinations list
- Respond ONLY with JSON, no explanation

Output format:
{"hallucinations": [{"text": "exact text", "label": "category", "prob": 0.67}]}

If no hallucinations:
{"hallucinations": []}"""

USER_TEMPLATE = """Look carefully at the image.

The AI was asked: {prompt}

The AI responded: "{response}"

Find ALL parts of the AI response that are factually wrong based on what you see. Copy exact text. Output ONLY JSON."""


# ── Helpers ───────────────────────────────────────────────────────────────────
def load_and_resize_image(image_path, size=IMAGE_SIZE):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((size, size), Image.LANCZOS)
    return img


def find_span(response, span_text):
    idx = response.find(span_text)
    if idx != -1:
        return idx, idx + len(span_text)
    lower_response = response.lower()
    lower_span = span_text.lower()
    idx = lower_response.find(lower_span)
    if idx != -1:
        return idx, idx + len(span_text)
    if len(span_text) >= 10:
        for length in range(len(span_text), 9, -1):
            for start_pos in range(len(span_text) - length + 1):
                substr = span_text[start_pos:start_pos + length]
                idx = response.lower().find(substr.lower())
                if idx != -1:
                    return idx, idx + length
    return -1, -1


def parse_output(raw_text, response):
    raw_text = re.sub(r"```json\s*", "", raw_text)
    raw_text = re.sub(r"```\s*", "", raw_text)
    raw_text = raw_text.strip()

    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError:
        match = re.search(r'\{.*\}', raw_text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
            except Exception:
                return []
        else:
            return []

    labels = []
    for h in data.get("hallucinations", []):
        span_text = h.get("text", "").strip()
        label     = h.get("label", "other")
        prob_raw  = float(h.get("prob", 0.5))

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

    labels.sort(key=lambda x: x["start"])
    return labels


# ── Load model ────────────────────────────────────────────────────────────────
def load_model():
    print(f"Loading base 7B model from {MODEL_PATH}...")
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    )

    print(f"Loading LoRA weights from {LORA_DIR}...")
    model = PeftModel.from_pretrained(base_model, LORA_DIR)
    model.eval()

    processor = AutoProcessor.from_pretrained(
        MODEL_PATH,
        min_pixels=IMAGE_SIZE * IMAGE_SIZE,
        max_pixels=IMAGE_SIZE * IMAGE_SIZE,
    )

    print("\nGPU memory after model load:")
    for i in range(torch.cuda.device_count()):
        used  = torch.cuda.memory_allocated(i) / 1e9
        total = torch.cuda.get_device_properties(i).total_memory / 1e9
        print(f"  GPU {i}: {used:.1f}GB / {total:.0f}GB used")

    return model, processor


# ── Inference ─────────────────────────────────────────────────────────────────
def detect_hallucinations(model, processor, record):
    image_path = os.path.join(IMAGES_DIR, record["image_name"])
    if not os.path.exists(image_path):
        return []

    image = load_and_resize_image(image_path, IMAGE_SIZE)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "image", "image": image},
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
            **inputs, max_new_tokens=512,
            do_sample=False, temperature=None, top_p=None,
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
    print("\nStarting inference with finetuned 7B model...\n")

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
    parser.add_argument("--lora_dir",   default=LORA_DIR)
    parser.add_argument("--test_file",  default=TEST_FILE)
    parser.add_argument("--images_dir", default=IMAGES_DIR)
    parser.add_argument("--output",     default=OUTPUT_FILE)
    args = parser.parse_args()
    IMAGES_DIR = args.images_dir
    LORA_DIR   = args.lora_dir
    main(args)