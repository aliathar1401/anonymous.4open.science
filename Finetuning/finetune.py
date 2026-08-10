"""
SHROOM-visions 2026 — LoRA Finetuning Script (v8 - resume 2 more epochs)
Resizes images before processing to control sequence length safely.

Usage:
    python finetune.py
"""

import os
import json
import argparse
import torch
from torch.utils.data import Dataset
from PIL import Image

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, TaskType
from qwen_vl_utils import process_vision_info

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH   = "/mnt/ssd2tb/models/Qwen2.5-VL-7B-Instruct"
TRAIN_FILE   = "../dataset/shroom-visions-data/distrib/english/shroom-vision.train.en.labeled.jsonl"
IMAGES_DIR   = "../dataset/shroom-visions-images"
OUTPUT_DIR   = "./lora_weights_7b"
EPOCHS       = 3       # ← total 3 epochs (1 done + 2 more)
BATCH_SIZE   = 1
GRAD_ACCUM   = 8
LR           = 2e-4
IMAGE_SIZE   = 448
LORA_RANK    = 16
LORA_ALPHA   = 32
LORA_DROPOUT = 0.05

# ── Prompts ───────────────────────────────────────────────────────────────────
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


def labels_to_json(record):
    labels   = record.get("labels", [])
    response = record["response"]
    if not labels:
        return '{"hallucinations": []}'
    hallucinations = []
    for lbl in labels:
        span_text = response[lbl["start"]:lbl["end"]]
        hallucinations.append({
            "text":  span_text,
            "label": lbl["label"],
            "prob":  lbl["prob"],
        })
    return json.dumps({"hallucinations": hallucinations})


def load_and_resize_image(image_path, size=IMAGE_SIZE):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((size, size), Image.LANCZOS)
    return img


# ── Dataset ───────────────────────────────────────────────────────────────────
class ShroomDataset(Dataset):
    def __init__(self, records, processor, images_dir):
        self.records = []
        for r in records:
            img_path = os.path.join(images_dir, r["image_name"])
            if os.path.exists(img_path):
                self.records.append(r)
            else:
                print(f"  [SKIP] Image not found: {r['image_name']}")
        self.processor  = processor
        self.images_dir = images_dir
        print(f"Dataset: {len(self.records)} records loaded")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record     = self.records[idx]
        image_path = os.path.join(self.images_dir, record["image_name"])
        target     = labels_to_json(record)

        image = load_and_resize_image(image_path, IMAGE_SIZE)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text",  "text": USER_TEMPLATE.format(
                        prompt=record["prompt"],
                        response=record["response"],
                    )},
                ],
            },
            {"role": "assistant", "content": target},
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            truncation=False,
            return_tensors="pt",
        )

        result = {k: v.squeeze(0) for k, v in inputs.items()}
        result["labels"] = result["input_ids"].clone()
        return result


# ── Collator ──────────────────────────────────────────────────────────────────
class ShroomCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        max_len = max(b["input_ids"].shape[0] for b in batch)

        input_ids_list      = []
        attention_mask_list = []
        labels_list         = []

        for b in batch:
            seq_len = b["input_ids"].shape[0]
            pad_len = max_len - seq_len

            input_ids_list.append(torch.cat([
                b["input_ids"],
                torch.full((pad_len,), self.pad_token_id, dtype=torch.long)
            ]))
            attention_mask_list.append(torch.cat([
                b["attention_mask"],
                torch.zeros(pad_len, dtype=torch.long)
            ]))
            lbl = b["labels"].clone()
            labels_list.append(torch.cat([
                lbl, torch.full((pad_len,), -100, dtype=torch.long)
            ]))

        result = {
            "input_ids":      torch.stack(input_ids_list),
            "attention_mask": torch.stack(attention_mask_list),
            "labels":         torch.stack(labels_list),
        }

        if "pixel_values" in batch[0]:
            result["pixel_values"] = torch.cat(
                [b["pixel_values"].unsqueeze(0) if b["pixel_values"].dim() == 3
                 else b["pixel_values"] for b in batch], dim=0
            )

        if "image_grid_thw" in batch[0]:
            grids = []
            for b in batch:
                g = b["image_grid_thw"]
                if g.dim() == 1:
                    g = g.unsqueeze(0)
                grids.append(g)
            result["image_grid_thw"] = torch.cat(grids, dim=0)

        return result


# ── Load 7B model with LoRA ───────────────────────────────────────────────────
def load_model_for_training():
    print(f"Loading 7B model from {MODEL_PATH}...")

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    model.print_trainable_parameters()

    print("\nGPU memory after model load:")
    for i in range(torch.cuda.device_count()):
        used  = torch.cuda.memory_allocated(i) / 1e9
        total = torch.cuda.get_device_properties(i).total_memory / 1e9
        print(f"  GPU {i}: {used:.1f}GB / {total:.0f}GB used")

    processor = AutoProcessor.from_pretrained(
        MODEL_PATH,
        min_pixels=IMAGE_SIZE * IMAGE_SIZE,
        max_pixels=IMAGE_SIZE * IMAGE_SIZE,
    )
    return model, processor


# ── Main ──────────────────────────────────────────────────────────────────────
def main(args):
    with open(args.train_file) as f:
        records = [json.loads(l) for l in f]
    print(f"Total training records: {len(records)}")

    split         = int(0.9 * len(records))
    train_records = records[:split]
    val_records   = records[split:]
    print(f"Train: {len(train_records)}, Val: {len(val_records)}")

    model, processor = load_model_for_training()

    train_dataset = ShroomDataset(train_records, processor, args.images_dir)
    val_dataset   = ShroomDataset(val_records,   processor, args.images_dir)

    pad_id   = processor.tokenizer.pad_token_id or 0
    collator = ShroomCollator(pad_token_id=pad_id)

    # Find latest checkpoint to resume from
    checkpoint_dir = args.output_dir
    latest_checkpoint = None
    if os.path.isdir(checkpoint_dir):
        checkpoints = [
            os.path.join(checkpoint_dir, d)
            for d in os.listdir(checkpoint_dir)
            if d.startswith("checkpoint-")
        ]
        if checkpoints:
            latest_checkpoint = max(checkpoints, key=os.path.getmtime)
            print(f"Resuming from checkpoint: {latest_checkpoint}")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=GRAD_ACCUM,
        gradient_checkpointing=True,
        learning_rate=LR,
        weight_decay=0.01,
        warmup_steps=50,
        lr_scheduler_type="cosine",
        bf16=True,
        fp16=False,
        optim="paged_adamw_8bit",
        max_grad_norm=0.3,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,
        load_best_model_at_end=True,
        dataloader_num_workers=0,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
    )

    print("\nStarting finetuning on 7B model...")
    trainer.train(resume_from_checkpoint=latest_checkpoint)

    print(f"\nSaving LoRA weights to {args.output_dir}...")
    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print("Done! ✅")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file",  default=TRAIN_FILE)
    parser.add_argument("--images_dir",  default=IMAGES_DIR)
    parser.add_argument("--output_dir",  default=OUTPUT_DIR)
    parser.add_argument("--epochs",      type=int, default=EPOCHS)
    parser.add_argument("--batch_size",  type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    main(args)