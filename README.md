# SKstars at SHROOM-Visions 2026

**Agreement-Guided Ensembling of Zero-Shot and LoRA-Adapted Vision–Language Models**

This repository contains the code and predictions for the SKstars English-track submission to [SHROOM-Visions 2026](https://helsinki-nlp.github.io/shroom/2026), a shared task on fine-grained hallucination detection in large vision-language model (VLM) outputs.

---

## 📊 Official Results (English Track)

| Metric | Score | EN Rank |
|--------|-------|---------|
| Cor+Lbl | 0.2902 | 15th / 29 |
| Cor | 0.3642 | 18th / 28 |
| IoU | 0.3151 | 18th |

---

## 🔧 System Overview

Our system is a three-stage pipeline:

```
Image + Prompt + Response
        ↓
Stage 1: Zero-Shot Inference (Qwen2.5-VL-72B)
        ↓
Stage 2: LoRA Finetuning + Inference (Qwen2.5-VL-7B)
        ↓
Stage 3: Ensemble + Post-processing
        ↓
predictions_en_ensemble.jsonl  ← submitted file
```

**Stage 1 — Zero-Shot 72B Inference:**
Qwen2.5-VL-72B-Instruct loaded in 4-bit NF4 quantization across 3× RTX A6000 GPUs. A prompt engineering approach guides the model to identify hallucinated character spans in JSON format.

**Stage 2 — LoRA Finetuning (7B):**
Qwen2.5-VL-7B-Instruct finetuned on 3,417 English training samples using LoRA (rank=16, α=32), training only 0.57% of parameters for 1 epoch (~80 minutes).

**Stage 3 — Ensemble:**
Span-level agreement between both models determines confidence: agreed spans receive prob=0.67, 72B-only spans receive prob=0.33, and 7B-only spans are discarded. Final confidence values are redistributed to match gold annotator distribution (71% at 0.33, 21% at 0.67, 8% at 1.0).

---

## 🖥️ Hardware Requirements

- **GPUs:** 3× NVIDIA RTX A6000 (48 GB each, 144 GB total VRAM)
- **Storage:** ~200 GB free (for model weights)
- **CUDA:** 12.0+
- **RAM:** 64 GB+ recommended

---

## 📦 Installation

```bash
# Clone the repository
git clone https://anonymous.4open.science/r/shroom-visions-2026-skstars
cd shroom-visions-2026

# Create conda environment
conda create -n shroom-visions python=3.11 -y
conda activate shroom-visions

# Install dependencies
pip install -r requirements.txt
```

---

## 📁 Dataset Setup

Download the SHROOM-Visions 2026 dataset from the [shared task website](https://helsinki-nlp.github.io/shroom/2026) and organize as follows:

```
shroom-visions-2026/
├── dataset/
│   ├── shroom-visions-data/
│   │   └── distrib/
│   │       └── english/
│   │           ├── shroom-vision.train.en.labeled.jsonl
│   │           └── shroom-vision.test.en.unlabeled.jsonl
│   └── shroom-visions-images/
│       ├── image1.jpg
│       └── ...
```

---

## 🚀 Running the Pipeline

### Stage 1 — Zero-Shot 72B Inference

Download the model from HuggingFace (requires ~150 GB):

```bash
export HF_HOME=/path/to/your/cache

python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='Qwen/Qwen2.5-VL-72B-Instruct',
    local_dir='/path/to/models/Qwen2.5-VL-72B-Instruct',
    local_dir_use_symlinks=False
)
"
```

Run inference:

```bash
python hallucination_detector_local.py \
    --test_file dataset/shroom-visions-data/distrib/english/shroom-vision.test.en.unlabeled.jsonl \
    --images_dir dataset/shroom-visions-images \
    --output predictions_en_72b.jsonl
```

### Stage 2 — LoRA Finetuning (7B)

Download the 7B base model (~16 GB):

```bash
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='Qwen/Qwen2.5-VL-7B-Instruct',
    local_dir='/path/to/models/Qwen2.5-VL-7B-Instruct',
    local_dir_use_symlinks=False
)
"
```

Finetune:

```bash
cd finetuning
python finetune.py \
    --train_file ../dataset/shroom-visions-data/distrib/english/shroom-vision.train.en.labeled.jsonl \
    --images_dir ../dataset/shroom-visions-images \
    --output_dir ./lora_weights_7b
```

Run 7B inference:

```bash
python inference_lora.py \
    --test_file ../dataset/shroom-visions-data/distrib/english/shroom-vision.test.en.unlabeled.jsonl \
    --images_dir ../dataset/shroom-visions-images \
    --output ../predictions_en_7b.jsonl
```

### Stage 3 — Ensemble + Post-processing

```bash
cd ..
python ensemble.py
```

This reads `predictions_en_72b.jsonl` and `predictions_en_7b.jsonl` and outputs `predictions_en_ensemble.jsonl`.

---

## 📂 Repository Structure

```
shroom-visions-2026/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── hallucination_detector_local.py    # Stage 1: 72B zero-shot inference
├── ensemble.py                        # Stage 3: ensemble + post-processing
├── Compare.py                         # Validation and comparison script
├── finetuning/
│   ├── finetune.py                    # Stage 2: LoRA finetuning
│   └── inference_lora.py             # Stage 2: 7B finetuned inference
└── predictions/
    └── predictions_en_ensemble.jsonl  # Final submitted predictions
```

---

## 📋 Prediction Format

Each line in the prediction file follows the SHROOM-Visions submission format:

```json
{
  "id": "test-en-428",
  "labels": [
    {
      "start": 45,
      "end": 67,
      "label": "miscounting",
      "prob": 0.67
    }
  ]
}
```

Empty `"labels": []` means no hallucination detected.

**Label categories:**
- `invention` — content absent from the image
- `mischaracterization` — visible content described incorrectly
- `ocr_problem` — misreading of text visible in the image
- `miscounting` — wrong quantity of visible items
- `other` — hallucination not fitting above categories

---

## 🔑 Key Configuration

Edit the config section at the top of each script to set paths:

```python
# hallucination_detector_local.py
MODEL_PATH  = "/path/to/models/Qwen2.5-VL-72B-Instruct"
TEST_FILE   = "dataset/.../shroom-vision.test.en.unlabeled.jsonl"
IMAGES_DIR  = "dataset/shroom-visions-images"
OUTPUT_FILE = "predictions_en_72b.jsonl"

# finetuning/finetune.py
MODEL_PATH  = "/path/to/models/Qwen2.5-VL-7B-Instruct"
TRAIN_FILE  = "../dataset/.../shroom-vision.train.en.labeled.jsonl"
IMAGES_DIR  = "../dataset/shroom-visions-images"
OUTPUT_DIR  = "./lora_weights_7b"
```

---

## 📖 Citation

If you use this code, please cite:

```bibtex
@inproceedings{skstars2026shroom,
  title     = {{SKstars} at {SHROOM}-Visions 2026: Agreement-Guided
               Ensembling of Zero-Shot and {LoRA}-Adapted
               Vision--Language Models},
  author    = {Anonymous},
  booktitle = {Proceedings of the 20th International Workshop on
               Semantic Evaluation},
  year      = {2026}
}
```

---

## 📜 License

The code in this repository is released under the MIT License.
The dataset is subject to the SHROOM-Visions 2026 shared task terms and conditions.

---

## 🔗 Links

- [SHROOM-Visions 2026 Shared Task](https://helsinki-nlp.github.io/shroom/2026)
- [Qwen2.5-VL on HuggingFace](https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct)
- [SHEEP Dataset](https://arxiv.org/abs/2608.01021)
