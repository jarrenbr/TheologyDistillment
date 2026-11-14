from pathlib import Path

import torch
from datasets import load_dataset
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
MODEL_DIR = Path(__file__).resolve().parent / "models" / "Qwen3-4B-Instruct-2507"
TRAIN_FILE = Path(__file__).resolve().parent / "data" / "train" / "nicene.draft.jsonl"

model_path = snapshot_download(
    repo_id=MODEL_ID,
    local_dir=MODEL_DIR,
)

tokenizer = AutoTokenizer.from_pretrained(model_path)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    dtype="auto",
    device_map=0, # "auto" maps onto GPU, Memory, and Disk
).eval()

print(f"Model {MODEL_ID} on Device: {model.device}")

dataset = load_dataset("json", data_files=str(TRAIN_FILE), split="train")
print(f"Loaded {len(dataset)} training examples")

