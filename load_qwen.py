
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
MODEL_DIR = Path(__file__).resolve().parent / "models" / "Qwen3-4B-Instruct-2507"

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

# Smoke Test
messages : list[dict[str,str]] = [
    {"role": "user", "content": "What is moral responsibility?"}
]

inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

with torch.inference_mode():
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=False,
    )

response = tokenizer.decode(
    outputs[0][inputs["input_ids"].shape[-1]:],
    skip_special_tokens=True,
)

print(response)
