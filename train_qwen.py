"""A small, local QLoRA training example. Running this file starts training.
"""

import json
from importlib.metadata import version
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)

ROOT = Path(__file__).resolve().parent
MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
MODEL_DIR = ROOT / "models" / "Qwen3-4B-Instruct-2507"
TRAIN_FILE = ROOT / "data/train/nicene.draft.jsonl"
OUTPUT_DIR = ROOT / "models/nicene-smoke-adapter"
NUM_EXAMPLES = 8
MAX_LENGTH = 1024
MAX_STEPS = 10
LEARNING_RATE = 2e-4
SEED = 42

# get records
dataset = load_dataset(
    "json",
    data_files=str(TRAIN_FILE),
    split="train",
)
subset = dataset.select(range(min(NUM_EXAMPLES, len(dataset))))
records = list(subset)

# Data and device error handling
if not records:
    raise ValueError("Select a nonempty training dataset first.")
if OUTPUT_DIR.exists():
    raise ValueError("Choose a new OUTPUT_DIR so an earlier run is not overwritten.")
if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
    raise ValueError("This configuration requires a CUDA GPU with bfloat16 support.")

set_seed(SEED)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

# 2. Format each conversation and mask everything before the assistant answer.
train_data = []
for row in records:
    messages = row["messages"]
    if [message["role"] for message in messages] != ["user", "assistant"]:
        raise ValueError("Each example must contain one user/assistant pair.")
    prompt_ids = tokenizer.apply_chat_template(
        messages[:-1], tokenize=True, add_generation_prompt=True, return_dict=False,
    )
    input_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False, return_dict=False,
    )
    if input_ids[:len(prompt_ids)] != prompt_ids:
        raise ValueError("The prompt tokens do not match the start of the conversation.")

    # Keep the final end-of-turn token (EOS), but omit the template's trailing newline.
    eos_position = len(input_ids) - 1 - input_ids[::-1].index(tokenizer.eos_token_id)
    input_ids = input_ids[:eos_position + 1]
    answer_ids = input_ids[len(prompt_ids):]
    if tokenizer.decode(answer_ids) != messages[-1]["content"] + tokenizer.eos_token:
        raise ValueError("The supervised tokens must be exactly the answer plus EOS.")
    if len(input_ids) > MAX_LENGTH:
        raise ValueError("An example exceeds MAX_LENGTH; inspect it instead of truncating it.")

    # Cross-entropy ignores labels equal to -100. The model shifts labels internally.
    labels = [-100] * len(prompt_ids) + answer_ids
    train_data.append({"input_ids": input_ids, "attention_mask": [1] * len(input_ids),
                       "labels": labels})

first = train_data[0]
answer_start = sum(label == -100 for label in first["labels"])
print("Masked prompt:\n", tokenizer.decode(first["input_ids"][:answer_start]))
print("Supervised answer:\n", tokenizer.decode(first["input_ids"][answer_start:]))

# 3. Load frozen base weights in 4-bit NF4; compute using bfloat16.
quantization = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR, local_files_only=True, quantization_config=quantization,
    dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="sdpa",
)
model.config.use_cache = False
model = prepare_model_for_kbit_training(
    model, gradient_checkpointing_kwargs={"use_reentrant": False},
)

# 4. Add trainable low-rank matrices to the linear layers.
lora = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules="all-linear", bias="none", task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora)
model.print_trainable_parameters()

# 5. Batch size one needs no padding. Accumulate four batches per optimizer step.
training_args = TrainingArguments(
    output_dir=str(OUTPUT_DIR),
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    max_steps=MAX_STEPS,
    learning_rate=LEARNING_RATE,
    lr_scheduler_type="constant",
    optim="adamw_torch",
    bf16=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    seed=SEED,
    data_seed=SEED,
    dataloader_num_workers=0,
    logging_steps=1,
    save_strategy="no",
    eval_strategy="no",
    report_to="none",
)

# Save the inputs and settings so this small experiment can be traced later.
OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
(OUTPUT_DIR / "training_records.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
(OUTPUT_DIR / "training_args.json").write_text(training_args.to_json_string(), encoding="utf-8")
download_metadata = MODEL_DIR / ".cache/huggingface/download"
run_info = {
    "model_id": MODEL_ID,
    "download_metadata": {str(path.relative_to(download_metadata)): path.read_text()
                          for path in download_metadata.rglob("*.metadata")},
    "versions": {name: version(name) for name in
                 ("torch", "transformers", "peft", "accelerate", "bitsandbytes")},
    "gpu": torch.cuda.get_device_name(0),
    "cuda": torch.version.cuda,
    "max_length": MAX_LENGTH,
    "quantization": quantization.to_dict(),
}
(OUTPUT_DIR / "run_info.json").write_text(json.dumps(run_info, indent=2), encoding="utf-8")

trainer = Trainer(
    model=model, args=training_args, train_dataset=train_data,
    data_collator=default_data_collator, processing_class=tokenizer,
)
result = trainer.train()

# 6. Save the adapter and tokenizer, not a second copy of the base weights.
model.save_pretrained(OUTPUT_DIR / "adapter")
tokenizer.save_pretrained(OUTPUT_DIR / "adapter")
trainer.save_metrics("train", result.metrics)
print("Saved adapter to", OUTPUT_DIR / "adapter")
