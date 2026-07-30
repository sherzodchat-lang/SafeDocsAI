#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
from typing import List

import unsloth
from datasets import load_dataset
from transformers import EarlyStoppingCallback, TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel, is_bfloat16_supported


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Gemma-3 270M IT with Unsloth LoRA"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="google/gemma-3-270m-it",
        help="Base model name on Hugging Face",
    )
    parser.add_argument(
        "--data-files",
        type=str,
        nargs="+",
        required=True,
        help="One or more JSONL files with {instruction, output}",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/home/ubuntu/models/gemma3_tajik_unsloth",
        help="Directory for checkpoints and final artifacts",
    )
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default="auto",
        help='Checkpoint path, "auto" to use latest checkpoint, or "none" to disable resume',
    )
    return parser.parse_args()


def validate_columns(dataset) -> None:
    missing = {"instruction", "output"} - set(dataset.column_names)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")


def ensure_files_exist(paths: List[str]) -> None:
    for path in paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Data file not found: {path}")


def find_latest_checkpoint(output_dir: str) -> str | None:
    root = Path(output_dir)
    if not root.exists():
        return None

    checkpoints = []
    for p in root.glob("checkpoint-*"):
        if not p.is_dir():
            continue
        suffix = p.name.split("-")[-1]
        if suffix.isdigit():
            checkpoints.append((int(suffix), str(p)))

    if not checkpoints:
        return None

    checkpoints.sort(key=lambda x: x[0])
    return checkpoints[-1][1]


def main() -> None:
    args = parse_args()

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise EnvironmentError("HF_TOKEN is not set")

    ensure_files_exist(args.data_files)
    os.makedirs(args.output_dir, exist_ok=True)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=True,
        token=hf_token,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
        use_rslora=False,
        loftq_config=None,
    )

    raw_dataset = load_dataset("json", data_files=args.data_files, split="train")
    validate_columns(raw_dataset)

    raw_dataset = raw_dataset.filter(
        lambda ex: ex["instruction"] is not None and ex["output"] is not None
    )

    split = raw_dataset.train_test_split(test_size=0.1, seed=args.seed, shuffle=True)
    eos_token = tokenizer.eos_token or ""

    def format_batch(batch):
        texts = []
        for instruction, output in zip(batch["instruction"], batch["output"]):
            inst = str(instruction).strip()
            out = str(output).strip()
            texts.append(f"### Instruction:\n{inst}\n\n### Response:\n{out}{eos_token}")
        return {"text": texts}

    train_dataset = split["train"].map(
        format_batch,
        batched=True,
        remove_columns=split["train"].column_names,
    )
    eval_dataset = split["test"].map(
        format_batch,
        batched=True,
        remove_columns=split["test"].column_names,
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=2,
        warmup_steps=100,
        learning_rate=args.learning_rate,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=args.seed,
        save_strategy="steps",
        save_steps=100,
        eval_strategy="steps",
        eval_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        packing=False,
        args=training_args,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    resume_arg = (args.resume_from_checkpoint or "").strip().lower()
    if resume_arg in {"", "auto"}:
        resume_path = find_latest_checkpoint(args.output_dir)
    elif resume_arg in {"none", "no", "false"}:
        resume_path = None
    else:
        resume_path = args.resume_from_checkpoint

    if resume_path:
        print(f"Resuming from checkpoint: {resume_path}")
        trainer.train(resume_from_checkpoint=resume_path)
    else:
        trainer.train()

    adapter_dir = os.path.join(args.output_dir, "adapter")
    merged_dir = os.path.join(args.output_dir, "merged")

    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(merged_dir, safe_serialization=True)
    tokenizer.save_pretrained(merged_dir)

    print("Training complete")
    print(f"Adapter saved to: {adapter_dir}")
    print(f"Merged model saved to: {merged_dir}")


if __name__ == "__main__":
    main()
