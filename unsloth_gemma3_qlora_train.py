#!/usr/bin/env python3
import argparse
import os

from unsloth import FastLanguageModel, is_bfloat16_supported
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="QLoRA 4-bit fine-tuning for Gemma 3 270M with Unsloth"
    )
    parser.add_argument("--model_name", type=str, default="google/gemma-3-270m-it")
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/ubuntu/ml/outputs/gemma3-270m-qlora-tajik",
    )
    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation", type=int, default=4)
    parser.add_argument("--lora_r", type=int, default=16, choices=[8, 16])
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--eval_split", type=float, default=0.1)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--seed", type=int, default=3407)
    return parser.parse_args()


def format_sample(example, tokenizer):
    instruction = str(example.get("instruction", "")).strip()
    output = str(example.get("output", "")).strip()

    messages = [
        {"role": "user", "content": instruction},
        {"role": "assistant", "content": output},
    ]

    if hasattr(tokenizer, "apply_chat_template"):
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
    else:
        text = f"Instruction:\n{instruction}\n\nResponse:\n{output}"

    return {"text": text}


def main():
    args = parse_args()

    if args.batch_size < 2 or args.batch_size > 4:
        raise ValueError("batch_size must be in range 2..4 for this setup")

    if not os.path.exists(args.dataset_path):
        raise FileNotFoundError(f"Dataset not found: {args.dataset_path}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )

    dataset = load_dataset("json", data_files=args.dataset_path, split="train")
    split = dataset.train_test_split(test_size=args.eval_split, seed=args.seed)
    train_dataset = split["train"].map(
        format_sample,
        fn_kwargs={"tokenizer": tokenizer},
        num_proc=1,
    )
    eval_dataset = split["test"].map(
        format_sample,
        fn_kwargs={"tokenizer": tokenizer},
        num_proc=1,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        dataset_num_proc=2,
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation,
            warmup_steps=args.warmup_steps,
            num_train_epochs=args.epochs,
            max_steps=args.max_steps,
            learning_rate=args.learning_rate,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=args.logging_steps,
            save_steps=args.save_steps,
            eval_steps=args.eval_steps,
            eval_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type=args.lr_scheduler_type,
            seed=args.seed,
            output_dir=args.output_dir,
            report_to="none",
        ),
    )

    train_result = trainer.train()
    print(train_result)

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved adapter and tokenizer to {args.output_dir}")


if __name__ == "__main__":
    main()
