#!/usr/bin/env python3
import argparse
import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-merge LoRA adapter into fp16 model"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="google/gemma-3-270m-it",
    )
    parser.add_argument(
        "--adapter-dir",
        type=str,
        default="/home/ubuntu/models/gemma3_tajik_unsloth/adapter",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/home/ubuntu/models/gemma3_tajik_unsloth/merged_fp16_fixed",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise EnvironmentError("HF_TOKEN is required")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, token=hf_token)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        token=hf_token,
        device_map="cpu",
        dtype=torch.float16,
        low_cpu_mem_usage=False,
    )

    peft_model = PeftModel.from_pretrained(base_model, args.adapter_dir)
    merged_model = peft_model.merge_and_unload()

    merged_model.save_pretrained(args.output_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.output_dir)

    print("Re-merge complete")
    print(f"Saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
