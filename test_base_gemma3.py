#!/usr/bin/env python3
import argparse
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test base Gemma3 model")
    parser.add_argument("--model-name", type=str, default="google/gemma-3-270m-it")
    parser.add_argument("--instruction", type=str, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise EnvironmentError("HF_TOKEN required")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, token=token, use_fast=False
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        token=token,
        device_map="auto",
        load_in_4bit=True,
    )

    prompt = f"### Instruction:\n{args.instruction.strip()}\n\n### Response:\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    generate_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.do_sample,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if args.do_sample:
        generate_kwargs["temperature"] = args.temperature
        generate_kwargs["top_p"] = args.top_p

    with torch.no_grad():
        outputs = model.generate(**inputs, **generate_kwargs)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "### Response:" in text:
        text = text.split("### Response:", 1)[1].strip()
    print(text)


if __name__ == "__main__":
    main()
