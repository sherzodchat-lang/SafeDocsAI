#!/usr/bin/env python3
import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test merged Gemma3 model")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="/home/ubuntu/models/gemma3_tajik_unsloth/merged_fp16_fixed",
        help="Path to merged model directory",
    )
    parser.add_argument("--instruction", type=str, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--do-sample",
        action="store_true",
        help="Enable sampling; default is greedy decoding",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Inference device",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    target_device = args.device
    if target_device == "auto":
        target_device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=False)
    if target_device == "cpu":
        model = AutoModelForCausalLM.from_pretrained(
            args.model_dir,
            device_map=None,
            dtype=torch.float32,
            low_cpu_mem_usage=False,
        )
        model.to("cpu")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_dir,
            device_map="auto",
            dtype=torch.float32,
        )

    model.generation_config.do_sample = args.do_sample
    if not args.do_sample:
        model.generation_config.top_p = None
        model.generation_config.top_k = None

    prompt = f"### Instruction:\n{args.instruction.strip()}\n\n### Response:\n"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    generate_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.do_sample,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.eos_token_id,
        "use_cache": True,
    }
    if args.repetition_penalty != 1.0:
        generate_kwargs["repetition_penalty"] = args.repetition_penalty
    if args.no_repeat_ngram_size > 0:
        generate_kwargs["no_repeat_ngram_size"] = args.no_repeat_ngram_size
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
