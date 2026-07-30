#!/usr/bin/env python3
import argparse

import torch
from unsloth import FastLanguageModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test merged Gemma3 model via Unsloth")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="/home/ubuntu/models/gemma3_tajik_unsloth/merged",
    )
    parser.add_argument("--instruction", type=str, required=True)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--load-in-4bit", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_dir,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=args.load_in_4bit,
    )
    FastLanguageModel.for_inference(model)

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
