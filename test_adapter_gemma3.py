#!/usr/bin/env python3
import argparse
import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test Gemma3 adapter model")
    parser.add_argument(
        "--base-model",
        type=str,
        default="google/gemma-3-270m-it",
        help="Base model name",
    )
    parser.add_argument(
        "--adapter-dir",
        type=str,
        default="/home/ubuntu/models/gemma3_tajik_unsloth/adapter",
        help="Path to adapter directory",
    )
    parser.add_argument("--instruction", type=str, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
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
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise EnvironmentError("HF_TOKEN is required")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, token=hf_token)
    target_device = args.device
    if target_device == "auto":
        target_device = "cuda" if torch.cuda.is_available() else "cpu"

    if target_device == "cpu":
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            token=hf_token,
            device_map=None,
            dtype=torch.float32,
            low_cpu_mem_usage=False,
        )
        model.to("cpu")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            token=hf_token,
            device_map="auto",
            load_in_4bit=True,
        )

    model = PeftModel.from_pretrained(model, args.adapter_dir)
    model.eval()

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
