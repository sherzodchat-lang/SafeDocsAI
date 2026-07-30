#!/usr/bin/env bash
set -euo pipefail

# Required for gated Gemma repositories on Hugging Face.
# Export your token before running, for example:
# export HF_TOKEN=hf_xxx

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN is not set"
  exit 1
fi

export HF_HOME="${HF_HOME:-/home/ubuntu/.cache/huggingface}"
export MODEL_NAME="${MODEL_NAME:-google/gemma-3-270m-it}"
export OUTPUT_DIR="${OUTPUT_DIR:-/home/ubuntu/ml/outputs/gemma3-270m-qlora-tajik-v2}"

/home/ubuntu/ml/venv/bin/python /home/ubuntu/ml/unsloth_gemma3_qlora_train.py \
  --model_name "$MODEL_NAME" \
  --dataset_path /home/ubuntu/ml/data/tajik_sft_20260221_200850.jsonl \
  --output_dir "$OUTPUT_DIR" \
  --max_seq_length 1024 \
  --batch_size 4 \
  --gradient_accumulation 4 \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --learning_rate 2e-4 \
  --epochs 2 \
  --warmup_steps 100 \
  --lr_scheduler_type cosine \
  --eval_split 0.1 \
  --eval_steps 100 \
  --save_steps 100
