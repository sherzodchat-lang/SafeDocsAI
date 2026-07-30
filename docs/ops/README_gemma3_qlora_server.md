# Gemma 3 270M QLoRA setup

Server paths:

- Unsloth repo: `/home/ubuntu/ml/unsloth`
- Python venv: `/home/ubuntu/ml/venv`
- Dataset: `/home/ubuntu/ml/data/tajik_sft_20260221_200850.jsonl`
- Trainer script: `/home/ubuntu/ml/unsloth_gemma3_qlora_train.py`
- Runner script: `/home/ubuntu/ml/run_gemma3_qlora.sh`

Default training parameters in the runner script:

- `max_seq_length = 1024`
- `batch_size = 2`
- `gradient_accumulation = 4`
- `LoRA r = 16` (script also supports `8`)
- `learning_rate = 2e-4`

Start training:

```bash
export HF_TOKEN=hf_your_token
/home/ubuntu/ml/run_gemma3_qlora.sh
```

Note:

- `google/gemma-3-270m-it` is a gated Hugging Face model, so `HF_TOKEN` with approved Gemma access is required.
