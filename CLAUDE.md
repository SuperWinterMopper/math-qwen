# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a homework project (GenAI-ML HW8) studying **fine-tuning leading to forgetting** in LLMs. It fine-tunes Qwen3 models on GSM8K math reasoning, then evaluates both math accuracy and safety (AILuminate) to measure catastrophic forgetting of safety properties.

## Environment

- **Python**: 3.13 via `.venv/` (managed with `uv`)
- **HF cache**: `/projects/p33139/classes/qwen-math/.hf_cache`
- **Python interpreter**: `/projects/p33139/classes/qwen-math/.venv/bin/python`
- **Cluster**: SLURM on `gengpu` partition with A100 GPUs (account `p33139`)
- Required modules: `gcc/11.2.0`, `cuda/12.1.0-gcc-11.2.0`

## Commands

### Submit SLURM jobs
```bash
make train-quest   # sbatch grid_search job (trains all 12 hyperparameter configs)
make eval-quest    # sbatch evaluate.py job (standalone evaluation)
```

### Run training directly (requires GPU)
```bash
.venv/bin/python train.py --model Qwen/Qwen3-8B --warmup-ratio 0.05 --train-n-shot 8
.venv/bin/python train.py --model Qwen/Qwen3-1.7B --warmup-ratio 0.1 --train-n-shot 1
```

### Run evaluation directly (requires GPU)
```bash
# GSM8K + AILuminate safety:
.venv/bin/python evaluate.py --adapter <checkpoint_path> --base-model Qwen/Qwen3-8B --gsm --safety

# GSM8K only or safety only:
.venv/bin/python evaluate.py --adapter <path> --gsm
.venv/bin/python evaluate.py --adapter <path> --safety
```

### Run full grid search (12 configs: 2 models × 2 warmup ratios × 3 n-shots)
```bash
.venv/bin/python grid_search.py --results-dir ./grid_results
```

### Gather results summary
```bash
make gather-res    # concatenates first 15 lines of each result file into grid_results/summary.txt
```

### Linting
```bash
.venv/bin/ruff check .
.venv/bin/ruff format .
```

### Setup (download datasets)
```bash
./setup.sh   # downloads datasets from NTU server; set HF_TOKEN env var first for HF login
```

## Architecture

### Pipeline Flow
```
grid_search.py
  └─ subprocess: train.py  (outputs "ADAPTER_PATH:<path>" sentinel to stdout)
  └─ subprocess: evaluate.py  (reads adapter path, runs both evaluations)
```

### Key Scripts

**`train.py`** — Fine-tunes a Qwen3 model on GSM8K using QLoRA (4-bit NF4 + LoRA r=32). Outputs the adapter checkpoint path to stdout with `ADAPTER_PATH:` prefix for parsing by `grid_search.py`. Training uses the shortest 1/3 of examples by letter count. Checkpoints saved to `runs/run_<config>_<timestamp>/`.

**`evaluate.py`** — Loads a trained LoRA adapter and evaluates on:
- **GSM8K**: public (100 questions) + private (100 questions) test sets. Extracts integer answers after `####`.
- **AILuminate safety**: public (items 0–39) + private (items 120–159) prompts from `ailuminate_test.csv`. Uses `meta-llama/Llama-Guard-3-8B` as the safety classifier (4-bit quantized).

**`grid_search.py`** — Orchestrates the 12-configuration grid search (2 models × 2 warmup ratios × 3 n-shots) by launching `train.py` and `evaluate.py` as subprocesses sequentially. Results saved to `grid_results/results_<config>_<timestamp>.txt`.

### Training Configuration (fixed)
- Quantization: 4-bit NF4 + double quant, bfloat16
- LoRA: r=32, alpha=64, dropout=0.1, all projection layers targeted
- Optimizer: paged_adamw_8bit, lr=5e-5, weight_decay=0.01
- Epochs: 2, batch size: 4, gradient accumulation: 1
- Dataset: `gsm8k_train_self-instruct.jsonl`, filtered to shortest 1/3 by letter count

### Data Files
- `gsm8k_train_self-instruct.jsonl` — training data (LLaMA-refined self-instruct version)
- `gsm8k_train.jsonl` — original GSM8K training data (used for few-shot examples during eval)
- `gsm8k_test_public.jsonl` / `gsm8k_test_private.jsonl` — test sets
- `ailuminate_test.csv` — safety evaluation prompts (`prompt_text` column)

### Output Structure
- `runs/run_<config>_<timestamp>/checkpoint-<N>/` — LoRA adapter checkpoints
- `grid_results/results_<config>_<timestamp>.txt` — per-config eval results
- `grid_results/summary.txt` — aggregated summary (via `make gather-res`)

### N-shot Prompt Format
Both train and eval use chat-template formatted prompts:
- Few-shot examples: `Q: <question>` / `A: <answer>` pairs (randomly sampled from train set)
- Target question: `Q: <question> Let's think step by step. At the end, you MUST write the answer as an integer after '####'.`
