#!/usr/bin/env python3
import os
import argparse
import torch

from huggingface_hub import snapshot_download
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset


def ensure_snapshot(repo_id: str, cache_dir: str, allow_patterns=None, revision=None) -> str:
    """
    Download a repo snapshot if needed and return the local path.
    Uses HF cache and resumes automatically.
    """
    path = snapshot_download(
        repo_id=repo_id,
        cache_dir=cache_dir,
        revision=revision,
        allow_patterns=allow_patterns,
        resume_download=True,
        local_files_only=False,  # allow downloading if not cached
    )
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_id", default="Qwen/Qwen3-8B")
    parser.add_argument("--adapter_id", default="SuperWinterMopper/Qwen3-8B-sft-gsm8k")
    parser.add_argument("--ckpt", default="checkpoint-1246")
    parser.add_argument("--dataset", default="gsm8k")
    parser.add_argument("--dataset_config", default="main")
    parser.add_argument("--split", default="test")
    parser.add_argument("--n", type=int, default=3)
    args = parser.parse_args()

    # Put caches somewhere persistent on HPC
    HF_HOME = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    HF_CACHE = os.environ.get("HF_HUB_CACHE", os.path.join(HF_HOME, "hub"))
    DATASETS_CACHE = os.environ.get("HF_DATASETS_CACHE", os.path.join(HF_HOME, "datasets"))

    os.makedirs(HF_CACHE, exist_ok=True)
    os.makedirs(DATASETS_CACHE, exist_ok=True)

    # Optional but helpful on HPC
    os.environ.setdefault("HF_HOME", HF_HOME)
    os.environ.setdefault("HF_HUB_CACHE", HF_CACHE)
    os.environ.setdefault("HF_DATASETS_CACHE", DATASETS_CACHE)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    print(f"HF_HOME={HF_HOME}")
    print(f"HF_HUB_CACHE={HF_CACHE}")
    print(f"HF_DATASETS_CACHE={DATASETS_CACHE}")

    # ---- 1) Download base model snapshot ----
    print(f"\nDownloading base model snapshot: {args.base_id}")
    base_path = ensure_snapshot(
        repo_id=args.base_id,
        cache_dir=HF_CACHE,
        # You can leave allow_patterns=None to get everything.
        # If you want to reduce download size, keep it broad enough not to break loading.
        allow_patterns=None,
    )
    print(f"Base snapshot path: {base_path}")

    # ---- 2) Download adapter snapshot (must include the ckpt subfolder) ----
    print(f"\nDownloading adapter snapshot: {args.adapter_id}")
    adapter_path = ensure_snapshot(
        repo_id=args.adapter_id,
        cache_dir=HF_CACHE,
        allow_patterns=[f"{args.ckpt}/**", "*.json", "*.safetensors", "*.bin", "*.pt"],
    )
    print(f"Adapter snapshot path: {adapter_path}")

    # ---- 3) Download dataset ----
    print(f"\nDownloading dataset: {args.dataset} ({args.dataset_config}) split={args.split}")
    ds = load_dataset(
        args.dataset,
        args.dataset_config,
        split=args.split,
        cache_dir=DATASETS_CACHE,
    )
    print(f"Dataset loaded: {len(ds)} rows")

    # ---- 4) Load PEFT config (from adapter ckpt) ----
    print("\nLoading PEFT config...")
    peft_cfg = PeftConfig.from_pretrained(adapter_path, subfolder=args.ckpt, local_files_only=True)
    print(f"Adapter declares base: {peft_cfg.base_model_name_or_path}")

    # ---- 5) Load base model (from local snapshot path) ----
    print("\nLoading base model...")
    # If you're on an A100, bf16 is ideal. If you might be on CPU, use float32.
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    base = AutoModelForCausalLM.from_pretrained(
        base_path,
        device_map="auto",          # change to {"": 0} if you want forced single GPU
        torch_dtype=dtype,
        trust_remote_code=True,
        local_files_only=True,
    )

    # ---- 6) Load adapter onto base ----
    print("Loading adapter...")
    model = PeftModel.from_pretrained(
        base,
        adapter_path,
        subfolder=args.ckpt,
        local_files_only=True,
    )

    # ---- 7) Load tokenizer (from base snapshot) ----
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        base_path,
        trust_remote_code=True,
        local_files_only=True,
    )

    model.eval()

    # ---- 8) Run inference on first N examples ----
    print(f"\nRunning inference on {args.n} examples...\n")
    for i in range(min(args.n, len(ds))):
        q = ds[i].get("question", "")
        prompt = f"Question: {q}\nAnswer:"

        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
            )

        text = tokenizer.decode(out[0], skip_special_tokens=True)
        print(f"=== Example {i} ===")
        print(text)
        print()

    print("Done.")


if __name__ == "__main__":
    main()