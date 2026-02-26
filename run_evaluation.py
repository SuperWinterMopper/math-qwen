#!/usr/bin/env python3
"""run_evaluation.py — Single-command evaluation of the fine-tuned Qwen3-8B LoRA adapter.

Reproduces all evaluation metrics (GSM8K accuracy + AILuminate safety rate)
for the SuperWinterMopper/Qwen3-8B-sft-gsm8k adapter (checkpoint-1246).

Prerequisites:
  - NVIDIA GPU with >= 20GB VRAM (A100 recommended)
  - NVIDIA driver supporting CUDA 12+ (>= 525.60)
  - Python 3.13+ (required by pyproject.toml)
  - HF_TOKEN env var set, with Llama-Guard-3-8B license accepted on huggingface.co

Usage:
  python run_evaluation.py
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

ADAPTER_REPO = "SuperWinterMopper/Qwen3-8B-sft-gsm8k"
CHECKPOINT = "checkpoint-1246"
BASE_MODEL = "Qwen/Qwen3-8B"
N_SHOT = 8
OUTPUT_CSV = "eval_results.csv"


def banner(text: str) -> None:
    print(f"\n{'=' * 60}")
    print(text)
    print("=" * 60)


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    """Run a command, printing it first. Raises on failure."""
    print(f"  $ {' '.join(cmd)}")
    return subprocess.run(cmd, check=True, **kwargs)


# ============================================================================
# Phase 1: Preflight checks
# ============================================================================
def preflight() -> None:
    banner("Phase 1: Preflight checks")

    # Load HPC modules if available
    if shutil.which("module") or os.path.isfile("/usr/share/Modules/init/bash"):
        print("HPC environment detected — loading modules...")
        # module is a shell function; source it via bash
        subprocess.run(
            'eval "$(modulecmd bash load gcc/11.2.0 2>/dev/null)" 2>/dev/null; '
            'eval "$(modulecmd bash load cuda/12.1.0-gcc-11.2.0 2>/dev/null)" 2>/dev/null',
            shell=True,
        )

    # Check GPU
    if not shutil.which("nvidia-smi"):
        print("ERROR: nvidia-smi not found. A CUDA-capable GPU is required.")
        print(
            "If on an HPC cluster, make sure you are on a GPU node "
            "(e.g., srun or sbatch to gengpu)."
        )
        sys.exit(1)

    result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
    if result.returncode != 0:
        print("ERROR: nvidia-smi failed. GPU may not be accessible.")
        sys.exit(1)

    print("\nGPU info:")
    run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader",
        ]
    )


# ============================================================================
# Phase 2: Dependencies
# ============================================================================
def install_dependencies() -> Path:
    banner("Phase 2: Installing dependencies")

    # Install uv if not available
    if not shutil.which("uv"):
        print("uv not found — installing...")
        subprocess.run(
            "curl -LsSf https://astral.sh/uv/install.sh | sh",
            shell=True,
            check=True,
        )
        # Add common install locations to PATH for this session
        for p in [
            Path.home() / ".local" / "bin",
            Path.home() / ".cargo" / "bin",
        ]:
            if str(p) not in os.environ.get("PATH", ""):
                os.environ["PATH"] = f"{p}:{os.environ.get('PATH', '')}"

        if not shutil.which("uv"):
            print("ERROR: uv installation failed.")
            sys.exit(1)

    uv = shutil.which("uv")
    print(f"Using uv: {uv}")

    # Create venv and install all locked deps
    print("Running uv sync (creates .venv and installs locked dependencies)...")
    run([uv, "sync"], cwd=str(SCRIPT_DIR))
    print("Dependencies installed.")

    python = SCRIPT_DIR / ".venv" / "bin" / "python"
    if not python.is_file():
        print(f"ERROR: Expected Python at {python} but it does not exist.")
        sys.exit(1)

    return python


# ============================================================================
# Phase 3: HuggingFace authentication
# ============================================================================
def check_hf_auth() -> None:
    banner("Phase 3: Checking HuggingFace authentication")

    if os.environ.get("HF_TOKEN"):
        print("HF_TOKEN environment variable is set.")
        return

    token_path = Path.home() / ".cache" / "huggingface" / "token"
    if token_path.is_file():
        print(f"Cached HF token found at {token_path}")
        return

    print(
        "\nERROR: No HuggingFace authentication found.\n"
        "\n"
        "Llama-Guard-3-8B (used for safety evaluation) is a gated model.\n"
        "You need to:\n"
        "  1. Create a HuggingFace account at https://huggingface.co\n"
        "  2. Accept the Llama-Guard-3-8B license at\n"
        "     https://huggingface.co/meta-llama/Llama-Guard-3-8B\n"
        "  3. Create an access token at https://huggingface.co/settings/tokens\n"
        "  4. Set it: export HF_TOKEN=hf_your_token_here\n"
        "\n"
        "Then re-run this script."
    )
    sys.exit(1)


# ============================================================================
# Phase 4: Download adapter from HuggingFace
# ============================================================================
def download_adapter(python: Path) -> str:
    banner("Phase 4: Downloading adapter from HuggingFace")
    print(f"Repo: {ADAPTER_REPO}")
    print(f"Checkpoint: {CHECKPOINT}")

    # Run snapshot_download in the venv (which has huggingface_hub installed)
    download_script = f"""\
from huggingface_hub import snapshot_download
import os

path = snapshot_download(
    repo_id={ADAPTER_REPO!r},
    allow_patterns=[{CHECKPOINT!r} + '/**'],
    resume_download=True,
)

adapter_dir = os.path.join(path, {CHECKPOINT!r})
config_file = os.path.join(adapter_dir, 'adapter_config.json')

if not os.path.isfile(config_file):
    raise FileNotFoundError(
        f'adapter_config.json not found at {{adapter_dir}}. '
        f'Contents: {{os.listdir(path)}}'
    )

print(adapter_dir)
"""
    result = subprocess.run(
        [str(python), "-c", download_script],
        capture_output=True,
        text=True,
        check=True,
    )
    adapter_path = result.stdout.strip().splitlines()[-1]
    print(f"Adapter downloaded to: {adapter_path}")
    return adapter_path


# ============================================================================
# Phase 5: Run evaluation
# ============================================================================
def run_evaluation(python: Path, adapter_path: str) -> None:
    banner("Phase 5: Running evaluation (GSM8K + AILuminate safety)")
    print(f"Base model : {BASE_MODEL}")
    print(f"Adapter    : {adapter_path}")
    print(f"N-shot     : {N_SHOT}")
    print(f"Output     : {OUTPUT_CSV}")
    print()

    evaluate_py = SCRIPT_DIR / "evaluate.py"
    run(
        [
            str(python),
            str(evaluate_py),
            "--adapter",
            adapter_path,
            "--base-model",
            BASE_MODEL,
            "--gsm",
            "--safety",
            "--n-shot",
            str(N_SHOT),
            "--output",
            OUTPUT_CSV,
        ],
        cwd=str(SCRIPT_DIR),
    )


# ============================================================================
# Phase 6: Summary
# ============================================================================
def summary() -> None:
    banner("DONE")
    print(f"Results CSV : {SCRIPT_DIR / OUTPUT_CSV}")
    print(f"Base model  : {BASE_MODEL}")
    print(f"Adapter     : {ADAPTER_REPO} ({CHECKPOINT})")
    print(f"N-shot      : {N_SHOT}")
    print("=" * 60)


# ============================================================================
# Main
# ============================================================================
def main() -> None:
    os.chdir(SCRIPT_DIR)

    preflight()
    python = install_dependencies()
    check_hf_auth()
    adapter_path = download_adapter(python)
    run_evaluation(python, adapter_path)
    summary()


if __name__ == "__main__":
    main()
