#!/bin/bash
#SBATCH -A p33139
#SBATCH -p gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 48:00:00
#SBATCH --mem=80G
#SBATCH -o /projects/p33139/other/qwen-math/slurm-%j.out
#SBATCH -e /projects/p33139/other/qwen-math/slurm-%j.err

# Load GCC 11.2+ before cuda - required for DeepSpeed C++ extensions (shm_comm)
# PyTorch/deepspeed need GCC 9+ for JIT compilation; system default is GCC 4.8
module load gcc/11.2.0
module load cuda/12.1.0-gcc-11.2.0

cd /projects/p33139/other/qwen-math

# Ensure CUDA is visible to PyTorch/DeepSpeed (avoids fallback to CPU/Gloo which triggers shm_comm build)
export CUDA_HOME="${CUDA_ROOT:-$CUDA_HOME}"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

date
echo "Running in directory: $(pwd)"
nvidia-smi

# Use uv to run the script - it manages the venv and dependencies automatically
/projects/p33139/other/qwen-math/.venv/bin/python copy_of_genai_ml_hw8_fine_tuning_leads_to_forgetting.py