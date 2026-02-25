train:
	sbatch --gres=gpu:a100:1 /projects/p33139/other/qwen-math/sbatch_train.sh

try:
	uv run try_loading_model.py --ckpt checkpoint-1246 --n 2
