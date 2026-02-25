from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

model = AutoPeftModelForCausalLM.from_pretrained(
    "SuperWinterMopper/Qwen3-8B-sft-gsm8k",
    device_map="auto",
    torch_dtype="auto",
    subfolder="checkpoint-1246",  # the latest checkpoint
)
tokenizer = AutoTokenizer.from_pretrained("SuperWinterMopper/Qwen3-8B-sft-gsm8k")