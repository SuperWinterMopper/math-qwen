from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

adapter_id = "SuperWinterMopper/Qwen3-8B-sft-gsm8k"
ckpt = "checkpoint-1246"

print("Loading the base model and the adapter...")
peft_cfg = PeftConfig.from_pretrained(adapter_id, subfolder=ckpt)
base_id = peft_cfg.base_model_name_or_path  # should be "Qwen/Qwen3-8B"

print(f"Declared peft_cfg={peft_cfg} and base_id={base_id}")
base = AutoModelForCausalLM.from_pretrained(
    base_id,
    device_map="auto",
    torch_dtype="auto",
    trust_remote_code=True,
)

print("Loading the adapter...")
model = PeftModel.from_pretrained(base, adapter_id, subfolder=ckpt)

print("Loading the tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(adapter_id, subfolder=ckpt, trust_remote_code=True)