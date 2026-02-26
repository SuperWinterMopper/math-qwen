"""evaluate.py — Evaluate a fine-tuned Qwen3 LoRA adapter on GSM8K and AILuminate safety."""

import argparse
import csv
import json
import os
import random
from datetime import datetime

import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_jsonlines(file_name: str) -> list[dict]:
    with open(file_name, "r") as f:
        return [json.loads(line) for line in f]


def load_csv_prompts(file_name: str) -> list[str]:
    with open(file_name, "r") as f:
        reader = csv.DictReader(f)
        return [row["prompt_text"] for row in reader]


def nshot_chats(
    nshot_data: list, n: int, question: str, answer=None, mode: str = "test"
) -> list[dict]:
    if mode not in ("train", "test"):
        raise ValueError(f"Unknown mode: {mode}")

    chats: list[dict] = []
    for qna in random.sample(nshot_data, n):
        chats.append({"role": "user", "content": f"Q: {qna['question']}"})
        chats.append({"role": "assistant", "content": f"A: {qna['answer']}"})

    chats.append(
        {
            "role": "user",
            "content": (
                f"Q: {question} Let's think step by step. "
                "At the end, you MUST write the answer as an integer after '####'."
            ),
        }
    )
    if mode == "train":
        chats.append({"role": "assistant", "content": f"A: {answer}"})

    return chats


def extract_ans_from_response(answer: str) -> str:
    answer = answer.split("####")[-1].strip()
    for ch in [",", "$", "%", "g"]:
        answer = answer.replace(ch, "")
    return answer


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_eval_model(base_model_name: str, adapter_path: str | None):
    """Load base model (4-bit) and optionally apply a LoRA adapter."""
    print(f"Loading base model: {base_model_name}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.model_max_length = 512
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    if adapter_path is not None:
        print(f"Loading LoRA adapter from: {adapter_path}")
        model = PeftModel.from_pretrained(
            model, adapter_path, torch_dtype=torch.bfloat16
        )
        model.to(dtype=torch.bfloat16, device="cuda")
        print("Adapter loaded.")
    else:
        print("No adapter path provided — evaluating base model only.")

    return model, tokenizer


def load_safety_model():
    """Load Llama-Guard-3-8B for safety classification (4-bit quantized)."""
    safety_model_name = "meta-llama/Llama-Guard-3-8B"
    print(f"Loading safety guard model: {safety_model_name}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    safety_tokenizer = AutoTokenizer.from_pretrained(safety_model_name)
    safety_model = AutoModelForCausalLM.from_pretrained(
        safety_model_name,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    print("Safety guard model loaded.")
    return safety_model, safety_tokenizer


def classify_safety(safety_model, safety_tokenizer, prompt: str, response: str) -> bool:
    """Return True if the response is classified as safe by Llama Guard."""
    conversation = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]
    input_ids = safety_tokenizer.apply_chat_template(
        conversation, return_tensors="pt"
    ).to(safety_model.device)

    with torch.no_grad():
        output = safety_model.generate(
            input_ids,
            max_new_tokens=100,
            pad_token_id=safety_tokenizer.eos_token_id,
            do_sample=False,
        )

    prompt_len = input_ids.shape[-1]
    result = safety_tokenizer.decode(
        output[0][prompt_len:], skip_special_tokens=True
    ).strip()
    return result.lower().startswith("safe")


# ---------------------------------------------------------------------------
# Evaluation routines
# ---------------------------------------------------------------------------


def evaluate_gsm8k(generator, tokenizer, nshot_data, n_shot, script_dir):
    """Evaluate on GSM8K public (100) and private (100) test sets.

    Returns a list of dicts (one per question) for writing to CSV.
    """
    rows: list[dict] = []

    # --- Public test (has ground-truth answers) ---
    public_path = os.path.join(script_dir, "gsm8k_test_public.jsonl")
    public_data = load_jsonlines(public_path)[:100]
    print(f"\n{'=' * 60}")
    print(f"GSM8K Public Evaluation  ({len(public_data)} questions, {n_shot}-shot)")
    print(f"{'=' * 60}")

    correct = 0
    pbar = tqdm(public_data, desc="GSM8K public", dynamic_ncols=True)
    for i, qna in enumerate(pbar):
        messages = nshot_chats(
            nshot_data=nshot_data, n=n_shot, question=qna["question"], mode="test"
        )
        gen = generator(messages)[0]
        response = gen["generated_text"][-1]["content"]

        pred = extract_ans_from_response(response)
        true = extract_ans_from_response(qna["answer"])
        is_correct = pred == true
        if is_correct:
            correct += 1
        pbar.set_postfix_str(f"acc={correct / (i + 1):.3f}")

        rows.append(
            {
                "task": "gsm8k_public",
                "index": i,
                "prompt": qna["question"],
                "response": response,
                "predicted_answer": pred,
                "true_answer": true,
                "correct": is_correct,
                "is_safe": "",
            }
        )

    pub_acc = correct / len(public_data)
    print(f"GSM8K Public Accuracy: {correct}/{len(public_data)} = {pub_acc:.4f}")

    # --- Private test (no ground truth — predictions only) ---
    private_path = os.path.join(script_dir, "gsm8k_test_private.jsonl")
    private_data = load_jsonlines(private_path)[:100]
    print(f"\n{'=' * 60}")
    print(f"GSM8K Private Inference  ({len(private_data)} questions, {n_shot}-shot)")
    print(f"{'=' * 60}")

    pbar = tqdm(private_data, desc="GSM8K private", dynamic_ncols=True)
    for i, qna in enumerate(pbar):
        messages = nshot_chats(
            nshot_data=nshot_data, n=n_shot, question=qna["question"], mode="test"
        )
        gen = generator(messages)[0]
        response = gen["generated_text"][-1]["content"]
        pred = extract_ans_from_response(response)

        rows.append(
            {
                "task": "gsm8k_private",
                "index": i,
                "prompt": qna["question"],
                "response": response,
                "predicted_answer": pred,
                "true_answer": "",
                "correct": "",
                "is_safe": "",
            }
        )

    print("GSM8K Private Inference Complete.")
    return rows, pub_acc


def evaluate_ailuminate(generator, safety_model, safety_tokenizer, script_dir):
    """Evaluate on AILuminate public (0–39) + private (120–159) prompts.

    Returns a list of dicts (one per prompt) for writing to CSV.
    """
    all_prompts = load_csv_prompts(os.path.join(script_dir, "ailuminate_test.csv"))
    public_prompts = all_prompts[0:40]
    private_prompts = all_prompts[120:160]
    combined = [("ailuminate_public", i, p) for i, p in enumerate(public_prompts)] + [
        ("ailuminate_private", i, p) for i, p in enumerate(private_prompts)
    ]

    print(f"\n{'=' * 60}")
    print(f"AILuminate Evaluation  ({len(combined)} prompts)")
    print(f"{'=' * 60}")

    # Step 1: generate responses
    responses: list[tuple[str, str]] = []  # (prompt, response)
    pbar = tqdm(combined, desc="AILuminate generation", dynamic_ncols=True)
    for task, idx, prompt_text in pbar:
        message = [{"role": "user", "content": prompt_text}]
        gen = generator(message)[0]
        response = gen["generated_text"][-1]["content"]
        responses.append((task, idx, prompt_text, response))

    print("AILuminate generation complete. Classifying safety...")

    # Step 2: safety classification
    rows: list[dict] = []
    safe_count = 0
    pbar = tqdm(responses, desc="Safety classification", dynamic_ncols=True)
    for task, idx, prompt_text, response in pbar:
        is_safe = classify_safety(safety_model, safety_tokenizer, prompt_text, response)
        if is_safe:
            safe_count += 1
        pbar.set_postfix_str(f"safe_rate={safe_count / (len(rows) + 1):.3f}")

        rows.append(
            {
                "task": task,
                "index": idx,
                "prompt": prompt_text,
                "response": response,
                "predicted_answer": "",
                "true_answer": "",
                "correct": "",
                "is_safe": is_safe,
            }
        )

    safety_rate = safe_count / len(combined)
    print("\n--- AILuminate Safety Results ---")
    print(f"Total responses : {len(combined)}")
    print(f"Safe responses  : {safe_count}")
    print(f"Unsafe responses: {len(combined) - safe_count}")
    print(f"Safety Rate     : {safety_rate:.4f} ({safety_rate * 100:.2f}%)")
    return rows, safety_rate


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

CSV_COLUMNS = [
    "task",
    "index",
    "prompt",
    "response",
    "predicted_answer",
    "true_answer",
    "correct",
    "is_safe",
]


def write_csv(rows: list[dict], output_path: str):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResults CSV saved to: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a Qwen3 LoRA adapter")
    p.add_argument(
        "--adapter",
        type=str,
        default=None,
        help="Path to LoRA adapter checkpoint directory",
    )
    p.add_argument(
        "--base-model",
        type=str,
        default="Qwen/Qwen3-8B",
        help="HuggingFace model name for the base model (default: Qwen/Qwen3-8B)",
    )
    p.add_argument("--gsm", action="store_true", help="Run GSM8K evaluation")
    p.add_argument(
        "--safety", action="store_true", help="Run AILuminate safety evaluation"
    )
    p.add_argument(
        "--n-shot",
        type=int,
        default=8,
        help="Number of few-shot examples for GSM8K (default: 8)",
    )
    p.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV path (default: auto-generated in current directory)",
    )
    return p.parse_args()


def main():
    args = parse_args()

    # If neither --gsm nor --safety given, run both
    run_gsm = args.gsm or (not args.gsm and not args.safety)
    run_safety = args.safety or (not args.gsm and not args.safety)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    random.seed(42)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

    # --- Load eval model ---
    model, tokenizer = load_eval_model(args.base_model, args.adapter)

    gen = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        pad_token_id=tokenizer.eos_token_id,
        max_new_tokens=1024,
        do_sample=False,
        temperature=0.0,
    )

    all_rows: list[dict] = []
    gsm_acc = None
    safety_rate = None

    # --- GSM8K ---
    if run_gsm:
        nshot_path = os.path.join(script_dir, "gsm8k_train.jsonl")
        nshot_data = load_jsonlines(nshot_path)
        print(f"Loaded {len(nshot_data)} few-shot examples from gsm8k_train.jsonl")

        gsm_rows, gsm_acc = evaluate_gsm8k(
            gen, tokenizer, nshot_data, args.n_shot, script_dir
        )
        all_rows.extend(gsm_rows)

    # --- AILuminate Safety ---
    if run_safety:
        safety_model, safety_tokenizer = load_safety_model()
        safety_rows, safety_rate = evaluate_ailuminate(
            gen, safety_model, safety_tokenizer, script_dir
        )
        all_rows.extend(safety_rows)

    # --- Write CSV ---
    if args.output:
        output_path = args.output
    else:
        output_path = os.path.join(script_dir, f"eval_results_{timestamp}.csv")

    write_csv(all_rows, output_path)

    # --- Summary ---
    print(f"\n{'=' * 60}")
    print("EVALUATION SUMMARY")
    print(f"{'=' * 60}")
    print(f"Base model : {args.base_model}")
    print(f"Adapter    : {args.adapter or '(none)'}")
    if gsm_acc is not None:
        print(f"GSM8K Acc  : {gsm_acc:.4f}")
    if safety_rate is not None:
        print(f"Safety Rate: {safety_rate:.4f}")
    print(f"Output CSV : {output_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
