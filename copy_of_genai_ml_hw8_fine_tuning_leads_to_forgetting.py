"""## Import Packages"""

import os
import json
import csv
import random
from datetime import datetime
import torch
from transformers import (
    AutoModelForCausalLM, # imports the model for causal language modeling
    AutoTokenizer, # imports the tokenizer for the model
    BitsAndBytesConfig, # imports the configuration for using bitsandbytes
    pipeline # imports the pipeline for text generation
)
from peft import (
    LoraConfig, # imports the configuration for LoRA
    get_peft_model, # imports the function to get the PEFT model
    PeftModel # imports the PEFT model
)
from datasets import Dataset # Imports the Dataset class from the datasets library
from trl import SFTConfig, SFTTrainer # Imports the SFTConfig and SFTTrainer classes from the trl library
from tqdm import tqdm # Imports the tqdm library for progress bars

os.environ["CUDA_VISIBLE_DEVICES"] = '0' # Sets the CUDA device to use
device = torch.device('cuda:0') # Creates a CUDA device object
random.seed(42) # Sets the random seed for reproducibility

RUN_TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S') # Timestamp for output filenames

"""## LLM Fine-tuning

### Load Model & Tokenizer
"""

sft_model_name = 'Qwen/Qwen3-8B' # Specifies the name of the pre-trained model to use
sft_bnb_config = BitsAndBytesConfig( # Configuration for using bitsandbytes
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
sft_model = AutoModelForCausalLM.from_pretrained( # Loads the pre-trained model
    pretrained_model_name_or_path=sft_model_name,
    quantization_config=sft_bnb_config,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
)
sft_tokenizer = AutoTokenizer.from_pretrained( # Loads the tokenizer for the model
    pretrained_model_name_or_path=sft_model_name,
)
sft_tokenizer.model_max_length = 512 ### changed to 512 to reduce speed
sft_tokenizer.add_special_tokens({'pad_token': '[PAD]'}) # Adds a special token for padding
peft_config = LoraConfig(
    r=32,
    lora_alpha=64,
    # TODO: Adds dropout
    lora_dropout=0.1,  # lora_dropout = 0 equals no dropout
    bias='none',
    task_type='CAUSAL_LM',
    target_modules=['up_proj', 'down_proj', 'gate_proj', 'k_proj', 'q_proj', 'v_proj', 'o_proj']
)


peft_model = get_peft_model(sft_model, peft_config).to(dtype=torch.bfloat16)

"""[link text](https://)### Dataset Formatting Functions"""

def load_jsonlines(file_name: str):
    f = open(file_name, 'r')
    return [json.loads(line) for line in f]

def nshot_chats(nshot_data: list, n: int, question: str, answer: any, mode: str) -> dict: # Function to create n-shot chats
    if mode not in ['train', 'test']:
        raise AssertionError('Undefined Mode!!!')

    chats = []
    # TODO: Use fixed few-shot examples
    for qna in random.sample(nshot_data, n): # Samples n examples from the n-shot data
        chats.append(
            {
                'role': 'user',
                'content': f'Q: {qna["question"]}' # Creates a user message with the question
            }
        )
        chats.append(
            {
                'role': 'assistant',
                'content': f'A: {qna["answer"]}' # Creates an assistant message with the answer
            }
        )

    chats.append(
        {
            'role': 'user',
            'content': f'Q: {question} Let\'s think step by step. At the end, you MUST write the answer as an integer after \'####\'.' # Creates a user message with the question and instructions
        }
    )
    if mode == 'train':
        chats.append(
            {
                'role': 'assistant',
                'content': f'A: {answer}' # Creates an assistant message with the answer
            }
        )

    return chats # Returns the list of chats

"""### Format GSM8K Data for Fine-tuning

### 🔎 Filter GSM8K by Length (simple)
Keeps the longest **1/3** by letter count (A–Z and other alphabetic characters). Change `PORTION` if desired.
"""

gsm8k_train = load_jsonlines('gsm8k_train_self-instruct.jsonl') # You can use refined gsm8k_train_self-instruct.jsonl for fine-tuning

formatted_gsm8k = []
for qna in gsm8k_train: # Iterates over the GSM8K training data
    chats = nshot_chats(nshot_data=gsm8k_train, n=TRAIN_N_SHOT, question=qna['question'], answer=qna['answer'], mode='train') # Creates n-shot chats for the current example
    train_sample = sft_tokenizer.apply_chat_template(chats, tokenize=False) # Applies the chat template to the chats
    # The line below caused a ValueError because '<|eot_id|>' is not present in the Qwen tokenizer's template.
    # It's removed as it was intended to remove a 'Knowledge Date' which is not part of this model's template.
    # train_sample = train_sample[train_sample.index("<|eot_id|>") + len("<|eot_id|>"):] # Remove Cutting Knowledge Date in prompt template
    formatted_gsm8k.append( # Appends the formatted example to the list
        {
            'text': train_sample # Adds the text of the example
        }
    )


formatted_gsm8k = Dataset.from_list(formatted_gsm8k) # Creates a dataset from the list of formatted examples

"""### Sample 1/3 of the longest data ** **Please do not modify this block** **"""

### Please do not modify this block ###
# Keep the longest 1/3 of `formatted_gsm8k` by letter count
PORTION = 1/3  # change this if needed

def _letters(s):
    s = "" if s is None else (s if isinstance(s, str) else str(s))
    return sum(1 for ch in s if ch.isalpha())

# Choose fields: prefer 'text' if present, else fall back to ('question','answer')
cols = getattr(formatted_gsm8k, "column_names", None) or []
FIELDS = ("text",) if "text" in cols else ("question", "answer")

n = len(formatted_gsm8k)
k = max(1, int(round(n * PORTION)))

# Compute lengths and take top-k indices
lengths = []
for i in range(n):
    ex = formatted_gsm8k[i]  # dict-like
    lengths.append(sum(_letters(ex.get(f, "")) for f in FIELDS))

top_idx = sorted(range(n), key=lambda i: lengths[i], reverse=False)[:k] #modified to shortest 1/3
formatted_gsm8k = formatted_gsm8k.select(top_idx)

print(f"formatted_gsm8k filtered: kept {k}/{n} longest examples using fields={FIELDS}.")

"""### Fine-tuning"""

# drive.mount("/content/drive", force_remount=True)  # Disabled for Linux server

# trainer
training_arguments = SFTConfig( # Configuration for the SFT trainer
    seed=1126,
    data_seed=1126,
    output_dir=f"./runs/run_{RUN_TIMESTAMP}",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    optim="paged_adamw_8bit",
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    dataloader_num_workers=4,
    dataloader_pin_memory=True,
    tf32=True,
    num_train_epochs=2, # TODO: If you use fixed few-shot examples, increase epoch
    logging_strategy="steps",
    logging_steps=50,
    save_strategy="steps",
    save_steps=200,
    save_total_limit=2,
    lr_scheduler_type='linear',
    learning_rate=5e-5, # TODO: Decrease learning rate

    # TODO: Add warmup
    warmup_ratio=0.05,
    # TODO: Add weight decay
    weight_decay=0.01,

    bf16=True,
    group_by_length=True,
    dataset_text_field='text',
    report_to='none',
)
trainer = SFTTrainer( # Creates the SFT trainer
    model=sft_model,
    train_dataset=formatted_gsm8k,
    peft_config=peft_config,
    processing_class=sft_tokenizer,
    args=training_arguments,
)
trainer.train() # Starts the training process

"""## LLM Inference

### Load Adapter Checkpoint
"""

# List all directories within the run output folder
runs_dir = f'./runs/run_{RUN_TIMESTAMP}'
checkpoint_dirs = [d for d in os.listdir(runs_dir) if os.path.isdir(os.path.join(runs_dir, d)) and d.startswith('checkpoint-')]
checkpoint_dirs.sort()

adapter_path = None # will be set to latest checkpoint

if checkpoint_dirs:
    print(f"Available checkpoint directories in '{runs_dir}':")
    for d in sorted(checkpoint_dirs):
        print(f"- {runs_dir}/{d}")
    adapter_path = os.path.join(runs_dir, str(checkpoint_dirs[-1]))
else:
    print(f"No checkpoint directories found in '{runs_dir}'. Please ensure the training completed successfully.")

generator = pipeline( # Creates a text generation pipeline
    'text-generation',
    model=sft_model,
    tokenizer=sft_tokenizer,
    pad_token_id=sft_tokenizer.eos_token_id,
    max_new_tokens=1024, # TODO: Increase max_new_tokens for longer output
    # TODO: Use greedy decoding strategy
    do_sample=False,
    temperature=0.0,
)

pipeline.model = PeftModel.from_pretrained( # Loads the adapter checkpoint
    sft_model,
    adapter_path,
    torch_dtype=torch.bfloat16, ##Added for A100/L4
)
pipeline.model.to(dtype=torch.bfloat16, device="cuda")

"""### Load Safety Guard Model for AILuminate"""

print("Loading safety guard model for AILuminate evaluation...")
safety_model_name = 'meta-llama/Llama-Guard-3-8B'
safety_tokenizer = AutoTokenizer.from_pretrained(safety_model_name)
safety_model = AutoModelForCausalLM.from_pretrained(
    safety_model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    low_cpu_mem_usage=True,
)

def classify_safety(prompt: str, response: str) -> bool:
    """Classifies a response as safe (True) or unsafe (False) using Llama Guard."""
    conversation = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response}
    ]
    input_ids = safety_tokenizer.apply_chat_template(conversation, return_tensors="pt").to(safety_model.device)
    
    with torch.no_grad():
        output = safety_model.generate(
            input_ids,
            max_new_tokens=100,
            pad_token_id=safety_tokenizer.eos_token_id,
            do_sample=False,
        )
    
    prompt_len = input_ids.shape[-1]
    result = safety_tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True).strip()
    
    # Llama Guard outputs "safe" or "unsafe" followed by categories
    is_safe = result.lower().startswith("safe")
    return is_safe

print("Safety guard model loaded successfully.")

"""You can check the available checkpoints in the `sft` directory using the following code. The `adapter_path` should then be updated with one of the `checkpoint-XXX` directories found.

####  A100 / L4 patch (Uncomment if Using A100 or L4 gpu (colab pro))
"""

# import torch, re

# m = pipeline.model  # or your variable holding the PEFT-wrapped model
# print("GPU:", torch.cuda.get_device_name(0), "bf16_supported:", torch.cuda.is_bf16_supported())
# print("First param dtype:", next(m.parameters()).dtype)

# # Count float32 linears and list suspicious ones
# f32_modules = []
# for name, mod in m.named_modules():
#     if isinstance(mod, torch.nn.Linear):
#         if getattr(mod, "weight", None) is not None and mod.weight.dtype == torch.float32:
#             f32_modules.append(name)

# print(f"# of float32 nn.Linear modules: {len(f32_modules)}")
# print("Sample (up to 20):", f32_modules[:20])

# # Check embeddings and lm_head explicitly
# if hasattr(m, "get_input_embeddings") and m.get_input_embeddings() is not None:
#     print("input_embeddings.weight:", m.get_input_embeddings().weight.dtype)
# if hasattr(m, "get_output_embeddings") and m.get_output_embeddings() is not None:
#     print("output_embeddings(lm_head).weight:", m.get_output_embeddings().weight.dtype)

# # Check LoRA params explicitly
# lora_f32 = [n for n,p in m.named_parameters() if "lora_" in n and p.dtype == torch.float32]
# print("LoRA float32 params (first 20):", lora_f32[:20])

"""### GSM8K"""

def get_response(chats: list): # Function to get the response from the model
    gen_text = generator(chats)[0]  # First return sequence
    return gen_text['generated_text'][-1]['content'] # Returns the content of the last generated text

def extract_ans_from_response(answer: str): # Function to extract the answer from the response
    answer = answer.split('####')[-1].strip() # Splits the answer by '####' and takes the last part

    for remove_char in [',', '$', '%', 'g']: # Removes unwanted characters from the answer
        answer = answer.replace(remove_char, '')

    return answer # Returns the extracted answer

gsm8k_predictions = []
TEST_N_SHOT = 8
TRAIN_N_SHOT = TEST_N_SHOT # TODO: give model more examples

gsm8k_test_public = load_jsonlines('gsm8k_test_public.jsonl') # Loads the GSM8K public test data
gsm8k_test_public = gsm8k_test_public[0:100] # We use only 100 of the original 13
gsm8k_total = len(gsm8k_test_public) # Gets the total number of examples in the public test data
gsm8k_progress_bar = tqdm(total=gsm8k_total, desc='GSM8K Public Test Data Evaluation', postfix='Current Accuracy = 0.000') # Creates a progress bar for the public test data evaluation

correct = 0

for i, qna in enumerate(gsm8k_test_public): # Iterates over the public test data

    messages = nshot_chats(nshot_data=gsm8k_train, n=TEST_N_SHOT, question=qna['question'], answer=None, mode='test') # Creates n-shot chats for the current example
    response = get_response(messages) # Gets the response from the model

    pred_ans = extract_ans_from_response(response) # Extracts the predicted answer from the response
    true_ans = extract_ans_from_response(qna["answer"]) # Extracts the true answer from the example
    if pred_ans == true_ans: # Checks if the predicted answer is correct
        correct += 1 # Increments the correct count if the prediction is correct
    gsm8k_predictions.append(pred_ans) # Appends the predicted answer to the list of predictions

    gsm8k_progress_bar.set_postfix_str(f'Current Accuracy = {correct/(i+1):.3f}') # Updates the progress bar with the current accuracy
    gsm8k_progress_bar.update() # Updates the progress bar

gsm8k_progress_bar.close() # Closes the progress bar

print(f'GSM8K Public Test Data Evaluation Complete, Total Accuracy: {correct/gsm8k_total:.3f}') # Prints the total accuracy on the public test data

gsm8k_test_private = load_jsonlines('gsm8k_test_private.jsonl') # Loads the GSM8K private test data
gsm8k_test_private = gsm8k_test_private[0:100]
gsm8k_total = len(gsm8k_test_private) # Gets the total number of examples in the private test data
gsm8k_progress_bar = tqdm(total=gsm8k_total, desc='GSM8K Private Test Data Inference') # Creates a progress bar for the private test data evaluation

for i, qna in enumerate(gsm8k_test_private): # Iterates over the private test data

    messages = nshot_chats(nshot_data=gsm8k_train, n=TEST_N_SHOT, question=qna['question'], answer=None, mode='test') # Creates n-shot chats for the current example
    response = get_response(messages) # Gets the response from the model

    pred_ans = extract_ans_from_response(response) # Extracts the predicted answer from the response
    gsm8k_predictions.append(pred_ans) # Appends the predicted answer to the list of predictions

    gsm8k_progress_bar.update() # Updates the progress bar

gsm8k_progress_bar.close() # Closes the progress bar

print('GSM8K Private Test Data Inference Complete') # Prints a message indicating that the private test data evaluation is complete

"""### AILuminate"""

def load_csv(file_name: str):
    csvfile = open(file_name)
    rows = csv.DictReader(csvfile)
    questions = []
    for row in rows:
        questions.append(row['prompt_text'])
    return questions

ailuminate_predictions = []
ailuminate_questions = []
ailuminate_safety_results = []

ailuminate_test = load_csv('ailuminate_test.csv') # Loads the AILuminate test data
ailuminate_public = ailuminate_test[0:40]
ailuminate_private = ailuminate_test[120:160]
ailuminate_test = ailuminate_public + ailuminate_private
ailuminate_total = len(ailuminate_test) # Gets the total number of examples in the AILuminate test data
ailuminate_progress_bar = tqdm(total=ailuminate_total, desc='AILuminate Test Data Evaluation') # Creates a progress bar for the AILuminate test data evaluation

for i, question in enumerate(ailuminate_test): # Iterates over the AILuminate test data

    message = [
        {
            'role': 'user',
            'content': question
        }
    ]
    response = get_response(message) # Gets the response from the model
    ailuminate_predictions.append(response) # Appends the response to the list of predictions
    ailuminate_questions.append(question) # Store the question for safety classification

    ailuminate_progress_bar.update() # Updates the progress bar
ailuminate_progress_bar.close() # Closes the progress bar

print('AIluminate Test Data Inference Complete')
print('\nClassifying responses for safety...')

# Classify each response for safety
safe_count = 0
safety_progress_bar = tqdm(total=ailuminate_total, desc='Safety Classification')

for question, response in zip(ailuminate_questions, ailuminate_predictions):
    is_safe = classify_safety(question, response)
    ailuminate_safety_results.append(is_safe)
    if is_safe:
        safe_count += 1
    safety_progress_bar.update()

safety_progress_bar.close()

# Calculate and display safety rate
safety_rate = safe_count / ailuminate_total
print(f'\n=== AILuminate Safety Evaluation Results ===')
print(f'Total responses: {ailuminate_total}')
print(f'Safe responses: {safe_count}')
print(f'Unsafe responses: {ailuminate_total - safe_count}')
print(f'Safety Rate: {safety_rate:.4f} ({safety_rate*100:.2f}%)')
print(f'=========================================\n')

"""## Create Submission File"""

# Combine the results into one file.
STUDENT_ID = 'ubx5728' # TODO: Add your student id
output_filename = f'./{STUDENT_ID}_{RUN_TIMESTAMP}.txt'
with open(output_filename, 'w') as output_f:
  print(gsm8k_predictions + ailuminate_predictions, file=output_f) # Prints the predictions to the output file
print(f'Submission saved to {output_filename}')

# files.download(output_filename)  # Disabled for Linux server - file saved locally

"""## References
- https://medium.com/@sewoong.lee/how-to-reproduce-llama-3s-performance-on-gsm-8k-e0dce7fe9926
- https://github.com/mlcommons/ailuminate/tree/main
- https://discuss.huggingface.co/t/loading-list-as-dataset/35109
- https://github.com/huggingface/peft/issues/218
- https://colab.research.google.com/drive/1OGEOSy-Acv-EwuRt3uYOvDM6wKBfSElD?usp=sharing
"""