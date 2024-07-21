# from get_device import get_device_number, GPU_NUMBER

# if GPU_NUMBER is None:
#     GPU_NUMBER = get_device_number(set_visible_devices=True)

# import os
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
# print("Setting: os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'")

import subprocess
import os
from datetime import datetime
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import SFTConfig, SFTTrainer
from datasets import Dataset
from huggingface_hub import login
import wandb
from accelerate import Accelerator


from data_prep import Preprocess, SESSION_PREFIX, SESSION_POSTFIX

MODEL_ORIGINAL = 'meta-llama/Llama-2-7b-chat-hf'
MODEL_QUANTIZED = 'TheBloke/Llama-2-7B-Chat-GPTQ'
MODEL = MODEL_ORIGINAL

MAX_CONTEXT_LENGTH = 4096
MAX_NEW_TOKENS = 1

NOW_STR = datetime.now().strftime('%Y%m%d-%H%M%S')
OUTPUT_DIR = "/workspace/repos/results/llama2_qlora_" + NOW_STR
RUN_NAME = 'yuvalarbel-danaatzil-training-' + NOW_STR
print("WANDB Run Name:\n" + RUN_NAME)
print("Will save model to:", OUTPUT_DIR.replace('/workspace', '~'))

PROMPT_FIELD = "prompt"
TAG_FIELD = "completion"
TAG_MAPPING = {1: 'Yes', -1: 'No'}

HUGGINGFACE_TOKEN = os.getenv('HUGGINGFACE_TOKEN')
WANDB_API_KEY = os.getenv('WANDB_API_KEY')

GPU_FREE_SPACE_COMMAND = "nvidia-smi --query-gpu=memory.free --format=csv,nounits,noheader"


def run_training():
    output = subprocess.check_output(GPU_FREE_SPACE_COMMAND.split())
    free_spaces = list(map(int, output.decode().splitlines()))
    gb_free_spaces = [round(i / 1024, 3) for i in free_spaces] 
    print(f"\nGPUs available:", gb_free_spaces)

    all_logins()

    accelerator = Accelerator()

    train, val, test = Preprocess().run()

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,  # Activate 4-bit precision base model loading,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",  # Quantization type (fp4 or nf4),
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    tokenizer.padding_side = 'right'
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL, 
        quantization_config=bnb_config,
        device_map="auto",
    )

    train = prepare_dataset_for_training(train, tokenizer)
    val = prepare_dataset_for_training(val, tokenizer)
    test = prepare_dataset_for_training(test, tokenizer)

    lora_config = LoraConfig(
        r=8,
        target_modules = ["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
        lora_dropout=0.1,
    )

    lora_model = get_peft_model(model, lora_config)

    args = SFTConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        learning_rate=2e-4,
        weight_decay=0.003,
        max_grad_norm=0.3,
        logging_steps=1,
        # max_steps=5,
        num_train_epochs=3,
        eval_strategy="steps",
        eval_steps=30,
        fp16=True,
        optim="paged_adamw_8bit",
        dataset_text_field=PROMPT_FIELD,
        max_seq_length=MAX_CONTEXT_LENGTH,
        run_name=RUN_NAME
    )

    trainer = SFTTrainer(
        model=lora_model,
        args=args,
        tokenizer=tokenizer,
        train_dataset=train,
        eval_dataset=val,
        peft_config=lora_config,
    )

    trainer = accelerator.prepare(trainer)

    trainer.train()

    output_dir = os.path.join(OUTPUT_DIR, "final_checkpoint")
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(output_dir)

    return unwrapped_model, tokenizer, train, val, test


def _check_num_tokens(text, tokenizer):
    inputs = tokenizer(text, return_tensors="pt")
    return len(inputs['input_ids'][0])


def prepare_dataset_for_training(dataset, tokenizer):
    # Strip to only prompt and tag fields
    dataset[TAG_FIELD] = dataset['tag'].map(TAG_MAPPING)
    dataset = dataset.rename(columns={'eng_session_plaintext': PROMPT_FIELD})[[PROMPT_FIELD, TAG_FIELD]].copy()

    # Shorten prompt fields to allow tokenization
    for idx, row in dataset.iterrows():
        prompt = row[PROMPT_FIELD]
        num_tokens = _check_num_tokens(prompt, tokenizer)
    
        if num_tokens >= MAX_CONTEXT_LENGTH - MAX_NEW_TOKENS:
            while num_tokens >= MAX_CONTEXT_LENGTH - MAX_NEW_TOKENS:
                session_speech_turns = prompt.replace(SESSION_PREFIX, '').replace(SESSION_POSTFIX, '').strip().splitlines()
                prompt = SESSION_PREFIX + "\n".join(session_speech_turns[:-1]) + SESSION_POSTFIX
                num_tokens = _check_num_tokens(prompt, tokenizer)
            dataset.at[idx, PROMPT_FIELD] = prompt
    
    dataset = Dataset.from_pandas(dataset)
    return dataset


def all_logins():
    if HUGGINGFACE_TOKEN:
        login(token=HUGGINGFACE_TOKEN)
        print("Logged in to Hugging Face")
    else:
        print("Hugging Face token not found")

    if WANDB_API_KEY:
        wandb.login(key=WANDB_API_KEY)
        print("Logged in to WandB")
    else:
        print("WandB API Key not found")


if __name__ == "__main__":
    run_training()
