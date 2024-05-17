from get_device import get_device_number, GPU_NUMBER

if GPU_NUMBER is None:
    GPU_NUMBER = get_device_number(set_visible_devices=True)

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
print("Setting: os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'")

import torch
from transformers import TrainingArguments
from peft import LoraConfig
from trl import ModelConfig, SFTTrainer, get_kbit_device_map, get_peft_config, get_quantization_config
from datasets import Dataset
from unsloth import FastLanguageModel

from data_prep import Preprocess, SESSION_PREFIX, SESSION_POSTFIX

MODEL_ORIGINAL = 'meta-llama/Llama-2-7b-chat-hf'
MODEL_QUANTIZED = 'TheBloke/Llama-2-7B-Chat-GPTQ'
MODEL = MODEL_ORIGINAL

MAX_CONTEXT_LENGTH = 4096
MAX_NEW_TOKENS = 1

OUTPUT_DIR = "/workspace/repos/results/llama2_qlora_try1"

PROMPT_FIELD = "prompt"
TAG_FIELD = "completion"
TAG_MAPPING = {1: 'Yes', -1: 'No'}


def run_training():
    train, val, test = Preprocess().run()
    # device = torch.device('cuda')

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL, 
        max_seq_length=MAX_CONTEXT_LENGTH, 
        dtype=None,  # torch.float16, 
        load_in_4bit=True,
        # device_map=device,
    )

    train = prepare_dataset_for_training(train, tokenizer)
    val = prepare_dataset_for_training(val, tokenizer)

    model = FastLanguageModel.get_peft_model(
        model,
        r = 16,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 16,
        lora_dropout = 0, # Dropout = 0 is currently optimized
        bias = "none",    # Bias = "none" is currently optimized
        use_gradient_checkpointing = True,
        random_state = 3407,
    )

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=10,
        max_steps=500
    )

    tokenizer.padding_side = 'right'
    tokenizer.pad_token = tokenizer.eos_token

    # model = model.to(device)

    trainer = SFTTrainer(
        model=model,
        args=args,
        tokenizer=tokenizer,
        train_dataset=train,
        eval_dataset=val,
        dataset_text_field=PROMPT_FIELD,
        max_seq_length=MAX_CONTEXT_LENGTH,
    )

    trainer.train()

    # Problem: ptxas /tmp/compile-ptx-src-deebe9, line 406; error   : Feature '.bf16' requires .target sm_80 or higher

    output_dir = os.path.join(OUTPUT_DIR, "final_checkpoint")
    trainer.model.save_pretrained(output_dir)


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


# def get_trainer(train, val, tokenizer):
    # peft_config = LoraConfig(
    #     lora_alpha=16,
    #     lora_dropout=0.1,
    #     r=64,
    #     bias="none",
    #     task_type="CAUSAL_LM",
    # )

    # training_args = TrainingArguments(
    #     output_dir=OUTPUT_DIR,
    #     per_device_train_batch_size=4,
    #     gradient_accumulation_steps=4,
    #     learning_rate=2e-4,
    #     logging_steps=10,
    #     max_steps=500
    # )

    # tokenizer.padding_side = 'right'
    # tokenizer.pad_token = tokenizer.eos_token

    # trainer = SFTTrainer(
    #     model=base_model,
    #     train_dataset=train,
    #     eval_dataset=val,
    #     dataset_text_field=PROMPT_FIELD,
    #     peft_config=peft_config,
    #     max_seq_length=MAX_CONTEXT_LENGTH,
    #     tokenizer=tokenizer,
    #     args=training_args,
    # )

    # return trainer


if __name__ == "__main__":
    run_training()
