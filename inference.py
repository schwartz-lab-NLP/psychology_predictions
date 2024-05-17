from get_device import get_device_number, GPU_NUMBER

if GPU_NUMBER is None:
    GPU_NUMBER = get_device_number(set_visible_devices=True)

from data_prep import Preprocess, SESSION_PREFIX, SESSION_POSTFIX
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

assert torch.cuda.is_available(), "CUDA is not available"

MODEL_ORIGINAL = 'meta-llama/Llama-2-7b-chat-hf'
MODEL_QUANTIZED = 'TheBloke/Llama-2-7B-Chat-GPTQ'
DEFAULT_MODEL = MODEL_QUANTIZED

MAX_CONTEXT_LENGTH = 4096
MAX_NEW_TOKENS = 1



def get_model(model, device="auto", fp4=False):
    if not fp4:
        return AutoModelForCausalLM.from_pretrained(
            model, 
            device_map=device
        )

    from transformers import BitsAndBytesConfig

    use_4bit = True  # Activate 4-bit precision base model loading
    bnb_4bit_compute_dtype = "float16"  # Compute dtype for 4-bit base models
    bnb_4bit_quant_type = "fp4"  # Quantization type (fp4 or nf4)
    use_nested_quant = False  # Activate nested quantization for 4-bit base models (double quantization)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=use_4bit,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=bnb_4bit_compute_dtype,
        bnb_4bit_use_double_quant=use_nested_quant,
    )

    return AutoModelForCausalLM.from_pretrained(
        model,
        quantization_config=bnb_config,
        device_map=device
    )


def get_all_assets(model_name=DEFAULT_MODEL, fp4=False):
    if GPU_NUMBER is None:
        raise Exception("First you should run get_device_number before importing torch.")
    device = torch.device('cuda')

    train, val, test = Preprocess().run()
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = get_model(model_name, device="auto", fp4=fp4)

    return train, val, test, tokenizer, model, device


def tokenize(prompt, tokenizer, device):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    num_tokens = len(inputs['input_ids'][0])
    
    if num_tokens >= MAX_CONTEXT_LENGTH - MAX_NEW_TOKENS:
        session_speech_turns = prompt.replace(SESSION_PREFIX, '').replace(SESSION_POSTFIX, '').strip().splitlines()
        new_prompt = SESSION_PREFIX + "\n".join(session_speech_turns[:-1]) + SESSION_POSTFIX
        tokenized_new_prompt, level = tokenize(new_prompt, tokenizer, device)
        return tokenized_new_prompt, level + 1
    return inputs, 0


def run_check(dataset, tokenizer, model, device):
    yes_ids = [idx for vocab_word, idx in tokenizer.vocab.items() 
                if vocab_word.strip(chr(9601)).strip().lower() == 'yes']
    no_ids = [idx for vocab_word, idx in tokenizer.vocab.items() 
                if vocab_word.strip(chr(9601)).strip().lower() == 'no']
    print(f"Assessing {len(yes_ids)} optional 'yes' tokens and {len(no_ids)} optional 'no' tokens.")

    results = []
    for example in dataset.itertuples():
        prompt = example.eng_session_plaintext
        inputs, shortened = tokenize(prompt, tokenizer, device)
        num_tokens = len(inputs['input_ids'][0])

        outputs = model.generate(**inputs, 
                                 max_new_tokens=MAX_NEW_TOKENS,
                                 output_scores=True,
                                 return_dict_in_generate=True)
        next_word_vocab_logits = outputs.scores[0][0]

        max_score = -np.inf
        final_prediction = None
        for token_list, prediction in ((yes_ids, 1), (no_ids, -1)):
            for token in token_list:
                pred = next_word_vocab_logits[token]
                if pred > max_score:
                    final_prediction = prediction
                    max_score = pred
                
        correct = ("X", "V")[final_prediction == example.tag]
        print(correct, end='', flush=True)

        result = {
            "name": example.idx, 
            "num_tokens": num_tokens, 
            "shortened": shortened, 
            "tag": example.tag, 
            "pred": final_prediction, 
            "correct": correct,
        }
        results.append(result)
    
    return results


def print_results(results):
    print("\n")
    print("Correct:", len(list(filter(lambda res: res['correct'] == 'V', results))))
    print("Mistake:", len(list(filter(lambda res: res['correct'] == 'X', results))))
    print()
    print("TP:", len(list(filter(lambda res: res['pred'] == 1 and res['correct'] == 'V', results))))
    print("FP:", len(list(filter(lambda res: res['pred'] == 1 and res['correct'] == 'X', results))))
    print("TN:", len(list(filter(lambda res: res['pred'] == -1 and res['correct'] == 'V', results))))
    print("FN:", len(list(filter(lambda res: res['pred'] == -1 and res['correct'] == 'X', results))))


def run_all():
    train, val, test, tokenizer, model, device = get_all_assets()
    results = run_check(val, tokenizer, model, device)
    print_results(results)
    return results
