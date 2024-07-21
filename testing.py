import numpy as np
from training import run_training, MAX_NEW_TOKENS, PROMPT_FIELD, TAG_FIELD


TAG_MAPPING = {'Yes': 1, 'No': -1}

YES_IDS = None
NO_IDS = None


def get_completion_vocab_ids(tokenizer):
    global YES_IDS, NO_IDS
    if YES_IDS is None or NO_IDS is None:
        YES_IDS = [idx for vocab_word, idx in tokenizer.vocab.items() 
                   if vocab_word.strip(chr(9601)).strip().lower() == 'yes']
        NO_IDS = [idx for vocab_word, idx in tokenizer.vocab.items() 
                  if vocab_word.strip(chr(9601)).strip().lower() == 'no']
        print(f"Assessing {len(YES_IDS)} optional 'yes' tokens and {len(NO_IDS)} optional 'no' tokens.")
    return YES_IDS, NO_IDS


def test_model():
    model, tokenizer, train, val, test = run_training()

    print("Finished training of model!")
    
    print("Evaluating on test set...")
    results, scores = evaluate_model_on_test_set(test, model, tokenizer)

    print("Evaluating on val set...")
    evaluate_model_on_test_set(val, model, tokenizer)

    print("Evaluating on train set...")
    evaluate_model_on_test_set(train, model, tokenizer)

    return results, scores, model, tokenizer, train, val, test


def evaluate_model_on_test_set(test_set, model, tokenizer):
    results = []
    for example in test_set:
        prediction = inference(example[PROMPT_FIELD], model, tokenizer)
        tag = TAG_MAPPING[example[TAG_FIELD]]
        correct = prediction == tag
        print(("X", "V")[correct], end='', flush=True)
        results.append((tag, prediction, correct))
    
    print(results)
    scores = get_scores(results)

    return results, scores

def get_scores(results):
    total = len(results)

    num_correct = len([l for l in results if l[2]])
    print("Num correct:", num_correct)
    percent_correct = round(num_correct / total, 2)
    print(f"{percent_correct}%")
    num_ones = len([l for l in results if l[1] == 1])
    print("Num ones:", num_ones)
    percent_ones = round(num_ones / total, 2)
    print(f"{percent_ones}%")

    tp, tn, fp, fn = 0, 0, 0, 0
    for tag, prediction, correct in results:
        if tag == 1 and correct:
            tp += 1
        elif tag == -1 and correct:
            tn += 1
        elif prediction == 1 and not correct:
            fp += 1
        elif prediction == -1 and not correct:
            fn += 1
        else:
            raise ValueError(f"Huh?: {tag}, {prediction}, {correct}")
    
    accuracy = (tp + tn) / total
    precision = tp / (tp + fp) if tp + fp != 0 else 0
    recall = tp / (tp + fn) if tp + fn != 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if precision + recall != 0 else 0

    print("Accuracy:", round(accuracy, 2))
    print("Precision:", round(precision, 2))
    print("Recall:", round(recall, 2))
    print("F1:", round(f1, 2))
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }



def inference(prompt, model, tokenizer):
    yes_ids, no_ids = get_completion_vocab_ids(tokenizer)

    inputs = tokenizer(prompt, return_tensors="pt")
    
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
    
    return final_prediction


if __name__ == "__main__":
    test_model()
