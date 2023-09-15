import torch

model=torch.load("./models/mt5_plain_full")
print("model loaded")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from transformers import  AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('google/mt5-base')

from datasets import load_dataset

raw_datasets = load_dataset("pn_summary")


val_texts = [] # list of validation texts
val_summaries = [] # list of validation summaries

for i in range(len(raw_datasets["test"])):
    val_texts.append(raw_datasets["test"][i]["article"])
    val_summaries.append(raw_datasets["test"][i]["summary"])


test_data = [] # list of test examples (text and summary pairs)
for i in range(len(val_texts)):
    test_data.append({'text':val_texts[i],'summary':val_summaries[i]})

from rouge_score import rouge_scorer
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True,tokenizer=tokenizer)
model.to(device)



# Define the evaluation function for calculating rouge
def evaluate(model, tokenizer, data):
    total_scores = {'rouge1': 0, 'rouge2': 0, 'rougeL': 0}

    for example in data:
        text = example['text']
        summary = example['summary']

        # Generate the summary using the model
        inputs = tokenizer(text, padding='max_length', truncation=True, max_length=512, return_tensors='pt').to(device)
        outputs = model.generate(inputs['input_ids'], attention_mask=inputs['attention_mask'], max_length=128, early_stopping=True)
        generated_summary = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Calculate the ROUGE scores
        scores = scorer.score(summary, generated_summary)
        for metric, score in scores.items():
            total_scores[metric] += score.fmeasure

    # Calculate the average ROUGE scores
    num_examples = len(data)
    avg_scores = {metric: score / num_examples for metric, score in total_scores.items()}
    return avg_scores




test_scores = evaluate(model, tokenizer, test_data)
print(f"ROUGE-1: {test_scores['rouge1']:.4f}")
print(f"ROUGE-2: {test_scores['rouge2']:.4f}")
print(f"ROUGE-L: {test_scores['rougeL']:.4f}")