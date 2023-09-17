from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import load_metric
from cleaning import cleanData
from transformers import AutoTokenizer, DataCollatorWithPadding
from datasets import load_dataset
import torch

MAX_INPUT_LENGTH = 1024  # Maximum length of the input to the model
MIN_TARGET_LENGTH = 5  # Minimum length of the output by the model
MAX_TARGET_LENGTH = 128  # Maximum length of the output by the model
BATCH_SIZE = 1    # Batch-size for training our model
LEARNING_RATE = 2e-5  # Learning-rate for training our model
MAX_EPOCHS = 3  # Maximum number of epochs we will train the model for


device = "cuda:0" if torch.cuda.is_available() else "cpu"

print(f"device:{device}")


raw_datasets = load_dataset("pn_summary")
checkpoint = "google/mt5-base"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


print("dataset loaded")
# just for test
# raw_datasets["train"]=raw_datasets["train"].select(range(10000,11000))
# raw_datasets["validation"]=raw_datasets["validation"].select(range(0, 100))
# raw_datasets["test"]=raw_datasets["test"].select(range(0, 100))
# just for test


prefix = "summurize:"


def preprocess_function(examples):
    inputs = [prefix + doc for doc in examples["article"]]
    inputs = cleanData(inputs)
    model_inputs = tokenizer(inputs, max_length=MAX_INPUT_LENGTH,
                             return_tensors="pt", truncation=True, padding=True)

    labels = tokenizer(
        examples["summary"], max_length=MAX_TARGET_LENGTH, return_tensors="pt", truncation=True, padding=True)

    model_inputs["labels"] = labels["input_ids"]

    return model_inputs


tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)
columns_to_return = ['input_ids', 'labels', 'attention_mask']
tokenized_datasets.set_format(type='torch', columns=columns_to_return)
tokenized_datasets = tokenized_datasets.remove_columns(
    raw_datasets["train"].column_names)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


print("preproccing done")
print("Loading the model ")
model = torch.load("mymodel")  # load mT5 model form local
print("Model Loaded")


# model= model.to(device)
for name, param in model.named_parameters():
    if 'bias' in name:
        param.requires_grad = False

for name, param in model.base_model.encoder.named_parameters():
    param.requires_grad = False


rouge2 = load_metric('./rouge.py')


def metric_fn(eval_predictions):
    predictions, labels = eval_predictions
    decoded_predictions = tokenizer.batch_decode(
        predictions, skip_special_tokens=True)
    for label in labels:
        # Replace masked label tokens
        label[label < 0] = tokenizer.pad_token_id
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    scores = rouge2.compute(
        predictions=decoded_labels,
        references=decoded_predictions,
        rouge_types=["rouge1", "rouge2", "rougeL"],
        use_stemmer=True,
        lang='fa'
    )

    result = {
        "Rouge1": scores["rouge1"].mid.fmeasure,
        "Rouge2": scores["rouge2"].mid.fmeasure,
        "RougeL": scores["rougeL"].mid.fmeasure


    }

    # result=  scores["rougeL"].mid.recall

    return result

    # /////////////////////////////////////////////////////////////


training_args = Seq2SeqTrainingArguments(
    "test-trainer",
    evaluation_strategy="epoch",
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=MAX_EPOCHS,
    predict_with_generate=True,


)


trainer = Seq2SeqTrainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["evaluation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=metric_fn
)


print("start training encoder")

trainer.train()


for name, param in model.base_model.encoder.named_parameters():
    param.requires_grad = True

for name, param in model.base_model.decoder.named_parameters():
    param.requires_grad = False

for name, param in model.named_parameters():
    if 'bias' in name:
        param.requires_grad = False


training_args = Seq2SeqTrainingArguments(
    "test-trainer",
    evaluation_strategy="epoch",
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=MAX_EPOCHS,
    predict_with_generate=True,


)


trainer = Seq2SeqTrainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["evaluation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=metric_fn
)

trainer.train()

for name, param in model.base_model.encoder.named_parameters():
    param.requires_grad = False

for name, param in model.base_model.decoder.named_parameters():
    param.requires_grad = False

for name, param in model.named_parameters():
    if 'bias' in name:
        param.requires_grad = True


training_args = Seq2SeqTrainingArguments(
    "test-trainer",
    evaluation_strategy="epoch",
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=MAX_EPOCHS,
    predict_with_generate=True,


)


trainer = Seq2SeqTrainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["evaluation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=metric_fn
)

trainer.train()


torch.save(model, "./models/mt5_finetuend.pth")

print("done!!!!!!")
