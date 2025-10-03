from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    TrainingArguments,
    Trainer
)
import evaluate
import torch

# Load CNN/DailyMail dataset
dataset = load_dataset("cnn_dailymail", "3.0.0")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")

# Preprocess
def preprocess_function(examples):
    inputs = [doc for doc in examples["article"]]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True)
    labels = tokenizer(examples["highlights"], max_length=128, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_datasets = dataset.map(preprocess_function, batched=True)

# Create train/test/validation splits with smaller subset
split_dataset = tokenized_datasets["train"].train_test_split(test_size=0.1)

tokenized_datasets = DatasetDict({
    "train": split_dataset["train"].select(range(500)),        # Small subset for fast training
    "validation": split_dataset["test"].select(range(100)),     # Small validation set
    "test": tokenized_datasets["test"]                           # Full test set
})

# Load model
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")

# Data collator (compatible with older and newer transformers)
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=32,
    per_device_eval_batch_size=16,
    learning_rate=2e-5,
    weight_decay=0.01,
    num_train_epochs=3,
    logging_dir="./logs",
    fp16=True
)

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator
)

# Train
trainer.train()

# Generate summary from one sample
def generate_summary(text):
    inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
    inputs = {k: v.to("cuda") for k, v in inputs.items()}
    model.to("cuda")
    model.eval()
    with torch.no_grad():
        summary_ids = model.generate(inputs["input_ids"], max_length=128, num_beams=4, early_stopping=True)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

sample_text = dataset["test"][0]["article"]
print("Generated Summary:", generate_summary(sample_text))

# ROUGE evaluation using `evaluate`
rouge = evaluate.load("rouge")

def compute_rouge(predictions, references):
    scores = rouge.compute(predictions=predictions, references=references)
    return scores

# Generate predictions on first 10 test articles
pred_summaries = [generate_summary(article) for article in dataset["test"]["article"][:10]]
ref_summaries = dataset["test"]["highlights"][:10]

rouge_scores = compute_rouge(pred_summaries, ref_summaries)
print("ROUGE Scores:", rouge_scores)

# Save fine-tuned model and tokenizer
model.save_pretrained("./fine_tuned_bart")
tokenizer.save_pretrained("./fine_tuned_bart")
