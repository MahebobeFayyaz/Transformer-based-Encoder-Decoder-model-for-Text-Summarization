from load_model import model, tokenizer
from utils import generate_summary
from datasets import load_dataset
import evaluate
import json

# Load dataset
dataset = load_dataset("cnn_dailymail", "3.0.0")
rouge = evaluate.load("rouge")

# Generate summaries and compute scores
preds = [generate_summary(article, tokenizer, model) for article in dataset["test"]["article"][:10]]
refs = dataset["test"]["highlights"][:10]
scores = rouge.compute(predictions=preds, references=refs)

# Save to JSON
scores_clean = {k: float(v) for k, v in scores.items()}
with open("rouge_scores.json", "w") as f:
    json.dump(scores_clean, f, indent=4)

print("✅ ROUGE scores saved to rouge_scores.json")
