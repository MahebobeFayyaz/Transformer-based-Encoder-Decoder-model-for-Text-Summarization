from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Load fine-tuned model and tokenizer
model = AutoModelForSeq2SeqLM.from_pretrained("fine_tuned_bart")
tokenizer = AutoTokenizer.from_pretrained("fine_tuned_bart")
model.to("cuda")

print("✅ Model and tokenizer loaded.")
