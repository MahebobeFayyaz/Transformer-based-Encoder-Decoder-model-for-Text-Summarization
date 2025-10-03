from load_model import model, tokenizer
from utils import generate_summary

while True:
    user_input = input("\nEnter an article to summarize (or type 'exit' to quit):\n")
    if user_input.strip().lower() == 'exit':
        break

    summary = generate_summary(user_input, tokenizer, model)
    print("\n📝 Summary:\n", summary)
