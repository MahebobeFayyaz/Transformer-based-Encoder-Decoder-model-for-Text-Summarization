import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch, re, json, pandas as pd, os

# ——— Configuration ———
MAX_WORDS = 200     # word limit
CHAR_LIMIT = 1200   # approx. chars for 200 words

# ——— 1) Load your fine-tuned BART model & tokenizer ———
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("fine_tuned_bart")
    model     = AutoModelForSeq2SeqLM.from_pretrained("fine_tuned_bart").to("cuda")
    return tokenizer, model

# ——— 2) Summarization function ———
def generate_summary(text: str, tokenizer, model) -> str:
    words = text.split()
    # Echo very short inputs
    if len(words) < 10:
        return text.strip().capitalize().rstrip('.') + '.'

    # Tokenize & move to GPU
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
    inputs = {k: v.to("cuda") for k, v in inputs.items()}

    model.eval()
    with torch.no_grad():
        summary_ids = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            num_beams=4,
            length_penalty=2.0,
            max_length=120,
            min_length=60,
            no_repeat_ngram_size=3,
            do_sample=True,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.5,
            early_stopping=True
        )

    # Decode and clean up
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True).strip()
    if summary and summary[-1] not in ".!?":
        summary = summary.rstrip('.') + '.'

    # Deduplicate exact sentences
    sentences = re.split(r'(?<=[\.!?])\s+', summary)
    seen, unique = set(), []
    for s in sentences:
        norm = s.lower().strip()
        if norm and norm not in seen:
            unique.append(s.strip())
            seen.add(norm)

    return " ".join(unique)

# ——— 3) Streamlit UI setup ———
st.set_page_config(page_title="Text Summarizer", page_icon="📝", layout="wide")
st.title("📝 Text Summarizer (Fine-Tuned BART)")

# ——— File uploader with 200-word limit & title extraction ———
uploaded = st.file_uploader("📄 Upload a .txt file (max 200 words)", type="txt")
text_input, title = "", "Untitled"

if uploaded:
    raw = uploaded.read().decode("utf-8")
    words = raw.split()
    if len(words) > MAX_WORDS:
        st.warning(f"Uploaded file has {len(words)} words; truncating to {MAX_WORDS}.")
        raw = " ".join(words[:MAX_WORDS])
    lines = [l.strip() for l in raw.splitlines() if l.strip()]
    title = lines[0] if lines else "Untitled"
    body  = "\n".join(lines[1:]) if len(lines) > 1 else ""
    st.markdown(f"### 📰 Detected Title: `{title}`")
    st.code("\n".join(lines[:5]), language="markdown")
    text_input = body

# ——— Manual input area with character cap & live word count ———
text_input = st.text_area(
    "✍️ Input Text (or paste here, max 200 words)",
    value=text_input,
    height=300,
    max_chars=CHAR_LIMIT,
    help="Approx. 200-word limit (max ~1200 characters)."
)
word_count = len(text_input.split())
st.caption(f"📝 Word count: {word_count}/200")

if word_count > MAX_WORDS:
    st.warning("⛔️ 200-word limit reached. Extra words have been removed.")
    text_input = " ".join(text_input.split()[:MAX_WORDS])

# ——— Save format choice ———
save_format = st.radio("💾 Save summary as:", [".txt", ".json"], horizontal=True)

# ——— Sidebar: history display & clear ———
hist_file = "summary_history.csv"
if st.sidebar.button("📜 Show Summary History") and os.path.exists(hist_file):
    st.sidebar.dataframe(pd.read_csv(hist_file))

with st.sidebar.expander("🗑️ Clear Summary History"):
    confirm = st.checkbox("⚠️ Yes, I'm sure I want to clear history", key="confirm_clear")
    if confirm and st.button("🗑️ Clear History Now", key="clear_history"):
        if os.path.exists(hist_file):
            os.remove(hist_file)
        st.sidebar.success("✅ Summary history cleared")
        st.experimental_rerun()

# ——— 4) Summarize on button click ———
if st.button("🚀 Summarize") and text_input.strip():
    tokenizer, model = load_model()
    summary = generate_summary(text_input, tokenizer, model)

    # Side-by-side comparison
    st.markdown("### 🔍 Comparison")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Original Input**")
        st.info(text_input)
    with c2:
        st.markdown("**Generated Summary**")
        st.success(summary)

    # Token metrics
    st.markdown("### 📊 Token Counts")
    m1, m2 = st.columns(2)
    m1.metric("📝 Input Tokens", len(text_input.split()))
    m2.metric("📄 Summary Tokens", len(summary.split()))

    # Append to history
    new_row = pd.DataFrame([{
        "Title":   title,
        "Input":   text_input,
        "Summary": summary,
        "Length":  len(summary.split())
    }])
    df_hist = pd.read_csv(hist_file) if os.path.exists(hist_file) else pd.DataFrame()
    pd.concat([df_hist, new_row], ignore_index=True).to_csv(hist_file, index=False)

    # Download options
    if save_format == ".txt":
        st.download_button(
            "📥 Download .txt",
            summary,
            file_name=f"{title[:20]}.txt",
            mime="text/plain"
        )
    else:
        payload = {"title": title, "input": text_input, "summary": summary}
        st.download_button(
            "📥 Download .json",
            json.dumps(payload, indent=2),
            file_name=f"{title[:20]}.json",
            mime="application/json"
        )
