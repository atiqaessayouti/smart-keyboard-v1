"""Lab 02 - Smart Keyboard V1 (Streamlit UI). Run:  streamlit run app.py"""
import streamlit as st

from src.pipeline import NGramModel, EOS, load_documents, backoff_top_k, split_by_document

st.set_page_config(
    page_title="Smart Keyboard V1",
    page_icon="⌨️",
    layout="wide",
)

st.title("⌨️ Smart Keyboard V1")
st.caption("Next-word suggestions from an n-gram model with add-k smoothing and back-off.")


@st.cache_resource
def get_data():
    docs = load_documents()
    train, test = split_by_document(docs)
    return train, test


@st.cache_resource
def get_models(train, n, k):
    return NGramModel(train, n, k)


train, test = get_data()

# ---------- Sidebar ----------
st.sidebar.header("⚙️ Settings")
n = st.sidebar.selectbox("N-gram order", [1, 2, 3], index=2,
                         help="1 = unigram, 2 = bigram, 3 = trigram")
k = st.sidebar.slider("Smoothing k", 0.0, 1.0, 0.1, 0.01,
                      help="0 = no smoothing (unseen → probability 0)")

tri = get_models(train, 3, k)
bi = get_models(train, 2, k)
uni = get_models(train, 1, k)

# ---------- Input ----------
st.subheader("Type a sentence")
text = st.text_input(
    " ",
    "the happy girls",
    label_visibility="collapsed",
    placeholder="e.g. the happy girls",
)

words = text.lower().replace(".", " . ").split()
unknown = [w for w in words if w not in uni.vocab]

if unknown:
    st.warning(f"⚠️ Not in the training vocabulary (treated as `<unk>`): {', '.join(unknown)}")
if not words:
    st.info("Type at least one word — showing how sentences usually start.")

# ---------- Suggestions ----------
suggestions, order = backoff_top_k(tri, bi, uni, words, 3)

st.subheader(f"💡 Top-3 suggestions  ·  used the **{order} model**")
cols = st.columns(3)
for col, (w, p) in zip(cols, suggestions):
    label = "(end of sentence)" if w == EOS else w
    col.metric(label, f"{p:.1%}")

st.caption("Probabilities are model estimates on a tiny synthetic corpus — not real-world typing statistics.")

# ---------- Model info ----------
st.divider()
st.subheader("📊 Model info")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Vocabulary size", uni.V)

with col2:
    pp, zero_rate = tri.perplexity(test)
    pp_display = "∞" if pp == float("inf") else f"{pp:.2f}"
    st.metric("Test perplexity (trigram)", pp_display,
              delta="infinite!" if pp == float("inf") else None,
              delta_color="inverse")

with col3:
    acc, _ = __import__("src.pipeline", fromlist=["top_k_accuracy"]).top_k_accuracy(tri, test, 3)
    st.metric("Top-3 accuracy (trigram)", f"{acc:.1%}")

if zero_rate > 0:
    st.error(f"🚨 {zero_rate:.1%} of test tokens have probability 0 → perplexity = ∞. Increase k!")

# ---------- Comparison table ----------
st.divider()
st.subheader("🔍 Model comparison (current k)")

from src.pipeline import top_k_accuracy

rows = []
for name, model in [
    ("Unigram", uni),
    ("Bigram", bi),
    ("Trigram", tri),
]:
    ppl, z = model.perplexity(test)
    acc, _ = top_k_accuracy(model, test, 3)
    rows.append({
        "Model": name,
        "Perplexity": "∞" if ppl == float("inf") else f"{ppl:.2f}",
        "Zero-prob tokens": f"{z:.1%}",
        "Top-3 accuracy": f"{acc:.1%}",
    })

st.dataframe(rows, use_container_width=True, hide_index=True)

st.divider()
st.caption("Lab 02 — Smart Keyboard V1 · n-gram language model")