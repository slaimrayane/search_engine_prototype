import os
import time

import streamlit as st

from indexer import build_index
from llm_local import generate_answer
from search import load_index, retrieve_documents
from utils import rank_paragraphs


st.set_page_config(page_title="Local RAG Search", page_icon="🔎", layout="wide")


def index_is_ready():
    return os.path.exists("embeddings/index.faiss") and os.path.exists("embeddings/meta.pkl")


@st.cache_resource(show_spinner=False)
def cached_index():
    return load_index()


def reset_index_cache():
    cached_index.clear()


st.title("Local Document Search")
st.caption("RAG over your local documents using FAISS, local embeddings, and a cached small language model.")

with st.sidebar:
    st.header("Index")
    data_files = [
        name
        for name in sorted(os.listdir("data")) if name.lower().endswith((".pdf", ".docx"))
    ] if os.path.isdir("data") else []
    st.write(f"{len(data_files)} source documents in `data/`")

    if st.button("Rebuild index", type="primary"):
        with st.spinner("Parsing documents and rebuilding FAISS index..."):
            build_index()
            reset_index_cache()
        st.success("Index rebuilt.")

    st.header("Generation")
    use_slm = st.toggle("Use local SLM", value=True)
    st.caption("Falls back to extractive answers if the model is not cached.")

if not index_is_ready():
    st.warning("No FAISS index found. Click **Rebuild index** in the sidebar first.")
    st.stop()

query = st.text_input("Ask a question", placeholder="How do I reset my password?")

if query:
    started = time.time()
    index, filenames = cached_index()
    docs = retrieve_documents(query, index, filenames, top_k=6, threshold=2.0)

    if not docs:
        st.error("No relevant document chunks found.")
        st.stop()

    raw_context = "\n\n".join(doc[2] for doc in docs[:4])
    focused_context = rank_paragraphs(raw_context, query, top_n=4)
    answer = generate_answer(query, focused_context, prefer_local_model=use_slm)
    elapsed = time.time() - started

    st.subheader("Answer")
    st.write(answer)
    st.caption(f"Answered in {elapsed:.2f}s from {len(docs)} retrieved chunks.")

    st.subheader("Sources")
    for rank, (name, distance, content) in enumerate(docs, start=1):
        similarity = 1 / (1 + float(distance))
        with st.expander(f"{rank}. {name} · similarity {similarity:.3f}"):
            st.write(content)
else:
    st.info("Ask a question to search the indexed documents.")
