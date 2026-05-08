# 🔍 Search Engine Prototype for Internal Document Retrieval

A lightweight **RAG (Retrieval-Augmented Generation)** prototype built for internal document search, using semantic chunk indexing and a local small language model.

## ✨ Features

- 📁 Converts internal documents (`.pdf`, `.docx`) to text
- ✂️ Smart document chunking (`line`, `paragraph`, or `word-based`)
- 🔍 FAISS vector search over semantic chunks
- 📊 Streamlit interface for easy interaction
- ✅ Answers with a local SLM (`google/flan-t5-small` by default) and no API key
- ✅ Falls back to extractive local answers if the model is not cached

---

## 📦 Requirements

- Python 3.10+
- PyTorch with CPU support (MPS optional for macOS)
- See `requirements.txt` for full dependencies

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/slaimrayane/search_engine_prototype.git
cd search_engine_prototype
```

### 2. Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Add your documents

Place your `.pdf` or `.docx` files inside the `data/` folder.

### 4. Build the index

```bash
python indexer.py
```

### 5. Launch the web app

```bash
streamlit run app.py
```

You can also use the CLI:

```bash
python search.py
```

### Local model settings

The app uses local models only. By default:

- Embeddings: `all-MiniLM-L6-v2`
- Answer generation: `google/flan-t5-small`

To use another cached small model:

```bash
export LOCAL_LLM_MODEL="google/flan-t5-small"
export EMBEDDING_MODEL="all-MiniLM-L6-v2"
```

---

## ✅ Example Questions

- *"How do I reset my password?"*
- *"What is the vendor approval process?"*
- *"How is incident reporting handled?"*

---

## 🛡️ Notes

- ⚠️ Avoid committing large files (models, venv, chunks) — use `.gitignore`
- No hosted LLM API key is required
- Tested on macOS with MPS and Intel CPU

---

## 🧑‍💻 Author

**Rayane Slaim** — Computer Science Student | L3 MIAGE (Université Paris Dauphine)

Feel free to ⭐️ the repo if you find it useful!
