# 🔍 Search Engine Prototype for Internal Document Retrieval

A lightweight and efficient **RAG (Retrieval-Augmented Generation)** prototype built for internal document search, using semantic chunk indexing and external LLMs.

## ✨ Features

- 📁 Converts internal documents (`.pdf`, `.docx`) to text
- ✂️ Smart document chunking (`line`, `paragraph`, or `word-based`)
- 🔍 FAISS vector search over semantic chunks
- 📊 Streamlit interface for easy interaction
- ✅ Supports external LLMs (e.g., Google GenAI)

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

---

## ✅ Example Questions

- *"How do I reset my password?"*
- *"What is the vendor approval process?"*
- *"How is incident reporting handled?"*

---

## 🛡️ Notes

- ⚠️ Avoid committing large files (models, venv, chunks) — use `.gitignore`
- Supports external LLMs for answering queries
- Tested on macOS with MPS and Intel CPU

---

## 🧑‍💻 Author

**Rayane Slaim** — Computer Science Student | L3 MIAGE (Université Paris Dauphine)

Feel free to ⭐️ the repo if you find it useful!