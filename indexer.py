import os
import faiss
import pickle
import numpy as np
from embedder import embed_documents
from parser import load_documents

TEXT_DIR = "texts"
INDEX_PATH = "embeddings/index.faiss"
META_PATH = "embeddings/meta.pkl"



def build_index():
    documents = load_documents()

    if not documents:
        print("❌ Aucun document trouvé dans texts/")
        return

    texts = [text for _, text in documents]
    vectors = embed_documents(documents)  # shape (n, dim)

    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)

    os.makedirs("embeddings", exist_ok=True)
    faiss.write_index(index, INDEX_PATH)

    filenames = [title for title, _ in documents]
    with open(META_PATH, "wb") as f:
        pickle.dump(filenames, f)

    # Save each chunk into a separate file
    os.makedirs("chunks", exist_ok=True)
    for title, text in documents:
        chunk_filename = os.path.join("chunks", f"{title}.txt")
        with open(chunk_filename, "w", encoding="utf-8") as chunk_file:
            chunk_file.write(text)
    print(f"✅ Index construit avec {len(filenames)} lignes indexées.")

if __name__ == "__main__":
    build_index()
