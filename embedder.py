from sentence_transformers import SentenceTransformer
import numpy as np

# Charge un modèle local préentraîné (léger et rapide)
model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_text(text):
    return model.encode(text, convert_to_numpy=True)

def embed_documents(documents):
    embeddings = []
    for _, text in documents:
        vector = embed_text(text)
        embeddings.append(vector)
    return np.array(embeddings)
