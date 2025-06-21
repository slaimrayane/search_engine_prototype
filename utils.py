# utils.py

import numpy as np
from sentence_transformers import SentenceTransformer

# Embedding model léger et rapide
embedder = SentenceTransformer("all-MiniLM-L6-v2")

def cosine_similarity(a, b):
    """Calcule la similarité cosinus entre deux vecteurs"""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return np.dot(a, b)

def rank_paragraphs(context: str, question: str, top_n: int = 2) -> str:
    """
    Sélectionne les top_n paragraphes les plus proches de la question.
    """
    paragraphs = [p.strip() for p in context.split("\n") if len(p.strip()) > 20]
    if not paragraphs:
        return context

    q_embed = embedder.encode(question)
    scored = [
        (cosine_similarity(q_embed, embedder.encode(p)), p)
        for p in paragraphs
    ]

    # Tri décroissant
    scored.sort(reverse=True)
    return "\n\n".join([p for _, p in scored[:top_n]])

def chunk_document(text: str, mode="word", max_words=100, overlap=20):
    """
    Découpe un document selon la méthode choisie :
    - 'line' : par lignes
    - 'paragraph' : par paragraphes (sauts de ligne doubles)
    - 'word' : par blocs de mots avec chevauchement
    """
    if mode == "line":
        return [line.strip() for line in text.split("\n") if line.strip()]

    elif mode == "paragraph":
        return [p.strip() for p in text.split("\n\n") if p.strip()]

    elif mode == "word":
        words = text.split()
        chunks = []
        start = 0

        while start < len(words):
            end = start + max_words
            chunk = " ".join(words[start:end])
            chunks.append(chunk)
            start += max_words - overlap

        return chunks

    else:
        raise ValueError("chunking must be 'word', 'line', or 'paragraph'")
