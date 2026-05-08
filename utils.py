# utils.py

import numpy as np

from embedder import embed_text

def cosine_similarity(a, b):
    """Calcule la similarité cosinus entre deux vecteurs"""
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if not a_norm or not b_norm:
        return 0.0
    a = a / a_norm
    b = b / b_norm
    return np.dot(a, b)

def rank_paragraphs(context: str, question: str, top_n: int = 2) -> str:
    """
    Sélectionne les top_n paragraphes les plus proches de la question.
    """
    paragraphs = [p.strip() for p in context.split("\n") if len(p.strip()) > 20]
    if not paragraphs:
        return context

    q_embed = embed_text(question)
    scored = [
        (cosine_similarity(q_embed, embed_text(p)), p)
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
