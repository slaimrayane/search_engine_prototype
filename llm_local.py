import os
import re
from functools import lru_cache

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


DEFAULT_LOCAL_MODEL = os.getenv("LOCAL_LLM_MODEL", "google/flan-t5-small")


def _keywords(text):
    return {
        word
        for word in re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{2,}", text.lower())
        if word
        not in {
            "the",
            "and",
            "for",
            "with",
            "from",
            "that",
            "this",
            "how",
            "what",
            "are",
            "you",
            "your",
            "our",
            "can",
        }
    }


def _split_sentences(context):
    compact = re.sub(r"\s+", " ", context).strip()
    return [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", compact) if sentence.strip()]


def extractive_answer(question, context, max_sentences=3):
    """Offline fallback that returns the most relevant context sentences."""
    sentences = _split_sentences(context)
    if not sentences:
        return "I could not find enough context to answer that."

    question_terms = _keywords(question)
    scored = []
    for index, sentence in enumerate(sentences):
        sentence_terms = _keywords(sentence)
        overlap = len(question_terms & sentence_terms)
        density = overlap / max(len(sentence_terms), 1)
        scored.append((overlap, density, -index, sentence))

    ranked = [sentence for overlap, _, _, sentence in sorted(scored, reverse=True) if overlap > 0]
    if not ranked:
        ranked = sentences[:max_sentences]

    return " ".join(ranked[:max_sentences])


@lru_cache(maxsize=1)
def _load_local_generator(model_name=DEFAULT_LOCAL_MODEL):
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, local_files_only=True)

    if torch.backends.mps.is_available():
        model = model.to("mps")

    return tokenizer, model


def generate_answer(question, context, max_tokens=160, prefer_local_model=True):
    """
    Answer from provided context using a local SLM when it is cached.

    No API key is required. If the local model cannot be loaded or generate a
    useful answer, the function falls back to extractive context snippets.
    """
    context = (context or "").strip()
    if not context:
        return "I could not find enough context to answer that."

    if not prefer_local_model:
        return extractive_answer(question, context)

    try:
        tokenizer, model = _load_local_generator()
        prompt = (
            "Answer the question using only the context. "
            "If the context does not contain the answer, say you do not know.\n\n"
            f"Context:\n{context[:3500]}\n\n"
            f"Question: {question}\n"
            "Answer:"
        )
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        inputs = {key: value.to(model.device) for key, value in inputs.items()}
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            num_beams=2,
        )
        answer = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
        if answer and answer.lower() not in {"i don't know", "i do not know", "unknown"}:
            return answer
    except Exception:
        pass

    return extractive_answer(question, context)
