import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 🔧 Désactiver le parallélisme source d'erreurs sur macOS
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

MODEL_NAME = "microsoft/phi-1_5"

print(f"🔄 Loading chat model: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

def generate_answer(question, context, max_tokens=256):
    prompt = f"""
You are an assistant that answers questions based only on the context provided.

Context:
{context}

Question: {question}
Answer:
"""

    inputs = tokenizer(prompt.strip(), return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Nettoyer : retourner uniquement le texte après "Answer:"
    if "Answer:" in output_text:
        return output_text.split("Answer:")[-1].strip().split("\n\n")[0]
    return output_text.strip().split("\n\n")[0]
