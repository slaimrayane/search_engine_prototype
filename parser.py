import os
import pdfplumber
import docx
from utils import chunk_document
def extract_text_from_pdf(path):
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def extract_text_from_docx(path):
    doc = docx.Document(path)
    return "\n".join([para.text for para in doc.paragraphs])



def load_documents(data_dir="data", texts_dir="texts", chunks_dir="chunks"):
    # Ensure output directories exist
    os.makedirs(texts_dir, exist_ok=True)
    os.makedirs(chunks_dir, exist_ok=True)

    documents = []

    for filename in os.listdir(data_dir):
        file_path = os.path.join(data_dir, filename)
        name, ext = os.path.splitext(filename)
        if ext.lower() == ".pdf":
            text = extract_text_from_pdf(file_path)
        elif ext.lower() == ".docx":
            text = extract_text_from_docx(file_path)
        else:
            continue

        text = text.strip()
        if text:
            # Save the full text to the texts directory
            txt_path = os.path.join(texts_dir, name + ".txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(text)

            # Split the text into chunks and save them to the chunks directory
            chunks = chunk_by_words(text, max_words=100, overlap=20)
            for i, chunk in enumerate(chunks):
                chunk_title = f"{name} [chunk {i+1}]"
                chunk_path = os.path.join(chunks_dir, f"{chunk_title}.txt")
                with open(chunk_path, "w", encoding="utf-8") as chunk_file:
                    chunk_file.write(chunk)
                documents.append((chunk_title, chunk))

    return documents