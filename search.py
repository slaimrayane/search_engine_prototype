# search.py
import faiss
import numpy as np
import pickle
import os
from llm_local import generate_answer
from embedder import embed_text
from utils import rank_paragraphs  # ✅ Added for intelligent filtering

# Function to load the FAISS index and metadata
def load_index(index_path="embeddings/index.faiss", meta_path="embeddings/meta.pkl"):
    # Check if the index and metadata files exist
    if not os.path.exists(index_path) or not os.path.exists(meta_path):
        print("❌ Index does not exist. Run: python indexer.py first.")
        exit()
    # Load the FAISS index
    index = faiss.read_index(index_path)
    # Load the metadata (filenames) from the pickle file
    with open(meta_path, "rb") as f:
        filenames = pickle.load(f)
    return index, filenames

# Function to retrieve documents based on a query
def retrieve_documents(query, index, filenames, top_k=5, threshold=1.3):
    # Embed the query into a vector
    query_vector = embed_text(query).reshape(1, -1)
    # Search the FAISS index for the top_k closest matches
    distances, indices = index.search(query_vector, top_k)

    results = []
    # Iterate through the retrieved indices and distances
    for i, idx in enumerate(indices[0]):
        file_name = filenames[idx]  # Get the filename from metadata
        distance = distances[0][i]  # Get the distance score
        file_path = os.path.join("chunks", f"{file_name}.txt")  # Ensure '.txt' extension is added

        # Only include results below the threshold distance
        if distance < threshold:
            try:
                # Read the content of the file
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                    results.append((file_name, distance, content))
            except FileNotFoundError:
                print(f"⚠️ File not found: {file_path}")
            except Exception as e:
                print(f"⚠️ Error reading {file_name}: {e}")
    return results

# Main function to handle user interaction
def main():
    print("🔄 Loading the index...")
    # Load the FAISS index and metadata
    index, filenames = load_index()

    print("🤖 Ask your questions (type 'exit' to quit)\n")
    while True:
        # Prompt the user for a query
        query = input("💬 Question: ").strip()
        if query.lower() in {"exit", "quit", ""}:  # Exit condition
            print("👋 Ending session.")
            break

        # Retrieve relevant documents for the query
        docs = retrieve_documents(query, index, filenames)
        if not docs:  # If no documents are found
            print("❌ No relevant documents found.\n")
            continue

        print("\n📄 Selected documents:")
        # Display the retrieved documents and their distances
        for i, (name, dist, _) in enumerate(docs):
            print(f"{i+1}. {name} (distance: {dist:.4f})")

        # ✅ Build a focused context based on similarity
        raw_context = "\n\n".join([doc[2] for doc in docs[:3]])  # Combine document contents
        focused_context = rank_paragraphs(raw_context, query, top_n=2)  # Rank paragraphs for relevance

        if not focused_context.strip():  # If no relevant context is found
            print("\n🧠 Response:\nI couldn't find a relevant answer in the documents.")
            continue

        # Generate an answer using the focused context
        answer = generate_answer(query, focused_context)

        print("\n🧠 Response:")
        print(answer)
        print("\n" + "-" * 50 + "\n")  # Separator for readability

# Entry point of the script
if __name__ == "__main__":
    main()