from pathlib import Path
import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer

# Project root
BASE_DIR = Path(__file__).resolve().parent.parent

# FAISS index lives inside embeddings/
INDEX_DIR = BASE_DIR / "embeddings" / "vector_db" / "faiss_index"

model = SentenceTransformer("all-MiniLM-L6-v2")

index = faiss.read_index(str(INDEX_DIR / "index.faiss"))

with open(INDEX_DIR / "metadata.json") as f:
    metadata = json.load(f)

def search(query, top_k=5):
    query_embedding = model.encode([query])
    distances, indices = index.search(
        np.array(query_embedding), top_k
    )

    results = []
    for idx in indices[0]:
        results.append(metadata[idx])

    return results

if __name__ == "__main__":
    query = "What is retrieval augmented generation?"
    results = search(query)

    for r in results:
        print("\n---")
        print(r["text"])
