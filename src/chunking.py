import json
import os
import tiktoken

INPUT_PATH = "data/processed/documents.json"
OUTPUT_PATH = "data/processed/chunks.json"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 80

tokenizer = tiktoken.get_encoding("cl100k_base")

def tokenize(text):
    return tokenizer.encode(text)

def detokenize(tokens):
    return tokenizer.decode(tokens)

def chunk_text(text, chunk_size, overlap):
    tokens = tokenize(text)
    chunks = []

    start = 0
    while start < len(tokens):
        end = start + chunk_size
        chunk_tokens = tokens[start:end]
        chunk_text = detokenize(chunk_tokens)
        chunks.append(chunk_text)
        start += chunk_size - overlap

    return chunks

def create_chunks():
    with open(INPUT_PATH, "r") as f:
        documents = json.load(f)

    all_chunks = []
    chunk_id = 0

    for doc in documents:
        text_chunks = chunk_text(doc["text"], CHUNK_SIZE, CHUNK_OVERLAP)

        for chunk in text_chunks:
            all_chunks.append({
                "chunk_id": chunk_id,
                "source": doc["source"],
                "page": doc["page"],
                "text": chunk
            })
            chunk_id += 1

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(all_chunks, f, indent=2)

    print(f"Created {len(all_chunks)} chunks")

if __name__ == "__main__":
    create_chunks()
