import os
import pdfplumber
import re
import json

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"

os.makedirs(PROCESSED_DIR, exist_ok=True)

def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\n+", " ", text)
    text = text.strip()
    return text

def extract_text_from_pdf(pdf_path):
    pages_text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                pages_text.append({
                    "page": page_num + 1,
                    "text": clean_text(text)
                })
    return pages_text

def process_all_pdfs():
    all_docs = []

    for file in os.listdir(RAW_DIR):
        if file.endswith(".pdf"):
            pdf_path = os.path.join(RAW_DIR, file)
            print(f"Processing {file}...")

            pages = extract_text_from_pdf(pdf_path)

            for p in pages:
                all_docs.append({
                    "source": file,
                    "page": p["page"],
                    "text": p["text"]
                })

    output_path = os.path.join(PROCESSED_DIR, "documents.json")
    with open(output_path, "w") as f:
        json.dump(all_docs, f, indent=2)

    print(f"Saved {len(all_docs)} pages to {output_path}")

if __name__ == "__main__":
    process_all_pdfs()
