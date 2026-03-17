from pathlib import Path
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from utils.monitoring import timed

# ---------------- Paths ----------------
BASE_DIR = Path(__file__).resolve().parent.parent
INDEX_DIR = BASE_DIR / "embeddings" / "vector_db" / "faiss_index"

# ---------------- Models ----------------
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
client = OpenAI()  # uses OPENAI_API_KEY env var
from sentence_transformers import CrossEncoder

reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# ---------------- Load FAISS ----------------
index = faiss.read_index(str(INDEX_DIR / "index.faiss"))

with open(INDEX_DIR / "metadata.json") as f:
    metadata = json.load(f)
from rank_bm25 import BM25Okapi

# Prepare BM25 corpus
tokenized_corpus = [doc["text"].lower().split() for doc in metadata]
bm25 = BM25Okapi(tokenized_corpus)
# ---------------- Agent: Question Understanding ----------------
def classify_question(question: str):
    q = question.lower()
    if any(w in q for w in ["compare", "vs", "difference"]):
        return "comparison"
    if any(w in q for w in ["risk", "challenge", "limitation"]):
        return "risk"
    if any(w in q for w in ["trend", "future", "next", "emerging"]):
        return "trend_analysis"
    return "general"

def rewrite_query(question: str, intent: str):
    if intent == "risk":
        return f"What risks or challenges are discussed regarding {question} in the documents?"
    if intent == "trend_analysis":
        return f"What technology trends or future implications are discussed regarding {question}?"
    if intent == "comparison":
        return f"What differences or comparisons are mentioned regarding {question}?"
    return question

# ---------------- Retrieval ----------------
def retrieve_dense_with_scores(query, top_k=8):
    query_vec = embed_model.encode([query])
    distances, indices = index.search(np.array(query_vec), top_k)

    results = []
    for score, idx in zip(distances[0], indices[0]):
        results.append({
            "doc": metadata[idx],
            "score": float(score)
        })

    return results
def retrieve_bm25_with_scores(query, top_k=8):
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)

    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        results.append({
            "doc": metadata[idx],
            "score": float(scores[idx])
        })

    return results
def normalize_scores(results, reverse=False):
    scores = [r["score"] for r in results]

    if reverse:  # for FAISS distances
        scores = [-s for s in scores]

    min_score = min(scores)
    max_score = max(scores)

    if max_score == min_score:
        return [1.0] * len(scores)

    return [(s - min_score) / (max_score - min_score) for s in scores]


    # return [(s - min_score) / (max_score - min_score) for s in scores]

@timed
def hybrid_retrieve(query, top_k=8, alpha=0.6):
    dense_results = retrieve_dense_with_scores(query, top_k)
    sparse_results = retrieve_bm25_with_scores(query, top_k)

    dense_norm = normalize_scores(dense_results, reverse=True)
    sparse_norm = normalize_scores(sparse_results, reverse=False)

    fusion_dict = {}

    # Dense contribution
    for i, r in enumerate(dense_results):
        text = r["doc"]["text"]
        fusion_dict[text] = {
            "doc": r["doc"],
            "score": alpha * dense_norm[i]
        }

    # Sparse contribution
    for i, r in enumerate(sparse_results):
        text = r["doc"]["text"]

        if text in fusion_dict:
            fusion_dict[text]["score"] += (1 - alpha) * sparse_norm[i]
        else:
            fusion_dict[text] = {
                "doc": r["doc"],
                "score": (1 - alpha) * sparse_norm[i]
            }

    # Sort by fused score
    ranked = sorted(fusion_dict.values(), key=lambda x: x["score"], reverse=True)

    return [r["doc"] for r in ranked[:top_k]]

@timed
def rerank_documents(query, docs, top_n=4):
    if not docs:
        return []

    pairs = [(query, doc["text"]) for doc in docs]

    scores = reranker_model.predict(pairs)

    scored_docs = list(zip(scores, docs))
    scored_docs.sort(key=lambda x: x[0], reverse=True)

    return [doc for _, doc in scored_docs[:top_n]]


def has_sufficient_context(chunks, min_chunks=2):
    return len(chunks) >= min_chunks

# ---------------- Prompt ----------------
def build_prompt(chunks, question):
    context = "\n\n".join([
    f"Source: {c['source']}, Page: {c['page']}\n{c['text']}"
    for c in chunks
])

    return f"""
If the context does NOT explicitly contain the answer, say:
"Not explicitly stated in the provided documents."

You are a technology trends research assistant.

Use ONLY the information provided below.
Do not use prior knowledge.

For the question:
"{question}"

Provide:
1. A clear answer
2. Key insights (bullet points)
3. Risks and implications
4. Cite sources in the format: (paper2.pdf, Page 4)

Context:
{context}
# You are a research assistant.

# Answer the question using ONLY the context below.
# If the answer is not present, say "I don't know based on the provided documents."

# Context:
# {context}

# Question:
# {question}

# Answer:
"""

# ---------------- Generation ----------------
def generate_answer(prompt):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    return response.choices[0].message.content
def verify_answer(context, answer):
    verification_prompt = f"""
Context:
{context}

Answer:
{answer}

Is the answer fully supported by the context?
Reply with only YES or NO.
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": verification_prompt}],
        temperature=0
    )
    return response.choices[0].message.content.strip()
def optimize_query_for_retrieval(query):
    rewrite_prompt = f"""
Rewrite the following user question into a concise search query
optimized for semantic retrieval from technology trend documents.

Keep it short.
Remove unnecessary words.
Do NOT answer the question.

Question:
{query}

Optimized search query:
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": rewrite_prompt}],
        temperature=0
    )

    return response.choices[0].message.content.strip()


# ---------------- Full RAG ----------------
# def rag(query):
#     chunks = retrieve(query)
#     prompt = build_prompt(chunks, query)
#     return generate_answer(prompt)
# ---------------- Agentic RAG ----------------
def agentic_rag(query):
    # intent = classify_question(query)
    # rewritten_query = rewrite_query(query, intent)
    # rewritten_query = query
    # rewritten_query = query

    # if intent != "general":
        # rewritten_query = query + " " + rewrite_query(query, intent)
    optimized_query = optimize_query_for_retrieval(query)




    # chunks = retrieve(optimized_query, top_k=8)
    chunks,retrieval_time = hybrid_retrieve(optimized_query, top_k=8)

    chunks,rerank_time = rerank_documents(optimized_query, chunks, top_n=4)
    

    if not has_sufficient_context(chunks):
        return "I don’t have enough information in the provided documents to answer this confidently."

    prompt = build_prompt(chunks, query)
    answer = generate_answer(prompt)

    context_text = "\n\n".join([c["text"] for c in chunks])
    verification = verify_answer(context_text, answer)

    if verification == "NO":
        return "The generated answer is not fully supported by the retrieved context."

    return answer
# ---------------- Retrieval Evaluation ----------------
def evaluate_retrieval(question, expected_keywords, top_k=8):
    optimized_query = optimize_query_for_retrieval(question)
    # chunks = retrieve(optimized_query, top_k=top_k)
    chunks = hybrid_retrieve(optimized_query, top_k=8)

    combined_text = " ".join([c["text"].lower() for c in chunks])

    hits = sum(1 for kw in expected_keywords if kw.lower() in combined_text)
    recall_score = hits / len(expected_keywords)

    return {
        "question": question,
        "hits": hits,
        "total_expected": len(expected_keywords),
        "recall_score": round(recall_score, 2)
    }


evaluation_questions = [
    {
        "question": "What challenges are mentioned in AI adoption?",
        "expected_keywords": ["cybersecurity", "privacy", "skill", "cost", "workflow"]
    },
    {
        "question": "What drives optimism for 2026?",
        "expected_keywords": ["operational efficiency", "AI for productivity"]
    }
]


if __name__ == "__main__":
    # question = "What is context engineering and why is it important?"
    question = "What are the biggest challenges organizations face in adopting AI?"

    answer = agentic_rag(question)
    print(answer)

if __name__ == "__main__":
    for item in evaluation_questions:
        result = evaluate_retrieval(
            item["question"],
            item["expected_keywords"]
        )
        print(result)


# if __name__ == "__main__":
#     question = "What is context engineering and why is it important?"
#     answer = rag(question)
#     print(answer)
