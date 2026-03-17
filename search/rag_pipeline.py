from pathlib import Path
import json
import faiss
import numpy as np
import time
import math
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from utils.monitoring import timed, log_metrics, generate_request_id
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi


# ---------------- Paths ----------------
BASE_DIR = Path(__file__).resolve().parent.parent
INDEX_DIR = BASE_DIR / "embeddings" / "vector_db" / "faiss_index"


# ---------------- Models ----------------
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
client = OpenAI()
reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


# ---------------- Load FAISS ----------------
index = faiss.read_index(str(INDEX_DIR / "index.faiss"))

with open(INDEX_DIR / "metadata.json") as f:
    metadata = json.load(f)


# ---------------- BM25 Setup ----------------
tokenized_corpus = [doc["text"].lower().split() for doc in metadata]
bm25 = BM25Okapi(tokenized_corpus)

CONFIDENCE_THRESHOLD = 0.70


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

def is_analytical_question(question: str) -> bool:
    analytical_keywords = [
        "analyze",
        "compare",
        "evaluate",
        "why",
        "impact",
        "systemic",
        "barriers",
        "implications",
        "risks",
        "challenges"
    ]
    q_lower = question.lower()
    return any(word in q_lower for word in analytical_keywords)
def calculate_confidence(faithfulness, relevance, answer, question):
    base = (faithfulness * 0.6) + (relevance * 0.4)

    if is_analytical_question(question):
        structure_bonus = 0
        if "Key Insights" in answer:
            structure_bonus += 0.05
        if "Risks" in answer:
            structure_bonus += 0.05
        if "Implications" in answer:
            structure_bonus += 0.05

        base += structure_bonus

    return round(min(base, 1.0), 3)
def generate_search_queries(question, n_queries=3):

    prompt = f"""
Generate {n_queries} different search queries that could retrieve
relevant documents for the question.

The queries should capture different ways the question might appear in text.

Return each query on a new line.

Question:
{question}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    queries = response.choices[0].message.content.strip().split("\n")

    queries = [q.strip("- ").strip() for q in queries if q.strip()]

    queries.insert(0, question)

    return list(set(queries))
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

    if reverse:
        scores = [-s for s in scores]

    min_score = min(scores)
    max_score = max(scores)

    if max_score == min_score:
        return [1.0] * len(scores)

    return [(s - min_score) / (max_score - min_score) for s in scores]


@timed
def hybrid_retrieve(query, top_k=8, alpha=0.6):
    dense_results = retrieve_dense_with_scores(query, top_k)
    sparse_results = retrieve_bm25_with_scores(query, top_k)

    dense_norm = normalize_scores(dense_results, reverse=True)
    sparse_norm = normalize_scores(sparse_results, reverse=False)

    fusion_dict = {}

    for i, r in enumerate(dense_results):
        text = r["doc"]["text"]
        fusion_dict[text] = {
            "doc": r["doc"],
            "score": alpha * dense_norm[i]
        }

    for i, r in enumerate(sparse_results):
        text = r["doc"]["text"]
        if text in fusion_dict:
            fusion_dict[text]["score"] += (1 - alpha) * sparse_norm[i]
        else:
            fusion_dict[text] = {
                "doc": r["doc"],
                "score": (1 - alpha) * sparse_norm[i]
            }

    ranked = sorted(fusion_dict.values(), key=lambda x: x["score"], reverse=True)

    return [r["doc"] for r in ranked[:top_k]]

def multi_query_retrieve(question, top_k=8):

    queries = generate_search_queries(question)

    all_chunks = []

    for q in queries:
        chunks, _ = hybrid_retrieve(q, top_k=top_k)
        all_chunks.extend(chunks)

    # remove duplicates
    seen = set()
    unique_chunks = []

    for c in all_chunks:
        text = c["text"]
        if text not in seen:
            seen.add(text)
            unique_chunks.append(c)

    return unique_chunks[:top_k]


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

    if is_analytical_question(question):
        return f"""
If the context does NOT explicitly contain the answer, say:
"Not explicitly stated in the provided documents."

You are a senior research analyst.

Use ONLY the information provided below.
Do not use prior knowledge.

For the question:
"{question}"

Provide:

1. A clear analytical answer
2. Group findings into logical categories
3. Identify systemic patterns across sources
4. Explain relationships between findings
5. Risks and long-term implications
# #6. Cite sources in the format: (paper2.pdf, Page 4)
6.Always format citations like this:
📄 paper_name — Page X

Context:
{context}
"""
    else:
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
4. Always format citations like this:
📄 paper_name — Page X

Context:
{context}
"""
# ---------------- Prompt ----------------
# def build_prompt(chunks, question):
#     context = "\n\n".join([
#     f"Source: {c['source']}, Page: {c['page']}\n{c['text']}"
#     for c in chunks
# ])

#     return f"""
# If the context does NOT explicitly contain the answer, say:
# "Not explicitly stated in the provided documents."

# You are a technology trends research assistant.

# Use ONLY the information provided below.
# Do not use prior knowledge.

# For the question:
# "{question}"

# Provide:
# 1. A clear answer
# 2. Key insights (bullet points)
# 3. Risks and implications
# 4. Cite sources in the format: (paper2.pdf, Page 4)

# Context:
# {context}
# # You are a research assistant.

# # Answer the question using ONLY the context below.
# # If the answer is not present, say "I don't know based on the provided documents."

# # Context:
# # {context}

# # Question:
# # {question}

# # Answer:
# """


# ---------------- Generation ----------------
def generate_answer(prompt):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    return response.choices[0].message.content


# def verify_answer(context, answer):
#     verification_prompt = f"""
# Context:
# {context}

# Answer:
# {answer}

# Is the answer fully supported by the context?
# Reply YES or NO only.
# """
#     response = client.chat.completions.create(
#         model="gpt-4o-mini",
#         messages=[{"role": "user", "content": verification_prompt}],
#         temperature=0
#     )
#     return response.choices[0].message.content.strip()
def faithfulness_score(context, answer):
    verification_prompt = f"""
Context:
{context}

Answer:
{answer}

On a scale of 0 to 1, how well is the answer supported by the context?
0 = Not supported at all
1 = Fully supported

Reply with only a number.
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": verification_prompt}],
        temperature=0
    )
    try:
        return float(response.choices[0].message.content.strip())
    except:
        return 0.0
    
def answer_relevance_score(question, answer):
    prompt = f"""
Question:
{question}

Answer:
{answer}

On a scale of 0 to 1, how well does the answer address the question?
Reply with only a number.
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    try:
        return float(response.choices[0].message.content.strip())
    except:
        return 0.0


def optimize_query_for_retrieval(query):
    rewrite_prompt = f"""
Rewrite the following user question into a concise search query.
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


# ---------------- Agentic RAG (DAY 13 MONITORED) ----------------
# def agentic_rag(query):

#     request_id = generate_request_id()
#     optimized_query = optimize_query_for_retrieval(query)

#     # Retrieval
#     chunks, retrieval_time = hybrid_retrieve(optimized_query, top_k=8)

#     # Rerank
#     chunks, rerank_time = rerank_documents(optimized_query, chunks, top_n=4)

#     if not has_sufficient_context(chunks):
#         log_metrics({
#             "request_id": request_id,
#             "query": query,
#             "status": "insufficient_context"
#         })
#         return "I don’t have enough information in the provided documents."

#     prompt = build_prompt(chunks, query)

#     # LLM timing
#     start_llm = time.time()
#     answer = generate_answer(prompt)
#     relevance_score = answer_relevance_score(query, answer)
#     llm_time = (time.time() - start_llm) * 1000

#     context_text = "\n\n".join([c["text"] for c in chunks])
#     verification = verify_answer(context_text, answer)

#     hallucination_flag = True if verification == "NO" else False

#     # Rerank stats
#     pairs = [(optimized_query, doc["text"]) for doc in chunks]
#     rerank_scores = reranker_model.predict(pairs)

#     avg_rerank_score = float(sum(rerank_scores) / len(rerank_scores))
#     max_rerank_score = float(max(rerank_scores))

#     # Log everything
#     log_metrics({
#         "request_id": request_id,
#         "query": query,
#         "optimized_query": optimized_query,
#         "retrieval_time_ms": retrieval_time,
#         "rerank_time_ms": rerank_time,
#         "llm_time_ms": llm_time,
#         "avg_rerank_score": avg_rerank_score,
#         "max_rerank_score": max_rerank_score,
#         "num_chunks": len(chunks),
#         "hallucination_flag": hallucination_flag
        
#     })

#     if hallucination_flag:
#         return "The generated answer is not fully supported by retrieved context."

#     return answer

# ---------------- Agentic RAG (DAY 14 EVALUATED) ----------------
def agentic_rag(query):

    request_id = generate_request_id()
    optimized_query = optimize_query_for_retrieval(query)

    # ---------------- Retrieval ----------------
    chunks, retrieval_time = hybrid_retrieve(optimized_query, top_k=8)

    # ---------------- Rerank ----------------
    chunks, rerank_time = rerank_documents(optimized_query, chunks, top_n=4)

    if not has_sufficient_context(chunks):
        log_metrics({
            "request_id": request_id,
            "query": query,
            "status": "insufficient_context"
        })
        return "I don’t have enough information in the provided documents."

    prompt = build_prompt(chunks, query)

    # ---------------- Generation ----------------
    start_llm = time.time()
    answer = generate_answer(prompt)
    llm_time = (time.time() - start_llm) * 1000

    # ---------------- Evaluation ----------------
    context_text = "\n\n".join([c["text"] for c in chunks])

    faithfulness = faithfulness_score(context_text, answer)
    relevance = answer_relevance_score(query, answer)

    hallucination_flag = True if faithfulness < 0.6 else False

    # ---------------- Rerank Stats ----------------
    pairs = [(optimized_query, doc["text"]) for doc in chunks]
    rerank_scores = reranker_model.predict(pairs)

    avg_rerank_score = float(sum(rerank_scores) / len(rerank_scores))
    max_rerank_score = float(max(rerank_scores))

    # Normalize rerank score roughly (MS MARCO range ~0-10 typically)
    # normalized_rerank = min(avg_rerank_score / 10, 1.0)
    # Use sigmoid normalization
    # import math

    normalized_rerank = 1 / (1 + math.exp(-avg_rerank_score))

    # ---------------- Final Confidence Score ----------------
    base_confidence = calculate_confidence(
    faithfulness,
    relevance,
    answer,
    question
)

    confidence_score = (
        0.7 * base_confidence +
        0.3 * normalized_rerank
    )
    # ---------------- Logging ----------------
    log_metrics({
        "request_id": request_id,
        "query": query,
        "optimized_query": optimized_query,
        "retrieval_time_ms": retrieval_time,
        "rerank_time_ms": rerank_time,
        "llm_time_ms": llm_time,
        "avg_rerank_score": avg_rerank_score,
        "max_rerank_score": max_rerank_score,
        "num_chunks": len(chunks),
        "faithfulness_score": round(faithfulness, 3),
        "answer_relevance_score": round(relevance, 3),
        "confidence_score": round(confidence_score, 3),
        "hallucination_flag": hallucination_flag
    })

    # ---------------- Safety Gate ----------------
    if hallucination_flag:
        return "The generated answer may not be fully supported by the retrieved context."

    return answer
def run_rag_attempt(question, top_k=8, top_n=4):

    optimized_query = optimize_query_for_retrieval(question)

    # ---------------- Retrieval ----------------
    start_retrieval = time.time()

    chunks = multi_query_retrieve(optimized_query, top_k=top_k)

    retrieval_time = (time.time() - start_retrieval) * 1000

    # ---------------- Rerank ----------------
    chunks, rerank_time = rerank_documents(optimized_query, chunks, top_n=top_n)

    if not has_sufficient_context(chunks):
        return {
            "answer": "Insufficient context.",
            "confidence": 0.0,
            "faithfulness": 0.0,
            "relevance": 0.0,
            "latency": retrieval_time + rerank_time,
            "top_k": top_k,
            "top_n": top_n
        }

    prompt = build_prompt(chunks, question)

    # ---------------- Generation ----------------
    start_llm = time.time()
    answer = generate_answer(prompt)
    llm_time = (time.time() - start_llm) * 1000

    # ---------------- Evaluation ----------------
    context_text = "\n\n".join([c["text"] for c in chunks])

    faithfulness = faithfulness_score(context_text, answer)
    relevance = answer_relevance_score(question, answer)

    # Keep same confidence logic as Day 14
    pairs = [(optimized_query, doc["text"]) for doc in chunks]
    rerank_scores = reranker_model.predict(pairs)
    avg_rerank_score = float(sum(rerank_scores) / len(rerank_scores))
    normalized_rerank = 1 / (1 + math.exp(-avg_rerank_score))
    # normalized_rerank = min(avg_rerank_score / 10, 1.0)

    base_confidence = calculate_confidence(
    faithfulness,
    relevance,
    answer,
    question
)

    confidence_score = (
        0.7 * base_confidence +
        0.3 * normalized_rerank
    )

    total_latency = retrieval_time + rerank_time + llm_time

    return {
        "answer": answer,
        "confidence": round(confidence_score, 3),
        "faithfulness": round(faithfulness, 3),
        "relevance": round(relevance, 3),
        "latency": round(total_latency, 2),
        "top_k": top_k,
        "top_n": top_n
    }
def adaptive_rag(question):
    print("\n--- Attempt 1 (Standard Depth) ---")

    attempt_1 = run_rag_attempt(
        question,
        top_k=8,
        top_n=4
    )

    print("Attempt 1 Confidence:", attempt_1["confidence"])

    # If good enough → return
    if attempt_1["confidence"] >= CONFIDENCE_THRESHOLD:
        attempt_1["fallback_used"] = False
        return attempt_1

    print("\n--- Attempt 2 (Deeper Retrieval) ---")

    attempt_2 = run_rag_attempt(
        question,
        top_k=14,
        top_n=6
    )

    print("Attempt 2 Confidence:", attempt_2["confidence"])

    # Pick best
    best_attempt = max(
        [attempt_1, attempt_2],
        key=lambda x: x["confidence"]
    )

    best_attempt["fallback_used"] = True
    return best_attempt


# ---------------- Retrieval Evaluation ----------------

def precision_at_k(question, expected_keywords, top_k=8):
    optimized_query = optimize_query_for_retrieval(question)
    chunks, _ = hybrid_retrieve(optimized_query, top_k=top_k)

    relevant_count = 0
    for c in chunks:
        text = c["text"].lower()
        if any(kw.lower() in text for kw in expected_keywords):
            relevant_count += 1

    precision = relevant_count / top_k
    return round(precision, 2)


# ---------------- Retrieval Evaluation ----------------
def evaluate_retrieval(question, expected_keywords, top_k=8):
    optimized_query = optimize_query_for_retrieval(question)
    chunks, _ = hybrid_retrieve(optimized_query, top_k=top_k)

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


# ---------------- MAIN ----------------
if __name__ == "__main__":

    question = "Analyze the systemic barriers preventing enterprise AI scale."
    # print(agentic_rag(question))
    result = adaptive_rag(question)

    print("\nFINAL RESULT")
    print("Answer:\n", result["answer"])
    print("\nConfidence:", result["confidence"])
    print("Faithfulness:", result["faithfulness"])
    print("Relevance:", result["relevance"])
    print("Latency:", result["latency"])
    print("Fallback Used:", result["fallback_used"])

    for item in evaluation_questions:
        result = evaluate_retrieval(
            item["question"],
            item["expected_keywords"]
        )
        print(result)