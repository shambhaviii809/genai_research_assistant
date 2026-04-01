# 🧠 GenAI Technical Research Assistant

An end-to-end **Agentic RAG (Retrieval-Augmented Generation)** system designed to solve the challenge of extracting **accurate, structured, and trustworthy insights from large technical documents**.

---

# ❗ Problem Statement

Modern organizations deal with massive volumes of **technical reports, research papers, PDFs, and internal documentation**. Extracting meaningful insights from these sources is difficult due to:

### 🔴 Core Challenges

- **Information Overload**  
  Users must manually read long documents to find relevant information.

- **Lack of Context-Aware Search**  
  Traditional keyword search fails to capture semantic meaning and intent.

- **Hallucinations in LLMs**  
  Large Language Models often generate plausible but **incorrect or unsupported answers**.

- **No Reliability Metrics**  
  Existing systems rarely provide:
  - Confidence scores
  - Faithfulness checks
  - Source grounding

- **Fragmented Knowledge**  
  Insights are spread across multiple documents with no unified reasoning layer.

---

# 💡 Solution

This project builds a **Agentic RAG system** that:

✅ Retrieves relevant context using hybrid search  
✅ Reranks results using cross-encoder models  
✅ Generates structured, research-grade answers  
✅ Evaluates its own output for reliability  
✅ Adapts retrieval depth dynamically  

# 🚀 Features

## 🔍 Advanced Retrieval
- Dense retrieval using FAISS + Sentence Transformers
- Sparse retrieval using BM25
- Hybrid fusion scoring (semantic + keyword)

## 🧠 Intelligent Reasoning
- Query rewriting for better retrieval
- Intent classification:
  - Comparison
  - Risk analysis
  - Trend analysis
- Structured analytical outputs

## 📊 Evaluation Layer (Key Differentiator)
- Faithfulness scoring (context grounding)
- Answer relevance scoring
- Confidence scoring (with reranker signal)

## 🔁 Adaptive RAG
- Multi-pass retrieval
- Automatically retries with deeper search if confidence is low

## 💬 Chat Interface
- Multi-chat sessions
- Persistent history (localStorage)
- Markdown + code rendering
- ChatGPT-like UX


# 🏗️ Architecture
# 🧠 GenAI Technical Research Assistant

An end-to-end **Agentic RAG (Retrieval-Augmented Generation)** system designed to solve the challenge of extracting **accurate, structured, and trustworthy insights from large technical documents**.

---

# ❗ Problem Statement

Modern organizations deal with massive volumes of **technical reports, research papers, PDFs, and internal documentation**. Extracting meaningful insights from these sources is difficult due to:

### 🔴 Core Challenges

- **Information Overload**  
  Users must manually read long documents to find relevant information.

- **Lack of Context-Aware Search**  
  Traditional keyword search fails to capture semantic meaning and intent.

- **Hallucinations in LLMs**  
  Large Language Models often generate plausible but **incorrect or unsupported answers**.

- **No Reliability Metrics**  
  Existing systems rarely provide:
  - Confidence scores
  - Faithfulness checks
  - Source grounding

- **Fragmented Knowledge**  
  Insights are spread across multiple documents with no unified reasoning layer.

---

# 💡 Solution

This project builds a **production-style Agentic RAG system** that:

✅ Retrieves relevant context using hybrid search  
✅ Reranks results using cross-encoder models  
✅ Generates structured, research-grade answers  
✅ Evaluates its own output for reliability  
✅ Adapts retrieval depth dynamically  

---

# 🚀 Features

## 🔍 Advanced Retrieval
- Dense retrieval using FAISS + Sentence Transformers
- Sparse retrieval using BM25
- Hybrid fusion scoring (semantic + keyword)

## 🧠 Intelligent Reasoning
- Query rewriting for better retrieval
- Intent classification:
  - Comparison
  - Risk analysis
  - Trend analysis
- Structured analytical outputs

## 📊 Evaluation Layer (Key Differentiator)
- Faithfulness scoring (context grounding)
- Answer relevance scoring
- Confidence scoring (with reranker signal)

## 🔁 Adaptive RAG
- Multi-pass retrieval
- Automatically retries with deeper search if confidence is low

## 💬 Chat Interface
- Multi-chat sessions
- Persistent history (localStorage)
- Markdown + code rendering
- ChatGPT-like UX

---

# 🏗️ Architecture
User Query
↓
Query Optimization (LLM)
↓
Hybrid Retrieval (FAISS + BM25)
↓
Cross Encoder Reranking
↓
Context Selection
↓
LLM Answer Generation
↓
Evaluation Layer
├── Faithfulness Score
├── Relevance Score
└── Confidence Score
↓
Adaptive Retry (if needed)


# 📂 Project Structure
genai_research_assistant/

├── frontend/ # Next.js Chat UI
│
├── search/ # Backend (FastAPI + RAG)
│ ├── api.py
│ ├── rag_pipeline.py
│ ├── requirements.txt
│
├── embeddings/ # FAISS index (ignored)
├── data/ # Source documents
├── vector_db/ # Vector storage
│
└── .gitignore

# ⚙️ Tech Stack

### Backend
- FastAPI
- FAISS
- Sentence Transformers
- BM25 (rank-bm25)
- OpenAI API

### Frontend
- Next.js (App Router)
- React
- React Markdown
