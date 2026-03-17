from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from rag_pipeline import adaptive_rag
import time

app = FastAPI(title="GenAI Research Assistant API")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------- Request Schema --------
class QuestionRequest(BaseModel):
    question: str


# -------- Health Check --------
@app.get("/")
def root():
    return {"status": "RAG API is running"}


# -------- Detailed API (for debugging) --------
@app.post("/ask")
def ask_question(request: QuestionRequest):
    return adaptive_rag(request.question)


# -------- Chat Endpoint (Streaming for frontend) --------
@app.post("/chat")
def chat(request: QuestionRequest):

    result = adaptive_rag(request.question)

    answer = result["answer"]

    def stream_answer():

        words = answer.split(" ")

        for word in words:
            yield word + " "
            time.sleep(0.02)  # simulate token streaming

    return StreamingResponse(stream_answer(), media_type="text/plain")
@app.get("/health")
def health():
    return {"status": "ok"}
# from fastapi import FastAPI
# from pydantic import BaseModel
# from rag_pipeline import adaptive_rag
# from fastapi.middleware.cors import CORSMiddleware
# import time

# app = FastAPI(title="GenAI Research Assistant API")
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://localhost:3000"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # -------- Request Schema --------
# class QuestionRequest(BaseModel):
#     question: str


# # -------- Response Schema --------
# class AnswerResponse(BaseModel):
#     answer: str
#     confidence: float
#     faithfulness: float
#     relevance: float
#     latency: float
#     fallback_used: bool


# # -------- Health Check --------
# @app.get("/")
# def root():
#     return {"status": "RAG API is running"}


# # -------- Main Endpoint --------
# @app.post("/ask", response_model=AnswerResponse)
# def ask_question(request: QuestionRequest):
#     start_time = time.time()

#     result = adaptive_rag(request.question)

#     total_time = time.time() - start_time

#     return {
#         "answer": result["answer"],
#         "confidence": result["confidence"],
#         "faithfulness": result["faithfulness"],
#         "relevance": result["relevance"],
#         "latency": result["latency"],
#         "fallback_used": result["fallback_used"]
#     }