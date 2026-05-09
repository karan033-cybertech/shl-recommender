"""
FastAPI entrypoint (skeleton).

Endpoints:
- GET /health: health check
- POST /chat: chat with SHL assessment recommender agent

TODO:
- Load env vars (GEMINI_API_KEY) and initialize global agent/retriever instances.
- Add request logging, CORS, and error handling.
- Add startup events to build/load FAISS index.
"""

from __future__ import annotations

from fastapi import FastAPI

from agent import RecommenderAgent
from models import ChatRequest, ChatResponse


app = FastAPI(
    title="SHL Assessment Recommender API",
    version="0.1.0",
)


# TODO: Initialize dependencies once (LLM, retriever, vector store)
agent = RecommenderAgent()


@app.get("/health")
def health() -> dict:
    """
    Lightweight health endpoint.

    TODO:
    - Optionally verify that FAISS index is loaded and LLM key is present.
    """
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    """
    Chat endpoint.

    Response schema is enforced by `ChatResponse` and MUST match:
    {
      "reply": "string",
      "recommendations": [{"name": "...", "url": "...", "test_type": "..."}],
      "end_of_conversation": boolean
    }

    TODO:
    - Add conversation memory (session id / chat history)
    - Implement agent behavior routing and retrieval
    """
    return agent.chat(req)

