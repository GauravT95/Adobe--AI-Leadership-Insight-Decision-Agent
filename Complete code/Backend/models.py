"""
Pydantic models for API request/response schemas.
"""
from pydantic import BaseModel, Field
from typing import List, Optional


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, description="The leadership question to answer")
    method: str = Field(
        default="agentic",
        description="RAG method: naive_dense | bm25 | hybrid | hybrid_reranked | agentic"
    )


class AskResponse(BaseModel):
    answer: str
    sources: List[str]
    confidence: float
    question_type: str
    latency_seconds: float
    cache_hit: bool = False
    matched_query: Optional[str] = None
    method: str


class HealthResponse(BaseModel):
    status: str
    llm_ok: bool
    embeddings_ok: bool
    documents_loaded: int
    chunks_indexed: int


class DocumentInfo(BaseModel):
    filename: str
    size_bytes: int
    chunks: int
