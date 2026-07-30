from typing import List

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str
    notebook_id: int | None = None
    domain_profile: str | None = None


class RetrievalRequest(BaseModel):
    question: str
    notebook_id: int | None = None
    domain_profile: str | None = None
    retrieval_top_k: int | None = Field(default=None, ge=1, le=50)
    top_k: int | None = Field(default=None, ge=1, le=20)


class SourceItem(BaseModel):
    source_type: str | None = None
    doc_id: int | None = None
    doc_name: str | None = None
    page: int | None = None
    chunk_id: str | None = None
    category: str | None = None
    quote: str | None = None


class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceItem]
    log_id: int


class RetrievalChunkItem(BaseModel):
    rank: int | None = None
    retrieval_method: str | None = None
    doc_id: int | None = None
    doc_name: str | None = None
    page: int | None = None
    chunk_id: str | None = None
    quote: str | None = None
    distance: float | None = None
    lexical_score: float | None = None
    rrf_score: float | None = None
    rerank_score: float | None = None


class RetrievalResponse(BaseModel):
    question: str
    search_query: str
    retrieval_top_k: int
    top_k: int
    vector_candidates: List[RetrievalChunkItem] = Field(default_factory=list)
    lexical_candidates: List[RetrievalChunkItem] = Field(default_factory=list)
    fused_candidates: List[RetrievalChunkItem] = Field(default_factory=list)
    chunks: List[RetrievalChunkItem]
