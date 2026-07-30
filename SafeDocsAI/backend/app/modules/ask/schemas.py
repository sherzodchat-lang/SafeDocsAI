from typing import List

from pydantic import BaseModel


class AskRequest(BaseModel):
    question: str
    notebook_id: int | None = None
    top_k: int | None = None
    domain_profile: str | None = None


class CitationItem(BaseModel):
    source_id: int | None = None
    source_name: str | None = None
    page: int | None = None
    chunk_id: str | None = None
    quote: str | None = None


class AskResponse(BaseModel):
    answer: str
    citations: List[CitationItem]
    log_id: int
