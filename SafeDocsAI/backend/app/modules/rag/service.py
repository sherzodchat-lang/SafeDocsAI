from app.modules.rag.chroma_gateway import ChromaGateway
from app.modules.rag.constants import (
    DEFAULT_CHAT_MODEL,
    DEFAULT_EMBEDDING_MODEL,
    MULTILINGUAL_EMBEDDING_MODEL,
    RELEVANCE_DISTANCE_THRESHOLD,
)
from app.modules.rag.generation_service import GenerationService
from app.modules.rag import text_utils


class RAGService:
    def __init__(self) -> None:
        self._gateway = ChromaGateway()
        self._generation = GenerationService()
        self.chroma_client = self._gateway.chroma_client
        self.collection = self._gateway.collection
        self.chroma_error = self._gateway.chroma_error

    def _init_chroma(self) -> None:
        self._gateway._init_chroma()
        self.chroma_client = self._gateway.chroma_client
        self.collection = self._gateway.collection
        self.chroma_error = self._gateway.chroma_error

    @staticmethod
    def _get_embedding_function():
        return ChromaGateway().get_embedding_function()

    add_documents = lambda self, documents, metadatas, ids: self._gateway.add_documents(
        documents, metadatas, ids
    )
    delete_documents = lambda self, ids: self._gateway.delete_documents(ids)
    query_documents = (
        lambda self, query_text, n_results=5, where=None: self._gateway.query_documents(
            query_text, n_results=n_results, where=where
        )
    )

    normalize_query = staticmethod(text_utils.normalize_query)
    detect_language = staticmethod(text_utils.detect_language)
    is_prompt_injection_attempt = staticmethod(text_utils.is_prompt_injection_attempt)
    _looks_like_no_data = staticmethod(text_utils.looks_like_no_data)
    stem_simple = staticmethod(text_utils.stem_simple)
    tokenize = staticmethod(text_utils.tokenize)
    _query_tokens = staticmethod(text_utils.query_tokens)
    _is_reasoning_question = staticmethod(text_utils.is_reasoning_question)
    _has_reasoning_markers = staticmethod(text_utils.has_reasoning_markers)
    tajik_query_to_russian_hint = staticmethod(text_utils.tajik_query_to_russian_hint)
    _sanitize_answer_text = staticmethod(text_utils.sanitize_answer_text)
    _is_numeric_question = staticmethod(text_utils.is_numeric_question)
    _detect_article_reference = staticmethod(text_utils.detect_article_reference)
    _boost_article_chunks = staticmethod(text_utils.boost_article_chunks)

    async def condense_query(self, query, chat_history, model=DEFAULT_CHAT_MODEL):
        return await self._generation.condense_query(query, chat_history, model=model)

    async def generate_answer(self, *args, **kwargs):
        return await self._generation.generate_answer(*args, **kwargs)

    async def stream_answer(self, *args, **kwargs):
        async for token in self._generation.stream_answer(*args, **kwargs):
            yield token


__all__ = [
    "DEFAULT_CHAT_MODEL",
    "DEFAULT_EMBEDDING_MODEL",
    "MULTILINGUAL_EMBEDDING_MODEL",
    "RELEVANCE_DISTANCE_THRESHOLD",
    "RAGService",
]
