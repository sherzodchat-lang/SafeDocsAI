"""Раздел презентаций.

Здесь переэкспортированы только схемы и константы — то, что нужно и HTTP-слою,
и тестам, и скриптам. Сервис и воркер сознательно НЕ импортируются: они тянут
за собой ретривал, ChromaDB и ModelManager, и любой импорт `app.modules.
presentations` (например, ради предела длины описания в схеме запроса)
поднимал бы половину RAG-стека.
"""

from app.modules.presentations.constants import (
    DEFAULT_LANGUAGE,
    DESCRIPTION_MAX,
    LANGUAGE_RU,
    LANGUAGE_TJ,
    PRESENTATION_JOB_TIMEOUT,
    SLIDE_COUNT_DEFAULT,
    SLIDE_COUNT_MAX,
    SLIDE_COUNT_MIN,
    SLIDE_RETRIEVAL_CANDIDATE_POOL,
    SLIDE_RETRIEVAL_TOP_K,
    STATUS_ERROR,
    STATUS_GENERATING,
    STATUS_QUEUED,
    STATUS_READY,
    SUPPORTED_LANGUAGES,
    normalize_language,
)
from app.modules.presentations.llm_schemas import (
    LlmResponseError,
    MIN_SLIDE_COUNT,
    PlanSection,
    PresentationPlan,
    PresentationSlide,
    RENDERER_ADDED_SLIDES,
    SlideCitation,
    content_section_count,
    parse_model_json,
    strip_code_fences,
    validate_plan,
    validate_slide,
)

__all__ = [
    "DEFAULT_LANGUAGE",
    "DESCRIPTION_MAX",
    "LANGUAGE_RU",
    "LANGUAGE_TJ",
    "LlmResponseError",
    "MIN_SLIDE_COUNT",
    "PRESENTATION_JOB_TIMEOUT",
    "PlanSection",
    "PresentationPlan",
    "PresentationSlide",
    "RENDERER_ADDED_SLIDES",
    "SLIDE_COUNT_DEFAULT",
    "SLIDE_COUNT_MAX",
    "SLIDE_COUNT_MIN",
    "SLIDE_RETRIEVAL_CANDIDATE_POOL",
    "SLIDE_RETRIEVAL_TOP_K",
    "STATUS_ERROR",
    "STATUS_GENERATING",
    "STATUS_QUEUED",
    "STATUS_READY",
    "SUPPORTED_LANGUAGES",
    "SlideCitation",
    "content_section_count",
    "normalize_language",
    "parse_model_json",
    "strip_code_fences",
    "validate_plan",
    "validate_slide",
]
