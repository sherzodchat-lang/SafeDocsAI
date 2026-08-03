from app.shared.settings.config import settings

from app.modules.rag.chunker_config import (  # noqa: F401  (re-export)
    CHUNKER_TARGET_TOKENS,
    CHUNKER_MAX_TOKENS,
    CHUNKER_MIN_TOKENS,
    CHUNKER_OVERLAP_TOKENS,
    CHUNKER_MAX_CHARS,
    RU_TJ_STOPWORDS,
    REASONING_MARKERS,
    TAJIK_TO_RU_HINTS,
)


DEFAULT_CHAT_MODEL = settings.OLLAMA_MODEL_CHAT
# У embedding-модели умолчания нет: пусто здесь — это «не задана», и такой
# ответ система обязана дать честно (SettingsErrors.EMBEDDING_MODEL_UNSET), а
# не подставить имя, из которого выведется пустая коллекция ChromaDB. Само
# разрешение (файл настроек -> переменная окружения) живёт в
# RuntimeSettingsService, а не здесь: значение, прочитанное на импорте, не
# переживает ни правку настроек, ни подмену переменной в тестах.
DEFAULT_EMBEDDING_MODEL = ""
MULTILINGUAL_EMBEDDING_MODEL = DEFAULT_EMBEDDING_MODEL

# Метрика расстояния коллекции ChromaDB. Это дефолт ChromaDB, и существующие
# коллекции уже созданы с ним — здесь он зафиксирован явно, чтобы смена дефолта
# в новой версии ChromaDB не сдвинула молча шкалу расстояний.
# Ollama отдаёт L2-нормированные эмбеддинги (проверено на qwen3-embedding:8b,
# |v| = 1.0), а chroma-шный "l2" — это КВАДРАТ евклидова расстояния, поэтому на
# нормированных векторах d = 2 * (1 - cos), то есть ровно удвоенное косинусное
# расстояние. Порог ниже задан в этой шкале: 1.0 == косинусная близость 0.5.
# Переход на "cosine" ополовинит шкалу (порог стал бы 0.5) и потребует
# пересоздания коллекции с переиндексацией: hnsw:space существующей коллекции
# get_or_create не меняет.
COLLECTION_DISTANCE_SPACE = "l2"
RELEVANCE_DISTANCE_THRESHOLD = 1.0
