from pathlib import Path
import logging
import re

import chromadb

from app.core.exceptions import ExternalServiceError
from app.modules.rag.constants import (
    COLLECTION_DISTANCE_SPACE,
    DEFAULT_EMBEDDING_MODEL,
)
from app.modules.rag.model_manager import ModelManager
from app.shared.settings.config import settings

logger = logging.getLogger(__name__)


class OllamaEmbeddingFunction:
    def __init__(self, manager: ModelManager, model: str) -> None:
        self._manager = manager
        self._model = model

    @staticmethod
    def _normalize_input(
        input_data: str | list[str] | tuple[str, ...] | object,
    ) -> list[str]:
        if isinstance(input_data, str):
            return [input_data]

        if isinstance(input_data, (list, tuple)):
            normalized: list[str] = []
            for item in input_data:
                if isinstance(item, str):
                    normalized.append(item)
                elif isinstance(item, (list, tuple)):
                    normalized.extend(
                        str(nested) for nested in item if nested is not None
                    )
                elif item is not None:
                    normalized.append(str(item))
            return normalized

        if input_data is None:
            return []

        return [str(input_data)]

    def __call__(self, input: list[str]) -> list[list[float]]:
        return self.embed_documents(input)

    def embed_documents(self, input: list[str]) -> list[list[float]]:
        normalized = self._normalize_input(input)
        return self._manager.embed(normalized, model=self._model)

    def embed_query(self, input: str | list[str]) -> list[list[float]]:
        normalized = self._normalize_input(input)
        if not normalized:
            return []
        return self._manager.embed([normalized[0]], model=self._model)

    @staticmethod
    def name() -> str:
        return "ollama"


class ChromaGateway:
    ADD_BATCH_SIZE = 20

    def __init__(self) -> None:
        from app.shared.settings.runtime_settings import RuntimeSettingsService

        self.model_manager = ModelManager()
        runtime_settings = RuntimeSettingsService.get_settings()
        self.embedding_model = self.model_manager.resolve_embedding_model(
            runtime_settings.get("embedding_model", DEFAULT_EMBEDDING_MODEL)
        )
        self.chroma_client = None
        self.collection = None
        self.chroma_error: Exception | None = None
        self._init_chroma()

    @classmethod
    def _collection_name(cls, embedding_model: str) -> str:
        suffix = re.sub(r"[^a-z0-9]+", "_", embedding_model.lower()).strip("_")
        return f"andozai_docs_{suffix or 'default'}"

    def get_embedding_function(self):
        return OllamaEmbeddingFunction(self.model_manager, self.embedding_model)

    @staticmethod
    def _warn_on_distance_space_mismatch(collection, collection_name: str) -> None:
        # metadata на уже существующей коллекции get_or_create не переписывает,
        # поэтому фактическую метрику читаем из конфигурации: порог релевантности
        # (RELEVANCE_DISTANCE_THRESHOLD) задан в шкале COLLECTION_DISTANCE_SPACE.
        try:
            configuration = getattr(collection, "configuration_json", None) or {}
            actual_space = (configuration.get("hnsw") or {}).get("space")
        except Exception:
            actual_space = None
        if actual_space and actual_space != COLLECTION_DISTANCE_SPACE:
            logger.warning(
                "ChromaDB collection '%s' uses distance space '%s' while the relevance "
                "threshold is calibrated for '%s'. Recreate the collection and run "
                "reindex_documents.py, otherwise relevance filtering is off-scale.",
                collection_name,
                actual_space,
                COLLECTION_DISTANCE_SPACE,
            )

    def _init_chroma(self) -> None:
        if self.collection is not None or self.chroma_error is not None:
            return
        ef = self.get_embedding_function()
        collection_name = self._collection_name(self.embedding_model)
        attempts = [
            lambda: chromadb.HttpClient(
                host=settings.CHROMA_HOST, port=settings.CHROMA_PORT
            ),
        ]
        if settings.ENVIRONMENT == "development":
            persist_dir = Path(settings.CHROMA_PERSIST_DIR)
            if not persist_dir.is_absolute():
                backend_dir = Path(__file__).resolve().parents[3]
                persist_dir = backend_dir / persist_dir
            persist_dir.mkdir(parents=True, exist_ok=True)
            attempts.append(lambda: chromadb.PersistentClient(path=str(persist_dir)))
            # Фолбэки в /tmp и в память намеренно убраны: они молча уводили
            # запись векторов в никуда, документ при этом получал status='indexed',
            # а после рестарта поиск по нему просто ничего не находил.
        last_error: Exception | None = None
        for create_client in attempts:
            try:
                client = create_client()
                collection = client.get_or_create_collection(
                    name=collection_name,
                    embedding_function=ef,
                    metadata={"hnsw:space": COLLECTION_DISTANCE_SPACE},
                )
                self.chroma_client = client
                self.collection = collection
                self.chroma_error = None
                self._warn_on_distance_space_mismatch(collection, collection_name)
                if collection.count() == 0:
                    logger.warning(
                        "ChromaDB collection '%s' is empty. "
                        "If the embedding model was recently changed, run reindex_documents.py.",
                        collection_name,
                    )
                return
            except Exception as exc:
                last_error = exc
        self.chroma_error = last_error or RuntimeError("Failed to initialize ChromaDB")
        self.collection = None
        logger.error(
            "ChromaDB is unavailable at %s:%s (%s). Indexing and retrieval will fail.",
            settings.CHROMA_HOST,
            settings.CHROMA_PORT,
            self.chroma_error,
        )

    def add_documents(
        self, documents: list[str], metadatas: list[dict], ids: list[str]
    ) -> None:
        self._init_chroma()
        if self.collection is None:
            raise ExternalServiceError(
                "ChromaDB is unavailable",
                service="ChromaDB",
                status_code=503,
                cause=self.chroma_error,
            )
        if not documents:
            return
        if not (len(documents) == len(metadatas) == len(ids)):
            raise ValueError("documents, metadatas and ids must have the same length")

        for start in range(0, len(documents), self.ADD_BATCH_SIZE):
            end = start + self.ADD_BATCH_SIZE
            self._add_documents_batch(
                documents[start:end],
                metadatas[start:end],
                ids[start:end],
            )

    def _add_documents_batch(
        self, documents: list[str], metadatas: list[dict], ids: list[str]
    ) -> None:
        try:
            embeddings = self.model_manager.embed(documents, model=self.embedding_model)
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids,
                embeddings=embeddings,
            )
        except ExternalServiceError:
            raise
        except Exception as exc:
            message = str(exc).lower()
            if "input length exceeds the context length" in message:
                logger.warning(
                    "Embedding batch exceeded context length: batch_size=%s max_chars=%s model=%s",
                    len(documents),
                    max((len(doc) for doc in documents), default=0),
                    self.embedding_model,
                )
            if (
                "input length exceeds the context length" in message
                and len(documents) > 1
            ):
                midpoint = max(1, len(documents) // 2)
                self._add_documents_batch(
                    documents[:midpoint],
                    metadatas[:midpoint],
                    ids[:midpoint],
                )
                self._add_documents_batch(
                    documents[midpoint:],
                    metadatas[midpoint:],
                    ids[midpoint:],
                )
                return
            raise ExternalServiceError(
                "ChromaDB request failed",
                service="ChromaDB",
                status_code=503,
                cause=exc,
            ) from exc

    def delete_documents(self, ids: list[str]) -> None:
        self._init_chroma()
        if self.collection is None:
            raise ExternalServiceError(
                "ChromaDB is unavailable",
                service="ChromaDB",
                status_code=503,
                cause=self.chroma_error,
            )
        if ids:
            try:
                self.collection.delete(ids=ids)
            except Exception as exc:
                raise ExternalServiceError(
                    "ChromaDB request failed",
                    service="ChromaDB",
                    status_code=503,
                    cause=exc,
                ) from exc

    def query_documents(
        self, query_text: str, n_results: int = 5, where: dict | None = None
    ) -> dict:
        self._init_chroma()
        if self.collection is None:
            raise ExternalServiceError(
                "ChromaDB is unavailable",
                service="ChromaDB",
                status_code=503,
                cause=self.chroma_error,
            )
        query_embeddings = self.model_manager.embed(
            [query_text], model=self.embedding_model
        )
        query_kwargs = {
            "query_embeddings": query_embeddings,
            "n_results": n_results,
        }
        if where:
            query_kwargs["where"] = where
        try:
            return self.collection.query(**query_kwargs)
        except ExternalServiceError:
            raise
        except Exception as exc:
            raise ExternalServiceError(
                "ChromaDB request failed",
                service="ChromaDB",
                status_code=503,
                cause=exc,
            ) from exc
