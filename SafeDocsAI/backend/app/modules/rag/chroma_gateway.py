from pathlib import Path
import logging
import re

import chromadb

from app.core.exceptions import (
    EmbeddingModelNotConfigured,
    ExternalServiceError,
    SettingsErrors,
)
from app.modules.rag.constants import COLLECTION_DISTANCE_SPACE
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
    # Общий префикс всех наших коллекций: имя коллекции выводится из
    # embedding-модели, поэтому их в одном ChromaDB столько, сколько моделей
    # успели побывать в настройках. По префиксу они и находятся — см.
    # _stale_collections.
    COLLECTION_PREFIX = "andozai_docs_"

    def __init__(self) -> None:
        """Шлюз к активной коллекции ChromaDB.

        Проверка «модель задана» стоит в КОНСТРУКТОРЕ, а не в add/query/delete:
        через него проходит каждый путь к векторам без исключений — чат, поиск,
        загрузка, удаление документа и блокнота, переиндексация, воркер,
        скрипты в backend/*.py, — потому что имя коллекции выводится здесь и
        больше нигде. Проверка в трёх методах пропустила бы четвёртый, а
        добавленный завтра путь — тем более.

        Отказ мягкий по объёму, а не по громкости: приложение стартует, раздел
        настроек живёт (RuntimeSettingsService сюда не ходит), но всё, чему
        нужна векторная база, отвечает 503 с кодом
        settings.embedding_model_unset. Молча взять «какую-нибудь» модель
        нельзя: get_or_create_collection на незнакомое имя создаёт ПУСТУЮ
        коллекцию, и продукт превращается в пустой поиск при полной базе.
        """
        from app.shared.settings.runtime_settings import RuntimeSettingsService

        self.model_manager = ModelManager()
        # Разрешение целиком (файл настроек -> переменная окружения) сделано в
        # RuntimeSettingsService; сюда приходит результат, и пусто здесь
        # означает «не задана нигде».
        self.embedding_model = self.model_manager.resolve_embedding_model(
            RuntimeSettingsService.embedding_model()
        )
        if not self.embedding_model:
            self._report_unset_model()
            raise EmbeddingModelNotConfigured()
        self.chroma_client = None
        self.collection = None
        self.chroma_error: Exception | None = None
        self._init_chroma()

    # Об отсутствующей модели жалуемся один раз за жизнь процесса: шлюз
    # создаётся на каждый вопрос к ассистенту и на каждую проверку /ready, и
    # без фильтра одна ненастроенная строка залила бы журнал. Состояние и так
    # названо при старте (app/main.py) и приходит клиенту кодом на каждый
    # отказ — вот там оно и считается.
    _unset_model_reported = False

    @classmethod
    def _report_unset_model(cls) -> None:
        if cls._unset_model_reported:
            return
        cls._unset_model_reported = True
        logger.error(
            "Embedding model is not set (neither in runtime_settings.json nor "
            "in OLLAMA_MODEL_EMBEDDING): vector search, indexing and deletion "
            "are refused with %s until a model is chosen in the admin panel.",
            SettingsErrors.EMBEDDING_MODEL_UNSET,
        )

    @classmethod
    def _collection_name(cls, embedding_model: str) -> str:
        suffix = re.sub(r"[^a-z0-9]+", "_", embedding_model.lower()).strip("_")
        return f"{cls.COLLECTION_PREFIX}{suffix or 'default'}"

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

    @staticmethod
    def _collection_names(client) -> list[str]:
        """Имена коллекций клиента.

        list_collections отдаёт объекты Collection в chromadb 1.x и голые
        строки в 0.6.x — здесь и там из ответа берётся имя.
        """
        names: list[str] = []
        for item in client.list_collections() or []:
            name = item if isinstance(item, str) else getattr(item, "name", None)
            if isinstance(name, str) and name:
                names.append(name)
        return names

    def _stale_collections(self) -> list:
        """Коллекции прежних embedding-моделей, кроме активной.

        Отказ list_collections сюда не поднимается: он означает, что ChromaDB
        недоступна целиком, и удаление из активной коллекции упадёт следом уже
        с ExternalServiceError. Молчать про сам список нельзя — предупреждение
        в лог остаётся.
        """
        client = self.chroma_client
        if client is None:
            return []
        active_name = self._collection_name(self.embedding_model)
        try:
            names = [
                name
                for name in self._collection_names(client)
                if name.startswith(self.COLLECTION_PREFIX) and name != active_name
            ]
        except Exception as exc:
            logger.warning("Could not list ChromaDB collections: %s", exc)
            return []
        # embedding_function на удалении по id не используется (ничего не
        # считается), но без него chromadb подставляет свою дефолтную и тянет
        # onnxruntime.
        ef = self.get_embedding_function()
        collections = []
        for name in names:
            try:
                collections.append(
                    client.get_collection(name=name, embedding_function=ef)
                )
            except Exception as exc:
                raise ExternalServiceError(
                    "ChromaDB request failed",
                    service="ChromaDB",
                    status_code=503,
                    cause=exc,
                ) from exc
        return collections

    def delete_documents(self, ids: list[str]) -> None:
        """Убрать векторы из ВСЕХ наших коллекций, а не только из активной.

        Имя коллекции выводится из embedding-модели, а модель меняется в
        настройках на живой системе. Удаление только из активной коллекции
        после такой смены — no-op: документ индексировался в другую, и его
        векторы оставались там навсегда. Вернув настройку назад, админ снова
        находил в поиске цитаты из давно удалённых документов — зомби,
        переживающий откат.
        Чистка по id безопасна: id вектора — это Chunk.id из PostgreSQL, он
        глобально уникален, поэтому в чужой коллекции такого id либо нет
        (delete по несуществующему id у ChromaDB — no-op), либо это ровно тот
        же чанк, записанный другой моделью.
        """
        self._init_chroma()
        if self.collection is None:
            raise ExternalServiceError(
                "ChromaDB is unavailable",
                service="ChromaDB",
                status_code=503,
                cause=self.chroma_error,
            )
        if not ids:
            return
        for collection in [self.collection, *self._stale_collections()]:
            try:
                collection.delete(ids=ids)
            except Exception as exc:
                # Отказ не проглатываем и на неактивной коллекции: тихо
                # оставленные векторы — это и есть исходный дефект. Вызывающие
                # уже умеют с этим жить (удаление блокнота ставит задачу
                # cleanup_embeddings, удаление документа отвечает 503).
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


def log_vector_store_state() -> Exception | None:
    """Записать в журнал одну строку о состоянии векторного хранилища.

    Формат: `embedding_model=X -> коллекция Y, векторов N`. До неё связать
    «поиск ничего не находит при полной базе» с настройкой было нечем: имя
    коллекции выводится из embedding-модели, а сама модель нигде не
    называлась — ни при старте, ни в ответах.

    Возвращает отказ ChromaDB, если она недоступна, и None во всех остальных
    случаях — включая незаданную модель. Так и задумано: незаданная модель
    старт не роняет (иначе некуда прийти и выбрать её), она отвечает 503 на
    RAG-операциях. Число векторов — сетевой запрос в ChromaDB: его отказ
    только меняет строку, но никогда не поднимается наверх.
    """
    from app.shared.settings.runtime_settings import RuntimeSettingsService

    embedding_model = RuntimeSettingsService.embedding_model()
    if not embedding_model:
        logger.error(
            "embedding_model не задан -> коллекции нет, векторов нет. "
            "Задайте модель в админ-панели (PUT /api/v1/settings/) или в "
            "переменной окружения OLLAMA_MODEL_EMBEDDING; до этого поиск, "
            "чат, индексация и удаление векторов отвечают 503 (%s).",
            SettingsErrors.EMBEDDING_MODEL_UNSET,
        )
        return None

    collection_name = ChromaGateway._collection_name(embedding_model)
    gateway = ChromaGateway()
    if gateway.collection is None:
        logger.error(
            "embedding_model=%s -> коллекция %s, векторов неизвестно: "
            "ChromaDB недоступна (%s)",
            embedding_model,
            collection_name,
            gateway.chroma_error,
        )
        return gateway.chroma_error or RuntimeError("ChromaDB is unavailable")

    try:
        count = gateway.collection.count()
    except Exception as exc:
        # Коллекция открылась, а счёт не сошёлся: ChromaDB отвалилась ровно
        # сейчас. Ронять старт из-за строки в журнале нельзя.
        logger.warning(
            "embedding_model=%s -> коллекция %s, векторов неизвестно: "
            "запрос в ChromaDB не удался (%s)",
            embedding_model,
            collection_name,
            exc,
        )
        return None

    logger.info(
        "embedding_model=%s -> коллекция %s, векторов %s",
        embedding_model,
        collection_name,
        count,
    )
    return None
