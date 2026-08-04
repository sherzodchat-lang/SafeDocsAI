import json
from typing import Any, AsyncIterator, Sequence
import urllib.request
from urllib.parse import urljoin

import ollama

from app.core.exceptions import (
    EmbeddingModelNotConfigured,
    ExternalServiceError,
    ExternalServiceErrorKind,
)
from app.modules.rag.constants import DEFAULT_CHAT_MODEL
from app.shared.settings.config import settings

# Имена классов исключений, а не сами классы: httpx приходит транзитом через
# ollama, а urllib и socket дают свои. Ловить по типу значило бы импортировать
# сюда три чужие библиотеки и проглядеть четвёртую при смене клиента.
#
# Разделение на две группы — это и есть то, что раньше было утеряно. Оба
# набора дают 503 (см. _wrap_provider_error), но повторять их можно
# по-разному, а рассказывать пользователю — тем более: «сервис не поднят» и
# «сервис занят и не успел ответить» чинятся разными людьми.
_CONNECTION_ERROR_NAMES = {
    "ConnectError",
    "ConnectTimeout",
    "ConnectionError",
    "ConnectionRefusedError",
    "RemoteProtocolError",
    "URLError",
}
_TIMEOUT_ERROR_NAMES = {
    "PoolTimeout",
    "ReadTimeout",
    "TimeoutError",
    "TimeoutException",
    "WriteTimeout",
}


def _error_names(exc: Exception) -> set[str]:
    """Имена самого исключения и того, что его породило.

    Цепочка нужна потому, что ollama оборачивает httpx: снаружи бывает
    ResponseError, а причина отказа — в __cause__.
    """
    names = {exc.__class__.__name__}
    for attr in ("__cause__", "__context__"):
        nested = getattr(exc, attr, None)
        if nested is not None:
            names.add(nested.__class__.__name__)
    return names


def _classify_provider_error(exc: Exception) -> str:
    """Причина отказа провайдера -> ExternalServiceErrorKind.

    Порядок проверок значим: у ollama.ResponseError есть свой status_code, но
    в него же заворачиваются сетевые беды, поэтому сеть смотрится первой.
    ConnectTimeout считается НЕДОСТУПНОСТЬЮ, а не таймаутом: соединение не
    установилось вовсе, то есть отвечать было некому.
    """
    names = _error_names(exc)
    if names & _CONNECTION_ERROR_NAMES:
        return ExternalServiceErrorKind.UNAVAILABLE
    if names & _TIMEOUT_ERROR_NAMES:
        return ExternalServiceErrorKind.TIMEOUT

    status = getattr(exc, "status_code", None)
    if isinstance(status, int) and status > 0:
        # 5xx — беда на стороне Ollama (модель не влезла в память, воркер
        # умер), она проходит сама. 4xx — осмысленный отказ запросу («нет
        # такой модели»), и повтор вернёт ровно тот же ответ.
        if status >= 500:
            return ExternalServiceErrorKind.SERVER_ERROR
        return ExternalServiceErrorKind.REQUEST_REJECTED

    return ExternalServiceErrorKind.UNKNOWN


# Причины, которым положен 503, а не 502. Набор тот же, что давал прежний
# _is_service_unavailable: статус уезжает в HTTP-ответы чата и настроек, и
# двигать его заодно с классификацией значило бы чинить одно, ломая другое.
_UNAVAILABLE_KINDS = (
    ExternalServiceErrorKind.UNAVAILABLE,
    ExternalServiceErrorKind.TIMEOUT,
)


class ModelManager:
    def __init__(self, timeout: float | None = None) -> None:
        """timeout — бюджет ОДНОГО HTTP-обращения к Ollama.

        Параметр существует потому, что бюджетов честно два, и сводить их к
        одному нельзя ни в одну сторону:

        * ЧАТ (умолчание, settings.OLLAMA_TIMEOUT_SECONDS = 120) —
          интерактивный бюджет. За ним сидит человек и смотрит на экран;
          поднять его до пяти минут ради фоновой джобы значит заставить
          зависший чатовский запрос висеть впятеро дольше;
        * ПРЕЗЕНТАЦИИ (передаётся явно, LLM_CALL_TIMEOUT = 300) — фоновая
          джоба, откалиброванная по замерам приёмки: худший наблюдённый вызов
          (план) — 69.9 с.

        Пока значение было одно, откалиброванный LLM_CALL_TIMEOUT был
        недостижим: клиент сдавался на 120-й секунде («max 120.0с» в строке
        статистики воркера — ровное число, то есть не замер), и wait_for
        вокруг вызова не срабатывал никогда. Не «унифицировать» обратно:
        разные величины здесь — решение, а не недосмотр.
        """
        self._timeout = (
            settings.OLLAMA_TIMEOUT_SECONDS if timeout is None else float(timeout)
        )
        self._ollama_client = ollama.Client(
            host=settings.OLLAMA_API_BASE,
            timeout=self._timeout,
        )
        self._ollama_async_client = ollama.AsyncClient(
            host=settings.OLLAMA_API_BASE,
            timeout=self._timeout,
        )

    @property
    def timeout(self) -> float:
        """Бюджет одного обращения — тем, кто обязан его знать.

        Читается сторожевым тестом связи (tests/
        test_presentation_call_timeout_single_source.py): без внешнего доступа
        к значению проверить, что пути презентаций достался именно
        LLM_CALL_TIMEOUT, можно было бы только через приватные потроха httpx.
        """
        return self._timeout

    @staticmethod
    def resolve_chat_model(model: str | None = None) -> str:
        selected = (model or "").strip()
        return selected or DEFAULT_CHAT_MODEL

    @staticmethod
    def resolve_embedding_model(model: str | None = None) -> str:
        """Имя embedding-модели; "" означает «не задана».

        Умолчания в коде нет намеренно (см. OLLAMA_MODEL_EMBEDDING в
        app/shared/settings/config.py): неверно угаданное имя выводит имя
        коллекции ChromaDB и молча уводит поиск в пустоту. Переменная
        окружения читается на каждый вызов, а не берётся из константы
        модуля: константа замерзает на импорте.
        """
        selected = (model or "").strip()
        if selected:
            return selected
        return str(settings.OLLAMA_MODEL_EMBEDDING or "").strip()

    @staticmethod
    def _wrap_provider_error(service: str, exc: Exception) -> ExternalServiceError:
        """Чужое исключение -> ошибка проекта, С СОХРАНЁННОЙ ПРИЧИНОЙ.

        Причина (kind) важнее текста: по ней вызывающий решает, повторять ли
        обращение и что сказать пользователю. Без неё «соединение отвергнуто»
        и «ответ не пришёл вовремя» — одна и та же ошибка с одним и тем же
        503, и презентации отдавали «Ollama недоступна» на живую, но занятую
        Ollama.
        """
        kind = _classify_provider_error(exc)
        status_code = 503 if kind in _UNAVAILABLE_KINDS else 502
        # Текст уезжает в presentation.error_text и в detail HTTP-ответов, то
        # есть его читает человек: он обязан совпадать с тем, что произошло.
        if kind == ExternalServiceErrorKind.TIMEOUT:
            message = f"{service} did not answer in time"
        elif kind == ExternalServiceErrorKind.UNAVAILABLE:
            message = f"{service} is unavailable"
        else:
            message = f"{service} request failed"
        return ExternalServiceError(
            message,
            service=service,
            status_code=status_code,
            cause=exc,
            kind=kind,
        )

    @staticmethod
    def _extract_embeddings(response: Any) -> list[list[float]]:
        embeddings = getattr(response, "embeddings", None)
        if embeddings is None and isinstance(response, dict):
            embeddings = response.get("embeddings")
        if embeddings is None:
            try:
                embeddings = response["embeddings"]
            except Exception:
                embeddings = None
        if not embeddings:
            raise ExternalServiceError(
                "Ollama returned empty embeddings",
                service="Ollama",
                status_code=502,
            )
        return [list(item) for item in embeddings]

    @staticmethod
    def _extract_openai_embeddings(response: dict[str, Any]) -> list[list[float]]:
        data = response.get("data") or []
        embeddings: list[list[float]] = []
        for item in data:
            embedding = item.get("embedding") if isinstance(item, dict) else None
            if embedding:
                embeddings.append(list(embedding))
        if not embeddings:
            raise ExternalServiceError(
                "Ollama returned empty embeddings",
                service="Ollama",
                status_code=502,
            )
        return embeddings

    @staticmethod
    def _extract_model_names(response: Any) -> list[str]:
        models = getattr(response, "models", None)
        if models is None and isinstance(response, dict):
            models = response.get("models")
        if models is None:
            try:
                models = response["models"]
            except Exception:
                models = []

        names: list[str] = []
        for item in models or []:
            model_name = getattr(item, "model", None)
            if model_name is None and isinstance(item, dict):
                model_name = item.get("model")
            if isinstance(model_name, str) and model_name.strip():
                names.append(model_name.strip())
        return names

    async def chat(
        self, messages: list[dict[str, str]], model: str | None = None, max_tokens: int | None = None, think: bool = False, num_ctx: int | None = None
    ) -> str:
        resolved_model = self.resolve_chat_model(model)

        try:
            kwargs: dict = dict(model=resolved_model, messages=messages, think=think, keep_alive=-1)
            options: dict = {"num_ctx": num_ctx if num_ctx is not None else 12288}
            if max_tokens is not None:
                options["num_predict"] = max_tokens
            kwargs["options"] = options
            response = await self._ollama_async_client.chat(**kwargs)
        except Exception as exc:
            raise self._wrap_provider_error("Ollama", exc) from exc

        content = response["message"]["content"].strip()
        return content

    async def chat_stream(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        max_tokens: int | None = None,
        think: bool = False,
        num_ctx: int | None = None,
    ) -> AsyncIterator[str]:
        resolved_model = self.resolve_chat_model(model)

        try:
            kwargs: dict = dict(
                model=resolved_model,
                messages=messages,
                think=think,
                keep_alive=-1,
                stream=True,
            )
            options: dict = {"num_ctx": num_ctx if num_ctx is not None else 12288}
            if max_tokens is not None:
                options["num_predict"] = max_tokens
            kwargs["options"] = options
            response = await self._ollama_async_client.chat(**kwargs)
        except Exception as exc:
            raise self._wrap_provider_error("Ollama", exc) from exc

        if hasattr(response, "__aiter__"):
            try:
                async for chunk in response:
                    if isinstance(chunk, dict):
                        message = chunk.get("message", {})
                        content = message.get("content") or chunk.get("response", "")
                    else:
                        message = getattr(chunk, "message", None)
                        if isinstance(message, dict):
                            content = message.get("content", "")
                        else:
                            content = getattr(message, "content", "") if message else ""
                        content = content or getattr(chunk, "response", "")
                    if content:
                        yield content
                return
            except Exception as exc:
                raise self._wrap_provider_error("Ollama", exc) from exc

        if isinstance(response, dict):
            message = response.get("message", {})
            content = (message.get("content") or response.get("response", "")).strip()
        else:
            message = getattr(response, "message", None)
            if isinstance(message, dict):
                content = message.get("content", "")
            else:
                content = getattr(message, "content", "") if message else ""
            content = (content or getattr(response, "response", "")).strip()
        if content:
            yield content

    def embed(
        self, texts: Sequence[str], model: str | None = None
    ) -> list[list[float]]:
        resolved_model = self.resolve_embedding_model(model)
        if not resolved_model:
            # Второй рубеж после ChromaGateway: без имени модели Ollama
            # ответила бы 400 на каждый чанк, и в журнале осталась бы жалоба
            # на Ollama вместо настоящей причины.
            raise EmbeddingModelNotConfigured()
        try:
            response = self._ollama_client.embed(
                model=resolved_model,
                input=list(texts),
            )
        except Exception as exc:
            # Older Ollama servers may not support the batch /api/embed route
            # used by newer client versions. Fall back to per-text embeddings.
            try:
                embeddings: list[list[float]] = []
                for text in texts:
                    legacy_response = self._ollama_client.embeddings(
                        model=resolved_model,
                        prompt=text,
                    )
                    embedding = getattr(legacy_response, "embedding", None)
                    if embedding is None and isinstance(legacy_response, dict):
                        embedding = legacy_response.get("embedding")
                    if not embedding:
                        raise ExternalServiceError(
                            "Ollama returned empty embeddings",
                            service="Ollama",
                            status_code=502,
                        )
                    embeddings.append(list(embedding))
                return embeddings
            except Exception:
                try:
                    request = urllib.request.Request(
                        urljoin(
                            settings.OLLAMA_API_BASE.rstrip("/") + "/",
                            "v1/embeddings",
                        ),
                        data=json.dumps(
                            {"model": resolved_model, "input": list(texts)}
                        ).encode("utf-8"),
                        headers={"Content-Type": "application/json"},
                    )
                    with urllib.request.urlopen(
                        request, timeout=self._timeout
                    ) as response:
                        return self._extract_openai_embeddings(
                            json.loads(response.read().decode("utf-8"))
                        )
                except Exception as fallback_exc:
                    raise self._wrap_provider_error("Ollama", fallback_exc) from exc
        return self._extract_embeddings(response)

    def list_ollama_models(self) -> list[str]:
        # Клиент создаётся ВНУТРИ try: ollama.Client разбирает адрес из
        # OLLAMA_API_BASE и на негодном значении бросает сам, ещё до запроса.
        # Мимо обёртки этот отказ уходил наверх голым и превращался в 500 на
        # GET /api/v1/settings/ — то есть гасил экран, на котором адрес и
        # правят. Отсюда же должен приходить обычный ExternalServiceError,
        # который вызывающие уже умеют показывать.
        try:
            discovery_client = ollama.Client(
                host=settings.OLLAMA_API_BASE,
                timeout=min(self._timeout, 5.0),
            )
            response = discovery_client.list()
        except Exception as exc:
            raise self._wrap_provider_error("Ollama", exc) from exc
        return self._extract_model_names(response)
