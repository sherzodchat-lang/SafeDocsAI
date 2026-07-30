import json
from typing import Any, AsyncIterator, Sequence
import urllib.request
from urllib.parse import urljoin

import ollama

from app.core.exceptions import ExternalServiceError
from app.modules.rag.constants import DEFAULT_CHAT_MODEL, DEFAULT_EMBEDDING_MODEL
from app.shared.settings.config import settings


def _is_service_unavailable(exc: Exception) -> bool:
    names = {exc.__class__.__name__}
    for attr in ("__cause__", "__context__"):
        nested = getattr(exc, attr, None)
        if nested is not None:
            names.add(nested.__class__.__name__)
    unavailable = {
        "ConnectError",
        "ConnectTimeout",
        "ConnectionError",
        "ReadTimeout",
        "RemoteProtocolError",
        "TimeoutError",
        "TimeoutException",
    }
    return any(name in unavailable for name in names)


class ModelManager:
    def __init__(self) -> None:
        self._timeout = settings.OLLAMA_TIMEOUT_SECONDS
        self._ollama_client = ollama.Client(
            host=settings.OLLAMA_API_BASE,
            timeout=self._timeout,
        )
        self._ollama_async_client = ollama.AsyncClient(
            host=settings.OLLAMA_API_BASE,
            timeout=self._timeout,
        )

    @staticmethod
    def resolve_chat_model(model: str | None = None) -> str:
        selected = (model or "").strip()
        return selected or DEFAULT_CHAT_MODEL

    @staticmethod
    def resolve_embedding_model(model: str | None = None) -> str:
        selected = (model or "").strip()
        return selected or DEFAULT_EMBEDDING_MODEL

    @staticmethod
    def _wrap_provider_error(service: str, exc: Exception) -> ExternalServiceError:
        status_code = 503 if _is_service_unavailable(exc) else 502
        message = (
            f"{service} is unavailable"
            if status_code == 503
            else f"{service} request failed"
        )
        return ExternalServiceError(
            message,
            service=service,
            status_code=status_code,
            cause=exc,
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
        discovery_client = ollama.Client(
            host=settings.OLLAMA_API_BASE,
            timeout=min(self._timeout, 5.0),
        )
        try:
            response = discovery_client.list()
        except Exception as exc:
            raise self._wrap_provider_error("Ollama", exc) from exc
        return self._extract_model_names(response)
