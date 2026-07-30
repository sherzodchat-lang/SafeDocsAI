import json
from pathlib import Path
from typing import Any

from app.core.exceptions import ExternalServiceError
from app.shared.settings.config import settings


class RuntimeSettingsService:
    DEFAULTS: dict[str, Any] = {
        "chat_model": settings.OLLAMA_MODEL_CHAT,
        "embedding_model": settings.OLLAMA_MODEL_EMBEDDING,
        "retrieval_top_k": 20,
        "top_k": 5,
        "default_domain_profile": "tax",
        "enable_condense_query": True,
        "contextual_embedding_enabled": False,
        "contextual_embedding_model": "gemma3:4b",
        "chat_model_num_ctx": 20000,
        "contextual_embedding_num_ctx": 8192,
        "reranker_enabled": False,
        "reranker_model": "gemma4:e4b",
    }

    @classmethod
    def _settings_path(cls) -> Path:
        backend_dir = Path(__file__).resolve().parents[3]
        path = backend_dir / "data" / "runtime_settings.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    @classmethod
    def available_models(cls) -> list[str]:
        return cls.model_catalog()["available_models"]

    @classmethod
    def model_catalog(cls) -> dict[str, Any]:
        ollama_available = True
        ollama_error: str | None = None
        candidates: list[str] = []

        try:
            candidates.extend(
                __import__("app.modules.rag.model_manager", fromlist=["ModelManager"])
                .ModelManager()
                .list_ollama_models()
            )
        except ExternalServiceError as exc:
            ollama_available = False
            ollama_error = exc.message

        available_embedding_models = cls._unique_models(candidates)
        available_chat_models = list(available_embedding_models)
        return {
            "available_models": cls._unique_models(
                [*available_chat_models, *available_embedding_models]
            ),
            "available_chat_models": available_chat_models,
            "available_embedding_models": available_embedding_models,
            "ollama_available": ollama_available,
            "ollama_error": ollama_error,
        }

    @classmethod
    def get_settings(cls) -> dict[str, Any]:
        path = cls._settings_path()
        if not path.exists():
            return dict(cls.DEFAULTS)

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return dict(cls.DEFAULTS)

        merged = dict(cls.DEFAULTS)
        merged.update(data if isinstance(data, dict) else {})
        legacy_model = str(merged.get("model") or "").strip()
        merged["chat_model"] = str(merged.get("chat_model") or legacy_model).strip()
        merged["embedding_model"] = str(merged.get("embedding_model") or "").strip()
        merged["retrieval_top_k"] = cls._normalize_retrieval_top_k(
            merged.get("retrieval_top_k")
        )
        merged["top_k"] = cls._normalize_top_k(merged.get("top_k"))
        merged["enable_condense_query"] = cls._normalize_bool(
            merged.get("enable_condense_query"), default=True
        )
        merged["contextual_embedding_enabled"] = cls._normalize_bool(
            merged.get("contextual_embedding_enabled"), default=False
        )
        merged["contextual_embedding_model"] = str(
            merged.get("contextual_embedding_model") or cls.DEFAULTS["contextual_embedding_model"]
        ).strip()
        merged["chat_model_num_ctx"] = cls._normalize_num_ctx(merged.get("chat_model_num_ctx"), 20000)
        merged["contextual_embedding_num_ctx"] = cls._normalize_num_ctx(merged.get("contextual_embedding_num_ctx"), 8192)
        merged["reranker_enabled"] = cls._normalize_bool(merged.get("reranker_enabled"), default=False)
        merged["reranker_model"] = str(merged.get("reranker_model") or cls.DEFAULTS["reranker_model"]).strip()
        if not merged["chat_model"]:
            merged["chat_model"] = cls.DEFAULTS["chat_model"]
        if not merged["embedding_model"]:
            merged["embedding_model"] = cls.DEFAULTS["embedding_model"]
        merged["model"] = merged["chat_model"]
        merged["default_domain_profile"] = cls._normalize_domain_profile(
            merged.get("default_domain_profile")
        )
        return merged

    @classmethod
    def update_settings(cls, patch: dict[str, Any]) -> dict[str, Any]:
        current = cls.get_settings()

        if "chat_model" in patch or "model" in patch:
            selected_model = str(
                patch.get("chat_model") or patch.get("model") or ""
            ).strip()
            if not selected_model:
                raise ValueError("Chat model must not be empty")
            if selected_model not in cls.model_catalog()["available_chat_models"]:
                raise ValueError(f"Unsupported chat model: {selected_model}")
            current["chat_model"] = selected_model
            current["model"] = selected_model
        if "embedding_model" in patch:
            embedding_model = str(patch["embedding_model"] or "").strip()
            if not embedding_model:
                raise ValueError("Embedding model must not be empty")
            if embedding_model not in cls.model_catalog()["available_embedding_models"]:
                raise ValueError(f"Unsupported embedding model: {embedding_model}")
            old_embedding = current.get("embedding_model", "")
            current["embedding_model"] = embedding_model
            if old_embedding and old_embedding != embedding_model:
                current["reindex_required"] = True
                import logging

                logging.getLogger(__name__).warning(
                    "Embedding model changed from %s to %s — reindex required. "
                    "Run reindex_documents.py to rebuild the vector store.",
                    old_embedding,
                    embedding_model,
                )
        if "retrieval_top_k" in patch:
            current["retrieval_top_k"] = cls._normalize_retrieval_top_k(
                patch["retrieval_top_k"]
            )
        if "top_k" in patch:
            current["top_k"] = cls._normalize_top_k(patch["top_k"])
        if "default_domain_profile" in patch:
            current["default_domain_profile"] = cls._normalize_domain_profile(
                patch["default_domain_profile"]
            )
        if "enable_condense_query" in patch:
            current["enable_condense_query"] = cls._normalize_bool(
                patch["enable_condense_query"], default=True
            )
        if "contextual_embedding_enabled" in patch:
            current["contextual_embedding_enabled"] = cls._normalize_bool(
                patch["contextual_embedding_enabled"], default=False
            )
        if "contextual_embedding_model" in patch:
            model = str(patch["contextual_embedding_model"] or "").strip()
            if model and model not in cls.model_catalog()["available_chat_models"]:
                raise ValueError(f"Unsupported contextual embedding model: {model}")
            current["contextual_embedding_model"] = model or cls.DEFAULTS["contextual_embedding_model"]
        if "chat_model_num_ctx" in patch:
            current["chat_model_num_ctx"] = cls._normalize_num_ctx(patch["chat_model_num_ctx"], 20000)
        if "contextual_embedding_num_ctx" in patch:
            current["contextual_embedding_num_ctx"] = cls._normalize_num_ctx(patch["contextual_embedding_num_ctx"], 8192)
        if "reranker_enabled" in patch:
            current["reranker_enabled"] = cls._normalize_bool(patch["reranker_enabled"], default=False)
        if "reranker_model" in patch:
            model = str(patch["reranker_model"] or "").strip()
            current["reranker_model"] = model or cls.DEFAULTS["reranker_model"]

        path = cls._settings_path()
        path.write_text(
            json.dumps(current, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return current

    @staticmethod
    def _normalize_top_k(value: Any) -> int:
        try:
            number = int(value)
        except (TypeError, ValueError):
            return 5
        return max(1, min(number, 20))

    @staticmethod
    def _normalize_retrieval_top_k(value: Any) -> int:
        try:
            number = int(value)
        except (TypeError, ValueError):
            return 20
        return max(1, min(number, 50))

    @staticmethod
    def _normalize_num_ctx(value: Any, default: int) -> int:
        try:
            number = int(value)
        except (TypeError, ValueError):
            return default
        return max(2048, min(number, 262144))

    @staticmethod
    def _normalize_domain_profile(value: Any) -> str:
        profile = str(value or "").strip().lower()
        from app.domain_profiles import list_domain_profiles as _list_profiles

        available = set(_list_profiles())
        return profile if profile in available else "tax"

    @staticmethod
    def _unique_models(candidates: list[str]) -> list[str]:
        seen: set[str] = set()
        unique_models: list[str] = []
        for model in candidates:
            normalized = str(model or "").strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            unique_models.append(normalized)
        return unique_models

    @staticmethod
    def _normalize_bool(value: Any, default: bool) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"1", "true", "yes", "on"}:
                return True
            if normalized in {"0", "false", "no", "off"}:
                return False
        if value is None:
            return default
        return bool(value)
