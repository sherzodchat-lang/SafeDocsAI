import secrets
from typing import Optional

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# Значения, которые исторически попадали в деплой как «временные».
# Ни одно из них не должно пережить старт в production.
_WEAK_SECRET_KEYS = {
    "change-me-in-production",
    "changeme",
    "secret",
    "aigov-secret-key-2026",
}


class Settings(BaseSettings):
    PROJECT_NAME: str = "SafeDocsAI"
    API_V1_STR: str = "/api/v1"
    ENVIRONMENT: str = "development"

    POSTGRES_USER: str = "andozai_user"
    POSTGRES_PASSWORD: str = "andozai_password"
    POSTGRES_SERVER: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "andozai_db"

    @property
    def SQLALCHEMY_DATABASE_URI(self) -> str:
        return f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_SERVER}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"

    SECRET_KEY: str = ""
    ALGORITHM: str = "HS256"
    # Access живёт минуты, а не дни: украденный токен нельзя отозвать иначе
    # как через token_version, поэтому длину его жизни держим короткой.
    # Долгую сессию обеспечивает refresh-токен с ротацией.
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # --- Куки сессии ----------------------------------------------------
    #
    # Токены отдаются и в теле ответа, и в httpOnly-куках. Кука — основное
    # хранилище: до localStorage дотягивается любой XSS, до httpOnly-куки —
    # нет. Тело оставлено ради клиентов, которые ещё ходят с заголовком.

    # Secure нельзя включать безусловно: в разработке фронт ходит на
    # http://localhost:8001, и браузер такую куку просто не сохранит.
    # None означает «решить по ENVIRONMENT», явное значение — перекрыть.
    COOKIE_SECURE: Optional[bool] = None

    @property
    def COOKIE_SECURE_FLAG(self) -> bool:
        if self.COOKIE_SECURE is not None:
            return self.COOKIE_SECURE
        return self.ENVIRONMENT == "production"

    # lax, а не strict: со strict пользователь, пришедший по внешней ссылке,
    # попадает на страницу без куки и выглядит разлогиненным. Порт в понятие
    # site не входит, поэтому lax работает и в связке 5173 -> 8001.
    COOKIE_SAMESITE: str = "lax"

    # Пусто — кука без атрибута Domain, то есть только на выдавший её хост.
    # Заполнять только если API и фронт разведены по поддоменам.
    COOKIE_DOMAIN: str = ""

    @property
    def COOKIE_DOMAIN_OR_NONE(self) -> Optional[str]:
        return self.COOKIE_DOMAIN or None

    # Окно, в котором повторное предъявление уже использованного refresh не
    # считается кражей. Нужно из-за гонки: две вкладки читают один токен из
    # localStorage и обновляют его одновременно. Секунды, а не минуты —
    # за пределами окна повтор по-прежнему гасит все сессии пользователя.
    REFRESH_TOKEN_REUSE_LEEWAY_SECONDS: int = 10

    # Кому можно верить в X-Forwarded-For / X-Real-IP. Заголовок задаёт
    # клиент, поэтому вне этого списка он игнорируется — иначе лимит на
    # попытки входа обходится сменой заголовка на каждом запросе.
    # По умолчанию только loopback: на стенде nginx стоит на том же хосте и
    # ходит на 127.0.0.1:8001. Для docker-compose (фронтовый nginx —
    # отдельный контейнер) добавить его подсеть, например 172.16.0.0/12.
    TRUSTED_PROXIES: str = "127.0.0.1,::1"

    @property
    def TRUSTED_PROXIES_LIST(self) -> list[str]:
        return [item.strip() for item in self.TRUSTED_PROXIES.split(",") if item.strip()]

    # Саморегистрация закрыта по умолчанию: пользователей заводит админ.
    # Включать осознанно и только там, где это действительно нужно.
    ALLOW_REGISTRATION: bool = False

    CORS_ORIGINS: str = "*"

    @property
    def CORS_ORIGINS_LIST(self) -> list[str]:
        if self.CORS_ORIGINS == "*":
            return ["*"]
        return [
            origin.strip() for origin in self.CORS_ORIGINS.split(",") if origin.strip()
        ]

    # Потолок размера загружаемого источника. 50 МБ — то, что обещает UI.
    # Проверка обязана быть здесь, а не только в nginx: client_max_body_size
    # различается между стендами (50M и 25M), а при прямом обращении к
    # uvicorn прокси в цепочке вообще нет.
    MAX_UPLOAD_SIZE_MB: int = 50

    @property
    def MAX_UPLOAD_SIZE_BYTES(self) -> int:
        return self.MAX_UPLOAD_SIZE_MB * 1024 * 1024

    CHROMA_HOST: str = "localhost"
    CHROMA_PORT: int = 8000
    CHROMA_PERSIST_DIR: str = "data/chroma"

    OLLAMA_API_BASE: str = "http://localhost:11434"
    OLLAMA_TIMEOUT_SECONDS: float = 120.0
    OLLAMA_MODEL_CHAT: str = "gemma3n:e4b"
    # Умолчания нет намеренно, и пустая строка здесь — не забывчивость.
    #
    # Имя коллекции ChromaDB выводится из embedding-модели
    # (ChromaGateway._collection_name), а get_or_create_collection на
    # незнакомое имя не отказывает, а СОЗДАЁТ пустую коллекцию. Здесь стояло
    # "nomic-embed-text", и любой процесс, стартовавший без
    # OLLAMA_MODEL_EMBEDDING (а стенд работает на qwen3-embedding:8b, который
    # выставляет скрипт запуска), заводил себе рядом пустую
    # andozai_docs_nomic_embed_text и отвечал на поиск пустотой при полной
    # базе — без единой ошибки в логах. Такое видели живьём.
    #
    # Embedding-модель — не порт и не таймаут: неверное тихое умолчание здесь
    # не деградация, а обнуление продукта. Поэтому порядок разрешения теперь
    # честный: runtime_settings.json -> эта переменная -> отказ
    # (SettingsErrors.EMBEDDING_MODEL_UNSET, 503). Приложение при этом
    # стартует и раздел настроек открывается — иначе модель негде выбрать.
    OLLAMA_MODEL_EMBEDDING: str = ""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @model_validator(mode="after")
    def _validate_deployment_secrets(self) -> "Settings":
        is_production = self.ENVIRONMENT == "production"

        samesite = self.COOKIE_SAMESITE.strip().lower()
        if samesite not in ("lax", "strict", "none"):
            raise ValueError(
                "COOKIE_SAMESITE must be one of: lax, strict, none "
                f"(got {self.COOKIE_SAMESITE!r})."
            )
        self.COOKIE_SAMESITE = samesite
        if samesite == "none" and not self.COOKIE_SECURE_FLAG:
            # SameSite=None без Secure браузеры отбрасывают молча: сессия
            # просто перестанет держаться, и причина будет неочевидна.
            raise ValueError(
                "COOKIE_SAMESITE='none' requires COOKIE_SECURE=true "
                "(browsers drop such cookies over plain http)."
            )

        if is_production:
            if not self.SECRET_KEY:
                raise ValueError(
                    "SECRET_KEY is required when ENVIRONMENT=production. "
                    "Generate one with: python -c \"import secrets; print(secrets.token_hex(32))\""
                )
            if self.SECRET_KEY in _WEAK_SECRET_KEYS or len(self.SECRET_KEY) < 32:
                raise ValueError(
                    "SECRET_KEY is a known placeholder or shorter than 32 characters. "
                    "Anyone who knows it can mint an admin token."
                )
            if self.CORS_ORIGINS == "*":
                raise ValueError(
                    "CORS_ORIGINS='*' is not allowed when ENVIRONMENT=production. "
                    "List the real origins, comma-separated."
                )
        elif not self.SECRET_KEY:
            # Только для локальной разработки. В многопроцессном режиме такой
            # ключ у каждого воркера свой, поэтому в production он запрещён выше.
            self.SECRET_KEY = secrets.token_hex(32)

        return self


settings = Settings()
