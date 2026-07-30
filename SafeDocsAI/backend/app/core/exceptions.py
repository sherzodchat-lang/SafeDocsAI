from fastapi import HTTPException


class SourceErrors:
    """Машинные коды ошибок раздела источников.

    Клиент показывает пользователю свой перевод по коду, а не `detail`:
    интерфейс переведён на три языка, а `detail` приходит на одном.
    Коды стабильны — менять их нельзя без синхронной правки словарей фронта.
    """

    # Приём файла (HTTP)
    FILENAME_REQUIRED = "source.filename_required"
    UNSUPPORTED_TYPE = "source.unsupported_type"
    INVALID_CONTENT_TYPE = "source.invalid_content_type"
    TOO_LARGE = "source.too_large"
    INVALID_UPLOAD = "source.invalid_upload"

    # Доступ к сущностям (HTTP)
    NOT_FOUND = "source.not_found"
    NOTEBOOK_NOT_FOUND = "source.notebook_not_found"
    CHUNK_NOT_FOUND = "source.chunk_not_found"
    FILE_MISSING = "source.file_missing"
    NO_IDS_PROVIDED = "source.no_ids_provided"

    # Удаление (HTTP)
    VECTOR_STORE_UNAVAILABLE = "source.vector_store_unavailable"
    DELETE_FILE_FAILED = "source.delete_file_failed"

    # Индексация: попадают не в HTTP-ответ, а в document.error_code
    INDEXING_FAILED = "source.indexing_failed"
    INDEXING_INTERRUPTED = "source.indexing_interrupted"
    TEXT_EXTRACTION_FAILED = "source.text_extraction_failed"
    ENCODING_NOT_UTF8 = "source.encoding_not_utf8"
    DELETED_BEFORE_INDEXING = "source.deleted_before_indexing"


class AuthErrors:
    """Машинные коды раздела аутентификации.

    Тексты `detail` намеренно неинформативны: по разнице ответов не должно
    быть видно, существует ли пользователь и чем именно плох токен.
    """

    INVALID_CREDENTIALS = "auth.invalid_credentials"
    INVALID_TOKEN = "auth.invalid_token"
    TOKEN_REVOKED = "auth.token_revoked"
    FORBIDDEN = "auth.forbidden"
    REGISTRATION_DISABLED = "auth.registration_disabled"
    REGISTRATION_REJECTED = "auth.registration_rejected"
    WEAK_PASSWORD = "auth.weak_password"


class ApiError(HTTPException):
    """HTTPException с машинным кодом рядом с `detail`.

    Тело ответа собирает обработчик в app/main.py:
    {"detail": "<текст для логов и фолбэка>", "error_code": "<код для i18n>"}.
    """

    def __init__(
        self,
        status_code: int,
        error_code: str,
        detail: str,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(status_code=status_code, detail=detail, headers=headers)
        self.error_code = error_code


class ExternalServiceError(Exception):
    def __init__(
        self,
        message: str,
        *,
        service: str,
        status_code: int = 502,
        cause: Exception | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.service = service
        self.status_code = status_code
        self.cause = cause
