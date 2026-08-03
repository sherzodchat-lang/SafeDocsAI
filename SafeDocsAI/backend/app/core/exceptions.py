from fastapi import HTTPException


class SourceErrors:
    """Машинные коды ошибок раздела источников.

    Клиент показывает пользователю свой перевод по коду, а не `detail`:
    интерфейс переведён на три языка, а `detail` приходит на одном.
    Коды стабильны — менять их нельзя без синхронной правки словарей фронта.
    """

    # Приём файла (HTTP)
    FILENAME_REQUIRED = "source.filename_required"
    # 400: имя не помещается в каталог загрузок даже после обрезки до предела
    # файловой системы. Не 413: 413 у нас означает «файл великоват, разбей его»,
    # а здесь размер ни при чём и лечится переименованием.
    FILENAME_TOO_LONG = "source.filename_too_long"
    UNSUPPORTED_TYPE = "source.unsupported_type"
    INVALID_CONTENT_TYPE = "source.invalid_content_type"
    TOO_LARGE = "source.too_large"
    INVALID_UPLOAD = "source.invalid_upload"

    # Доступ к сущностям (HTTP)
    NOT_FOUND = "source.not_found"
    NOTEBOOK_NOT_FOUND = "source.notebook_not_found"
    CHUNK_NOT_FOUND = "source.chunk_not_found"
    NOTE_NOT_FOUND = "source.note_not_found"
    INSIGHT_NOT_FOUND = "source.insight_not_found"
    FILE_MISSING = "source.file_missing"
    NO_IDS_PROVIDED = "source.no_ids_provided"

    # Блокноты (HTTP)
    UNSUPPORTED_DOMAIN_PROFILE = "source.unsupported_domain_profile"

    # Частичное обновление (HTTP)
    # 400: тело PATCH не содержит ни одного известного поля. Молча отвечать 200
    # нельзя: клиент считал бы, что правка применена.
    NOTHING_TO_UPDATE = "source.nothing_to_update"
    # 400: недопустимое значение Note.status (допустимы active/archived).
    INVALID_NOTE_STATUS = "source.invalid_note_status"

    # Удаление (HTTP)
    VECTOR_STORE_UNAVAILABLE = "source.vector_store_unavailable"
    DELETE_FILE_FAILED = "source.delete_file_failed"
    # 409: тот же блокнот удаляет параллельный запрос. Клиенту достаточно
    # обновить список — работу уже сделали за него.
    NOTEBOOK_DELETE_CONFLICT = "source.notebook_delete_conflict"
    # 409: в блокноте идёт индексация. Прервать её нечем — воркер живёт в
    # другом процессе, поэтому просим повторить удаление позже.
    NOTEBOOK_BUSY_INDEXING = "source.notebook_busy_indexing"

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


class ChatErrors:
    """Машинные коды раздела вопросов к ассистенту.

    Один префикс на все три точки входа (POST /chat/, /chat/stream, /ask/) и
    на разбор запроса (/chat/retrieve): поле там одно и то же — question, — и
    причина отказа тоже одна, поэтому и перевод на фронте должен быть один.
    Отдельного `ask.` не заводим: разными кодами пришлось бы объяснять
    пользователю одно и то же двумя строками.
    """

    # 422: вопрос пуст или состоит из одних пробелов. Отказ стоит на валидации
    # схемы, то есть до поиска и до генерации: пустой вопрос иначе занимал бы
    # GPU ровно как настоящий.
    QUESTION_REQUIRED = "chat.question_required"
    # 422: вопрос длиннее QUESTION_MAX_LENGTH (app/api/deps.py). Предел стоит
    # не ради БД, а ради контекста модели: вставленный в поле вопроса документ
    # вытесняет из промпта найденные фрагменты и историю диалога.
    QUESTION_TOO_LONG = "chat.question_too_long"


class LogErrors:
    """Машинные коды раздела журнала запросов.

    Отдельный префикс, а не source.*: журнал живёт своей сущностью, и
    словарь переводов на фронте разбит по тем же разделам.
    """

    NOT_FOUND = "log.not_found"


class InternalErrors:
    """Машинные коды, не привязанные ни к одному разделу.

    Отдельный префикс, а не source./auth./log.: непойманное исключение
    прилетает из любого эндпоинта, и раздел по нему не восстановить. Положить
    такой код в словарь одного из разделов значило бы, что клиент ищет
    перевод не там.
    """

    # 500: до последнего обработчика дошло исключение, которого никто не ждал.
    # Наружу уходит только этот код и request_id — по нему в логе ищется
    # трейсбек. Один код на все случаи намеренно: разбирать причину должен
    # разработчик по логу, а не пользователь по тексту на экране.
    INTERNAL_ERROR = "internal.error"


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
