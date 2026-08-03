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


class SettingsErrors:
    """Машинные коды раздела настроек (админ-панель).

    Раздел выпал из общей схемы целиком: все отказы уходили голым
    HTTPException без кода, и трёхъязычный интерфейс показывал английский
    `detail` как есть. Коды стабильны — менять их нельзя без синхронной правки
    словарей фронта.
    """

    # --- Смена роли (HTTP) ---
    # Строки дословно те, что временно жили в app/api/endpoints/settings.py.
    USER_NOT_FOUND = "settings.user_not_found"
    # 400: админ и правда последний. Повторять запрос бессмысленно — сначала
    # нужно кого-то назначить.
    LAST_ADMIN = "settings.last_admin"
    # 409: админов было больше, но пока запрос ждал блокировки, их снял
    # кто-то ещё. Список у клиента устарел.
    ROLE_CHANGE_CONFLICT = "settings.role_change_conflict"

    # --- Выбор модели (HTTP) ---
    # Три разных исхода, а не один «модель не подходит»: админу нужно разное
    # действие, и слить их в один код значит показать ему подсказку наугад.
    #
    # 400: поле модели пришло пустым. Лечится вводом значения.
    MODEL_REQUIRED = "settings.model_required"
    # 400: такой модели в Ollama нет. Лечится `ollama pull <модель>` —
    # выбирать из списка нечего, нужного там просто не будет.
    MODEL_NOT_INSTALLED = "settings.model_not_installed"
    # 400: модель установлена, но не того вида — embedding-модель в поле чата
    # или чат-модель в поле эмбеддингов. Лечится выбором другой модели из
    # списка available_chat_models / available_embedding_models; `ollama pull`
    # здесь не поможет, модель уже стоит.
    MODEL_WRONG_KIND = "settings.model_wrong_kind"
    # 503: список установленных моделей собрать не удалось — Ollama лежит, в
    # OLLAMA_API_BASE опечатка, клиент не импортируется. Сверять выбранную
    # модель не с чем, и отвечать MODEL_NOT_INSTALLED нельзя: модель может
    # стоять на месте и уже быть настроенной, а админ пошёл бы её «доставлять».
    # Тело запроса менять не нужно — повторить его как есть, когда каталог
    # снова соберётся.
    MODEL_CATALOG_UNAVAILABLE = "settings.model_catalog_unavailable"

    # --- Контекстное обогащение (HTTP) ---
    # 400: обогащение включают (или уже включено и правят модель), а модель для
    # него не выбрана. Пустое значение здесь не «умолчание», а тихое отключение
    # функции: индексация читает `if _ctx_enabled and _ctx_model`
    # (app/modules/documents/service.py) и без модели просто ничего не делает —
    # то есть переключатель стоял бы во «включено», не делая ничего.
    CONTEXTUAL_MODEL_REQUIRED = "settings.contextual_model_required"

    # --- Значения полей при ЗАПИСИ (HTTP) ---
    # Общее правило раздела: на чтении настройки чинятся (иначе испорченный
    # файл гасит и тот экран, на котором его чинят), при записи — отвергаются.
    # Тихая подмена при записи означала бы 200 OK на значение, которое админ не
    # выбирал: он видит успех, а работает система по-другому.
    #
    # 400: профиль домена не зарегистрирован. Свой код, а не
    # source.unsupported_domain_profile: тот про профиль блокнота, а перевод и
    # подсказка («выберите из available_domain_profiles») здесь свои.
    UNSUPPORTED_DOMAIN_PROFILE = "settings.unsupported_domain_profile"
    # 400: в числовом поле не целое число.
    INVALID_NUMBER = "settings.invalid_number"
    # 400: целое число вне допустимого диапазона поля. Границы названы в
    # `detail`; клиент показывает свой перевод и подставляет их из подсказки
    # поля.
    VALUE_OUT_OF_RANGE = "settings.value_out_of_range"
    # 400: в логическом поле значение, которое не является ни истиной, ни
    # ложью. `bool("banana")` — это True, и переключатель включался бы от
    # любого мусора.
    INVALID_BOOLEAN = "settings.invalid_boolean"

    # --- Смена embedding-модели (HTTP) ---
    # 409: запрос меняет embedding_model, но подтверждения в теле нет. Имя
    # коллекции ChromaDB выводится из embedding-модели, поэтому смена мгновенно
    # уводит поиск в другую (пустую) коллекцию — со стороны выглядит как
    # «все документы пропали». Тело валидно, менять в нём нечего: клиент
    # повторяет тот же запрос с confirm_reindex=true, объяснив последствия
    # пользователю. Тем же кодом отвечает сброс настроек, если он возвращает
    # embedding_model к умолчанию.
    REINDEX_CONFIRMATION_REQUIRED = "settings.reindex_confirmation_required"

    # 400: отказ уровня настроек без своего кода. Фолбэк для ValueError,
    # прилетевшего из RuntimeSettingsService мимо SettingsError.
    INVALID_VALUE = "settings.invalid_value"

    # --- Embedding-модель не задана (HTTP 503) ---
    # 503: ни в runtime_settings.json, ни в переменной окружения
    # OLLAMA_MODEL_EMBEDDING нет embedding-модели, а имя коллекции ChromaDB
    # выводится ровно из неё. Умолчания в коде нет намеренно: подставленное
    # молча имя уводило поиск в коллекцию, которую никто не заполнял, —
    # ноль найденных фрагментов при полной базе и ни одной ошибки в журнале.
    #
    # Отказ мягкий: приложение стартует, раздел настроек открывается и
    # сохраняется (иначе модель негде выбрать), а всё, чему нужна векторная
    # база — поиск, чат, индексация, удаление векторов, — отвечает этим кодом.
    # 503, а не 400: тело запроса верное, повторять его надо как есть, когда
    # админ выберет модель. Раздел settings., а не source.: чинится это в
    # админ-панели, и подсказка пользователю здесь та же, что у остальных
    # ненастроенных моделей.
    EMBEDDING_MODEL_UNSET = "settings.embedding_model_unset"


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


class RequestErrors:
    """Машинные коды, описывающие сам запрос, а не раздел API.

    Отдельный верхнеуровневый префикс по тому же доводу, что у internal.:
    отказ на разборе запроса прилетает из любого эндпоинта — из настроек, из
    чата, из загрузки источника, — и раздел по нему не восстановить. Положить
    такой код в словарь одного из разделов значило бы, что клиент ищет перевод
    не там.

    Префикс request., а не validation.: на фронте уже живёт
    request.unexpected_field (frontend/src/lib/apiError.js) — синтетический код
    для 422 от схем с extra="forbid". Это тот же класс ответов «запрос не
    разобрался», и второе верхнеуровневое пространство имён развело бы два
    соседних кода по разным словарям, хотя приходят они из одного и того же
    тела 422 и разбираются одной веткой клиента.
    """

    # 422: Pydantic не принял тело, параметры пути или строку запроса. Живой
    # пример: GET /api/v1/settings/users?limit=501 — до этого кода такой ответ
    # доезжал до трёхъязычного интерфейса английским «Input should be less
    # than or equal to 500», потому что тело собирал сам FastAPI.
    #
    # Массив detail обработчик (app/main.py) сохраняет в прежнем виде: код
    # говорит лишь «перед тобой отказ валидации», а КАКОЕ поле не принято,
    # клиент по-прежнему читает из detail (loc → имя поля). Там же он опознаёт
    # extra_forbidden, у которого на фронте свой код и своё сообщение.
    VALIDATION_FAILED = "request.validation_failed"


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


class EmbeddingModelNotConfigured(ApiError):
    """Операция требует векторной базы, а embedding-модель не задана.

    ApiError, а не SettingsError: бросает это ЛЮБОЙ путь к ChromaDB (чат,
    поиск, загрузка, удаление, переиндексация), а не раздел настроек, — и
    таблица кодов из app/api/endpoints/settings.py до них не дотягивается.
    Обработчик ApiError в app/main.py отдаёт наружу
    {"detail": ..., "error_code": ...} с нужным статусом на всех путях сразу,
    включая те, которые заведут завтра.

    За HTTP этот же объект работает как обычное исключение: воркер индексации
    (app/modules/jobs/worker.py) до него не доходит вовсе — он не берёт задачи,
    пока модель не выбрана, — а скрипты вне API получают внятный текст.
    """

    MESSAGE = (
        "Embedding model is not configured: set it in the admin panel "
        "(PUT /api/v1/settings/, field embedding_model) or in the "
        "OLLAMA_MODEL_EMBEDDING environment variable. Vector search, indexing "
        "and deletion are unavailable until then."
    )

    def __init__(self, detail: str | None = None) -> None:
        super().__init__(
            503, SettingsErrors.EMBEDDING_MODEL_UNSET, detail or self.MESSAGE
        )


class SettingsError(ValueError):
    """Отказ уровня настроек: машинный код без HTTP-семантики.

    Не ApiError намеренно. RuntimeSettingsService — слой настроек, а не HTTP:
    его зовут и мимо запроса (backend/reindex_documents.py, воркер индексации),
    где status_code некому прочитать, а поднять HTTPException значило бы
    протащить FastAPI в модуль, которому он не нужен. Код же — не HTTP: это
    ключ словаря переводов, общий у сервера и клиента.

    Статус выбирает HTTP-слой (app/api/endpoints/settings.py), сопоставляя код
    с таблицей: одна ошибка настроек — 409, остальные — 400.

    Наследник ValueError, а не Exception: `update_settings` испокон веку
    бросал ValueError, и вызывающие (в том числе скрипты вне API) ловят
    именно его. Код добавлен рядом, договор не сломан.
    """

    def __init__(self, error_code: str, message: str) -> None:
        super().__init__(message)
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
