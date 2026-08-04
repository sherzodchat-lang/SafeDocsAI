from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.datastructures import MutableHeaders
from app.core import security
from app.core.config import settings
from app.core.exceptions import (
    ApiError,
    ExternalServiceError,
    InternalErrors,
    RequestErrors,
)
from app.core.logging import setup_logging, get_logger

# Setup logging
setup_logging(level="DEBUG" if settings.ENVIRONMENT == "development" else "INFO")
logger = get_logger(__name__)
from app.api.endpoints import (
    auth,
    documents,
    sources,
    notebooks,
    notes,
    insights,
    chat,
    ask,
    logs,
    analytics,
    presentations,
    settings as runtime_settings,
)

@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Одна строка о том, чем система считает векторы и куда они легли:
    # `embedding_model=X -> коллекция Y, векторов N`. Имя коллекции ChromaDB
    # выводится из embedding-модели, поэтому её подмена (или её отсутствие) —
    # это другая коллекция, то есть пустой поиск при полной базе. Пишется
    # всегда, а не только в production: «как в README» поднимают именно
    # development-стенд, и именно там эта строка нужнее всего.
    from app.modules.rag.chroma_gateway import log_vector_store_state

    chroma_error = log_vector_store_state()

    # В production стартуем только с живой ChromaDB. Иначе загрузка документов
    # «успешно» проходит, документ помечается indexed, а векторов нет —
    # это тише и хуже, чем отказ на старте.
    #
    # Незаданная embedding-модель старт НЕ роняет (log_vector_store_state
    # вернёт None): выбирают её в админ-панели, и отказ на старте отобрал бы
    # единственное место, где эту неисправность можно починить. Отвечают
    # отказом сами RAG-операции — 503 settings.embedding_model_unset.
    if settings.ENVIRONMENT == "production" and chroma_error is not None:
        raise RuntimeError(
            f"ChromaDB is unreachable at {settings.CHROMA_HOST}:{settings.CHROMA_PORT}: "
            f"{chroma_error}. Refusing to start."
        )

    from app.modules.jobs.worker import IndexingWorker
    from app.modules.presentations.worker import PresentationWorker

    worker = IndexingWorker()
    # Задачи и документы, оставшиеся от убитого процесса, приводим в
    # осмысленное состояние до того, как воркер начнёт разбирать очередь.
    await worker.reconcile()
    worker.start()
    _app.state.indexing_worker = worker

    # Генерация презентаций — своя очередь и свой цикл, но тот же lifespan.
    # Отдельный воркер, а не задача в очереди индексации: генерация занимает
    # минуты и держит GPU, и в общей очереди она задерживала бы загрузку
    # документов ровно на своё время.
    presentation_worker = PresentationWorker()
    await presentation_worker.recover()
    presentation_worker.start()
    _app.state.presentation_worker = presentation_worker
    try:
        yield
    finally:
        # Останавливаемся штатно: воркер успевает вернуть текущую задачу
        # в очередь, а документ — из 'indexing' в 'pending'.
        #
        # Остановка презентаций идёт в своём try: отказ на одном воркере не
        # должен оставить второй работать в уже закрывающемся приложении.
        try:
            await presentation_worker.stop()
        finally:
            await worker.stop()


app = FastAPI(
    title=f"{settings.PROJECT_NAME} API",
    description="Backend for a grounded knowledge assistant over uploaded sources",
    version="0.1.0",
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    lifespan=lifespan,
)

# --- CSRF ---------------------------------------------------------------
#
# Пока токен ездил только в заголовке, CSRF был невозможен: заголовок
# ставит скрипт страницы, а чужой origin этого сделать не может. С кукой
# браузер прикладывает сессию к любому запросу, в том числе к отправленному
# со стороннего сайта, поэтому изменяющие запросы приходится подтверждать
# double-submit токеном: значение из не-httpOnly куки должно совпасть со
# значением в заголовке. Прочитать куку и выставить заголовок может только
# код с нашего origin.
#
# SameSite=lax сам по себе закрывает большую часть вектора, но опираться
# только на него нельзя: старые браузеры его игнорируют, а на верхнеуровневой
# навигации (форма с method=POST) lax куку всё же отдаёт.

_CSRF_PROTECTED_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})

# Проверяется по суффиксу: роутер auth подключён дважды — /api/v1/auth и
# совместимый /api/auth.
_CSRF_EXEMPT_SUFFIXES = (
    # Сессии ещё нет — подтверждать нечего.
    "/auth/login",
    "/auth/login/access-token",
    "/auth/register",
    # Здесь проверка перенесена внутрь эндпоинта: refresh-токен можно
    # предъявить и в теле запроса, а такой вызов CSRF не подвержен, и
    # снаружи по одним заголовкам это не отличить от вызова по куке.
    "/auth/refresh",
    "/auth/logout",
)


def _requires_csrf(request: Request) -> bool:
    if request.method not in _CSRF_PROTECTED_METHODS:
        return False
    # Запрос с Authorization CSRF не подвержен: этот заголовок браузер сам
    # не подставляет. Требовать с него токен значило бы сломать curl и
    # скрипты — и фронтенд, который на время выкатки ещё ходит с заголовком.
    if request.headers.get("authorization"):
        return False
    # Куки сессии нет — подделывать нечего.
    if not (
        request.cookies.get(security.ACCESS_COOKIE_NAME)
        or request.cookies.get(security.REFRESH_COOKIE_NAME)
    ):
        return False
    path = request.url.path.rstrip("/")
    return not path.endswith(_CSRF_EXEMPT_SUFFIXES)


class CSRFMiddleware:
    """Проверка double-submit токена.

    Написана как ASGI-слой, а не через @app.middleware: BaseHTTPMiddleware
    прогоняет ответ через собственный поток, а через этот слой проходит и
    SSE-стрим чата.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        request = Request(scope)
        if _requires_csrf(request) and not security.csrf_tokens_match(request):
            response = JSONResponse(
                status_code=status.HTTP_403_FORBIDDEN,
                content={
                    "detail": "CSRF token missing or invalid",
                    "error_code": security.CSRF_ERROR_CODE,
                },
            )
            return await response(scope, receive, send)

        return await self.app(scope, receive, send)


app.add_middleware(CSRFMiddleware)


# --- Идентификатор запроса ----------------------------------------------

REQUEST_ID_HEADER = "X-Request-ID"


class RequestIDMiddleware:
    """Присваивает каждому запросу идентификатор.

    Заведён ради обработчика непойманных исключений: наружу оттуда уходит
    только код ошибки, а трейсбек остаётся в логе, и связать жалобу
    пользователя с нужной строкой лога больше нечем.

    Как и CSRFMiddleware, написан ASGI-слоем, а не через @app.middleware:
    BaseHTTPMiddleware прогоняет ответ через собственный поток, а здесь
    проходит и SSE-стрим чата.

    Идентификатор всегда свой, входящий X-Request-ID не подхватывается:
    заголовок задаёт клиент, и в лог уходила бы произвольная строка от него.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        request_id = uuid4().hex
        # Кладём в scope, а не в contextvar: обработчик исключений вызывается
        # слоем снаружи этого (ServerErrorMiddleware), уже после раскрутки
        # стека, и scope — единственное, что доезжает до него неизменным.
        scope.setdefault("state", {})["request_id"] = request_id

        async def send_with_request_id(message):
            if message["type"] == "http.response.start":
                MutableHeaders(scope=message).append(REQUEST_ID_HEADER, request_id)
            await send(message)

        await self.app(scope, receive, send_with_request_id)


app.add_middleware(RequestIDMiddleware)


# CORS Middleware
# In production, set CORS_ORIGINS to specific domains
#
# Регистрируется последним намеренно: добавленный позже слой оказывается
# снаружи, и заголовки CORS навешиваются в том числе на отказ по CSRF.
# Иначе браузер в кросс-доменной разработке показал бы вместо 403 ошибку
# CORS и скрыл бы настоящую причину.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS_LIST,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    # Заголовок с CSRF-токеном обязан быть разрешён явно, если список
    # заголовков когда-нибудь перестанет быть '*'.
    allow_headers=["*", security.CSRF_HEADER_NAME],
    # Без expose_headers браузер не отдаёт X-Total-Count скрипту страницы,
    # и пагинация списка источников остаётся без общего числа записей.
    # X-Request-ID по той же причине: иначе клиент не сможет показать
    # пользователю идентификатор, по которому в логе ищется трейсбек.
    expose_headers=["X-Total-Count", REQUEST_ID_HEADER],
)

# Include Routers
app.include_router(auth.router, prefix=f"{settings.API_V1_STR}/auth", tags=["auth"])
# Compatibility alias for clients expecting /api/auth/*
app.include_router(auth.router, prefix="/api/auth", tags=["auth-compat"])
app.include_router(
    documents.router, prefix=f"{settings.API_V1_STR}/documents", tags=["documents"]
)
app.include_router(
    sources.router, prefix=f"{settings.API_V1_STR}/sources", tags=["sources"]
)
app.include_router(
    notebooks.router, prefix=f"{settings.API_V1_STR}/notebooks", tags=["notebooks"]
)
app.include_router(notes.router, prefix=f"{settings.API_V1_STR}/notes", tags=["notes"])
app.include_router(
    insights.router, prefix=f"{settings.API_V1_STR}/insights", tags=["insights"]
)
app.include_router(chat.router, prefix=f"{settings.API_V1_STR}/chat", tags=["chat"])
app.include_router(ask.router, prefix=f"{settings.API_V1_STR}/ask", tags=["ask"])
app.include_router(logs.router, prefix=f"{settings.API_V1_STR}/logs", tags=["logs"])
app.include_router(
    analytics.router, prefix=f"{settings.API_V1_STR}/analytics", tags=["analytics"]
)
app.include_router(
    runtime_settings.router, prefix=f"{settings.API_V1_STR}/settings", tags=["settings"]
)
# Единственный роутер раздела, подключённый БЕЗ собственного префикса: его пути
# живут в двух пространствах сразу — заказ и список привязаны к блокноту
# (/notebooks/{id}/presentations), а сама колода адресуется своим id
# (/presentations/{id}). Разрезать его на два роутера значило бы развести по
# файлам ответы одного и того же контракта.
app.include_router(
    presentations.router, prefix=settings.API_V1_STR, tags=["presentations"]
)


@app.exception_handler(ApiError)
async def api_error_handler(request: Request, exc: ApiError) -> JSONResponse:
    """Добавляет к detail машинный код: интерфейс переведён на три языка,
    а detail приходит на одном и годится только в лог и как фолбэк."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "error_code": exc.error_code},
        headers=exc.headers,
    )


@app.exception_handler(ExternalServiceError)
async def external_service_error_handler(
    request: Request, exc: ExternalServiceError
) -> JSONResponse:
    cause_text = f"; cause={exc.cause}" if exc.cause else ""
    logger.warning(
        "External service error on %s %s: %s%s",
        request.method,
        request.url.path,
        exc.message,
        cause_text,
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.message, "service": exc.service},
    )


@app.exception_handler(RequestValidationError)
async def request_validation_error_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """422 от Pydantic: прежний массив detail плюс машинный код.

    Последний класс ответов, который уходил без кода. Тело таким ответам
    собирал сам FastAPI (request_validation_exception_handler), и
    `GET /api/v1/settings/users?limit=501` доезжал до интерфейса с языком по
    умолчанию tg английским «Input should be less than or equal to 500».

    Массив detail сохраняется в прежнем виде, а не сворачивается в строку, и
    это не аккуратность, а условие правки. На нём держатся два разбора на
    клиенте (frontend/src/lib/apiError.js): имя непринятого поля берётся из
    loc, а 422 от схем с extra="forbid" опознаётся по type == extra_forbidden
    и получает собственное сообщение. Свернуть detail значило бы обменять
    «сервер не принял поле limit» на «проверьте введённые данные» — то есть
    сделать хуже, чем было без кода вовсе.

    Ни у ApiError, ни у HTTPException, ни у ExternalServiceError отобрать
    ответ этот обработчик не может: приложение хранит обработчики словарём с
    ключом-классом исключения, а выбирает их обходом __mro__ пойманного
    исключения (starlette/_exception_handler.py). RequestValidationError
    наследуется от ValidationException, а не от HTTPException, так что в его
    __mro__ чужих ключей нет, а в чужих __mro__ нет этого. Регистрация здесь
    вытесняет ровно одну запись — умолчание FastAPI по тому же ключу.

    Логировать нечего: причина отказа целиком в ответе, а не в трейсбеке, и
    ошибка ожидаемая — в отличие от обработчика Exception ниже.
    """
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        # jsonable_encoder — как в вытесняемом умолчании FastAPI: в ctx
        # ошибок Pydantic попадаются объекты, которые json.dumps не берёт
        # (само исключение у json_invalid, Decimal у числовых границ).
        content={
            "detail": jsonable_encoder(exc.errors()),
            "error_code": RequestErrors.VALIDATION_FAILED,
        },
    )


def _request_id(request: Request) -> str:
    """Идентификатор запроса, выданный RequestIDMiddleware.

    Фолбэк на случай, когда до обработчика дошёл запрос мимо слоя (например,
    приложение вызвали напрямую в тесте): ответ без идентификатора лишился бы
    единственной зацепки к строке лога.
    """
    return getattr(request.state, "request_id", None) or uuid4().hex


@app.exception_handler(Exception)
async def unhandled_exception_handler(
    request: Request, exc: Exception
) -> JSONResponse:
    """Последний рубеж: всё, что не разобрали обработчики выше.

    Без него непойманное исключение уходило клиенту голым 500 без тела —
    ни JSON, ни error_code, и показать пользователю было нечего. Так живьём
    выглядели KeyError в настройках и OSError на длинном имени файла.

    Подменить осмысленные ответы этот обработчик не может: Starlette отдаёт
    обработчик на Exception самому внешнему слою (ServerErrorMiddleware), а
    HTTPException, RequestValidationError, ApiError и ExternalServiceError
    разбираются слоем внутри (ExceptionMiddleware) и сюда просто не доходят.

    Наружу — только код и request_id. Текст исключения не отдаём никогда:
    в нём попадаются пути на сервере, куски SQL и строки подключения.
    Трейсбек целиком уходит в лог под тем же request_id.
    """
    request_id = _request_id(request)
    logger.error(
        "Unhandled exception on %s %s; request_id=%s",
        request.method,
        request.url.path,
        request_id,
        # exc_info=exc, а не logger.exception(): обработчик вызывается из
        # чужого except, и опираться на текущий sys.exc_info() незачем, когда
        # нужное исключение уже передано аргументом.
        exc_info=exc,
    )
    content = {
        "detail": "Internal server error",
        "error_code": InternalErrors.INTERNAL_ERROR,
        "request_id": request_id,
    }
    if settings.ENVIRONMENT != "production":
        # Вне production прячем меньше, иначе обработчик замаскирует свежий
        # баг под безликую пятисотку и найти его станет труднее, чем было до
        # него. Но только имя класса: ни текста, ни трейсбека — форма ответа
        # обязана совпадать с production, иначе клиент и тесты начнут
        # расходиться по режимам. Всё остальное разработчик берёт из лога.
        content["exception"] = type(exc).__name__
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content=content
    )


@app.get("/", tags=["root"])
async def root():
    return {"message": f"Welcome to {settings.PROJECT_NAME} API", "version": "0.1.0"}


@app.get("/health", tags=["health"])
async def health_check():
    """
    Health check endpoint for monitoring and load balancers.
    Returns 200 OK if the service is running.
    """
    return {
        "status": "healthy",
        "service": f"{settings.PROJECT_NAME.lower()}-api",
        "version": "0.1.0",
        "environment": settings.ENVIRONMENT,
    }


@app.get("/ready", tags=["health"])
async def readiness_check():
    """
    Readiness check endpoint for Kubernetes.
    Verifies database and external service connections.
    """
    from app.core.database import engine
    from app.modules.rag.service import RAGService

    checks = {
        "database": False,
        "chromadb": False,
    }

    # Check database connection
    try:
        from sqlalchemy import text

        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        checks["database"] = True
    except Exception as e:
        checks["database_error"] = str(e)
        logger.error(f"Database health check failed: {e}")

    # Check ChromaDB
    try:
        rag = RAGService()
        if rag.collection is not None:
            checks["chromadb"] = True
    except Exception as e:
        checks["chromadb_error"] = str(e)

    all_healthy = all(checks.values())

    return JSONResponse(
        status_code=status.HTTP_200_OK
        if all_healthy
        else status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"status": "ready" if all_healthy else "not_ready", "checks": checks},
    )
