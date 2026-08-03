from typing import Annotated, Any, Optional

from fastapi import Depends, Path, Request, status
from fastapi.security import OAuth2PasswordBearer
from pydantic import AfterValidator, BeforeValidator, StringConstraints
from sqlmodel.ext.asyncio.session import AsyncSession
from app.shared.settings import settings
from app.core import security
from app.core.database import get_session, session_context
from app.core.exceptions import ApiError, AuthErrors, ChatErrors, SourceErrors
from app.shared.models import User
from sqlmodel import select

# auto_error=False: без заголовка схема обязана промолчать, чтобы можно было
# заглянуть в куку. Сама схема остаётся объявленной — на ней держится кнопка
# Authorize в /docs и разметка securitySchemes в OpenAPI.
oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl=f"{settings.API_V1_STR}/auth/login/access-token",
    auto_error=False,
)


# --- Границы входных значений -------------------------------------------
#
# Одно место на весь API: одинаковые ограничения должны стоять и на путевых
# параметрах, и на query, и на полях тела запроса — иначе дыра остаётся там,
# где её забыли закрыть.

# Первичные ключи — PostgreSQL integer, то есть 32 бита. Значение больше
# диапазона asyncpg не может передать параметром запроса: вместо честного 404
# запрос падает OverflowError и превращается в 500. Поэтому диапазон режем
# на валидации: id за его пределами не может существовать по определению.
MAX_ID = 2147483647

# Имя блокнота и заголовки заметки/инсайта лежат в колонках под btree-индексом
# (см. shared/models/entities.py). Значение длиннее ~2700 байт Postgres не
# может проиндексировать и отвечает ошибкой размера индексной записи — снова
# 500 вместо 422. 255 символов с запасом покрывают осмысленный заголовок и в
# UTF-8 остаются втрое ниже предела индекса.
TITLE_MAX_LENGTH = 255
# Описание и тела заметок/инсайтов не индексируются, ограничение здесь только
# против неограниченного роста строки в БД и в ответе списка.
DESCRIPTION_MAX_LENGTH = 2000
BODY_MAX_LENGTH = 100_000
KIND_MAX_LENGTH = 50

# Вопрос к ассистенту (POST /chat/, /chat/stream, /chat/retrieve, /ask/).
#
# Здесь ограничение стоит не ради БД, а ради контекста модели: вопрос уходит
# в промпт целиком и тянет за собой найденные фрагменты и историю диалога.
# Бюджет считается так (числа — из кода, не на глаз):
#
#   * окно чат-модели: chat_model_num_ctx = 20000 токенов по умолчанию
#     (app/shared/settings/runtime_settings.py), 12288 у ModelManager без
#     явного значения (app/modules/rag/model_manager.py). Считаем по
#     меньшему — 12000 токенов;
#   * кириллица у нас оценивается как 3.6 символа на токен
#     (HybridChunker.CHARS_PER_TOKEN), то есть окно ≈ 43 000 символов;
#   * из них уходит на найденные фрагменты: top_k = 5 чанков по max_tokens =
#     800 (HybridChunker) — до 4000 токенов, ≈ 14 400 символов;
#   * на историю: последние 5 записей журнала парами «вопрос + ответ»
#     (app/modules/chat/service.py) — ещё несколько тысяч токенов;
#   * плюс системная часть промпта, правила профиля и место под сам ответ.
#
# На вопрос остаётся порядка 700 токенов. 2000 символов — это ≈ 555 токенов
# по той же оценке, то есть влезает с запасом и остаётся примерно 4-5% окна.
# Для живого вопроса этого много (2000 символов — около 300 слов), а
# вставленный целиком документ отсекается. Отдельный довод за предел: вопрос
# сохраняется в log.question и потом ещё пять запросов подряд едет в историю
# диалога — без границы одна вставка дорожает шестикратно.
QUESTION_MAX_LENGTH = 2000


def _require_non_blank(value: str) -> str:
    """Подрезать пробелы по краям; строка из одних пробелов — не значение.

    Проверяем именно подрезанное значение: strip() в эндпоинте без такой
    проверки молча превращал "   " в пустое имя.
    """
    trimmed = value.strip()
    if not trimmed:
        raise ValueError("must not be empty or whitespace only")
    return trimmed


# Имя блокнота, заголовок заметки и инсайта: непустые после подрезки.
TitleStr = Annotated[
    str,
    StringConstraints(max_length=TITLE_MAX_LENGTH),
    AfterValidator(_require_non_blank),
]
DescriptionStr = Annotated[str, StringConstraints(max_length=DESCRIPTION_MAX_LENGTH)]
BodyStr = Annotated[str, StringConstraints(max_length=BODY_MAX_LENGTH)]
KindStr = Annotated[str, StringConstraints(max_length=KIND_MAX_LENGTH)]


def _require_question(value: Any) -> Any:
    """Вопрос: непустой после подрезки и не длиннее QUESTION_MAX_LENGTH.

    Правило то же, что у имени блокнота (_require_non_blank), а вот отказ
    другой: ApiError вместо ValueError. Причина — машинный код. Пользователю
    интерфейс показывает свой перевод по error_code, а 422, который собирает
    из ValueError сам FastAPI, приходит без этого поля: в теле только массив
    detail с английским msg. Для имени блокнота это терпимо (кнопка «Создать»
    и так заблокирована при пустом поле), а здесь отказ должен быть объясним:
    именно он заменяет собой запуск поиска и генерации.

    Проверка стоит BeforeValidator'ом, то есть срабатывает раньше
    StringConstraints: иначе на слишком длинном значении первым сработал бы
    предел Pydantic и вернул бы то самое 422 без кода. Сам StringConstraints
    остаётся ради OpenAPI (maxLength в схеме) и как страховка.

    Не-строку пропускаем дальше: «question: 5» — ошибка типа, и отвечать на
    неё должен Pydantic своим обычным сообщением, а не наш код про пустоту.
    """
    if not isinstance(value, str):
        return value
    trimmed = value.strip()
    if not trimmed:
        raise ApiError(
            422,
            ChatErrors.QUESTION_REQUIRED,
            "Question must not be empty or whitespace only",
        )
    if len(trimmed) > QUESTION_MAX_LENGTH:
        raise ApiError(
            422,
            ChatErrors.QUESTION_TOO_LONG,
            f"Question must be at most {QUESTION_MAX_LENGTH} characters",
        )
    return trimmed


# Вопрос к ассистенту: подрезанный, непустой, в пределах бюджета контекста.
QuestionStr = Annotated[
    str,
    StringConstraints(max_length=QUESTION_MAX_LENGTH),
    BeforeValidator(_require_question),
]


def _invalid_token(error_code: str = AuthErrors.INVALID_TOKEN) -> ApiError:
    """Один и тот же ответ на любую негодность токена.

    В том числе на «пользователя из токена больше нет»: отдельный 404 на
    этот случай превращал эндпоинты в оракул для перебора имён.
    """
    return ApiError(
        status.HTTP_401_UNAUTHORIZED,
        error_code,
        "Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )


async def get_access_token(
    request: Request,
    header_token: Optional[str] = Depends(oauth2_scheme),
) -> str:
    """Access-токен: сначала заголовок Authorization, затем httpOnly-кука.

    Заголовок выигрывает намеренно. Его выставляют осознанно (curl, скрипты,
    Authorize в /docs), а куку браузер подставляет сам, и при работе в
    /docs из-под своей же залогиненной сессии кука иначе перебивала бы
    явно введённый токен. Тот же порядок делает безопасной выкатку: пока
    фронт ходит с заголовком, поведение не меняется вовсе.
    """
    token = header_token or request.cookies.get(security.ACCESS_COOKIE_NAME)
    if not token:
        raise _invalid_token()
    return token


async def _get_current_user_from_session(session: AsyncSession, token: str) -> User:
    payload = security.decode_token(token, security.ACCESS_TOKEN_TYPE)
    if payload is None:
        raise _invalid_token()

    username: str = payload.get("sub")
    result = await session.exec(select(User).where(User.username == username))
    user = result.first()

    if not user:
        raise _invalid_token()

    # Поколение токена должно совпадать с текущим поколением пользователя:
    # так смена пароля и принудительный выход обесценивают выданные ранее
    # токены, не дожидаясь их истечения.
    if payload.get("ver") != (user.token_version or 0):
        raise _invalid_token(AuthErrors.TOKEN_REVOKED)
    return user


async def get_current_user(
    session: AsyncSession = Depends(get_session), token: str = Depends(get_access_token)
) -> User:
    return await _get_current_user_from_session(session, token)


async def get_current_user_short_lived(
    token: str = Depends(get_access_token),
) -> User:
    async with session_context() as session:
        return await _get_current_user_from_session(session, token)


async def get_current_active_superuser(
    current_user: User = Depends(get_current_user),
) -> User:
    if current_user.role != "admin":
        raise ApiError(
            status.HTTP_403_FORBIDDEN,
            AuthErrors.FORBIDDEN,
            "The user doesn't have enough privileges",
        )
    return current_user


async def get_current_content_manager_or_admin(
    current_user: User = Depends(get_current_user),
) -> User:
    if current_user.role not in ("admin", "content_manager"):
        raise ApiError(
            status.HTTP_403_FORBIDDEN,
            AuthErrors.FORBIDDEN,
            "The user does not have enough privileges",
        )
    return current_user


# --- Владение ресурсами -------------------------------------------------
#
# Единое правило для всего API: админ видит всё, остальные — только своё.
# Ресурс без владельца доступен только админу: так безопаснее, чем трактовать
# это как «общее». Для блокнотов и документов такого состояния больше не
# существует (см. user_owns), остаётся оно только у журнала.
# Наружу всегда 404, а не 403, чтобы нельзя было перебором подтвердить
# существование чужого ресурса.


def user_owns(resource_owner_id: int | None, user: User) -> bool:
    """True, если пользователь вправе работать с ресурсом.

    None в resource_owner_id осмысленно ровно для одной сущности —
    log.user_id. Журнал хранит запись о событии, а не ресурс, и «у события
    нет автора» бывает правдой: legacy-строки, системное действие, возможный
    будущий фоновый писатель. Колонка остаётся nullable осознанно:
    переписать такие строки на админа значило бы заставить журнал врать, что
    админ делал то, чего не делал, а удалить — потерять историю, ради которой
    журнал и ведётся. Новые записи безавторскими не бывают, их автора требует
    require_log_author (app/modules/chat/service.py).

    Для блокнота и документа ветвь недостижима: notebook.owner_id и
    document.owner_id объявлены NOT NULL (app/core/database.py, init_db),
    потому что владение — обязательное свойство ресурса, и NULL там был
    багом, а не состоянием. Оставлена как защита в глубину: функция общая,
    сузить проверку до «не-None» на стороне этих сущностей нечем.
    """
    if user.role == "admin":
        return True
    return resource_owner_id is not None and resource_owner_id == user.id


async def get_owned_notebook(
    notebook_id: int = Path(..., ge=1, le=MAX_ID),
    session: AsyncSession = Depends(get_session),
    current_user: User = Depends(get_current_user),
) -> "Notebook":
    from app.shared.models import Notebook

    notebook = await session.get(Notebook, notebook_id)
    if not notebook or not user_owns(notebook.owner_id, current_user):
        raise ApiError(404, SourceErrors.NOTEBOOK_NOT_FOUND, "Notebook not found")
    return notebook


async def get_owned_notebook_for_update(
    notebook_id: int = Path(..., ge=1, le=MAX_ID),
    session: AsyncSession = Depends(get_session),
    current_user: User = Depends(get_current_user),
) -> "Notebook":
    """То же, что get_owned_notebook, но строка берётся SELECT ... FOR UPDATE.

    Нужна операциям, которые сносят блокнот со всем содержимым. Без
    блокировки два параллельных DELETE проходят проверку владения оба:
    второй доходит до commit, обнаруживает, что удалять уже нечего, и падает
    StaleDataError — то есть 500 на совершенно штатной гонке. С блокировкой
    второй запрос ждёт первого и после его commit не находит строки вовсе:
    отвечает тем же 404, что и на любой несуществующий блокнот.

    Отдельная зависимость, а не блокировка внутри обработчика: правило
    владения (админ видит всё, наружу всегда 404) должно оставаться в одном
    месте, а обработчик не должен иметь возможности случайно взять блокнот
    без блокировки. Сессия здесь та же, что у обработчика — Depends кэширует
    get_session на запрос, — поэтому блокировка живёт ровно до его commit.
    На 404 транзакция закрывается вместе с запросом и блокировку отпускает.
    """
    from app.shared.models import Notebook

    result = await session.exec(
        select(Notebook).where(Notebook.id == notebook_id).with_for_update()
    )
    notebook = result.first()
    if not notebook or not user_owns(notebook.owner_id, current_user):
        raise ApiError(404, SourceErrors.NOTEBOOK_NOT_FOUND, "Notebook not found")
    return notebook


async def get_owned_document(
    id: int = Path(..., ge=1, le=MAX_ID),
    session: AsyncSession = Depends(get_session),
    current_user: User = Depends(get_current_user),
) -> "Document":
    from app.shared.models import Document

    document = await session.get(Document, id)
    if not document or not user_owns(document.owner_id, current_user):
        raise ApiError(404, SourceErrors.NOT_FOUND, "Document not found")
    return document


async def assert_owns_notebook(
    notebook_id: int | None,
    session: AsyncSession,
    current_user: User,
    *,
    lock: bool = False,
) -> None:
    """Проверка владения блокнотом, когда id приходит в теле запроса.

    lock=True берёт строку блокнота SELECT ... FOR SHARE до конца транзакции
    запроса. Это нужно тем, кто вставляет в блокнот новые строки (загрузка и
    прикрепление документа): без блокировки параллельное удаление блокнота
    успевает пройти между проверкой и INSERT, и вставка падает по внешнему
    ключу document.notebook_id — 500 на ровном месте. FOR SHARE несовместим с
    FOR UPDATE удаления: либо удаление ждёт конца вставки и уносит уже новый
    документ, либо вставка ждёт конца удаления и получает честный 404.

    По умолчанию выключено намеренно: тот же помощник вызывает чат, а он
    держит транзакцию всё время стрима ответа — блокировка там задержала бы
    удаление блокнота на минуты.
    """
    if notebook_id is None:
        return
    # Последний рубеж на случай схемы без ограничения диапазона: id вне
    # диапазона PostgreSQL integer не может существовать, и запрос за ним
    # уронил бы asyncpg (OverflowError → 500) вместо честного 404.
    if not 1 <= notebook_id <= MAX_ID:
        raise ApiError(404, SourceErrors.NOTEBOOK_NOT_FOUND, "Notebook not found")
    from app.shared.models import Notebook

    if lock:
        result = await session.exec(
            select(Notebook)
            .where(Notebook.id == notebook_id)
            .with_for_update(read=True)
        )
        notebook = result.first()
    else:
        notebook = await session.get(Notebook, notebook_id)
    if not notebook or not user_owns(notebook.owner_id, current_user):
        raise ApiError(404, SourceErrors.NOTEBOOK_NOT_FOUND, "Notebook not found")
