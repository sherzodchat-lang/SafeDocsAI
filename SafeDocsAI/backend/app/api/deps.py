from typing import Optional

from fastapi import Depends, Request, status
from fastapi.security import OAuth2PasswordBearer
from sqlmodel.ext.asyncio.session import AsyncSession
from app.shared.settings import settings
from app.core import security
from app.core.database import get_session, session_context
from app.core.exceptions import ApiError, AuthErrors, SourceErrors
from app.shared.models import User
from sqlmodel import select

# auto_error=False: без заголовка схема обязана промолчать, чтобы можно было
# заглянуть в куку. Сама схема остаётся объявленной — на ней держится кнопка
# Authorize в /docs и разметка securitySchemes в OpenAPI.
oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl=f"{settings.API_V1_STR}/auth/login/access-token",
    auto_error=False,
)


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
# Отсутствие владельца (owner_id IS NULL) считается legacy-состоянием и
# доступно только админу: так безопаснее, чем трактовать это как «общее».
# Наружу всегда 404, а не 403, чтобы нельзя было перебором подтвердить
# существование чужого ресурса.


def user_owns(resource_owner_id: int | None, user: User) -> bool:
    """True, если пользователь вправе работать с ресурсом."""
    if user.role == "admin":
        return True
    return resource_owner_id is not None and resource_owner_id == user.id


async def get_owned_notebook(
    notebook_id: int,
    session: AsyncSession = Depends(get_session),
    current_user: User = Depends(get_current_user),
) -> "Notebook":
    from app.shared.models import Notebook

    notebook = await session.get(Notebook, notebook_id)
    if not notebook or not user_owns(notebook.owner_id, current_user):
        raise ApiError(404, SourceErrors.NOTEBOOK_NOT_FOUND, "Notebook not found")
    return notebook


async def get_owned_document(
    id: int,
    session: AsyncSession = Depends(get_session),
    current_user: User = Depends(get_current_user),
) -> "Document":
    from app.shared.models import Document

    document = await session.get(Document, id)
    if not document or not user_owns(document.owner_id, current_user):
        raise ApiError(404, SourceErrors.NOT_FOUND, "Document not found")
    return document


async def assert_owns_notebook(
    notebook_id: int | None, session: AsyncSession, current_user: User
) -> None:
    """Проверка владения блокнотом, когда id приходит в теле запроса."""
    if notebook_id is None:
        return
    from app.shared.models import Notebook

    notebook = await session.get(Notebook, notebook_id)
    if not notebook or not user_owns(notebook.owner_id, current_user):
        raise ApiError(404, SourceErrors.NOTEBOOK_NOT_FOUND, "Notebook not found")
