import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Optional
from fastapi import APIRouter, Depends, Header, Response, status, Request
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, Field, ConfigDict
from sqlalchemy import delete, update
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlmodel import select

from app.core import security
from app.core.config import settings
from app.core.exceptions import ApiError, AuthErrors
from app.core.rate_limit import (
    auth_limiter,
    check_rate_limit,
    refresh_limiter,
)
from app.api import deps
from app.shared.models import RefreshToken, User, utcnow

logger = logging.getLogger(__name__)

router = APIRouter()


class RegisterRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    username: str = Field(min_length=3, max_length=64)
    password: str = Field(min_length=security.MIN_PASSWORD_LENGTH, max_length=128)


class RegisterResponse(BaseModel):
    id: int
    username: str
    role: str
    created_at: datetime


class RefreshRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    refresh_token: Optional[str] = None


class LogoutRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    refresh_token: Optional[str] = None


class ChangePasswordRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    current_password: str = Field(min_length=1, max_length=128)
    new_password: str = Field(min_length=security.MIN_PASSWORD_LENGTH, max_length=128)


def _invalid_credentials() -> ApiError:
    """Один ответ на неверный логин и на неверный пароль: разные ответы
    позволяли бы перебором узнать, какие имена заведены."""
    return ApiError(
        status.HTTP_400_BAD_REQUEST,
        AuthErrors.INVALID_CREDENTIALS,
        "Incorrect username or password",
    )


def _invalid_refresh(error_code: str = AuthErrors.INVALID_TOKEN) -> ApiError:
    return ApiError(
        status.HTTP_401_UNAUTHORIZED,
        error_code,
        "Invalid or expired refresh token",
        headers={"WWW-Authenticate": "Bearer"},
    )


def _naive_utc(value: datetime) -> datetime:
    """Привести к тому же виду, в котором время лежит в остальных колонках."""
    return value.astimezone(timezone.utc).replace(tzinfo=None)


async def _revoke_all_refresh_tokens(session: AsyncSession, user_id: int) -> None:
    await session.exec(
        update(RefreshToken)
        .where(RefreshToken.user_id == user_id, RefreshToken.revoked_at.is_(None))
        .values(revoked_at=utcnow())
    )


async def _issue_tokens(
    session: AsyncSession,
    user: User,
    replaces: Optional[RefreshToken] = None,
) -> dict[str, Any]:
    """Выдать пару токенов. `replaces` — тот refresh, взамен которого выдаём:
    он гасится в этой же транзакции, то есть ротация атомарна."""
    version = user.token_version or 0
    access_token = security.create_access_token(user.username, version)
    refresh_token, jti, refresh_expires = security.create_refresh_token(
        user.username, version
    )

    # Мусор чистим по владельцу: строк на пользователя немного, а полная
    # уборка по таблице на каждом входе не нужна.
    await session.exec(
        delete(RefreshToken).where(
            RefreshToken.user_id == user.id,
            RefreshToken.expires_at < utcnow(),
        )
    )

    if replaces is not None:
        replaces.revoked_at = utcnow()
        replaces.replaced_by = jti
        session.add(replaces)

    session.add(
        RefreshToken(
            jti=jti,
            user_id=user.id,
            expires_at=_naive_utc(refresh_expires),
        )
    )
    await session.commit()

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "expires_in": settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        "refresh_token": refresh_token,
        "refresh_expires_in": settings.REFRESH_TOKEN_EXPIRE_DAYS * 24 * 3600,
    }


def _csrf_failed() -> ApiError:
    return ApiError(
        status.HTTP_403_FORBIDDEN,
        security.CSRF_ERROR_CODE,
        "CSRF token missing or invalid",
    )


def _extract_refresh_token(
    payload: Optional[RefreshRequest | LogoutRequest],
    authorization: Optional[str],
    request: Optional[Request] = None,
) -> tuple[str, str]:
    """Refresh-токен и то, откуда он взят: body, header или cookie.

    Порядок от явного к неявному: тело и заголовок клиент заполняет сам,
    куку подставляет браузер. Так клиент, предъявивший конкретный токен,
    всегда обменивает именно его, а не тот, что случайно лежит в куке.
    Источник нужен вызывающему: от него зависит, требовать ли CSRF.
    """
    if payload is not None and payload.refresh_token:
        return payload.refresh_token, "body"
    if authorization and authorization.lower().startswith("bearer "):
        return authorization[7:].strip(), "header"
    if request is not None:
        cookie_token = request.cookies.get(security.REFRESH_COOKIE_NAME)
        if cookie_token:
            return cookie_token, "cookie"
    raise _invalid_refresh()


def _assert_csrf_if_cookie(request: Optional[Request], source: str) -> None:
    """CSRF нужен только тогда, когда токен пришёл кукой.

    Общий middleware этого различить не может: он видит куку и не видит
    тела, а вызов с refresh-токеном в теле CSRF не подвержен — чужой сайт
    не знает его значения. Поэтому /auth/refresh и /auth/logout выведены
    из-под middleware и проверяются здесь, где источник уже известен.
    """
    if source != "cookie":
        return
    if request is None or not security.csrf_tokens_match(request):
        raise _csrf_failed()


def _apply_session_cookies(
    response: Response,
    tokens: dict[str, Any],
    csrf_token: Optional[str] = None,
) -> dict[str, Any]:
    """Разложить выданную пару по httpOnly-кукам и собрать тело ответа.

    Тело по-прежнему содержит сами токены: фронтенд переключается на куки
    отдельным шагом, и до этого момента он читает их именно оттуда.
    """
    csrf_token = csrf_token or security.create_csrf_token()
    security.set_auth_cookies(
        response, tokens["access_token"], tokens["refresh_token"], csrf_token
    )
    return {**tokens, "csrf_token": csrf_token}


@router.post("/login/access-token", response_model=dict)
async def login_access_token(
    response: Response,
    session: AsyncSession = Depends(deps.get_session),
    form_data: OAuth2PasswordRequestForm = Depends(),
    request: Request = None,
) -> Any:
    """
    OAuth2 compatible token login, get an access token for future requests.
    Rate limited to 10 attempts per minute per IP.
    """
    # Check rate limit
    if request:
        await check_rate_limit(request, auth_limiter)

    result = await session.exec(select(User).where(User.username == form_data.username))
    user = result.first()

    if not user or not security.verify_password(form_data.password, user.password_hash):
        raise _invalid_credentials()

    tokens = await _issue_tokens(session, user)
    # CSRF-токен на входе всегда новый: иначе тот, кто сумел заранее
    # подсунуть браузеру свою CSRF-куку, знал бы её значение и после входа.
    return _apply_session_cookies(response, tokens)


@router.post("/login", response_model=dict)
async def login_alias(
    response: Response,
    session: AsyncSession = Depends(deps.get_session),
    form_data: OAuth2PasswordRequestForm = Depends(),
    request: Request = None,
) -> Any:
    """
    Alias for /login/access-token to match clients expecting /auth/login.
    """
    return await login_access_token(
        response=response, session=session, form_data=form_data, request=request
    )


@router.post("/refresh", response_model=dict)
async def refresh_access_token(
    response: Response,
    session: AsyncSession = Depends(deps.get_session),
    payload: RefreshRequest = None,
    authorization: Optional[str] = Header(default=None),
    request: Request = None,
) -> Any:
    """
    Обменять refresh-токен на новую пару. Предъявленный refresh при этом
    гасится: украденная копия перестаёт работать после первого же честного
    обновления, а повторное предъявление обнаруживается.
    """
    if request:
        await check_rate_limit(request, refresh_limiter)

    # CSRF-токен на обновлении сохраняем: вкладки обновляются вразнобой, и
    # ротация оставила бы у одной из них заголовок от прежнего значения
    # куки — то есть ложный 403 на ровном месте.
    existing_csrf = (
        request.cookies.get(security.CSRF_COOKIE_NAME) if request else None
    )

    token, source = _extract_refresh_token(payload, authorization, request)
    _assert_csrf_if_cookie(request, source)

    claims = security.decode_token(token, security.REFRESH_TOKEN_TYPE)
    if claims is None:
        raise _invalid_refresh()

    result = await session.exec(
        select(RefreshToken).where(RefreshToken.jti == claims["jti"])
    )
    stored = result.first()
    if stored is None:
        raise _invalid_refresh()

    user_result = await session.exec(
        select(User).where(User.username == claims["sub"])
    )
    user = user_result.first()
    if user is None or claims.get("ver") != (user.token_version or 0):
        raise _invalid_refresh(AuthErrors.TOKEN_REVOKED)

    if stored.revoked_at is not None:
        # Гонка вкладок: обе прочитали один токен из localStorage и обновили
        # его почти одновременно. В пределах короткого окна это не кража, и
        # выкидывать пользователя на экран входа не за что.
        leeway = timedelta(seconds=settings.REFRESH_TOKEN_REUSE_LEEWAY_SECONDS)
        if (
            stored.replaced_by
            and leeway
            and utcnow() - stored.revoked_at <= leeway
        ):
            tokens = await _issue_tokens(session, user)
            return _apply_session_cookies(response, tokens, existing_csrf)

        # Токен уже использовали или отозвали. Раз копия ходит по рукам,
        # гасим все refresh-токены этого пользователя: остаются только
        # короткоживущие access, войти заново придётся честно.
        await _revoke_all_refresh_tokens(session, user.id)
        await session.commit()
        logger.warning(
            "Refresh token reuse detected for user_id=%s jti=%s", user.id, stored.jti
        )
        raise _invalid_refresh(AuthErrors.TOKEN_REVOKED)

    if stored.expires_at < utcnow():
        raise _invalid_refresh()

    tokens = await _issue_tokens(session, user, replaces=stored)
    return _apply_session_cookies(response, tokens, existing_csrf)


@router.post("/logout", response_model=dict)
async def logout(
    response: Response,
    session: AsyncSession = Depends(deps.get_session),
    payload: LogoutRequest = None,
    authorization: Optional[str] = Header(default=None),
    request: Request = None,
) -> Any:
    """
    Погасить предъявленный refresh-токен. Access-токен продолжает работать
    до истечения своего короткого срока; чтобы оборвать и его, нужен
    /logout-all.

    Отвечает 200 всегда: чужой или уже недействительный токен не повод
    сообщать клиенту, что такой токен когда-то существовал.
    """
    try:
        token, source = _extract_refresh_token(payload, authorization, request)
    except ApiError:
        # Куки гасим в любом случае, даже если предъявить было нечего:
        # иначе браузер продолжит носить с собой мёртвую сессию.
        security.clear_auth_cookies(response)
        return {"status": "ok"}

    _assert_csrf_if_cookie(request, source)
    security.clear_auth_cookies(response)

    claims = security.decode_token(token, security.REFRESH_TOKEN_TYPE)
    if claims is None:
        return {"status": "ok"}

    result = await session.exec(
        select(RefreshToken).where(RefreshToken.jti == claims["jti"])
    )
    stored = result.first()
    if stored is not None and stored.revoked_at is None:
        stored.revoked_at = utcnow()
        session.add(stored)
        await session.commit()
    return {"status": "ok"}


@router.post("/logout-all", response_model=dict)
async def logout_all_sessions(
    response: Response,
    session: AsyncSession = Depends(deps.get_session),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    """
    Выход со всех устройств: поднимаем поколение токенов, после чего все
    ранее выданные access и refresh перестают приниматься немедленно.
    """
    current_user.token_version = (current_user.token_version or 0) + 1
    session.add(current_user)
    await _revoke_all_refresh_tokens(session, current_user.id)
    await session.commit()
    security.clear_auth_cookies(response)
    return {"status": "ok"}


@router.post("/change-password", response_model=dict)
async def change_password(
    response: Response,
    session: AsyncSession = Depends(deps.get_session),
    payload: ChangePasswordRequest = None,
    current_user: User = Depends(deps.get_current_user),
    request: Request = None,
) -> Any:
    """
    Смена пароля. Все ранее выданные токены обесцениваются: смена пароля,
    не разрывающая уже открытые сессии, не спасает от угона токена.
    Вызывающему сразу выдаётся новая пара, чтобы не выкидывать его на экран
    входа.
    """
    if request:
        await check_rate_limit(request, auth_limiter)

    if not security.verify_password(
        payload.current_password, current_user.password_hash
    ):
        raise _invalid_credentials()

    policy_error = security.password_policy_error(payload.new_password)
    if policy_error:
        raise ApiError(
            status.HTTP_400_BAD_REQUEST, AuthErrors.WEAK_PASSWORD, policy_error
        )

    current_user.password_hash = security.get_password_hash(payload.new_password)
    current_user.token_version = (current_user.token_version or 0) + 1
    session.add(current_user)
    await _revoke_all_refresh_tokens(session, current_user.id)
    await session.commit()
    await session.refresh(current_user)

    tokens = await _issue_tokens(session, current_user)
    # Смена пароля обрывает все прежние сессии, поэтому CSRF-токен тоже
    # выдаётся новый.
    return _apply_session_cookies(response, tokens)


@router.post("/register", response_model=RegisterResponse)
async def register_user(
    session: AsyncSession = Depends(deps.get_session),
    payload: RegisterRequest = None,
    request: Request = None,
) -> Any:
    """
    Register a new user.
    Disabled unless ALLOW_REGISTRATION is set; admins create accounts instead.
    Rate limited to 10 attempts per minute per IP.
    """
    if not settings.ALLOW_REGISTRATION:
        raise ApiError(
            status.HTTP_403_FORBIDDEN,
            AuthErrors.REGISTRATION_DISABLED,
            "Self-registration is disabled. Contact an administrator.",
        )

    if request:
        # Check rate limit
        await check_rate_limit(request, auth_limiter)

    policy_error = security.password_policy_error(payload.password)
    if policy_error:
        raise ApiError(
            status.HTTP_400_BAD_REQUEST, AuthErrors.WEAK_PASSWORD, policy_error
        )

    result = await session.exec(select(User).where(User.username == payload.username))
    user = result.first()
    if user:
        # Формулировка намеренно не подтверждает, что имя занято: иначе
        # открытая регистрация превращается в список пользователей.
        raise ApiError(
            status.HTTP_400_BAD_REQUEST,
            AuthErrors.REGISTRATION_REJECTED,
            "Registration could not be completed",
        )
    user_in = User(
        username=payload.username,
        password_hash=security.get_password_hash(payload.password),
        role="user",
    )
    session.add(user_in)
    await session.commit()
    await session.refresh(user_in)
    return RegisterResponse(
        id=user_in.id,
        username=user_in.username,
        role=user_in.role,
        created_at=user_in.created_at,
    )
