import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional
from jose import JWTError, jwt
import bcrypt
from fastapi import Request, Response
from app.core.config import settings

# Тип токена пишется в полезную нагрузку и проверяется при разборе: без него
# refresh-токен принимается как access и живёт неделю вместо получаса.
ACCESS_TOKEN_TYPE = "access"
REFRESH_TOKEN_TYPE = "refresh"

MIN_PASSWORD_LENGTH = 8

# --- Куки сессии --------------------------------------------------------
#
# Префикс sd_ нужен, чтобы на localhost, где куки не разделяются по портам,
# соседний проект не перетирал нашу сессию.
ACCESS_COOKIE_NAME = "sd_access_token"
REFRESH_COOKIE_NAME = "sd_refresh_token"

# CSRF-кука намеренно НЕ httpOnly: её обязан прочитать наш же скрипт и
# положить значение в заголовок. Секретом она не является — защита в том,
# что чужой origin не может ни прочитать её, ни выставить заголовок.
CSRF_COOKIE_NAME = "sd_csrf_token"
CSRF_HEADER_NAME = "X-CSRF-Token"
CSRF_ERROR_CODE = "auth.csrf_failed"

def verify_password(plain_password, hashed_password):
    if not plain_password or not hashed_password:
        return False
    try:
        return bcrypt.checkpw(
            plain_password.encode("utf-8"),
            hashed_password.encode("utf-8"),
        )
    except (ValueError, TypeError):
        return False

def get_password_hash(password):
    return bcrypt.hashpw(
        password.encode("utf-8"),
        bcrypt.gensalt(),
    ).decode("utf-8")


def password_policy_error(password: str) -> Optional[str]:
    """Причина отказа в пароле или None, если пароль приемлем.

    Требования сознательно минимальны: длина и наличие двух разных классов
    символов. Более жёсткие правила состава не увеличивают стойкость, но
    гарантированно приводят к паролям на бумажке.
    """
    if len(password) < MIN_PASSWORD_LENGTH:
        return f"Password must be at least {MIN_PASSWORD_LENGTH} characters long"
    if not any(char.isalpha() for char in password):
        return "Password must contain at least one letter"
    if not any(char.isdigit() for char in password):
        return "Password must contain at least one digit"
    return None


def _encode_token(
    subject: str,
    token_type: str,
    token_version: int,
    expires_delta: timedelta,
) -> tuple[str, str, datetime]:
    """Собрать подписанный токен. Возвращает (токен, jti, момент истечения)."""
    now = datetime.now(timezone.utc)
    expire = now + expires_delta
    # jti нужен, чтобы конкретный токен можно было опознать и отозвать
    # поимённо: без него отзыв возможен только целыми поколениями (ver).
    jti = secrets.token_urlsafe(24)
    payload = {
        "sub": subject,
        "typ": token_type,
        "ver": token_version,
        "jti": jti,
        "iat": int(now.timestamp()),
        "exp": int(expire.timestamp()),
    }
    encoded_jwt = jwt.encode(
        payload, settings.SECRET_KEY, algorithm=settings.ALGORITHM
    )
    return encoded_jwt, jti, expire


def create_access_token(
    subject: str,
    token_version: int = 0,
    expires_delta: Optional[timedelta] = None,
) -> str:
    delta = expires_delta or timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    token, _jti, _expire = _encode_token(
        subject, ACCESS_TOKEN_TYPE, token_version, delta
    )
    return token


def create_refresh_token(
    subject: str,
    token_version: int = 0,
    expires_delta: Optional[timedelta] = None,
) -> tuple[str, str, datetime]:
    """Refresh-токен вместе с его jti и сроком: jti сохраняется в БД, и
    только он делает возможной ротацию (использованный токен помечается)."""
    delta = expires_delta or timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
    return _encode_token(subject, REFRESH_TOKEN_TYPE, token_version, delta)


def decode_token(token: str, expected_type: str) -> Optional[dict]:
    """Разобрать и проверить токен. None — если подпись, срок или тип не те."""
    try:
        payload = jwt.decode(
            token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM]
        )
    except JWTError:
        return None

    if payload.get("typ") != expected_type:
        return None
    if not payload.get("sub"):
        return None
    return payload


def create_csrf_token() -> str:
    return secrets.token_urlsafe(32)


def csrf_tokens_match(request: Request) -> bool:
    """Совпадает ли CSRF-токен из куки с токеном из заголовка.

    Сравнение постоянного времени: значение хоть и не секрет в обычном
    смысле, но и подбирать его по времени ответа незачем.
    """
    cookie_token = request.cookies.get(CSRF_COOKIE_NAME)
    header_token = request.headers.get(CSRF_HEADER_NAME)
    if not cookie_token or not header_token:
        return False
    return secrets.compare_digest(cookie_token, header_token)


def _cookie_attrs(http_only: bool, max_age: int) -> dict:
    """Общие атрибуты всех кук сессии.

    Path='/' — а не более узкий '/api/v1/auth' для refresh: тот же роутер
    подключён вторым префиксом '/api/auth', и кука, привязанная к одному из
    них, молча не доедет до другого. Сузить путь можно только вместе с
    отказом от совместимого алиаса.
    """
    return {
        "httponly": http_only,
        "secure": settings.COOKIE_SECURE_FLAG,
        "samesite": settings.COOKIE_SAMESITE,
        "domain": settings.COOKIE_DOMAIN_OR_NONE,
        "path": "/",
        "max_age": max_age,
    }


def set_auth_cookies(
    response: Response,
    access_token: str,
    refresh_token: str,
    csrf_token: str,
) -> None:
    """Положить пару токенов и CSRF-токен в куки.

    Срок жизни кук совпадает со сроком самих токенов: кука, пережившая
    токен, даёт 401 вместо тихого перелогина. CSRF-кука живёт столько же,
    сколько refresh, — она нужна ровно до конца сессии.
    """
    access_max_age = settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60
    refresh_max_age = settings.REFRESH_TOKEN_EXPIRE_DAYS * 24 * 3600

    response.set_cookie(
        ACCESS_COOKIE_NAME, access_token, **_cookie_attrs(True, access_max_age)
    )
    response.set_cookie(
        REFRESH_COOKIE_NAME, refresh_token, **_cookie_attrs(True, refresh_max_age)
    )
    response.set_cookie(
        CSRF_COOKIE_NAME, csrf_token, **_cookie_attrs(False, refresh_max_age)
    )


def clear_auth_cookies(response: Response) -> None:
    """Удалить куки сессии.

    Атрибуты path/domain обязаны совпадать с теми, что были при выдаче:
    иначе браузер заведёт вторую куку вместо удаления первой.
    """
    for name in (ACCESS_COOKIE_NAME, REFRESH_COOKIE_NAME, CSRF_COOKIE_NAME):
        response.delete_cookie(
            name,
            path="/",
            domain=settings.COOKIE_DOMAIN_OR_NONE,
            secure=settings.COOKIE_SECURE_FLAG,
            httponly=name != CSRF_COOKIE_NAME,
            samesite=settings.COOKIE_SAMESITE,
        )
