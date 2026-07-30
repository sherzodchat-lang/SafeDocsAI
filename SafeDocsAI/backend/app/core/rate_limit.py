"""
Simple in-memory rate limiting for API endpoints.
For production, consider using Redis with slowapi or fastapi-limiter.

Ограничение по устройству: счётчики живут в памяти процесса, поэтому при
запуске с несколькими воркерами (`--workers N`) каждый держит свой счётчик,
и фактический лимит умножается на N. Пока это осознанный компромисс; общий
лимит требует внешнего хранилища (Redis).
"""

import ipaddress
import time
from functools import wraps
from typing import Optional
from fastapi import HTTPException, Request

from app.core.config import settings


class RateLimiter:
    """Simple in-memory rate limiter."""

    def __init__(self, requests: int = 100, window: int = 60, max_clients: int = 10000):
        """
        Args:
            requests: Maximum number of requests allowed
            window: Time window in seconds
            max_clients: Верхняя граница числа отслеживаемых клиентов
        """
        self.requests = requests
        self.window = window
        self.max_clients = max_clients
        # Обычный dict, а не defaultdict: чтение состояния не должно создавать
        # запись — иначе get_remaining() сам порождает мусор.
        self.clients: dict[str, list[float]] = {}
        self._last_cleanup = time.monotonic()

    def _cleanup(self, now: float) -> None:
        """Убрать ключи, по которым в окне не осталось запросов.

        Без этого перебор поддельных адресов оставляет в памяти по записи на
        каждый — состояние росло бы неограниченно и никогда не очищалось.
        """
        window_start = now - self.window
        stale = [
            client_id
            for client_id, timestamps in self.clients.items()
            if not timestamps or timestamps[-1] <= window_start
        ]
        for client_id in stale:
            del self.clients[client_id]
        self._last_cleanup = now

        # Аварийный предел на случай шквала уникальных адресов внутри одного
        # окна: выкидываем наименее свежие. Отдельный клиент при этом теряет
        # историю, но остальные лимиты продолжают работать.
        if len(self.clients) > self.max_clients:
            excess = len(self.clients) - self.max_clients
            oldest = sorted(self.clients, key=lambda key: self.clients[key][-1])[:excess]
            for client_id in oldest:
                del self.clients[client_id]

    def _recent(self, client_id: str, now: float) -> list[float]:
        window_start = now - self.window
        timestamps = self.clients.get(client_id)
        if timestamps is None:
            return []
        timestamps[:] = [t for t in timestamps if t > window_start]
        return timestamps

    def is_allowed(self, client_id: str) -> bool:
        """Check if request from client is allowed."""
        now = time.time()

        # Уборку делаем не чаще раза в окно: она линейна по числу ключей.
        if time.monotonic() - self._last_cleanup >= self.window:
            self._cleanup(now)

        client_requests = self._recent(client_id, now)

        if len(client_requests) < self.requests:
            if not client_requests:
                self.clients[client_id] = client_requests
            client_requests.append(now)
            return True

        return False

    def get_remaining(self, client_id: str) -> int:
        """Get remaining requests for client."""
        return max(0, self.requests - len(self._recent(client_id, time.time())))

    def get_retry_after(self, client_id: str) -> int:
        """Get seconds until next request is allowed."""
        client_requests = self._recent(client_id, time.time())
        if not client_requests:
            return 0
        oldest = min(client_requests)
        return max(0, int(oldest + self.window - time.time()))


# Global rate limiters
auth_limiter = RateLimiter(requests=10, window=60)  # 10 auth attempts per minute
# Обновление сессии — не попытка подбора пароля, но и не безлимит: несколько
# вкладок одного пользователя не должны упираться в лимит входа.
refresh_limiter = RateLimiter(requests=30, window=60)
chat_limiter = RateLimiter(requests=30, window=60)  # 30 chat requests per minute
api_limiter = RateLimiter(requests=100, window=60)  # 100 general API requests per minute


def _parse_networks(values: list[str]) -> list[ipaddress._BaseNetwork]:
    networks = []
    for value in values:
        try:
            networks.append(ipaddress.ip_network(value, strict=False))
        except ValueError:
            continue
    return networks


_TRUSTED_PROXIES = _parse_networks(settings.TRUSTED_PROXIES_LIST)


def _is_trusted(address: str) -> bool:
    try:
        ip = ipaddress.ip_address(address)
    except ValueError:
        return False
    return any(ip in network for network in _TRUSTED_PROXIES)


def get_client_id(request: Request) -> str:
    """Адрес клиента для счётчика лимита.

    X-Forwarded-For и X-Real-IP задаёт кто угодно, поэтому им верим только
    когда запрос пришёл от известного прокси (TRUSTED_PROXIES). Иначе
    достаточно менять заголовок на каждом запросе, и лимит на подбор пароля
    не срабатывает никогда.
    """
    peer = request.client.host if request.client else None
    if not peer:
        return "unknown"

    if not _is_trusted(peer):
        return peer

    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        # Читаем справа налево: слева цепочку дописывает клиент, справа —
        # доверенные прокси. Первый недоверенный справа и есть отправитель.
        for candidate in reversed([part.strip() for part in forwarded.split(",")]):
            if candidate and not _is_trusted(candidate):
                return candidate

    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip.strip()

    return peer


def rate_limit(limiter: RateLimiter, error_message: Optional[str] = None):
    """
    Decorator for rate limiting FastAPI endpoints.

    Usage:
        @app.get("/api/some-endpoint")
        @rate_limit(chat_limiter)
        async def my_endpoint():
            ...
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Find request in args/kwargs
            request: Optional[Request] = None
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                    break
            if not request:
                for value in kwargs.values():
                    if isinstance(value, Request):
                        request = value
                        break

            if request:
                client_id = get_client_id(request)
                if not limiter.is_allowed(client_id):
                    raise HTTPException(
                        status_code=429,
                        detail=error_message or "Rate limit exceeded",
                        headers={"Retry-After": str(limiter.get_retry_after(client_id))}
                    )

            return await func(*args, **kwargs)
        return wrapper
    return decorator


async def check_rate_limit(request: Request, limiter: RateLimiter) -> None:
    """
    Check rate limit for a request.
    Raises HTTPException if limit exceeded.
    """
    client_id = get_client_id(request)
    if not limiter.is_allowed(client_id):
        raise HTTPException(
            status_code=429,
            detail="Too many requests",
            headers={
                "Retry-After": str(limiter.get_retry_after(client_id)),
                "X-RateLimit-Limit": str(limiter.requests),
                "X-RateLimit-Remaining": str(limiter.get_remaining(client_id)),
            }
        )
