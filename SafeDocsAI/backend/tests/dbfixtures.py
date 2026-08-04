"""Общая обвязка для тестов, идущих по настоящему PostgreSQL.

Тот же приём, что и в tests/test_notebook_delete_db.py: схему создаёт код
проекта (app.core.database.init_db) в отдельной базе, а не CREATE TABLE в
тесте, поэтому внешние ключи, индексы и ALTER — ровно те же, что в рабочей
базе. Рабочая andozai_db при этом не трогается.

Файл выделен отдельно, потому что проверок, которым нужны настоящие внешние
ключи, стало больше одной: доступ к источникам (test_source_access_db.py) и
отложенная очистка векторов (test_vector_cleanup_db.py). Имя не начинается с
test_, поэтому unittest discover не пытается искать здесь тесты.

Тестовая база не создаётся автоматически (у andozai_user нет CREATEDB):

    sudo -u postgres psql -c "CREATE DATABASE andozai_test OWNER andozai_user;"

Имя перекрывается переменной окружения SAFEDOCS_TEST_DB. Если базы нет,
наследники делают skipTest с этой подсказкой, а не падают.
"""

import asyncio
import logging
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from httpx import ASGITransport, AsyncClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

import app.core.database as database_module
from app.api import deps
from app.core.database import get_session
from app.main import app
from app.shared.models import User
from app.shared.settings import settings


# IsolatedAsyncioTestCase поднимает цикл событий в debug-режиме, а
# app.core.logging в development выставляет корневой уровень DEBUG: вдвоём они
# засыпают вывод трассировкой каждого соединения asyncpg.
logging.getLogger("asyncio").setLevel(logging.WARNING)


TEST_DB_NAME = os.environ.get("SAFEDOCS_TEST_DB", "andozai_test")

# Страховка от опечатки в переменной окружения: тесты удаляют и обрезают
# таблицы, и запускать их по рабочей базе нельзя ни при каких условиях.
if TEST_DB_NAME == settings.POSTGRES_DB:
    raise RuntimeError(
        f"SAFEDOCS_TEST_DB={TEST_DB_NAME!r} совпадает с рабочей базой "
        f"POSTGRES_DB={settings.POSTGRES_DB!r}. Заведите отдельную базу."
    )

TEST_DATABASE_URI = (
    f"postgresql+asyncpg://{settings.POSTGRES_USER}:{settings.POSTGRES_PASSWORD}"
    f"@{settings.POSTGRES_SERVER}:{settings.POSTGRES_PORT}/{TEST_DB_NAME}"
)

# Своя схема внутри тестовой базы, отдельная на каждый процесс прогона.
#
# Каждый тест начинается с TRUNCATE, поэтому два одновременных прогона по
# одной схеме вычищают данные друг у друга посреди теста. Падения при этом
# выглядят необъяснимо («Could not refresh instance» на только что
# закоммиченной строке) и списываются на «мигает». Одновременные прогоны — не
# экзотика: их запускают разработчик и CI, разработчик и соседний агент, две
# ветки на одной машине.
#
# Отсюда pid в имени: схема принадлежит процессу и живёт ровно столько,
# сколько он. Отдельных прав не нужно — схему создаёт владелец базы. Цена —
# один init_db на прогон (доли секунды), а не на тест.
#
# search_path указывает только на неё: если оставить в нём public, create_all
# нашёл бы уже существующие там таблицы и не создал бы свои.
SCHEMA_PREFIX = "safedocs_tests_"
TEST_SCHEMA = os.environ.get("SAFEDOCS_TEST_SCHEMA", f"{SCHEMA_PREFIX}{os.getpid()}")

# Порядок не важен: TRUNCATE перечисленных разом снимает вопрос ссылок между
# ними, CASCADE подхватывает всё, что могло появиться в схеме позже.
_TABLES = (
    "chunk",
    "job",
    "presentation",
    "insight",
    "note",
    "log",
    "document",
    "notebook",
    "refreshtoken",
    '"user"',
)

_SCHEMA_READY = False


def new_engine():
    # NullPool обязателен: IsolatedAsyncioTestCase заводит новый цикл событий
    # на каждый тест, а соединение asyncpg привязано к циклу, в котором
    # открыто. С обычным пулом второй тест получил бы соединение из закрытого
    # цикла.
    return create_async_engine(
        TEST_DATABASE_URI,
        poolclass=NullPool,
        future=True,
        connect_args={"server_settings": {"search_path": TEST_SCHEMA}},
    )


def _bootstrap_engine():
    """Соединение без search_path: своей схемы ещё может не быть."""
    return create_async_engine(TEST_DATABASE_URI, poolclass=NullPool, future=True)


def _process_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:  # чужой процесс, но живой
        return True
    return True


async def _drop_abandoned_schemas(conn) -> None:
    """Убрать схемы завершившихся прогонов.

    Уборка идёт на старте, а не на выходе, потому что на выходе она
    невозможна: к моменту atexit пул потоков интерпретатора уже закрыт, и
    новое соединение asyncpg не поднять («cannot schedule new futures after
    interpreter shutdown»). Схема живёт до следующего прогона — это несколько
    пустых таблиц в тестовой базе.

    Живые чужие схемы не трогаем: перед удалением проверяется, жив ли процесс
    с таким pid. Так параллельный прогон (соседний агент, CI) остаётся цел.
    """
    result = await conn.execute(
        text(
            "SELECT nspname FROM pg_namespace WHERE nspname LIKE :prefix || '%'"
        ),
        {"prefix": SCHEMA_PREFIX},
    )
    for (name,) in result.all():
        suffix = name[len(SCHEMA_PREFIX):]
        if not suffix.isdigit() or name == TEST_SCHEMA:
            continue
        if _process_is_alive(int(suffix)):
            continue
        await conn.execute(text(f'DROP SCHEMA IF EXISTS "{name}" CASCADE'))


async def ensure_schema(engine) -> None:
    """Таблицы создаёт код проекта (init_db), а не CREATE TABLE в тесте."""
    global _SCHEMA_READY
    if _SCHEMA_READY:
        return
    bootstrap = _bootstrap_engine()
    try:
        async with bootstrap.begin() as conn:
            await _drop_abandoned_schemas(conn)
            await conn.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{TEST_SCHEMA}"'))
    finally:
        await bootstrap.dispose()
    with patch.object(database_module, "engine", engine):
        await database_module.init_db()
    _SCHEMA_READY = True


async def truncate(engine) -> None:
    async with engine.begin() as conn:
        await conn.execute(
            text(f"TRUNCATE TABLE {', '.join(_TABLES)} RESTART IDENTITY CASCADE")
        )


async def _probe() -> str | None:
    engine = new_engine()
    try:
        await ensure_schema(engine)
        # init_db на пустой базе может завести блокнот-заглушку для миграции
        # старых источников (сейчас — только если в базе есть админ); тестам
        # он не нужен и мешал бы счётным проверкам. Заодно RESTART IDENTITY
        # приводит счётчики id к предсказуемому началу.
        await truncate(engine)
    except Exception as exc:  # pragma: no cover - зависит от окружения
        return (
            f"Тестовая база {TEST_DB_NAME!r} недоступна ({exc.__class__.__name__}: "
            f"{exc}). Создайте её: sudo -u postgres psql -c "
            f'"CREATE DATABASE {TEST_DB_NAME} OWNER {settings.POSTGRES_USER};"'
        )
    finally:
        await engine.dispose()
    return None


SETUP_ERROR = asyncio.run(_probe())


class DatabaseBackedTestCase(unittest.IsolatedAsyncioTestCase):
    """Настоящая сессия к PostgreSQL, подставленная в зависимости FastAPI.

    Пользователь запроса подменяется целиком (deps.get_current_user), поэтому
    токены и пароли в этих тестах не участвуют: проверяется то, что эндпоинт
    делает с уже известным ему пользователем. Подмена накрывает и вложенные
    зависимости — если эндпоинт вернётся к проверке роли
    (get_current_content_manager_or_admin), она увидит ту же подставленную
    роль и тест это заметит.
    """

    async def asyncSetUp(self) -> None:
        if SETUP_ERROR:
            self.skipTest(SETUP_ERROR)

        self.engine = new_engine()
        self.addAsyncCleanup(self.engine.dispose)
        await ensure_schema(self.engine)
        await truncate(self.engine)

        self.session_factory = sessionmaker(
            self.engine, class_=AsyncSession, expire_on_commit=False
        )

        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)

        self.current_user: User | None = None

        app.dependency_overrides[get_session] = self._session_dependency
        app.dependency_overrides[deps.get_session] = self._session_dependency
        app.dependency_overrides[deps.get_current_user] = self._current_user_dependency
        app.dependency_overrides[deps.get_current_user_short_lived] = (
            self._current_user_dependency
        )
        self.addCleanup(app.dependency_overrides.clear)

        # raise_app_exceptions по умолчанию True: непойманное исключение
        # приложения всплывает в тест настоящим трейсбеком, а не 500.
        self.client = AsyncClient(
            transport=ASGITransport(app=app), base_url="http://testserver"
        )
        self.addAsyncCleanup(self.client.aclose)

    # --- зависимости ---

    async def _session_dependency(self):
        # Повторяет app.core.database.session_context: откат на исключении.
        async with self.session_factory() as session:
            try:
                yield session
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()

    async def _current_user_dependency(self) -> User:
        return self.current_user

    def as_user(self, user: User) -> None:
        self.current_user = user

    # --- работа с данными ---

    async def seed(self, *rows):
        async with self.session_factory() as session:
            for row in rows:
                session.add(row)
            await session.commit()
            for row in rows:
                await session.refresh(row)
        return rows[0] if len(rows) == 1 else rows

    async def make_user(self, username: str, role: str) -> User:
        return await self.seed(
            User(username=username, password_hash="not-a-real-hash", role=role)
        )

    async def get_row(self, model, pk):
        async with self.session_factory() as session:
            return await session.get(model, pk)

    async def exists(self, model, pk) -> bool:
        return await self.get_row(model, pk) is not None

    async def rows_where(self, model, clause):
        async with self.session_factory() as session:
            result = await session.exec(select(model).where(clause))
            return list(result.all())

    async def all_rows(self, model):
        async with self.session_factory() as session:
            result = await session.exec(select(model))
            return list(result.all())

    def make_file(self, name: str, content: str = "содержимое источника") -> str:
        path = Path(self._tmpdir.name) / name
        path.write_text(content, encoding="utf-8")
        return str(path)
