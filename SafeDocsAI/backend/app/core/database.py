import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlmodel import SQLModel
from app.core.config import settings

logger = logging.getLogger(__name__)

# Echo SQL queries only in development
engine = create_async_engine(
    settings.SQLALCHEMY_DATABASE_URI,
    echo=settings.ENVIRONMENT == "development",
    future=True,
    pool_pre_ping=True,  # Verify connections before using
    pool_recycle=300,  # Recycle connections after 5 minutes
)

async_session_factory = sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)


async def init_db():
    """Initialize database tables."""
    try:
        async with engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)

            # Backward-compatible schema update for deployments that already
            # have the chunk table without this column.
            await conn.execute(
                text(
                    """
                    ALTER TABLE IF EXISTS chunk
                    ADD COLUMN IF NOT EXISTS chunk_index INTEGER
                    """
                )
            )

            # Без этого индекса каждый просмотр фрагментов документа и каждое
            # удаление (включая каскад от document) сканирует всю таблицу chunk.
            await conn.execute(
                text(
                    """
                    CREATE INDEX IF NOT EXISTS ix_chunk_doc_id ON chunk (doc_id)
                    """
                )
            )

            # Причина ошибки индексации. Пишется фоновым воркером и уходит
            # в API вместе со статусом документа.
            await conn.execute(
                text(
                    """
                    ALTER TABLE IF EXISTS document
                    ADD COLUMN IF NOT EXISTS error_text VARCHAR
                    """
                )
            )

            # Та же причина машинным кодом: под неё у клиента есть перевод,
            # а под error_text — нет.
            await conn.execute(
                text(
                    """
                    ALTER TABLE IF EXISTS document
                    ADD COLUMN IF NOT EXISTS error_code VARCHAR
                    """
                )
            )

            # GET /sources/ отдаёт страницу в порядке created_at DESC, id DESC.
            # Без индекса каждая страница сортирует таблицу целиком, а сам
            # порядок без ORDER BY не определён и ломает пагинацию.
            await conn.execute(
                text(
                    """
                    CREATE INDEX IF NOT EXISTS ix_document_created_at_id
                    ON document (created_at DESC, id DESC)
                    """
                )
            )

            # Воркер на каждой итерации ищет очередную задачу по
            # (status, job_type); без индекса это seq scan по всей истории.
            await conn.execute(
                text(
                    """
                    CREATE INDEX IF NOT EXISTS ix_job_queue
                    ON job (status, job_type, created_at)
                    """
                )
            )

            # Владение источниками. Колонка добавляется отдельно от create_all,
            # потому что на существующих базах таблица document уже есть.
            await conn.execute(
                text(
                    """
                    ALTER TABLE IF EXISTS document
                    ADD COLUMN IF NOT EXISTS owner_id INTEGER REFERENCES "user"(id)
                    """
                )
            )
            await conn.execute(
                text(
                    """
                    CREATE INDEX IF NOT EXISTS ix_document_owner_id
                    ON document (owner_id)
                    """
                )
            )
            # Порядок страницы GET /sources/ для обычного пользователя: его
            # выборка всегда сужена по owner_id, и ix_document_owner_id
            # сортировку не покрывает. Создаётся после ALTER выше — на старых
            # базах колонки owner_id до него ещё нет.
            await conn.execute(
                text(
                    """
                    CREATE INDEX IF NOT EXISTS ix_document_owner_created_at
                    ON document (owner_id, created_at DESC, id DESC)
                    """
                )
            )
            await conn.execute(
                text(
                    """
                    CREATE INDEX IF NOT EXISTS ix_notebook_owner_id
                    ON notebook (owner_id)
                    """
                )
            )

            # Бэкфилл: блокноты без владельца отдаём старейшему админу, иначе
            # после включения проверок они станут недоступны вообще никому.
            # Старейший (минимальный id) выбран как детерминированное правило:
            # результат не должен зависеть от порядка строк в таблице, а самый
            # ранний админ — это тот, кто и заводил эти блокноты до появления
            # колонки владельца.
            await conn.execute(
                text(
                    """
                    UPDATE notebook
                    SET owner_id = (
                        SELECT id FROM "user" WHERE role = 'admin' ORDER BY id LIMIT 1
                    )
                    WHERE owner_id IS NULL
                      AND EXISTS (SELECT 1 FROM "user" WHERE role = 'admin')
                    """
                )
            )
            # Документ наследует владельца своего блокнота; документы вне
            # блокнотов тоже уходят старейшему админу.
            await conn.execute(
                text(
                    """
                    UPDATE document d
                    SET owner_id = n.owner_id
                    FROM notebook n
                    WHERE d.notebook_id = n.id
                      AND d.owner_id IS NULL
                      AND n.owner_id IS NOT NULL
                    """
                )
            )
            await conn.execute(
                text(
                    """
                    UPDATE document
                    SET owner_id = (
                        SELECT id FROM "user" WHERE role = 'admin' ORDER BY id LIMIT 1
                    )
                    WHERE owner_id IS NULL
                      AND EXISTS (SELECT 1 FROM "user" WHERE role = 'admin')
                    """
                )
            )

            await conn.execute(
                text(
                    """
                    ALTER TABLE IF EXISTS document
                    ADD COLUMN IF NOT EXISTS notebook_id INTEGER REFERENCES notebook(id)
                    """
                )
            )

            # Поколение токенов пользователя. DEFAULT 0 обязателен: у уже
            # заведённых пользователей колонки нет, а NULL здесь означал бы
            # несовпадение с любым выданным токеном, то есть разлогин всех.
            await conn.execute(
                text(
                    """
                    ALTER TABLE IF EXISTS "user"
                    ADD COLUMN IF NOT EXISTS token_version INTEGER NOT NULL DEFAULT 0
                    """
                )
            )

            # Просроченные refresh-токены не нужны ни для проверки, ни для
            # обнаружения повторного использования — их уже отвергает срок.
            await conn.execute(
                text(
                    """
                    DELETE FROM refreshtoken WHERE expires_at < NOW()
                    """
                )
            )

            await conn.execute(
                text(
                    """
                    ALTER TABLE IF EXISTS log
                    ADD COLUMN IF NOT EXISTS notebook_id INTEGER REFERENCES notebook(id)
                    """
                )
            )

            await conn.execute(
                text(
                    """
                    ALTER TABLE IF EXISTS log
                    ADD COLUMN IF NOT EXISTS domain_profile VARCHAR(50)
                    """
                )
            )

            # Fill missing chunk indexes for old rows so ordering-dependent
            # features continue working after upgrades.
            await conn.execute(
                text(
                    """
                    WITH ranked AS (
                        SELECT
                            id,
                            ROW_NUMBER() OVER (PARTITION BY doc_id ORDER BY id) - 1 AS rn
                        FROM chunk
                    )
                    UPDATE chunk c
                    SET chunk_index = ranked.rn
                    FROM ranked
                    WHERE c.id = ranked.id AND c.chunk_index IS NULL
                    """
                )
            )

            # Блокнот-заглушка для источников, оставшихся без notebook_id.
            # Владелец — тот же старейший админ, что и в бэкфилле выше:
            # notebook.owner_id больше не может быть NULL. Без админа заглушку
            # не создаём вовсе — назначить владельца некому, а мигрировать на
            # такой базе всё равно нечего (документы там тоже ничьи).
            await conn.execute(
                text(
                    """
                    INSERT INTO notebook (name, description, domain_profile, owner_id, created_at)
                    SELECT 'Imported Tax Notebook', 'Migrated default notebook for existing sources', 'tax',
                           (SELECT id FROM "user" WHERE role = 'admin' ORDER BY id LIMIT 1), NOW()
                    WHERE NOT EXISTS (SELECT 1 FROM notebook)
                      AND EXISTS (SELECT 1 FROM "user" WHERE role = 'admin')
                    """
                )
            )

            # notebook.owner_id: NOT NULL вместо особого случая в коде.
            #
            # Ставится последним из шагов, затрагивающих notebook, и только
            # после того, как бэкфилл выше отработал: ограничение на колонке с
            # оставшимися NULL уронило бы старт приложения на живой базе.
            #
            # Если админа в базе нет (чистая установка до create_admin.py или
            # админа удалили), бэкфиллу некого назначить владельцем. Тогда шаг
            # пропускается с предупреждением, а не падает: приложение должно
            # подниматься, чтобы админа вообще можно было завести. Поведение
            # при этом не меняется — блокноты без владельца и сейчас доступны
            # только админу (deps.user_owns), — а ограничение доедет на первом
            # же старте после появления админа.
            orphan_notebooks = (
                await conn.execute(
                    text("SELECT COUNT(*) FROM notebook WHERE owner_id IS NULL")
                )
            ).scalar_one()
            if orphan_notebooks:
                logger.warning(
                    "Notebooks without an owner: %s. NOT NULL on notebook.owner_id "
                    "is postponed: no admin to inherit them. Run create_admin.py "
                    "and restart.",
                    orphan_notebooks,
                )
            else:
                # SET NOT NULL на уже помеченной колонке — no-op, поэтому шаг
                # безопасно повторяется на каждом старте.
                await conn.execute(
                    text(
                        """
                        ALTER TABLE IF EXISTS notebook
                        ALTER COLUMN owner_id SET NOT NULL
                        """
                    )
                )

            await conn.execute(
                text(
                    """
                    WITH default_notebook AS (
                        SELECT id FROM notebook ORDER BY id LIMIT 1
                    )
                    UPDATE document
                    SET notebook_id = (SELECT id FROM default_notebook)
                    WHERE notebook_id IS NULL
                    """
                )
            )

            # document.owner_id: NOT NULL, ровно по той же причине, что и у
            # блокнота — особый случай «ничей документ виден только админу»
            # уходит из кода в схему.
            #
            # Шаг идёт последним из затрагивающих document и обязательно после
            # notebook: бэкфилл документа наследует владельца своего блокнота,
            # а тот сам проставляется бэкфиллом блокнотов выше. В обратном
            # порядке документ блокнота, у которого владелец ещё NULL, ушёл бы
            # админу вместо настоящего хозяина.
            #
            # Без админа в базе (чистая установка до create_admin.py или админа
            # удалили) назначать владельца некому: документы вне блокнотов и
            # документы блокнотов, тоже оставшихся без владельца, остаются с
            # NULL. Тогда шаг пропускается с предупреждением, а не падает —
            # приложение должно подниматься, иначе админа не завести. Поведение
            # не меняется: документ без владельца и сейчас доступен только
            # админу (deps.user_owns), — а ограничение доедет на первом же
            # старте после появления админа.
            orphan_documents = (
                await conn.execute(
                    text("SELECT COUNT(*) FROM document WHERE owner_id IS NULL")
                )
            ).scalar_one()
            if orphan_documents:
                logger.warning(
                    "Documents without an owner: %s. NOT NULL on document.owner_id "
                    "is postponed: no admin to inherit them. Run create_admin.py "
                    "and restart.",
                    orphan_documents,
                )
            else:
                # SET NOT NULL на уже помеченной колонке — no-op, поэтому шаг
                # безопасно повторяется на каждом старте.
                await conn.execute(
                    text(
                        """
                        ALTER TABLE IF EXISTS document
                        ALTER COLUMN owner_id SET NOT NULL
                        """
                    )
                )

            await conn.execute(
                text(
                    """
                    UPDATE log
                    SET domain_profile = COALESCE(domain_profile, 'tax')
                    WHERE domain_profile IS NULL
                    """
                )
            )

            await conn.execute(
                text(
                    """
                    ALTER TABLE IF EXISTS note
                    ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP WITHOUT TIME ZONE
                    """
                )
            )

            await conn.execute(
                text(
                    """
                    UPDATE note
                    SET updated_at = COALESCE(updated_at, created_at, NOW())
                    WHERE updated_at IS NULL
                    """
                )
            )

            await conn.execute(
                text(
                    """
                    ALTER TABLE IF EXISTS insight
                    ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP WITHOUT TIME ZONE
                    """
                )
            )

            await conn.execute(
                text(
                    """
                    UPDATE insight
                    SET updated_at = COALESCE(updated_at, created_at, NOW())
                    WHERE updated_at IS NULL
                    """
                )
            )

        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Failed to create database tables: {e}")
        raise


@asynccontextmanager
async def session_context() -> AsyncIterator[AsyncSession]:
    """Get database session with automatic cleanup."""
    async with async_session_factory() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def get_session() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency for request-scoped database sessions."""
    async with session_context() as session:
        yield session


async def check_database_connection() -> bool:
    """Check if database connection is working."""
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return True
    except Exception as e:
        logger.error(f"Database connection check failed: {e}")
        return False
