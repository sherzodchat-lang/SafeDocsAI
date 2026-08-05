import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlmodel import SQLModel
from app.core.config import settings
# Статусы очереди презентаций — из объявления раздела, а не выписанными строками
# здесь: предикат частичного индекса ниже обязан описывать ровно то же
# множество «активных» статусов, что проверяет хендлер заказа
# (presentations.ACTIVE_STATUSES). Импорт дешёвый и не тянет ни ретривал, ни
# ChromaDB — пакет презентаций переэкспортирует только константы и схемы
# (см. его docstring).
from app.modules.presentations.constants import STATUS_GENERATING, STATUS_QUEUED

logger = logging.getLogger(__name__)

# --- Инвариант очереди презентаций ---------------------------------------
#
# «Не больше одной активной генерации на блокнот» держит БАЗА, а не хендлер.
# Имя индекса и множество статусов вынесены в константы, потому что на них
# ссылаются трое: сама миграция ниже, перехват нарушения уникальности в
# app/api/endpoints/presentations.py (по имени индекса он отличает СВОЁ
# нарушение от чужого) и проверка схемы в тестах.
PRESENTATION_ACTIVE_INDEX = "uq_presentation_active_notebook"
PRESENTATION_ACTIVE_STATUSES = (STATUS_QUEUED, STATUS_GENERATING)

# --- Инварианты раздела тем ----------------------------------------------
#
# Оба держит БАЗА, и оба объявлены здесь, а не в app/modules/topics/service.py,
# по одной причине: их читает миграция ниже, и импорт из раздела тем сюда
# завернул бы цикл (сервис тем берёт отсюда session_context). Обратное
# направление безопасно — сервис и эндпоинт импортируют эти имена отсюда, как
# эндпоинт презентаций импортирует PRESENTATION_ACTIVE_INDEX.

# Активная обученная модель ровно одна. Вторая означала бы, что «текущая тема
# документа» зависит от того, какую строку вернул SELECT без ORDER BY.
TOPIC_MODEL_ACTIVE_INDEX = "uq_topic_model_active"

# Переразметка — задача в общей таблице job, а не своя очередь: она нужна
# ровно за тем же, за чем очередь индексации (пережить перезапуск, атомарный
# захват при uvicorn --workers 2), и вторая реализация того же разошлась бы с
# первой на первой правке.
TOPIC_REASSIGN_JOB_TYPE = "reassign_topics"
# Не больше одной активной переразметки на всю систему. Предпроверка в
# хендлере — это то самое место, которое гонится: два клика в секунду проходят
# её оба, а переразметка ходит по всем документам разом и держит ChromaDB.
TOPIC_REASSIGN_ACTIVE_INDEX = "uq_topic_reassign_active"
TOPIC_REASSIGN_ACTIVE_STATUSES = ("queued", "running")

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

            # Очередь презентаций. Воркер на каждой итерации ищет самую раннюю
            # строку со status='queued' (ORDER BY created_at, id), а расчёт
            # позиции в очереди на каждый запрос статуса считает такие строки
            # «строго раньше этой». Оба запроса без индекса — seq scan по всей
            # истории генераций.
            #
            # Таблица presentation создаётся самим create_all выше (она новая,
            # ALTER для неё не нужен), а вот составной индекс SQLModel из
            # объявления модели не выводит: Field(index=True) умеет только
            # одноколоночные. Отсюда явный CREATE INDEX IF NOT EXISTS — тот же
            # приём и та же идемпотентность, что у ix_job_queue выше.
            await conn.execute(
                text(
                    """
                    CREATE INDEX IF NOT EXISTS ix_presentation_queue
                    ON presentation (status, created_at)
                    """
                )
            )

            # Не больше ОДНОЙ активной генерации на блокнот — инвариантом БАЗЫ.
            #
            # Предпроверка в хендлере заказа (SELECT активных строк, потом
            # INSERT) — это и есть то место, которое гонится: два клика в одну
            # секунду проходят проверку оба, а лимит частоты от них не спасает.
            # Цена промаха максимальная: двойная работа GPU, два файла-дубля и
            # спутанные позиции очереди. Частичный уникальный индекс закрывает
            # окно целиком, потому что проверку делает та же транзакция, что и
            # вставку.
            #
            # Предикат перечисляет статусы явно (подставить их параметром в DDL
            # нельзя): значения берутся из констант раздела, а тест сверяет
            # определение индекса в pg_indexes с ними же.
            active_statuses = ", ".join(
                f"'{status}'" for status in PRESENTATION_ACTIVE_STATUSES
            )
            # ГРЯЗНЫЕ ДАННЫЕ НЕ ДОЛЖНЫ РОНЯТЬ СТАРТ. На базе, пережившей гонку
            # до появления индекса, CREATE UNIQUE INDEX упадёт — и утащил бы за
            # собой весь init_db, то есть приложение вообще не поднялось бы, и
            # чинить данные пришлось бы вслепую из psql. Поэтому:
            #
            #   1) сначала считаем блокноты с несколькими активными заказами и
            #      при находке ПРОПУСКАЕМ создание индекса с ERROR в журнале —
            #      тот же приём, что у отложенных SET NOT NULL ниже (там шаг
            #      пропускается, когда бэкфиллу некого назначить владельцем);
            #   2) сам CREATE идёт под SAVEPOINT'ом: между подсчётом и
            #      созданием индекса соседний процесс uvicorn мог успеть
            #      поставить заказ. Без вложенной транзакции ошибка на этом
            #      шаге обрывает всю транзакцию init_db, включая уже сделанные
            #      миграции; с ней — гаснет только этот шаг.
            #
            # CREATE INDEX CONCURRENTLY здесь неприменим: PostgreSQL запрещает
            # его внутри транзакционного блока, а весь init_db выполняется в
            # одной транзакции (engine.begin()). Класть его в отдельное
            # соединение с autocommit ради онлайн-построения незачем: таблица
            # presentation маленькая (одна строка на заказ), а блокировка на
            # время построения — доли секунды на старте, когда очередь ещё не
            # запущена.
            conflicting = (
                await conn.execute(
                    text(
                        f"""
                        SELECT notebook_id, COUNT(*) AS active
                        FROM presentation
                        WHERE status IN ({active_statuses})
                        GROUP BY notebook_id
                        HAVING COUNT(*) > 1
                        ORDER BY notebook_id
                        """
                    )
                )
            ).all()
            if conflicting:
                logger.error(
                    "Unique index %s is postponed: notebooks with more than one "
                    "active presentation: %s. Keep one order per notebook "
                    "(DELETE FROM presentation WHERE ...) and restart; until "
                    "then the invariant rests on the pre-check in the handler "
                    "alone.",
                    PRESENTATION_ACTIVE_INDEX,
                    ", ".join(
                        f"notebook {row[0]}: {row[1]}" for row in conflicting
                    ),
                )
            else:
                try:
                    async with conn.begin_nested():
                        await conn.execute(
                            text(
                                f"""
                                CREATE UNIQUE INDEX IF NOT EXISTS
                                {PRESENTATION_ACTIVE_INDEX}
                                ON presentation (notebook_id)
                                WHERE status IN ({active_statuses})
                                """
                            )
                        )
                except Exception as exc:  # pragma: no cover - гонка на старте
                    logger.error(
                        "Could not create the unique index %s: %s. The "
                        "invariant rests on the pre-check in the handler; the "
                        "index will be created on the next start.",
                        PRESENTATION_ACTIVE_INDEX,
                        exc,
                    )

            # --- Раздел тем ------------------------------------------------
            #
            # Тема документа: номер кластера, хранимые подписи и версия модели,
            # которая это назначение сделала. Колонки добавляются отдельно
            # от create_all по той же причине, что и все остальные ALTER выше:
            # на существующих базах таблица document уже есть.
            #
            # Все они nullable и остаются такими навсегда: документ без темы —
            # нормальное состояние, а не незаполненная миграция. Тему может не
            # получить документ, загруженный до обучения модели, документ, чьи
            # векторы не отдала ChromaDB, и вообще любой документ, пока
            # переразметка до него не дошла. NOT NULL здесь означал бы, что
            # индексация обязана дождаться темы, — а она не обязана: тема
            # украшение, поиск по документу основная функция.
            #
            # topic_label_ru и topic_label_tg появились позже остальных и на
            # существующих документах остаются NULL до первой переразметки:
            # обратной засыпкой их не заполнить, потому что подпись принадлежит
            # той версии модели, которая делала назначение, а не текущей.
            for column in (
                "topic_cluster_index INTEGER",
                "topic_label VARCHAR",
                "topic_label_ru VARCHAR",
                "topic_label_tg VARCHAR",
                "topic_model_version INTEGER",
            ):
                await conn.execute(
                    text(
                        f"""
                        ALTER TABLE IF EXISTS document
                        ADD COLUMN IF NOT EXISTS {column}
                        """
                    )
                )

            # Реестр версий модели тоже пополнился колонкой: подписи кластеров
            # на языках интерфейса. Таблица создаётся самим create_all, но на
            # стендах, где она уже есть, create_all колонку не добавит.
            await conn.execute(
                text(
                    """
                    ALTER TABLE IF EXISTS topicmodelversion
                    ADD COLUMN IF NOT EXISTS labels_localized_json VARCHAR DEFAULT '{}'
                    """
                )
            )

            # Распределение тем считается GROUP BY topic_cluster_index при
            # фиксированной версии модели — и делает это КАЖДЫЙ показ раздела.
            # Без индекса это seq scan по всей таблице документов. Порядок
            # колонок именно такой: версия в запросе всегда задана равенством,
            # кластер только группируется.
            await conn.execute(
                text(
                    """
                    CREATE INDEX IF NOT EXISTS ix_document_topic
                    ON document (topic_model_version, topic_cluster_index)
                    """
                )
            )

            # Активная обученная модель ровно одна — инвариантом БАЗЫ, а не
            # дисциплиной кода. Регистрация модели идёт в две операции (снять
            # флаг со старой, вставить новую), и два процесса uvicorn,
            # обнаружившие новый артефакт одновременно, прошли бы её оба.
            #
            # Индекс частичный по is_active: в него попадают только активные
            # строки, а среди них значение колонки всегда одно и то же — то
            # есть уникальность по ней и означает «не больше одной». История
            # неактивных версий при этом не ограничена ничем, ради неё реестр
            # и заведён.
            #
            # Таблица topicmodelversion создаётся самим create_all выше (она
            # новая), поэтому грязных данных в ней быть не может; SAVEPOINT
            # оставлен на случай гонки со вторым процессом на старте — по тому
            # же доводу, что у индекса презентаций.
            try:
                async with conn.begin_nested():
                    await conn.execute(
                        text(
                            f"""
                            CREATE UNIQUE INDEX IF NOT EXISTS
                            {TOPIC_MODEL_ACTIVE_INDEX}
                            ON topicmodelversion (is_active)
                            WHERE is_active
                            """
                        )
                    )
            except Exception as exc:  # pragma: no cover - гонка на старте
                logger.error(
                    "Could not create the unique index %s: %s. Two active topic "
                    "models would become possible; the index will be created on "
                    "the next start.",
                    TOPIC_MODEL_ACTIVE_INDEX,
                    exc,
                )

            # Не больше ОДНОЙ активной переразметки тем. Довод тот же, что у
            # uq_presentation_active_notebook: предпроверка в хендлере (SELECT,
            # потом INSERT) гонится между двумя кликами, а цена промаха здесь —
            # два прохода по всем документам сразу, каждый со своими запросами
            # в ChromaDB.
            #
            # Уникальность по job_type, а не по чему-то содержательному: в
            # предикат уже входит и тип, и множество активных статусов, поэтому
            # в индекс попадает не больше одной строки на всю таблицу.
            reassign_statuses = ", ".join(
                f"'{status}'" for status in TOPIC_REASSIGN_ACTIVE_STATUSES
            )
            try:
                async with conn.begin_nested():
                    await conn.execute(
                        text(
                            f"""
                            CREATE UNIQUE INDEX IF NOT EXISTS
                            {TOPIC_REASSIGN_ACTIVE_INDEX}
                            ON job (job_type)
                            WHERE job_type = '{TOPIC_REASSIGN_JOB_TYPE}'
                              AND status IN ({reassign_statuses})
                            """
                        )
                    )
            except Exception as exc:  # pragma: no cover - гонка на старте
                logger.error(
                    "Could not create the unique index %s: %s. A second topic "
                    "reassignment could be queued in parallel; the index will "
                    "be created on the next start.",
                    TOPIC_REASSIGN_ACTIVE_INDEX,
                    exc,
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
