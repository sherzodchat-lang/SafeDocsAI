from datetime import datetime
import logging
import os
from typing import Any

from fastapi import APIRouter, Depends, Query, Request, Response
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, field_serializer, field_validator
from sqlalchemy.orm.exc import StaleDataError
from sqlmodel import delete, func, or_, select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.api import deps
from app.api.endpoints.documents import (
    MAX_PAGE_SIZE,
    TOTAL_COUNT_HEADER,
    serialize_utc,
)
from app.core.exceptions import ApiError, SourceErrors
from app.core.rate_limit import RateLimiter, check_rate_limit
from app.domain_profiles import list_domain_profiles
# Только константа статуса: пакет app.modules.presentations намеренно не тянет
# за собой сервис и воркер (см. его __init__), поэтому RAG-стек сюда не
# приезжает. Своего литерала 'generating' здесь заводить нельзя — он разъедется
# с тем, что пишет в колонку PresentationsService.
from app.modules.presentations import (
    STATUS_GENERATING as PRESENTATION_STATUS_GENERATING,
)
from app.shared.models import (
    Chunk,
    Document,
    Insight,
    Job,
    Log,
    Note,
    Notebook,
    Presentation,
    User,
)

router = APIRouter()
logger = logging.getLogger(__name__)

# Создание блокнота — одна строка в БД, то есть на порядки дешевле загрузки
# источника (30 за 5 минут, documents.upload_limiter): ни файла, ни очереди,
# ни GPU. Поэтому лимит заметно свободнее и рассчитан не на защиту ресурса, а
# на скрипт, который в цикле набивает таблицу. 30 блокнотов за минуту живой
# человек не создаёт даже случайным двойным кликом по «Создать».
create_limiter = RateLimiter(requests=30, window=60)

# Страница по умолчанию равна потолку намеренно: клиент параметров не
# передаёт, и любой меньший размер молча урезал бы уже отдаваемые списки.
# Потолок при этом закрывает выгрузку всей таблицы одним запросом.
DEFAULT_PAGE_SIZE = MAX_PAGE_SIZE

# Незавершённые статусы очереди (app/modules/jobs/service.py: claim_next
# переводит 'queued' → 'running', finish — в 'completed'/'failed').
JOB_STATUS_QUEUED = "queued"
JOB_STATUS_RUNNING = "running"
ACTIVE_JOB_STATUSES = (JOB_STATUS_QUEUED, JOB_STATUS_RUNNING)


class NotebookCreate(BaseModel):
    # strip() без проверки результата принимал "   " как имя, а без верхней
    # границы имя длиннее предела btree-индекса роняло INSERT (500 вместо 422).
    name: deps.TitleStr
    description: deps.DescriptionStr | None = None
    domain_profile: str = "general"


class NotebookUpdate(BaseModel):
    """Частичное обновление: применяются только присланные поля.

    Все поля необязательны, поэтому «не прислано» и «прислано null» надо
    различать — иначе PATCH с одним name затирал бы описание. Отличаем по
    model_fields_set: default=None означает ровно «поле не пришло».

    Ограничения те же, что у NotebookCreate: типы из deps держат их в одном
    месте, иначе переименование обходило бы проверки, действующие на создании.
    """

    name: deps.TitleStr | None = None
    # description=null — осмысленное значение: так описание очищается.
    description: deps.DescriptionStr | None = None
    domain_profile: str | None = None

    @field_validator("name", "domain_profile")
    @classmethod
    def _reject_explicit_null(cls, value: str | None) -> str | None:
        # Колонки NOT NULL: явный null в теле — ошибка клиента, а не «очистить».
        # Валидатор не вызывается для значения по умолчанию, поэтому непришедшее
        # поле сюда не попадает.
        if value is None:
            raise ValueError("must not be null")
        return value


class NotebookResponse(BaseModel):
    id: int
    name: str
    description: str | None = None
    domain_profile: str
    owner_id: int | None = None
    created_at: datetime

    @field_serializer("created_at")
    def _serialize_datetime(self, value: datetime) -> str:
        return serialize_utc(value)


@router.get(
    "/",
    response_model=list[NotebookResponse],
    responses={
        200: {
            "headers": {
                TOTAL_COUNT_HEADER: {
                    "description": (
                        "Общее число блокнотов, доступных вызывающему, "
                        "без учёта skip/limit."
                    ),
                    "schema": {"type": "integer"},
                }
            }
        }
    },
)
async def list_notebooks(
    response: Response,
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=DEFAULT_PAGE_SIZE, ge=1, le=MAX_PAGE_SIZE),
    current_user: User = Depends(deps.get_current_user),
    session: AsyncSession = Depends(deps.get_session),
) -> Any:
    """Страница блокнотов, свежие сверху.

    Тело осталось массивом, общее число уходит заголовком X-Total-Count —
    ровно как у GET /sources/, поэтому клиент, не знающий о пагинации,
    ничего не заметил.
    """

    def _filtered(statement):
        # Фильтруем в SQL, а не после выборки: иначе чужие блокноты всё равно
        # попадают в память и легко «протекают» при следующей правке.
        if current_user.role != "admin":
            statement = statement.where(Notebook.owner_id == current_user.id)
        return statement

    # id вторым ключом сортировки: без него блокноты, созданные в одну
    # миллисекунду, прыгают между соседними страницами.
    page = _filtered(select(Notebook)).order_by(
        Notebook.created_at.desc(), Notebook.id.desc()
    )
    result = await session.exec(page.offset(skip).limit(limit))
    notebooks = result.all()

    total_result = await session.exec(
        _filtered(select(func.count()).select_from(Notebook))
    )
    response.headers[TOTAL_COUNT_HEADER] = str(int(total_result.first() or 0))
    return [
        NotebookResponse(
            id=notebook.id,
            name=notebook.name,
            description=notebook.description,
            domain_profile=notebook.domain_profile,
            owner_id=notebook.owner_id,
            created_at=notebook.created_at,
        )
        for notebook in notebooks
        if notebook.id is not None
    ]


@router.get("/{notebook_id}", response_model=NotebookResponse)
async def get_notebook(
    notebook: Notebook = Depends(deps.get_owned_notebook),
) -> Any:
    return NotebookResponse(
        id=notebook.id,
        name=notebook.name,
        description=notebook.description,
        domain_profile=notebook.domain_profile,
        owner_id=notebook.owner_id,
        created_at=notebook.created_at,
    )


@router.post("/", response_model=NotebookResponse)
async def create_notebook(
    request: Request,
    payload: NotebookCreate,
    current_user: User = Depends(deps.get_current_user),
    session: AsyncSession = Depends(deps.get_session),
) -> Any:
    await check_rate_limit(request, create_limiter)
    if payload.domain_profile not in list_domain_profiles():
        raise ApiError(
            400,
            SourceErrors.UNSUPPORTED_DOMAIN_PROFILE,
            "Unsupported domain profile",
        )

    notebook = Notebook(
        # Подрезка уже сделана валидацией схемы: там же отсеяно имя из
        # одних пробелов, которое strip() здесь молча превращал в пустое.
        name=payload.name,
        description=payload.description,
        domain_profile=payload.domain_profile,
        owner_id=current_user.id,
    )
    session.add(notebook)
    await session.commit()
    await session.refresh(notebook)
    return NotebookResponse(
        id=notebook.id,
        name=notebook.name,
        description=notebook.description,
        domain_profile=notebook.domain_profile,
        owner_id=notebook.owner_id,
        created_at=notebook.created_at,
    )


@router.patch("/{notebook_id}", response_model=NotebookResponse)
async def update_notebook(
    payload: NotebookUpdate,
    # Тот же SELECT ... FOR UPDATE, что и у удаления, и по той же причине:
    # без блокировки удаление блокнота успевает пройти между проверкой владения
    # и UPDATE, и SQLAlchemy обнаруживает 0 изменённых строк уже на commit —
    # StaleDataError, то есть 500 на штатной гонке. С блокировкой два запроса
    # выстраиваются в очередь: переименование либо успевает до удаления, либо
    # не находит блокнота и отвечает обычным 404.
    notebook: Notebook = Depends(deps.get_owned_notebook_for_update),
    session: AsyncSession = Depends(deps.get_session),
) -> Any:
    """Частичное обновление блокнота: name, description, domain_profile.

    Смена domain_profile разрешена намеренно. Профиль (app/domain_profiles)
    влияет только на поведение ответа — расширение запроса, переранжирование и
    текст правил в промпте (см. DomainProfile.search_queries/rerank_results/
    answer_rules); он читается на каждом запросе чата и ask через
    resolve_profile. Индексация документов профиль не использует вовсе, так что
    уже построенные чанки и эмбеддинги от смены не устаревают и
    переиндексация не нужна. Запрет же означал бы, что ошибку в профиле,
    выбранном при создании, можно исправить только удалением блокнота вместе с
    источниками — ровно та беда, ради которой этот эндпоинт и появился.
    """
    # id запоминаем до commit: после rollback объекты сессии помечены
    # протухшими, и чтение поля потянуло бы за собой запрос к БД.
    notebook_id = notebook.id
    fields = payload.model_fields_set
    if not fields:
        # Пустое тело — почти наверняка ошибка клиента. Ответить 200 с
        # неизменённым блокнотом значит подтвердить правку, которой не было.
        raise ApiError(
            400,
            SourceErrors.NOTHING_TO_UPDATE,
            "No fields to update",
        )

    if "domain_profile" in fields:
        if payload.domain_profile not in list_domain_profiles():
            raise ApiError(
                400,
                SourceErrors.UNSUPPORTED_DOMAIN_PROFILE,
                "Unsupported domain profile",
            )
        notebook.domain_profile = payload.domain_profile
    if "name" in fields:
        # Подрезка и запрет имени из одних пробелов уже сделаны валидацией
        # TitleStr — теми же правилами, что и на создании.
        notebook.name = payload.name
    if "description" in fields:
        notebook.description = payload.description

    session.add(notebook)
    try:
        await session.commit()
    except StaleDataError as exc:
        # Практически недостижимо при живой FOR UPDATE, но 500 на конкурентном
        # удалении не ответ: блокнота больше нет, править нечего.
        await session.rollback()
        logger.info("Concurrent delete of notebook %s: %s", notebook_id, exc)
        raise ApiError(
            409,
            SourceErrors.NOTEBOOK_DELETE_CONFLICT,
            "Notebook is already being deleted",
        ) from exc
    await session.refresh(notebook)
    return NotebookResponse(
        id=notebook.id,
        name=notebook.name,
        description=notebook.description,
        domain_profile=notebook.domain_profile,
        owner_id=notebook.owner_id,
        created_at=notebook.created_at,
    )


@router.delete("/{notebook_id}")
async def delete_notebook(
    # Блокнот приходит уже под SELECT ... FOR UPDATE: без блокировки два
    # параллельных DELETE проходят проверку владения оба и второй падает на
    # commit (StaleDataError → 500), а загрузка документа успевает вставить
    # строку в блокнот, которого через мгновение не станет.
    notebook: Notebook = Depends(deps.get_owned_notebook_for_update),
    session: AsyncSession = Depends(deps.get_session),
) -> dict[str, Any]:
    notebook_id = notebook.id
    # Владельца запоминаем до удаления: после commit объект отмечен удалённым,
    # и обращение к его полям уже не гарантировано.
    owner_id = notebook.owner_id

    # 1. Документы блокнота — только id и путь.
    # Целые строки Document здесь не нужны, а на блокноте с полусотней файлов
    # это лишние мегабайты в памяти обработчика.
    docs_result = await session.exec(
        select(Document.id, Document.path).where(Document.notebook_id == notebook_id)
    )
    doc_rows = docs_result.all()
    doc_ids = [row[0] for row in doc_rows if row[0] is not None]

    # 2. Цели побочной очистки собираем, пока строки ещё читаются.
    # Сами побочные эффекты (файлы на диске и векторы в ChromaDB) выполняем
    # только после commit: при откате транзакции блокнот остался бы в БД, а
    # файл и эмбеддинги были бы уже уничтожены безвозвратно.
    # Из чанков берём одну колонку id: text (~2 КБ на строку, десятки тысяч
    # строк на блокнот) для удаления не нужен ни в БД, ни в ChromaDB.
    chunk_ids: list[str] = []
    if doc_ids:
        chunks_result = await session.exec(
            select(Chunk.id).where(Chunk.doc_id.in_(doc_ids))
        )
        chunk_ids = [str(cid) for cid in chunks_result.all() if cid is not None]
    # Тройки (сущность, id, путь): id нужен только логу — по нему осиротевший
    # файл сопоставляется с записями о его загрузке и индексации, а имя
    # сущности отвечает на вопрос, в какой таблице эту строку искать: файлы на
    # диске оставляют и документы (data/uploads), и готовые презентации
    # (data/presentations).
    file_rows: list[tuple[str, Any, str]] = [
        ("document", row[0], row[1]) for row in doc_rows if row[1]
    ]

    # 3. Задачи блокнота.
    # Берём их не только по notebook_id: у документа, загруженного вне
    # блокнота и потом прикреплённого, у задачи остаётся старый или NULL
    # notebook_id — такая задача не попала бы в выборку и обрушила удаление.
    job_filter = Job.notebook_id == notebook_id
    if doc_ids:
        job_filter = or_(job_filter, Job.source_id.in_(doc_ids))

    # 4. Незавершённые задачи. FOR UPDATE здесь обязателен и делает две вещи.
    # Во-первых, воркер забирает задачи запросом FOR UPDATE SKIP LOCKED
    # (jobs/service.claim_next), поэтому запертую нами 'queued' он просто
    # пропустит и не начнёт индексировать документ, который мы сейчас удалим.
    # Во-вторых, задачу, уже перешедшую в 'running', мы гарантированно увидим
    # именно как 'running'.
    # Уже работающую задачу отменить нечем: воркер живёт в другом процессе,
    # канала отмены у очереди нет, а смену статуса он затрёт своим finish().
    # Удалить документ у него из-под рук — ровно тот сценарий, где воркер
    # пишет чанки на исчезнувший doc_id и роняет задачу IntegrityError.
    # Поэтому единственный честный ответ — 409 «повторите позже»: индексация
    # конечна, а зависшую задачу очередь сама вернёт в строй по протухшей
    # аренде (jobs/service.reap_stale).
    active_result = await session.exec(
        select(Job.id, Job.status)
        .where(job_filter)
        .where(Job.status.in_(ACTIVE_JOB_STATUSES))
        .with_for_update()
    )
    running = [row[0] for row in active_result.all() if row[1] == JOB_STATUS_RUNNING]
    if running:
        logger.info(
            "Refusing to delete notebook %s: jobs %s are still running",
            notebook_id,
            ", ".join(str(job_id) for job_id in running),
        )
        raise ApiError(
            409,
            SourceErrors.NOTEBOOK_BUSY_INDEXING,
            "Notebook is being indexed, try again later",
        )

    # 5. Презентации блокнота. Приём ровно тот же, что с задачами очереди, и по
    # тем же двум причинам. FOR UPDATE: воркер презентаций забирает очередной
    # заказ через FOR UPDATE SKIP LOCKED (presentations/service.claim_next),
    # поэтому запертую нами 'queued' он пропустит и не начнёт генерацию по
    # блокноту, которого через мгновение не станет; а заказ, уже перешедший в
    # 'generating', мы гарантированно увидим именно как 'generating'.
    #
    # 'generating' -> 409 по тому же доводу, что и 'running'-индексация:
    # генерация идёт в другом процессе, канала отмены у очереди нет, и удалить
    # блокнот у неё из-под рук значит уронить джобу на пустом месте. Отказ
    # временный: пайплайн конечен, а хвост убитого процесса очередь сама
    # вернёт в 'queued' (presentations/service.requeue_stuck).
    #
    # 'queued' удаляется свободно: файла у него ещё нет, и запирать удаление
    # блокнота из-за заказа, который никто не начал, было бы отказом на пустом
    # месте — ровно как с 'queued'-задачей индексации.
    presentations_result = await session.exec(
        select(Presentation.id, Presentation.status, Presentation.file_path)
        .where(Presentation.notebook_id == notebook_id)
        .with_for_update()
    )
    presentation_rows = presentations_result.all()
    generating = [
        row[0]
        for row in presentation_rows
        if row[1] == PRESENTATION_STATUS_GENERATING
    ]
    if generating:
        logger.info(
            "Refusing to delete notebook %s: presentations %s are still generating",
            notebook_id,
            ", ".join(str(pid) for pid in generating),
        )
        # Кода «в блокноте генерируется презентация» в exceptions.py пока нет,
        # а заводить строковый литерал по месту нельзя: клиент ищет перевод по
        # коду. Берём ближайший существующий — тот же 409 «блокнот занят,
        # повторите позже» с той же природой отказа (работа идёт в другом
        # процессе и прервать её нечем). Заменить на собственный код нужно
        # вместе с правкой exceptions.py и словарей фронта.
        raise ApiError(
            409,
            SourceErrors.NOTEBOOK_BUSY_GENERATING,
            "Notebook has a presentation being generated, try again later",
        )
    # Пути готовых колод читаются ЗДЕСЬ, до удаления строк: после commit
    # file_path не хранится больше нигде, и файл стал бы неотличим от чужого.
    # Фильтра по статусу намеренно нет: заполненная колонка file_path и есть
    # запись о существовании файла (её пишет только mark_ready), а вторая
    # проверка «а тот ли статус» однажды разойдётся с ней — у 'error' файл
    # предыдущего успешного прогона остаться может, у 'queued' колонка пуста.
    file_rows.extend(
        ("presentation", row[0], row[2]) for row in presentation_rows if row[2]
    )

    # 6. Удаление в БД.
    # Порядок DELETE теперь задаётся здесь, а не unit of work SQLAlchemy:
    # bulk DELETE уходит в базу сразу, в порядке вызовов. Нарушить его нельзя —
    # именно на этом ловится job_source_id_fkey (см.
    # tests/test_notebook_delete_db.py).
    try:
        if doc_ids:
            await _delete_rows(session, Chunk, Chunk.doc_id.in_(doc_ids))
        # insight.note_id -> note.id: инсайты уходят раньше заметок.
        # presentation.notebook_id объявлен без ON DELETE, как и всё здесь,
        # поэтому заказы обязаны уйти раньше самого блокнота — иначе DELETE
        # блокнота с готовой колодой падает на presentation_notebook_id_fkey
        # (см. tests/test_notebook_delete_presentations_db.py). Ссылок на
        # presentation ниоткуда нет, так что место в этом списке любое.
        for model_cls, fk in (
            (Insight, Insight.notebook_id),
            (Note, Note.notebook_id),
            (Log, Log.notebook_id),
            (Presentation, Presentation.notebook_id),
        ):
            await _delete_rows(session, model_cls, fk == notebook_id)
        # Задачи строго до документов: job.source_id ссылается на document.id
        # без ON DELETE CASCADE.
        await _delete_rows(session, Job, job_filter)
        # Документы (файлы на диске удаляются после commit) — до блокнота.
        await _delete_rows(session, Document, Document.notebook_id == notebook_id)
        # Сам блокнот — обычным ORM-удалением: строка одна, а несовпадение
        # числа удалённых строк с ожидаемым здесь ценно как сигнал гонки.
        await session.delete(notebook)
        await session.commit()
    except StaleDataError as exc:
        # Блокнот исчез между нашей блокировкой и commit. При живой FOR UPDATE
        # это почти невозможно, но 500 на конкурентном удалении не ответ:
        # работа всё равно сделана, клиенту достаточно обновить список.
        await session.rollback()
        logger.info("Concurrent delete of notebook %s: %s", notebook_id, exc)
        raise ApiError(
            409,
            SourceErrors.NOTEBOOK_DELETE_CONFLICT,
            "Notebook is already being deleted",
        ) from exc

    # 7. Побочная очистка — только после успешного commit.
    # Отказ ChromaDB здесь нельзя ни проглотить, ни превратить в 503: блокнота
    # в БД уже нет, а id чанков после commit не хранятся больше нигде. Поэтому
    # отказ переводится в задачу cleanup_embeddings — воркер повторит удаление,
    # когда ChromaDB вернётся. Смысл тот же, что у 503 на удалении одиночного
    # документа (documents/service.delete_document): сбой векторного хранилища
    # не считается успехом и не теряется.
    vector_cleanup = "done"
    if chunk_ids:
        try:
            await run_in_threadpool(_drop_vectors, chunk_ids)
        except Exception as exc:
            logger.warning(
                "ChromaDB cleanup failed for notebook %s after commit "
                "(%d chunks), scheduling deferred cleanup: %s",
                notebook_id,
                len(chunk_ids),
                exc,
            )
            vector_cleanup = await _schedule_vector_cleanup(
                session, notebook_id, chunk_ids, owner_id
            )
    # Файлы — последними, и отказ здесь не превращается ни в ошибку, ни в
    # отложенную задачу. Отдельного уборщика по образцу cleanup_embeddings у
    # файлов намеренно нет: висячий вектор портит ответы (всплывает цитатой из
    # удалённого документа), а осиротевший файл только занимает место и ни на
    # что не влияет. Причина отказа у них тоже разная по природе: ChromaDB
    # обычно просто временно лежит и повтор её дожидается, а os.remove падает
    # на правах или read-only монтировании — то, что повтором не лечится и
    # требует человека. Наконец, список файлов, в отличие от id чанков,
    # восстановим и без задачи: каталог загрузок сверяется с колонкой
    # document.path, каталог презентаций — с presentation.file_path, всё
    # лишнее в них — сирота.
    #
    # Поэтому единственное требование к этому месту — чтобы лога хватило на
    # ручную уборку без раскопок: причина пишется по каждому файлу, а полный
    # список путей — одной строкой, готовой к вставке в rm.
    orphan_paths: list[str] = []
    for kind, row_id, path in file_rows:
        try:
            os.remove(path)
        except FileNotFoundError:
            # Файла и так нет — ровно то, чего мы добивались. Проверка
            # os.path.exists() перед удалением делала то же самое, но с
            # окном гонки между проверкой и вызовом.
            continue
        except OSError as exc:
            orphan_paths.append(path)
            logger.warning(
                "Could not remove file %s of %s %s from deleted "
                "notebook %s after commit: %s",
                path,
                kind,
                row_id,
                notebook_id,
                exc,
            )
    if orphan_paths:
        # Отдельная строка ERROR: по ней одним grep'ом собирается полный
        # список того, что осталось на диске после удаления блокнота.
        logger.error(
            "Orphan files left on disk after deleting notebook %s (%d): %s",
            notebook_id,
            len(orphan_paths),
            " ".join(orphan_paths),
        )

    return {
        "detail": "Notebook deleted",
        "id": notebook_id,
        "vector_cleanup": vector_cleanup,
    }


async def _delete_rows(session: AsyncSession, model_cls, clause) -> None:
    """Удалить строки одним DELETE ... WHERE, не поднимая их в память.

    Полные ORM-объекты ради удаления обходятся дорого: у чанка это колонка
    text целиком (~2 КБ × десятки тысяч строк на блокнот), у лога —
    question, answer и sources, и на каждую строку уходит отдельный DELETE.
    Приём тот же, что в documents/service._purge_chunks.

    Цена — порядок: bulk DELETE выполняется сразу и в порядке вызовов, а не
    сортируется unit of work по зависимостям таблиц. Очерёдность обязан
    задать вызывающий, поэтому функция работает с одной сущностью за раз.
    """
    await session.exec(delete(model_cls).where(clause))


def _drop_vectors(chunk_ids: list[str]) -> None:
    """Снести векторы из ChromaDB. Только через run_in_threadpool.

    Блокирующие здесь оба вызова: конструктор RAGService открывает соединение
    с ChromaDB, delete_documents делает синхронный HTTP-запрос. В event loop
    это останавливает весь процесс uvicorn на всё время удаления — вместе с
    SSE-стримами чата других пользователей. Ср. documents/service, где по той
    же причине в threadpool вынесены add_documents и delete_documents.
    """
    from app.modules.rag.service import RAGService

    RAGService().delete_documents(chunk_ids)


async def _schedule_vector_cleanup(
    session: AsyncSession,
    notebook_id: int | None,
    chunk_ids: list[str],
    owner_id: int | None,
) -> str:
    """Поставить отложенное удаление векторов и вернуть состояние очистки.

    Задача — единственное место, где id осиротевших векторов ещё существуют:
    строки chunk удалены вместе с блокнотом. Если не удалось даже это (лежит и
    БД), пишем id в лог как последний рубеж — дальше их найдёт только
    reconcile_chroma.py, сверяющий коллекцию с таблицей chunk.
    """
    try:
        from app.modules.jobs.service import JOB_CLEANUP_EMBEDDINGS, JobsService

        job = await JobsService.enqueue(
            session,
            job_type=JOB_CLEANUP_EMBEDDINGS,
            payload={"chunk_ids": chunk_ids, "notebook_id": notebook_id},
            # source_id и notebook_id не заполняем: обе строки уже удалены,
            # внешние ключи такую задачу просто не примут.
            created_by=owner_id,
        )
    except Exception as exc:
        logger.error(
            "Could not schedule ChromaDB cleanup for deleted notebook %s: %s. "
            "Orphan chunk ids: %s",
            notebook_id,
            exc,
            ", ".join(chunk_ids),
        )
        return "failed"
    logger.info(
        "Scheduled deferred ChromaDB cleanup job %s for %d chunks "
        "of deleted notebook %s",
        job.id,
        len(chunk_ids),
        notebook_id,
    )
    return "deferred"
