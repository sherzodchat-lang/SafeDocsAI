import os
from datetime import datetime
from typing import Annotated, Any, List, Optional
from fastapi import (
    APIRouter,
    Depends,
    Form,
    Path,
    Query,
    Request,
    Response,
    UploadFile,
)
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field, field_serializer
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.api import deps
from app.core.exceptions import ApiError, SourceErrors
from app.core.rate_limit import RateLimiter, check_rate_limit
from app.models.models import User, Document, Chunk, as_utc
from app.modules.documents import DocumentModuleService
from app.services.document_service import DocumentService
from app.shared.settings import RuntimeSettingsService

router = APIRouter()

# Загрузка источника — самая дорогая операция раздела: файл принимается
# целиком, а следом идёт извлечение текста и индексация на GPU. Права на неё
# теперь есть у любого зарегистрированного пользователя (внутри своих
# блокнотов), поэтому без ограничения частоты один аккаунт может занять
# очередь индексации целиком.
#
# 30 загрузок за 5 минут: окно длиннее, чем у чата (30/60с), намеренно —
# выбор нескольких файлов в диалоге загружается по одному файлу за запрос,
# и обычный сценарий «перетащил папку с документами» должен пройти одним
# махом. Устойчивая же скорость выходит 6 файлов в минуту, что индексация
# успевает переваривать.
upload_limiter = RateLimiter(requests=30, window=300)

# Страница по умолчанию — как раньше; потолок нужен, чтобы ?limit=100000000
# не выгружал всю таблицу одним запросом.
DEFAULT_PAGE_SIZE = 100
MAX_PAGE_SIZE = 500

# Общее число документов под фильтрами запроса. Заголовком, а не полем в теле:
# тело остаётся массивом DocumentRead, и клиенты, не знающие о пагинации,
# продолжают работать без изменений.
TOTAL_COUNT_HEADER = "X-Total-Count"


def serialize_utc(value: datetime) -> str:
    """Момент времени из БД — строкой с явным UTC.

    В колонке TIMESTAMP WITHOUT TIME ZONE лежит UTC, но без смещения строка
    "2026-07-30T09:15:00" по спецификации JS читается как местное время — в
    Душанбе (UTC+5) дата уезжала на пять часов назад.

    Общая для всех разделов: даты источников, блокнотов, заметок и инсайтов
    показываются рядом на одном экране и обязаны читаться одинаково.
    """
    return as_utc(value).isoformat().replace("+00:00", "Z")


class DocumentRead(BaseModel):
    """Документ в ответах API.

    Отдаём не саму модель: у Document есть path (абсолютный путь на сервере)
    и owner_id — наружу они не нужны.

    status: pending → indexing → indexed | error. Индексация асинхронная,
    поэтому клиент опрашивает GET /sources/ до терминального статуса;
    error_text объясняет, что именно пошло не так, а error_code даёт то же
    объяснение машинным кодом — под него у клиента есть переводы.
    """

    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    size: int
    language: str
    status: str
    notebook_id: int | None = None
    created_at: datetime
    error_text: str | None = None
    error_code: str | None = None
    # Тема документа. Оба поля необязательные и добавлены В КОНЕЦ: порядок
    # существующих полей — часть уже выданного обещания, а null здесь означает
    # честное «темы нет» (модель не обучена, документ ещё не размечен, векторов
    # не нашлось), а не ошибку.
    #
    # Подпись приходит ХРАНИМАЯ, а не собранная по номеру кластера из активной
    # модели: после переобучения номер 3 означает уже другую тему, и сборка на
    # лету переписала бы задним числом то, что пользователь видел вчера.
    #
    # Подписей несколько: устойчивая английская (ключ темы) и переводы на языки
    # интерфейса. null у перевода — «имени на этом языке у назначения нет»
    # (документ размечен моделью без переводов или до появления колонок), и
    # клиент откатывается к topic_label. Переводы НЕ достраиваются здесь по
    # topic_cluster_index из активной модели: у документа, размеченного прошлой
    # версией, тот же номер означает уже другую тему, и рядом с исторической
    # английской подписью встало бы чужое название.
    topic_label: str | None = None
    topic_label_ru: str | None = None
    topic_label_tg: str | None = None
    topic_cluster_index: int | None = None

    @field_serializer("created_at")
    def _serialize_created_at(self, value: datetime) -> str:
        return serialize_utc(value)


class AttachSourcesPayload(BaseModel):
    # Границы диапазона нужны и полям тела: id больше PostgreSQL integer
    # роняет запрос в asyncpg (OverflowError) и отдаёт 500 вместо 404.
    notebook_id: int = Field(ge=1, le=deps.MAX_ID)
    source_ids: list[Annotated[int, Field(ge=1, le=deps.MAX_ID)]]


class AttachSourcesResponse(BaseModel):
    updated_count: int
    documents: list[DocumentRead]


def _owner_filter(user: User) -> int | None:
    """id владельца для фильтрации выборок; None — админ, фильтр не нужен."""
    return None if user.role == "admin" else user.id


# --- Доступ к источникам ------------------------------------------------
#
# Раздел живёт на владении, а не на роли: источники принадлежат блокноту, а
# блокнот — своему владельцу. Любой аутентифицированный пользователь работает
# с источниками внутри собственных блокнотов; чужой блокнот и чужой документ
# отвечают 404 (см. deps.user_owns), документ без владельца — legacy-состояние
# и остаётся админским. Админ по-прежнему видит и правит всё, content_manager
# не теряет ничего: раньше он был ограничен теми же рамками владения.
# Исключение одно — POST /reindex: это обслуживание всего индекса разом,
# а не работа со своими источниками, и остаётся за админом.


@router.post("/upload", response_model=DocumentRead)
async def upload_document(
    request: Request,
    file: UploadFile,
    notebook_id: Optional[int] = Form(default=None, ge=1, le=deps.MAX_ID),
    current_user: User = Depends(deps.get_current_user),
    session: AsyncSession = Depends(deps.get_session),
) -> Any:
    await check_rate_limit(request, upload_limiter)
    # notebook_id=None — источник «вне блокнота»: владельцем становится сам
    # загрузивший, и дальше он виден только ему (и админу).
    #
    # lock=True держит блокнот под SELECT ... FOR SHARE до конца запроса: иначе
    # удаление блокнота успевает пройти между проверкой и INSERT документа, и
    # вставка падает по внешнему ключу document.notebook_id — 500 вместо
    # честного 404 или подождавшего своей очереди удаления.
    await deps.assert_owns_notebook(notebook_id, session, current_user, lock=True)
    return await DocumentModuleService.upload_document(
        session=session,
        file=file,
        notebook_id=notebook_id,
        owner_id=current_user.id,
    )


@router.get(
    "/",
    response_model=List[DocumentRead],
    responses={
        200: {
            "headers": {
                TOTAL_COUNT_HEADER: {
                    "description": (
                        "Общее число документов, доступных вызывающему под "
                        "текущим notebook_id, без учёта skip/limit."
                    ),
                    "schema": {"type": "integer"},
                }
            }
        }
    },
)
async def read_documents(
    response: Response,
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=DEFAULT_PAGE_SIZE, ge=1, le=MAX_PAGE_SIZE),
    notebook_id: int | None = Query(default=None, ge=1, le=deps.MAX_ID),
    # Номер кластера активной модели тем. ge=0, а не ge=1: нулевой кластер
    # существует и является такой же темой, как остальные. Верхняя граница —
    # общая для целых параметров запроса; несуществующий номер даст пустую
    # страницу, а не отказ, потому что «в этой теме ничего нет» — обычный ответ.
    topic: int | None = Query(default=None, ge=0, le=deps.MAX_ID),
    session: AsyncSession = Depends(deps.get_session),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    """Страница источников, свежие сверху.

    Порядок — created_at DESC, id DESC. Общее число записей уходит заголовком
    X-Total-Count: тело осталось массивом, поэтому клиент, не знающий о
    пагинации, ничего не заметил.

    Владение при заданном notebook_id проверяется до выборки — тем же
    assert_owns_notebook, что и в GET /notes/ и GET /insights/. Раньше здесь
    работал только фильтр по owner_id, и чужой либо несуществующий блокнот
    отвечал пустым списком: клиент не мог отличить «блокнота нет» от «в
    блокноте нет источников», а один экран блокнота получал на три запроса
    два разных ответа — 404 по заметкам и 200 по источникам.

    Оракулом это не делает: assert_owns_notebook отвечает одним и тем же
    404 source.notebook_not_found и на чужой блокнот, и на несуществующий
    (см. deps.user_owns), поэтому подтвердить существование чужого блокнота
    перебором по-прежнему нельзя. Пустой массив остаётся ответом ровно на
    один случай — свой блокнот без источников.

    Без notebook_id всё как было: выборка ограничена своими документами
    фильтром по owner_id.

    topic сужает страницу до одной темы, и делает это СЕРВЕР. Отбор на клиенте
    по уже загруженному списку врёт на больших объёмах: клиент долистывает
    ограниченное число страниц, и «источников по теме: 12» при сорока
    настоящих выглядит как правда. Здесь же фильтр стоит рядом с остальными
    условиями, то есть попадает и в X-Total-Count.

    Номер кластера осмыслен только вместе с версией модели, поэтому она
    подставляется здесь же — от АКТИВНОЙ модели. Без обученной модели ни у
    одного документа темы активной версии нет, и честный ответ — пустая
    страница: подставить вместо этого нефильтрованный список значило бы
    показать по запросу «покажи тему N» вообще всё.
    """
    if notebook_id is not None:
        await deps.assert_owns_notebook(notebook_id, session, current_user)

    topic_model_version: int | None = None
    if topic is not None:
        # Локальный импорт: раздел источников не должен тянуть numpy и артефакт
        # модели тем ради параметра, которым пользуются не все запросы.
        from app.modules.topics.service import TopicsService

        model = await TopicsService.active_model(session)
        if model is None:
            response.headers[TOTAL_COUNT_HEADER] = "0"
            return []
        topic_model_version = model.version

    documents, total = await DocumentModuleService.read_documents(
        session=session,
        skip=skip,
        limit=limit,
        notebook_id=notebook_id,
        owner_id=_owner_filter(current_user),
        topic_cluster_index=topic,
        topic_model_version=topic_model_version,
    )
    response.headers[TOTAL_COUNT_HEADER] = str(total)
    return documents


@router.post("/attach", response_model=AttachSourcesResponse)
async def attach_documents(
    payload: AttachSourcesPayload,
    current_user: User = Depends(deps.get_current_user),
    session: AsyncSession = Depends(deps.get_session),
) -> Any:
    if not payload.source_ids:
        raise ApiError(400, SourceErrors.NO_IDS_PROVIDED, "No source ids provided")

    # lock=True — по той же причине, что и в upload: прикрепление проставляет
    # документам notebook_id, и параллельное удаление блокнота между проверкой
    # и UPDATE оставило бы ссылку на исчезнувшую строку (500 по внешнему ключу).
    await deps.assert_owns_notebook(
        payload.notebook_id, session, current_user, lock=True
    )
    documents = await DocumentModuleService.attach_documents_to_notebook(
        session=session,
        notebook_id=payload.notebook_id,
        source_ids=payload.source_ids,
        owner_id=_owner_filter(current_user),
    )
    return AttachSourcesResponse(updated_count=len(documents), documents=documents)


@router.get("/{id}/chunks", response_model=List[Chunk])
async def get_document_chunks(
    document: Document = Depends(deps.get_owned_document),
    session: AsyncSession = Depends(deps.get_session),
) -> Any:
    # Аутентификация и владение — внутри get_owned_document; отдельная
    # зависимость на пользователя здесь была нужна только ради проверки роли.
    return await DocumentModuleService.get_document_chunks(
        session=session, document_id=document.id
    )


@router.delete("/{id}", response_model=DocumentRead)
async def delete_document(
    document: Document = Depends(deps.get_owned_document),
    session: AsyncSession = Depends(deps.get_session),
) -> Any:
    return await DocumentModuleService.delete_document(
        session=session, document_id=document.id
    )


@router.post("/reindex")
async def reindex_all_documents(
    current_user: User = Depends(deps.get_current_active_superuser),
    session: AsyncSession = Depends(deps.get_session),
) -> Any:
    """Перестроить весь индекс и, если получилось, погасить долг за смену модели.

    Флаг reindex_required (см. RuntimeSettingsService) ставится при смене
    embedding_model и до сих пор не снимался ничем: снять его может только
    переиндексация, а она о нём не знала.

    Успехом считается ровно status == "ok". _reindex_all возвращает "partial",
    если хотя бы один документ не переиндексировался (файла нет на диске, текст
    не извлёкся, упала индексация): его чанки удалены из БД и из ChromaDB, а
    новых векторов нет — в активной коллекции документ отсутствует целиком.
    Долг за сменой модели в этом случае не погашен, и гасить флаг нельзя:
    он — единственное, что напоминает админу о неполном индексе, а по ответу
    одного запроса этого потом не восстановить. Пустая база ("No documents to
    reindex") — тоже "ok" и тоже честный успех: устаревших векторов не
    осталось, потому что их нет вовсе.

    Флаг возвращается в ответе, чтобы клиент обновил предупреждение в
    интерфейсе, не перечитывая настройки отдельным запросом.
    """
    result = await DocumentModuleService.reindex_all_documents(session=session)
    if result.get("status") == "ok":
        reindex_required = await RuntimeSettingsService.clear_reindex_required()
    else:
        reindex_required = bool(
            RuntimeSettingsService.get_settings().get("reindex_required")
        )
    result["reindex_required"] = reindex_required
    return result


MIME_MAP = {
    ".pdf": "application/pdf",
    ".txt": "text/plain; charset=utf-8",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}


@router.get("/{id}/preview")
async def preview_document(
    doc: Document = Depends(deps.get_owned_document),
) -> FileResponse:
    if not doc.path or not os.path.exists(doc.path):
        raise ApiError(404, SourceErrors.FILE_MISSING, "File not found on disk")
    ext = os.path.splitext(doc.name or doc.path)[1].lower()
    media_type = MIME_MAP.get(ext, "application/octet-stream")
    return FileResponse(
        doc.path,
        media_type=media_type,
        # Новые документы приходят сюда уже с очищенным name (см.
        # DocumentModuleService.upload_document), но строки, загруженные до
        # этого, хранят имя как прислал клиент — вплоть до
        # `../../../../etc/passwd.txt`. Чистим и здесь, чтобы не заводить
        # миграцию ради заголовка Content-Disposition.
        filename=DocumentService.sanitize_display_name(doc.name),
    )


class ChunkContext(BaseModel):
    chunk_id: int
    text: str
    page: int
    chunk_index: int | None = None
    section: str | None = None
    doc_id: int
    doc_name: str
    highlight: bool = False


@router.get("/{id}/chunk/{chunk_id}/context")
async def get_chunk_context(
    chunk_id: int = Path(..., ge=1, le=deps.MAX_ID),
    neighbors: int = Query(default=2, ge=0, le=5),
    doc: Document = Depends(deps.get_owned_document),
    session: AsyncSession = Depends(deps.get_session),
) -> list[ChunkContext]:
    """Return the target chunk plus neighboring chunks for context."""
    target = await session.get(Chunk, chunk_id)
    if not target or target.doc_id != doc.id:
        raise ApiError(404, SourceErrors.CHUNK_NOT_FOUND, "Chunk not found")

    result = await session.exec(
        select(Chunk)
        .where(Chunk.doc_id == doc.id)
        .order_by(Chunk.page, Chunk.id)
    )
    all_chunks = result.all()

    target_idx = next(
        (i for i, c in enumerate(all_chunks) if c.id == chunk_id), None
    )
    if target_idx is None:
        raise ApiError(
            404, SourceErrors.CHUNK_NOT_FOUND, "Chunk not found in document"
        )

    start = max(0, target_idx - neighbors)
    end = min(len(all_chunks), target_idx + neighbors + 1)

    return [
        ChunkContext(
            chunk_id=c.id,
            text=c.text,
            page=c.page,
            chunk_index=c.chunk_index,
            section=c.section,
            doc_id=doc.id,
            doc_name=doc.name,
            highlight=(c.id == chunk_id),
        )
        for c in all_chunks[start:end]
    ]
