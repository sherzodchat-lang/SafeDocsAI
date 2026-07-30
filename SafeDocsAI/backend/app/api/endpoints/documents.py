import os
from datetime import datetime
from typing import Any, List, Optional
from fastapi import (
    APIRouter,
    Depends,
    Form,
    Query,
    Response,
    UploadFile,
)
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, field_serializer
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.api import deps
from app.core.exceptions import ApiError, SourceErrors
from app.models.models import User, Document, Chunk, as_utc
from app.modules.documents import DocumentModuleService

router = APIRouter()

# Страница по умолчанию — как раньше; потолок нужен, чтобы ?limit=100000000
# не выгружал всю таблицу одним запросом.
DEFAULT_PAGE_SIZE = 100
MAX_PAGE_SIZE = 500

# Общее число документов под фильтрами запроса. Заголовком, а не полем в теле:
# тело остаётся массивом DocumentRead, и клиенты, не знающие о пагинации,
# продолжают работать без изменений.
TOTAL_COUNT_HEADER = "X-Total-Count"


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

    @field_serializer("created_at")
    def _serialize_created_at(self, value: datetime) -> str:
        # В колонке TIMESTAMP WITHOUT TIME ZONE лежит UTC, но без смещения
        # строка "2026-07-30T09:15:00" по спецификации JS читается как местное
        # время — в Душанбе (UTC+5) дата уезжала на пять часов назад.
        return as_utc(value).isoformat().replace("+00:00", "Z")


class AttachSourcesPayload(BaseModel):
    notebook_id: int
    source_ids: list[int]


class AttachSourcesResponse(BaseModel):
    updated_count: int
    documents: list[DocumentRead]


def _owner_filter(user: User) -> int | None:
    """id владельца для фильтрации выборок; None — админ, фильтр не нужен."""
    return None if user.role == "admin" else user.id


@router.post("/upload", response_model=DocumentRead)
async def upload_document(
    file: UploadFile,
    notebook_id: Optional[int] = Form(default=None),
    current_user: User = Depends(deps.get_current_content_manager_or_admin),
    session: AsyncSession = Depends(deps.get_session),
) -> Any:
    await deps.assert_owns_notebook(notebook_id, session, current_user)
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
    notebook_id: int | None = Query(default=None),
    session: AsyncSession = Depends(deps.get_session),
    current_user: User = Depends(deps.get_current_content_manager_or_admin),
) -> Any:
    """Страница источников, свежие сверху.

    Порядок — created_at DESC, id DESC. Общее число записей уходит заголовком
    X-Total-Count: тело осталось массивом, поэтому клиент, не знающий о
    пагинации, ничего не заметил.
    """
    documents, total = await DocumentModuleService.read_documents(
        session=session,
        skip=skip,
        limit=limit,
        notebook_id=notebook_id,
        owner_id=_owner_filter(current_user),
    )
    response.headers[TOTAL_COUNT_HEADER] = str(total)
    return documents


@router.post("/attach", response_model=AttachSourcesResponse)
async def attach_documents(
    payload: AttachSourcesPayload,
    current_user: User = Depends(deps.get_current_content_manager_or_admin),
    session: AsyncSession = Depends(deps.get_session),
) -> Any:
    if not payload.source_ids:
        raise ApiError(400, SourceErrors.NO_IDS_PROVIDED, "No source ids provided")

    await deps.assert_owns_notebook(payload.notebook_id, session, current_user)
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
    current_user: User = Depends(deps.get_current_content_manager_or_admin),
) -> Any:
    return await DocumentModuleService.get_document_chunks(
        session=session, document_id=document.id
    )


@router.delete("/{id}", response_model=DocumentRead)
async def delete_document(
    document: Document = Depends(deps.get_owned_document),
    session: AsyncSession = Depends(deps.get_session),
    current_user: User = Depends(deps.get_current_content_manager_or_admin),
) -> Any:
    return await DocumentModuleService.delete_document(
        session=session, document_id=document.id
    )


@router.post("/reindex")
async def reindex_all_documents(
    current_user: User = Depends(deps.get_current_active_superuser),
    session: AsyncSession = Depends(deps.get_session),
) -> Any:
    return await DocumentModuleService.reindex_all_documents(session=session)


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
        filename=doc.name,
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
    chunk_id: int,
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
