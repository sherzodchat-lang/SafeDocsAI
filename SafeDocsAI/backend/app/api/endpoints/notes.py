from datetime import datetime
from typing import Any

import logging

from fastapi import APIRouter, Depends, Path, Query, Request, Response
from pydantic import BaseModel, Field, field_serializer, field_validator
from sqlalchemy.orm.exc import StaleDataError
from sqlmodel import func, select, update
from sqlmodel.ext.asyncio.session import AsyncSession

from app.api import deps
from app.api.endpoints.documents import (
    MAX_PAGE_SIZE,
    TOTAL_COUNT_HEADER,
    serialize_utc,
)
from app.core.exceptions import ApiError, SourceErrors
from app.core.rate_limit import RateLimiter, check_rate_limit
from app.shared.models import Insight, Note, Notebook, User, utcnow

router = APIRouter()
logger = logging.getLogger(__name__)

# Страница по умолчанию равна потолку намеренно: клиент параметров не
# передаёт, и любой меньший размер молча урезал бы уже отдаваемые списки.
# Потолок особенно нужен здесь: админский вызов без notebook_id иначе
# поднимает в память заметки всех пользователей вместе с телами.
DEFAULT_PAGE_SIZE = MAX_PAGE_SIZE

# Заметка создаётся вручную и стоит одну строку в БД — несравнимо дешевле
# загрузки источника (30 за 5 минут, documents.upload_limiter), где следом идут
# извлечение текста и индексация. Лимит поэтому свободный и рассчитан только на
# скрипт: 60 заметок в минуту — это по одной в секунду без единой паузы, чего
# набором текста не достичь.
create_limiter = RateLimiter(requests=60, window=60)

# Допустимые значения Note.status. В модели это обычный str с default="active"
# (shared/models/entities.py), проверять значение больше негде: в БД нет ни
# enum, ни CHECK, и «archved» с опечаткой осел бы в колонке навсегда.
NOTE_STATUS_ACTIVE = "active"
NOTE_STATUS_ARCHIVED = "archived"
NOTE_STATUSES = (NOTE_STATUS_ACTIVE, NOTE_STATUS_ARCHIVED)


class NoteCreate(BaseModel):
    notebook_id: int = Field(ge=1, le=deps.MAX_ID)
    # strip() без проверки результата принимал "   " как заголовок, а без
    # верхней границы заголовок длиннее предела btree-индекса ронял INSERT.
    title: deps.TitleStr
    body: deps.BodyStr = ""
    kind: deps.KindStr = "manual"


class NoteUpdate(BaseModel):
    """Частичное обновление: применяются только присланные поля.

    Различать «поле не пришло» и «пришло null» обязательно — иначе PATCH с
    одним title затирал бы тело заметки. Отличаем по model_fields_set:
    default=None здесь означает ровно «поле не пришло».

    Колонки title/body/kind/status объявлены NOT NULL, поэтому явный null
    отклоняется, а не трактуется как очистка. notebook_id менять нельзя:
    перенос заметки в другой блокнот — отдельная операция со своей проверкой
    владения принимающим блокнотом, и молча делать её частью переименования
    неправильно.
    """

    title: deps.TitleStr | None = None
    body: deps.BodyStr | None = None
    kind: deps.KindStr | None = None
    status: deps.KindStr | None = None

    @field_validator("title", "body", "kind", "status")
    @classmethod
    def _reject_explicit_null(cls, value: str | None) -> str | None:
        # Для непришедшего поля валидатор не вызывается (значение по умолчанию
        # не валидируется), поэтому сюда попадает только явный null из тела.
        if value is None:
            raise ValueError("must not be null")
        return value


class NoteResponse(BaseModel):
    id: int
    notebook_id: int
    title: str
    body: str
    kind: str
    status: str
    created_by: int | None = None
    created_at: datetime
    updated_at: datetime

    @field_serializer("created_at", "updated_at")
    def _serialize_datetime(self, value: datetime) -> str:
        return serialize_utc(value)


def _note_response(note: Note) -> NoteResponse:
    """Одна форма ответа на все эндпоинты раздела.

    Собирается вручную, а не from_attributes: у Note может появиться поле,
    которое наружу не нужно, и тогда оно молча утечёт во все ответы разом.
    """
    return NoteResponse(
        id=note.id,
        notebook_id=note.notebook_id,
        title=note.title,
        body=note.body,
        kind=note.kind,
        status=note.status,
        created_by=note.created_by,
        created_at=note.created_at,
        updated_at=note.updated_at,
    )


async def get_owned_note(
    note_id: int = Path(..., ge=1, le=deps.MAX_ID),
    session: AsyncSession = Depends(deps.get_session),
    current_user: User = Depends(deps.get_current_user),
) -> Note:
    """Заметка вызывающего или 404.

    Владение считается по блокноту — тем же правилом deps.user_owns, что и в
    GET и POST этого файла: created_by у Note nullable, а notebook_id объявлен
    NOT NULL, поэтому источник истины ровно один.

    Код ошибки одинаков и для несуществующей заметки, и для чужой: разные коды
    на один и тот же 404 позволяли бы перебором узнать, какие id существуют.
    """
    note = await session.get(Note, note_id)
    notebook = (
        await session.get(Notebook, note.notebook_id) if note is not None else None
    )
    if notebook is None or not deps.user_owns(notebook.owner_id, current_user):
        raise ApiError(404, SourceErrors.NOTE_NOT_FOUND, "Note not found")
    return note


@router.get(
    "/",
    response_model=list[NoteResponse],
    responses={
        200: {
            "headers": {
                TOTAL_COUNT_HEADER: {
                    "description": (
                        "Общее число заметок, доступных вызывающему под "
                        "текущим notebook_id, без учёта skip/limit."
                    ),
                    "schema": {"type": "integer"},
                }
            }
        }
    },
)
async def list_notes(
    response: Response,
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=DEFAULT_PAGE_SIZE, ge=1, le=MAX_PAGE_SIZE),
    notebook_id: int | None = Query(default=None, ge=1, le=deps.MAX_ID),
    status: str | None = Query(default=None, max_length=deps.KIND_MAX_LENGTH),
    current_user: User = Depends(deps.get_current_user),
    session: AsyncSession = Depends(deps.get_session),
) -> Any:
    """Страница заметок, свежие сверху.

    Тело осталось массивом, общее число уходит заголовком X-Total-Count —
    как у GET /sources/.

    status — необязательный фильтр (active | archived). По умолчанию отдаются
    заметки любого статуса: без этого архивация одним PATCH выкинула бы
    заметки из уже работающих списков клиента.
    """
    if notebook_id is not None:
        # Владение проверяем до выборки: иначе по чужому notebook_id
        # из query можно прочитать чужие заметки.
        await deps.assert_owns_notebook(notebook_id, session, current_user)
    if status is not None and status not in NOTE_STATUSES:
        # Молча отдать пустой список нельзя: опечатка в фильтре выглядела бы
        # как «заметок нет».
        raise ApiError(
            400,
            SourceErrors.INVALID_NOTE_STATUS,
            f"Unsupported note status, expected one of: {', '.join(NOTE_STATUSES)}",
        )

    def _filtered(statement):
        if status is not None:
            statement = statement.where(Note.status == status)
        if notebook_id is not None:
            return statement.where(Note.notebook_id == notebook_id)
        if current_user.role != "admin":
            # Без фильтра по блокноту отдаём только заметки из своих блокнотов.
            # Источник истины — владелец блокнота: created_by у Note nullable,
            # а notebook_id объявлен NOT NULL.
            return statement.where(
                Note.notebook_id.in_(
                    select(Notebook.id).where(Notebook.owner_id == current_user.id)
                )
            )
        return statement

    # id вторым ключом сортировки: без него заметки с одинаковым updated_at
    # прыгают между соседними страницами.
    page = _filtered(select(Note)).order_by(Note.updated_at.desc(), Note.id.desc())
    result = await session.exec(page.offset(skip).limit(limit))
    notes = result.all()

    total_result = await session.exec(_filtered(select(func.count()).select_from(Note)))
    response.headers[TOTAL_COUNT_HEADER] = str(int(total_result.first() or 0))
    return [_note_response(note) for note in notes if note.id is not None]


@router.post("/", response_model=NoteResponse)
async def create_note(
    request: Request,
    payload: NoteCreate,
    current_user: User = Depends(deps.get_current_user),
    session: AsyncSession = Depends(deps.get_session),
) -> Any:
    await check_rate_limit(request, create_limiter)
    await deps.assert_owns_notebook(payload.notebook_id, session, current_user)
    note = Note(
        notebook_id=payload.notebook_id,
        # Подрезка сделана валидацией схемы — там же отсеян заголовок из
        # одних пробелов, который strip() здесь молча делал пустым.
        title=payload.title,
        body=payload.body,
        kind=payload.kind,
        created_by=current_user.id,
    )
    session.add(note)
    await session.commit()
    await session.refresh(note)
    return _note_response(note)


@router.patch("/{note_id}", response_model=NoteResponse)
async def update_note(
    payload: NoteUpdate,
    note: Note = Depends(get_owned_note),
    session: AsyncSession = Depends(deps.get_session),
) -> Any:
    """Частичное обновление заметки: title, body, kind, status.

    status оставлен управляемым, а не убран из ответа: колонка существует в
    модели с самого начала, уже отдаётся клиенту и в задуманном виде означает
    архивацию. Отдать поле и не дать его менять — это и есть текущая поломка;
    убрать поле из ответа значит выкинуть из API состояние, которое в БД всё
    равно хранится. Допустимые значения проверяем здесь (NOTE_STATUSES): в БД
    это обычный текст без enum и CHECK.
    """
    fields = payload.model_fields_set
    if not fields:
        # Пустое тело — ошибка клиента: 200 подтвердил бы правку, которой нет.
        raise ApiError(400, SourceErrors.NOTHING_TO_UPDATE, "No fields to update")

    if "status" in fields and payload.status not in NOTE_STATUSES:
        raise ApiError(
            400,
            SourceErrors.INVALID_NOTE_STATUS,
            f"Unsupported note status, expected one of: {', '.join(NOTE_STATUSES)}",
        )
    for field in ("title", "body", "kind", "status"):
        if field in fields:
            # Подрезка и границы длины уже сделаны валидацией схемы — теми же
            # типами из deps, что и на создании.
            setattr(note, field, getattr(payload, field))
    # updated_at проставляем руками: в модели у колонки только default_factory,
    # onupdate там нет, а списки заметок сортируются именно по ней.
    note.updated_at = utcnow()

    note_id = note.id
    session.add(note)
    try:
        await session.commit()
    except StaleDataError as exc:
        # Заметку удалили (сама по себе или вместе с блокнотом) между выборкой
        # и commit: UPDATE не нашёл строки. Это не 500, а тот же 404.
        await session.rollback()
        logger.info("Concurrent delete of note %s: %s", note_id, exc)
        raise ApiError(404, SourceErrors.NOTE_NOT_FOUND, "Note not found") from exc
    await session.refresh(note)
    return _note_response(note)


@router.delete("/{note_id}")
async def delete_note(
    note: Note = Depends(get_owned_note),
    session: AsyncSession = Depends(deps.get_session),
) -> dict[str, Any]:
    note_id = note.id

    # Инсайты ссылаются на заметку внешним ключом insight.note_id без
    # ON DELETE, поэтому DELETE по заметке со связанными инсайтами упал бы
    # IntegrityError (500). Инсайт принадлежит блокноту и переживает свою
    # заметку, так что связь просто снимаем — удалять его молча значило бы
    # уносить данные, которых пользователь не выбирал.
    await session.exec(
        update(Insight).where(Insight.note_id == note_id).values(note_id=None)
    )
    await session.delete(note)
    try:
        await session.commit()
    except StaleDataError as exc:
        # Тот же двойной клик, что и на блокнотах: второй запрос обнаруживает
        # на commit, что удалять уже нечего.
        await session.rollback()
        logger.info("Concurrent delete of note %s: %s", note_id, exc)
        raise ApiError(404, SourceErrors.NOTE_NOT_FOUND, "Note not found") from exc
    return {"detail": "Note deleted", "id": note_id}
