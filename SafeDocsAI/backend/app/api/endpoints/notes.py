from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.api import deps
from app.shared.models import Note, Notebook, User

router = APIRouter()


class NoteCreate(BaseModel):
    notebook_id: int
    title: str
    body: str = ""
    kind: str = "manual"


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


@router.get("/", response_model=list[NoteResponse])
async def list_notes(
    notebook_id: int | None = Query(default=None),
    current_user: User = Depends(deps.get_current_user),
    session: AsyncSession = Depends(deps.get_session),
) -> Any:
    statement = select(Note).order_by(Note.updated_at.desc())
    if notebook_id is not None:
        # Владение проверяем до выборки: иначе по чужому notebook_id
        # из query можно прочитать чужие заметки.
        await deps.assert_owns_notebook(notebook_id, session, current_user)
        statement = statement.where(Note.notebook_id == notebook_id)
    elif current_user.role != "admin":
        # Без фильтра по блокноту отдаём только заметки из своих блокнотов.
        # Источник истины — владелец блокнота: created_by у Note nullable,
        # а notebook_id объявлен NOT NULL.
        statement = statement.where(
            Note.notebook_id.in_(
                select(Notebook.id).where(Notebook.owner_id == current_user.id)
            )
        )
    result = await session.exec(statement)
    notes = result.all()
    return [
        NoteResponse(
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
        for note in notes
        if note.id is not None
    ]


@router.post("/", response_model=NoteResponse)
async def create_note(
    payload: NoteCreate,
    current_user: User = Depends(deps.get_current_user),
    session: AsyncSession = Depends(deps.get_session),
) -> Any:
    await deps.assert_owns_notebook(payload.notebook_id, session, current_user)
    note = Note(
        notebook_id=payload.notebook_id,
        title=payload.title.strip(),
        body=payload.body,
        kind=payload.kind,
        created_by=current_user.id,
    )
    session.add(note)
    await session.commit()
    await session.refresh(note)
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
