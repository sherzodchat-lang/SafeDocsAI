from datetime import datetime
import logging
import os
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.api import deps
from app.domain_profiles import list_domain_profiles
from app.shared.models import Chunk, Document, Insight, Job, Log, Note, Notebook, User

router = APIRouter()
logger = logging.getLogger(__name__)


class NotebookCreate(BaseModel):
    name: str
    description: str | None = None
    domain_profile: str = "general"


class NotebookResponse(BaseModel):
    id: int
    name: str
    description: str | None = None
    domain_profile: str
    owner_id: int | None = None
    created_at: datetime


@router.get("/", response_model=list[NotebookResponse])
async def list_notebooks(
    current_user: User = Depends(deps.get_current_user),
    session: AsyncSession = Depends(deps.get_session),
) -> Any:
    statement = select(Notebook)
    # Фильтруем в SQL, а не после выборки: иначе чужие блокноты всё равно
    # попадают в память и легко «протекают» при следующей правке.
    if current_user.role != "admin":
        statement = statement.where(Notebook.owner_id == current_user.id)
    result = await session.exec(statement.order_by(Notebook.created_at.desc()))
    notebooks = result.all()
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
    payload: NotebookCreate,
    current_user: User = Depends(deps.get_current_user),
    session: AsyncSession = Depends(deps.get_session),
) -> Any:
    if payload.domain_profile not in list_domain_profiles():
        raise HTTPException(status_code=400, detail="Unsupported domain profile")

    notebook = Notebook(
        name=payload.name.strip(),
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


@router.delete("/{notebook_id}")
async def delete_notebook(
    notebook: Notebook = Depends(deps.get_owned_notebook),
    session: AsyncSession = Depends(deps.get_session),
) -> dict[str, Any]:
    notebook_id = notebook.id

    # 1. Collect documents belonging to this notebook
    docs_result = await session.exec(
        select(Document).where(Document.notebook_id == notebook_id)
    )
    docs = docs_result.all()
    doc_ids = [d.id for d in docs if d.id is not None]

    # 2. Delete chunk embeddings from ChromaDB
    if doc_ids:
        chunks_result = await session.exec(
            select(Chunk).where(Chunk.doc_id.in_(doc_ids))
        )
        chunks = chunks_result.all()
        chunk_ids = [str(c.id) for c in chunks if c.id is not None]
        if chunk_ids:
            try:
                from app.modules.rag.service import RAGService
                RAGService().delete_documents(chunk_ids)
            except Exception as exc:
                logger.warning("ChromaDB cleanup failed for notebook %s: %s", notebook_id, exc)

        # 3. Delete chunks from DB
        for chunk in chunks:
            await session.delete(chunk)

    # 4. Delete document files from disk + DB
    for doc in docs:
        if doc.path and os.path.exists(doc.path):
            try:
                os.remove(doc.path)
            except OSError:
                logger.warning("Could not remove file %s", doc.path)
        await session.delete(doc)

    # 5. Delete related entities: logs, notes, insights, jobs
    for model_cls, fk in [
        (Log, Log.notebook_id),
        (Note, Note.notebook_id),
        (Insight, Insight.notebook_id),
        (Job, Job.notebook_id),
    ]:
        result = await session.exec(select(model_cls).where(fk == notebook_id))
        for row in result.all():
            await session.delete(row)

    # 6. Delete notebook itself
    await session.delete(notebook)
    await session.commit()

    return {"detail": "Notebook deleted", "id": notebook_id}
