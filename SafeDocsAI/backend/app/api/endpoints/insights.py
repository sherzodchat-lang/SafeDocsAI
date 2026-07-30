from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.api import deps
from app.shared.models import Insight, Notebook, User

router = APIRouter()


class InsightCreate(BaseModel):
    notebook_id: int
    title: str
    body: str = ""
    insight_type: str = "summary"
    evidence_json: str | None = None


class InsightResponse(BaseModel):
    id: int
    notebook_id: int
    note_id: int | None = None
    title: str
    body: str
    insight_type: str
    evidence_json: str | None = None
    created_by: int | None = None
    created_at: datetime
    updated_at: datetime


@router.get("/", response_model=list[InsightResponse])
async def list_insights(
    notebook_id: int | None = Query(default=None),
    current_user: User = Depends(deps.get_current_user),
    session: AsyncSession = Depends(deps.get_session),
) -> Any:
    statement = select(Insight).order_by(Insight.updated_at.desc())
    if notebook_id is not None:
        # Владение проверяем до выборки: иначе по чужому notebook_id
        # из query можно прочитать чужие инсайты.
        await deps.assert_owns_notebook(notebook_id, session, current_user)
        statement = statement.where(Insight.notebook_id == notebook_id)
    elif current_user.role != "admin":
        # Без фильтра по блокноту отдаём только инсайты из своих блокнотов.
        # Источник истины — владелец блокнота: created_by у Insight nullable,
        # а notebook_id объявлен NOT NULL.
        statement = statement.where(
            Insight.notebook_id.in_(
                select(Notebook.id).where(Notebook.owner_id == current_user.id)
            )
        )
    result = await session.exec(statement)
    insights = result.all()
    return [
        InsightResponse(
            id=insight.id,
            notebook_id=insight.notebook_id,
            note_id=insight.note_id,
            title=insight.title,
            body=insight.body,
            insight_type=insight.insight_type,
            evidence_json=insight.evidence_json,
            created_by=insight.created_by,
            created_at=insight.created_at,
            updated_at=insight.updated_at,
        )
        for insight in insights
        if insight.id is not None
    ]


@router.post("/", response_model=InsightResponse)
async def create_insight(
    payload: InsightCreate,
    current_user: User = Depends(deps.get_current_user),
    session: AsyncSession = Depends(deps.get_session),
) -> Any:
    await deps.assert_owns_notebook(payload.notebook_id, session, current_user)
    insight = Insight(
        notebook_id=payload.notebook_id,
        title=payload.title.strip(),
        body=payload.body,
        insight_type=payload.insight_type,
        evidence_json=payload.evidence_json,
        created_by=current_user.id,
    )
    session.add(insight)
    await session.commit()
    await session.refresh(insight)
    return InsightResponse(
        id=insight.id,
        notebook_id=insight.notebook_id,
        note_id=insight.note_id,
        title=insight.title,
        body=insight.body,
        insight_type=insight.insight_type,
        evidence_json=insight.evidence_json,
        created_by=insight.created_by,
        created_at=insight.created_at,
        updated_at=insight.updated_at,
    )
