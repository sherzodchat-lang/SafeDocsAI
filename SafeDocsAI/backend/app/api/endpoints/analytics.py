from collections import Counter
from typing import Any, List

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlmodel import select

from app.api import deps
from app.core.database import get_session
from app.shared.models import Log, User

router = APIRouter()


class TopQuestion(BaseModel):
    question: str
    count: int


class AnalyticsResponse(BaseModel):
    total_requests: int
    no_data_count: int
    no_data_ratio: float
    avg_response_time_ms: float
    positive_feedback: int
    negative_feedback: int
    top_questions: List[TopQuestion]


@router.get("/", response_model=AnalyticsResponse)
async def get_analytics(
    current_user: User = Depends(deps.get_current_active_superuser),
    session: AsyncSession = Depends(get_session),
) -> Any:
    result = await session.exec(select(Log))
    logs = result.all()

    total_requests = len(logs)

    no_data_count = sum(
        1
        for item in logs
        if item.answer
        and (
            "Маълумот дар база мавҷуд нест" in item.answer
            or "Ответ не найден в базе" in item.answer
        )
    )

    no_data_ratio = (no_data_count / total_requests) if total_requests else 0.0
    avg_response_time_ms = (
        sum((item.time_ms or 0) for item in logs) / total_requests
        if total_requests
        else 0.0
    )

    positive_feedback = sum(1 for item in logs if item.rating == "up")
    negative_feedback = sum(1 for item in logs if item.rating == "down")

    question_counter = Counter(
        (item.question or "").strip() for item in logs if item.question
    )
    top_questions = [
        TopQuestion(question=question, count=count)
        for question, count in question_counter.most_common(5)
    ]

    return AnalyticsResponse(
        total_requests=total_requests,
        no_data_count=no_data_count,
        no_data_ratio=no_data_ratio,
        avg_response_time_ms=avg_response_time_ms,
        positive_feedback=positive_feedback,
        negative_feedback=negative_feedback,
        top_questions=top_questions,
    )
