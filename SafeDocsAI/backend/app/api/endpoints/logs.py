import csv
import io
from datetime import datetime, date, time
from typing import List, Any, Literal
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlmodel import select, desc
from sqlmodel.ext.asyncio.session import AsyncSession
from pydantic import BaseModel

from app.api import deps
from app.core.database import get_session
from app.shared.models import Log, User

router = APIRouter()


class RatingUpdate(BaseModel):
    rating: Literal["up", "down"]


class RatingResponse(BaseModel):
    # Узкий ответ: полная модель Log вернула бы question/answer/sources,
    # то есть содержимое чужого запроса при переборе log_id.
    id: int
    rating: str


@router.get("/", response_model=List[Log])
async def read_logs(
    skip: int = 0,
    limit: int = 100,
    start_date: date | None = None,
    end_date: date | None = None,
    current_user: User = Depends(deps.get_current_active_superuser),
    session: AsyncSession = Depends(get_session),
) -> Any:
    statement = select(Log)
    if start_date:
        statement = statement.where(
            Log.created_at >= datetime.combine(start_date, time.min)
        )
    if end_date:
        statement = statement.where(
            Log.created_at <= datetime.combine(end_date, time.max)
        )
    statement = statement.order_by(desc(Log.created_at)).offset(skip).limit(limit)

    result = await session.exec(statement)
    return result.all()


@router.post("/{log_id}/rating", response_model=RatingResponse)
async def rate_log(
    log_id: int,
    rating_in: RatingUpdate,
    current_user: User = Depends(deps.get_current_user),
    session: AsyncSession = Depends(get_session),
) -> Any:
    log = await session.get(Log, log_id)
    if not log or not deps.user_owns(log.user_id, current_user):
        raise HTTPException(status_code=404, detail="Log not found")

    log.rating = rating_in.rating
    session.add(log)
    await session.commit()
    await session.refresh(log)
    return RatingResponse(id=log.id, rating=log.rating)


@router.get("/export")
async def export_logs(
    start_date: date | None = None,
    end_date: date | None = None,
    current_user: User = Depends(deps.get_current_active_superuser),
    session: AsyncSession = Depends(get_session),
) -> StreamingResponse:
    """Export logs to CSV file."""
    statement = select(Log)
    if start_date:
        statement = statement.where(
            Log.created_at >= datetime.combine(start_date, time.min)
        )
    if end_date:
        statement = statement.where(
            Log.created_at <= datetime.combine(end_date, time.max)
        )
    statement = statement.order_by(desc(Log.created_at))

    result = await session.exec(statement)
    logs = result.all()

    # Create CSV in memory
    output = io.StringIO()
    writer = csv.writer(output, quoting=csv.QUOTE_ALL)

    # Write header
    writer.writerow(
        [
            "ID",
            "Вопрос",
            "Ответ",
            "Источники",
            "Время (мс)",
            "Отзыв",
            "ID пользователя",
            "Создано",
        ]
    )

    # Write data
    for log in logs:
        writer.writerow(
            [
                log.id,
                log.question,
                log.answer,
                log.sources or "",
                log.time_ms,
                log.rating or "",
                log.user_id or "",
                log.created_at.isoformat() if log.created_at else "",
            ]
        )

    output.seek(0)

    # Generate filename with current date
    filename = f"logs_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
