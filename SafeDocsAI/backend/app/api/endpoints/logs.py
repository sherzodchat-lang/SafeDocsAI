import csv
import io
from datetime import datetime, date, time
from typing import List, Any, Literal
from fastapi import APIRouter, Depends, Path, Query
from fastapi.responses import StreamingResponse
from sqlmodel import select, desc
from sqlmodel.ext.asyncio.session import AsyncSession
from pydantic import BaseModel, ConfigDict, field_serializer

from app.api import deps
from app.api.endpoints.documents import MAX_PAGE_SIZE, serialize_utc
from app.core.database import get_session
from app.core.exceptions import ApiError, LogErrors
from app.shared.models import Log, User

router = APIRouter()


class RatingUpdate(BaseModel):
    rating: Literal["up", "down"]


class LogRead(BaseModel):
    """Запись журнала в ответе GET /logs/.

    Появилась ради created_at: сырая модель Log отдавала его наивным, а
    строка без смещения по спецификации JS читается как местное время — в
    Душанбе (UTC+5) дата в журнале уезжала на пять часов. Та же поломка уже
    исправлена в блокнотах, заметках, инсайтах и источниках, и чинится она
    той же общей serialize_utc: даты этих разделов админ видит рядом и они
    обязаны читаться одинаково.

    Набор полей — ровно тот, что раньше отдавала модель Log: эндпоинт
    админский, ничего лишнего в модели нет, а сузить ответ значило бы менять
    контракт заодно с починкой дат.
    """

    model_config = ConfigDict(from_attributes=True)

    id: int
    question: str
    answer: str
    sources: str | None = None
    time_ms: int
    rating: str | None = None
    user_id: int | None = None
    notebook_id: int | None = None
    domain_profile: str | None = None
    created_at: datetime

    @field_serializer("created_at")
    def _serialize_created_at(self, value: datetime) -> str:
        return serialize_utc(value)


class RatingResponse(BaseModel):
    # Узкий ответ: полная модель Log вернула бы question/answer/sources,
    # то есть содержимое чужого запроса при переборе log_id.
    id: int
    rating: str


@router.get("/", response_model=List[LogRead])
async def read_logs(
    # Значения по умолчанию прежние; границы нужны, чтобы ?limit=100000000
    # не выгружал весь журнал, а отрицательный offset не ронял запрос.
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=100, ge=1, le=MAX_PAGE_SIZE),
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
    rating_in: RatingUpdate,
    log_id: int = Path(..., ge=1, le=deps.MAX_ID),
    current_user: User = Depends(deps.get_current_user),
    session: AsyncSession = Depends(get_session),
) -> Any:
    log = await session.get(Log, log_id)
    # log.user_id IS NULL — не баг, а «у события нет автора»: журнал хранит
    # запись о событии, и такое состояние бывает правдой (legacy-строки,
    # системное действие). Переписывать их на админа значило бы заставить
    # журнал врать, удалять — потерять историю, ради которой он и ведётся,
    # поэтому колонка остаётся nullable. Доступ к ним даёт user_owns: ничью
    # запись видит только админ. Новые записи безавторскими не бывают — их
    # автора требует chat/service.py, require_log_author.
    if not log or not deps.user_owns(log.user_id, current_user):
        # ApiError, а не голый HTTPException: error_code навешивается только
        # на него, и без кода фронтенд не может перевести сообщение.
        raise ApiError(404, LogErrors.NOT_FOUND, "Log not found")

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
                # Тот же serialize_utc, что и в JSON-ответе: колонка в БД
                # naive, и без явного UTC та же выгрузка читалась бы как
                # местное время. Два пути одного журнала не должны отдавать
                # один момент времени по-разному.
                serialize_utc(log.created_at) if log.created_at else "",
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
