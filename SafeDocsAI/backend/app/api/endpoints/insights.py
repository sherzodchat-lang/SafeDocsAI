from datetime import datetime
from typing import Any

import logging

from fastapi import APIRouter, Depends, Path, Query, Request, Response
from pydantic import BaseModel, Field, field_serializer, field_validator
from sqlalchemy.orm.exc import StaleDataError
from sqlmodel import func, select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.api import deps
from app.api.endpoints.documents import (
    MAX_PAGE_SIZE,
    TOTAL_COUNT_HEADER,
    serialize_utc,
)
from app.core.exceptions import ApiError, SourceErrors
from app.core.rate_limit import RateLimiter, check_rate_limit
from app.shared.models import Insight, Notebook, User, utcnow

router = APIRouter()
logger = logging.getLogger(__name__)

# Страница по умолчанию равна потолку намеренно: клиент параметров не
# передаёт, и любой меньший размер молча урезал бы уже отдаваемые списки.
# Потолок закрывает выгрузку всей таблицы одним запросом.
DEFAULT_PAGE_SIZE = MAX_PAGE_SIZE

# Инсайт, как и заметка, стоит одну строку в БД — несравнимо дешевле загрузки
# источника (30 за 5 минут, documents.upload_limiter) с её извлечением текста и
# индексацией. Лимит поэтому такой же свободный, как у заметок: 60 в минуту —
# это по одному в секунду без пауз, чего ручной работой не достичь.
create_limiter = RateLimiter(requests=60, window=60)


class InsightCreate(BaseModel):
    notebook_id: int = Field(ge=1, le=deps.MAX_ID)
    # strip() без проверки результата принимал "   " как заголовок, а без
    # верхней границы заголовок длиннее предела btree-индекса ронял INSERT.
    title: deps.TitleStr
    body: deps.BodyStr = ""
    insight_type: deps.KindStr = "summary"
    evidence_json: deps.BodyStr | None = None


class InsightUpdate(BaseModel):
    """Частичное обновление: применяются только присланные поля.

    «Поле не пришло» и «пришло null» различаются по model_fields_set, иначе
    PATCH с одним title затирал бы тело инсайта. Колонки title/body/
    insight_type объявлены NOT NULL и явный null не принимают, а
    evidence_json nullable — для него null означает «убрать обоснование».

    notebook_id и note_id не меняются: перенос инсайта в другой блокнот
    требует отдельной проверки владения принимающим блокнотом и не должен
    быть побочным эффектом правки заголовка.
    """

    title: deps.TitleStr | None = None
    body: deps.BodyStr | None = None
    insight_type: deps.KindStr | None = None
    evidence_json: deps.BodyStr | None = None

    @field_validator("title", "body", "insight_type")
    @classmethod
    def _reject_explicit_null(cls, value: str | None) -> str | None:
        # Для непришедшего поля валидатор не вызывается, поэтому сюда попадает
        # только явный null из тела запроса.
        if value is None:
            raise ValueError("must not be null")
        return value


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

    @field_serializer("created_at", "updated_at")
    def _serialize_datetime(self, value: datetime) -> str:
        return serialize_utc(value)


def _insight_response(insight: Insight) -> InsightResponse:
    """Одна форма ответа на все эндпоинты раздела.

    Собирается вручную, а не from_attributes: новое поле модели иначе молча
    утекло бы наружу во всех ответах разом.
    """
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


async def get_owned_insight(
    insight_id: int = Path(..., ge=1, le=deps.MAX_ID),
    session: AsyncSession = Depends(deps.get_session),
    current_user: User = Depends(deps.get_current_user),
) -> Insight:
    """Инсайт вызывающего или 404.

    Владение считается по блокноту — тем же правилом deps.user_owns, что и в
    GET и POST этого файла: created_by у Insight nullable, а notebook_id
    объявлен NOT NULL.

    Код ошибки одинаков для несуществующего и для чужого инсайта: разные коды
    на один и тот же 404 позволяли бы перебором узнать существующие id.
    """
    insight = await session.get(Insight, insight_id)
    notebook = (
        await session.get(Notebook, insight.notebook_id)
        if insight is not None
        else None
    )
    if notebook is None or not deps.user_owns(notebook.owner_id, current_user):
        raise ApiError(404, SourceErrors.INSIGHT_NOT_FOUND, "Insight not found")
    return insight


@router.get(
    "/",
    response_model=list[InsightResponse],
    responses={
        200: {
            "headers": {
                TOTAL_COUNT_HEADER: {
                    "description": (
                        "Общее число инсайтов, доступных вызывающему под "
                        "текущим notebook_id, без учёта skip/limit."
                    ),
                    "schema": {"type": "integer"},
                }
            }
        }
    },
)
async def list_insights(
    response: Response,
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=DEFAULT_PAGE_SIZE, ge=1, le=MAX_PAGE_SIZE),
    notebook_id: int | None = Query(default=None, ge=1, le=deps.MAX_ID),
    current_user: User = Depends(deps.get_current_user),
    session: AsyncSession = Depends(deps.get_session),
) -> Any:
    """Страница инсайтов, свежие сверху.

    Тело осталось массивом, общее число уходит заголовком X-Total-Count —
    как у GET /sources/.
    """
    if notebook_id is not None:
        # Владение проверяем до выборки: иначе по чужому notebook_id
        # из query можно прочитать чужие инсайты.
        await deps.assert_owns_notebook(notebook_id, session, current_user)

    def _filtered(statement):
        if notebook_id is not None:
            return statement.where(Insight.notebook_id == notebook_id)
        if current_user.role != "admin":
            # Без фильтра по блокноту отдаём только инсайты из своих блокнотов.
            # Источник истины — владелец блокнота: created_by у Insight
            # nullable, а notebook_id объявлен NOT NULL.
            return statement.where(
                Insight.notebook_id.in_(
                    select(Notebook.id).where(Notebook.owner_id == current_user.id)
                )
            )
        return statement

    # id вторым ключом сортировки: без него инсайты с одинаковым updated_at
    # прыгают между соседними страницами.
    page = _filtered(select(Insight)).order_by(
        Insight.updated_at.desc(), Insight.id.desc()
    )
    result = await session.exec(page.offset(skip).limit(limit))
    insights = result.all()

    total_result = await session.exec(
        _filtered(select(func.count()).select_from(Insight))
    )
    response.headers[TOTAL_COUNT_HEADER] = str(int(total_result.first() or 0))
    return [
        _insight_response(insight) for insight in insights if insight.id is not None
    ]


@router.post("/", response_model=InsightResponse)
async def create_insight(
    request: Request,
    payload: InsightCreate,
    current_user: User = Depends(deps.get_current_user),
    session: AsyncSession = Depends(deps.get_session),
) -> Any:
    await check_rate_limit(request, create_limiter)
    await deps.assert_owns_notebook(payload.notebook_id, session, current_user)
    insight = Insight(
        notebook_id=payload.notebook_id,
        # Подрезка сделана валидацией схемы — там же отсеян заголовок из
        # одних пробелов, который strip() здесь молча делал пустым.
        title=payload.title,
        body=payload.body,
        insight_type=payload.insight_type,
        evidence_json=payload.evidence_json,
        created_by=current_user.id,
    )
    session.add(insight)
    await session.commit()
    await session.refresh(insight)
    return _insight_response(insight)


@router.patch("/{insight_id}", response_model=InsightResponse)
async def update_insight(
    payload: InsightUpdate,
    insight: Insight = Depends(get_owned_insight),
    session: AsyncSession = Depends(deps.get_session),
) -> Any:
    """Частичное обновление инсайта: title, body, insight_type, evidence_json."""
    fields = payload.model_fields_set
    if not fields:
        # Пустое тело — ошибка клиента: 200 подтвердил бы правку, которой нет.
        raise ApiError(400, SourceErrors.NOTHING_TO_UPDATE, "No fields to update")

    for field in ("title", "body", "insight_type", "evidence_json"):
        if field in fields:
            # Подрезка и границы длины уже сделаны валидацией схемы — теми же
            # типами из deps, что и на создании.
            setattr(insight, field, getattr(payload, field))
    # updated_at проставляем руками: в модели у колонки только default_factory,
    # onupdate там нет, а списки инсайтов сортируются именно по ней.
    insight.updated_at = utcnow()

    insight_id = insight.id
    session.add(insight)
    try:
        await session.commit()
    except StaleDataError as exc:
        # Инсайт удалили (сам по себе или вместе с блокнотом) между выборкой и
        # commit: UPDATE не нашёл строки. Это не 500, а тот же 404.
        await session.rollback()
        logger.info("Concurrent delete of insight %s: %s", insight_id, exc)
        raise ApiError(
            404, SourceErrors.INSIGHT_NOT_FOUND, "Insight not found"
        ) from exc
    await session.refresh(insight)
    return _insight_response(insight)


@router.delete("/{insight_id}")
async def delete_insight(
    insight: Insight = Depends(get_owned_insight),
    session: AsyncSession = Depends(deps.get_session),
) -> dict[str, Any]:
    # На инсайт не ссылается никто: связь с заметкой хранится у него самого
    # (insight.note_id), поэтому удаление ничего за собой не тянет.
    insight_id = insight.id
    await session.delete(insight)
    try:
        await session.commit()
    except StaleDataError as exc:
        # Двойной клик по «удалить»: второй запрос обнаруживает на commit, что
        # удалять уже нечего.
        await session.rollback()
        logger.info("Concurrent delete of insight %s: %s", insight_id, exc)
        raise ApiError(
            404, SourceErrors.INSIGHT_NOT_FOUND, "Insight not found"
        ) from exc
    return {"detail": "Insight deleted", "id": insight_id}
