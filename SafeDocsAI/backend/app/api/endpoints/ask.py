from typing import Any

from fastapi import APIRouter, Depends, Request
from pydantic import Field
from sqlmodel.ext.asyncio.session import AsyncSession

from app.api import deps
from app.core.rate_limit import chat_limiter, check_rate_limit
from app.models.models import User
from app.modules.ask import AskRequest, AskResponse, handle_ask_request

router = APIRouter()


class AskRequestIn(AskRequest):
    """AskRequest с границами диапазона на notebook_id и проверкой вопроса.

    Ограничение стоит на слое HTTP, а не в модуле: id больше PostgreSQL
    integer asyncpg не может передать параметром запроса, и вместо 404
    запрос падал OverflowError, то есть 500.

    Вопрос проверяется тем же типом, что и в чате (deps.QuestionStr): пустой
    вопрос иначе доходил до поиска и генерации, а вопрос без верхней границы
    вытеснял из окна модели найденные фрагменты.
    """

    question: deps.QuestionStr
    notebook_id: int | None = Field(default=None, ge=1, le=deps.MAX_ID)


@router.post("/", response_model=AskResponse)
async def ask(
    request: Request,
    ask_request: AskRequestIn,
    current_user: User = Depends(deps.get_current_user),
    session: AsyncSession = Depends(deps.get_session),
) -> Any:
    await check_rate_limit(request, chat_limiter)
    return await handle_ask_request(
        ask_request=ask_request,
        current_user=current_user,
        session=session,
    )
