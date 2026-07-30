from typing import Any

from fastapi import APIRouter, Depends, Request
from sqlmodel.ext.asyncio.session import AsyncSession

from app.api import deps
from app.core.rate_limit import chat_limiter, check_rate_limit
from app.models.models import User
from app.modules.ask import AskRequest, AskResponse, handle_ask_request

router = APIRouter()


@router.post("/", response_model=AskResponse)
async def ask(
    request: Request,
    ask_request: AskRequest,
    current_user: User = Depends(deps.get_current_user),
    session: AsyncSession = Depends(deps.get_session),
) -> Any:
    await check_rate_limit(request, chat_limiter)
    return await handle_ask_request(
        ask_request=ask_request,
        current_user=current_user,
        session=session,
    )
