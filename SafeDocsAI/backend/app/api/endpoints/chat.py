from typing import Any

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from sqlmodel.ext.asyncio.session import AsyncSession

from app.api import deps
from app.core.rate_limit import chat_limiter, check_rate_limit
from app.models.models import User
from app.modules.chat.schemas import (
    ChatRequest,
    ChatResponse,
    RetrievalRequest,
    RetrievalResponse,
    SourceItem,
)
from app.modules.chat.service import (
    chat_request as handle_chat_request,
    chat_request_stream as handle_chat_request_stream,
    is_no_data_answer as _is_no_data_answer,
    retrieve_chunks as handle_retrieve_chunks,
    select_relevant_chunks as _select_relevant_chunks,
)

router = APIRouter()


@router.post("/", response_model=ChatResponse)
async def chat(
    request: Request,
    chat_request: ChatRequest,
    current_user: User = Depends(deps.get_current_user),
    session: AsyncSession = Depends(deps.get_session),
) -> Any:
    await check_rate_limit(request, chat_limiter)
    return await handle_chat_request(
        chat_request=chat_request,
        current_user=current_user,
        session=session,
    )


@router.post("/stream")
async def chat_stream(
    request: Request,
    chat_request: ChatRequest,
    current_user: User = Depends(deps.get_current_user_short_lived),
) -> StreamingResponse:
    await check_rate_limit(request, chat_limiter)
    return StreamingResponse(
        handle_chat_request_stream(
            chat_request=chat_request,
            current_user=current_user,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/retrieve", response_model=RetrievalResponse)
async def retrieve(
    request: Request,
    retrieval_request: RetrievalRequest,
    current_user: User = Depends(deps.get_current_user),
    session: AsyncSession = Depends(deps.get_session),
) -> Any:
    await check_rate_limit(request, chat_limiter)
    return await handle_retrieve_chunks(
        retrieval_request=retrieval_request,
        current_user=current_user,
        session=session,
    )


__all__ = [
    "ChatRequest",
    "ChatResponse",
    "RetrievalRequest",
    "RetrievalResponse",
    "SourceItem",
    "_is_no_data_answer",
    "_select_relevant_chunks",
]
