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


class ChatRequestIn(ChatRequest):
    """ChatRequest с проверкой вопроса.

    Ограничение стоит на слое HTTP, а не в модуле, по тому же доводу, что и у
    AskRequestIn: модуль чата описывает запрос, а границы входных значений
    живут в одном месте на весь API (app/api/deps.py).

    Проверка на слое схемы, а не в теле обработчика, — не стилистика.
    Пустой вопрос до неё проходил поиск и генерацию целиком, то есть занимал
    GPU ровно как настоящий. Валидация тела запроса срабатывает раньше, чем
    обработчик получает управление: до check_rate_limit, до обращения к
    ChromaDB и до StreamingResponse у /stream — поэтому отказ приходит
    обычным 422 с телом JSON, а не событием error внутри уже начатого SSE.
    """

    question: deps.QuestionStr


class RetrievalRequestIn(RetrievalRequest):
    """RetrievalRequest с той же проверкой вопроса.

    Разбор поиска не доходит до генерации, но эмбеддинг запроса и обращение к
    ChromaDB делает — на пустой строке это работа впустую.
    """

    question: deps.QuestionStr


@router.post("/", response_model=ChatResponse)
async def chat(
    request: Request,
    chat_request: ChatRequestIn,
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
    chat_request: ChatRequestIn,
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
    retrieval_request: RetrievalRequestIn,
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
    "ChatRequestIn",
    "RetrievalRequestIn",
    "ChatResponse",
    "RetrievalRequest",
    "RetrievalResponse",
    "SourceItem",
    "_is_no_data_answer",
    "_select_relevant_chunks",
]
