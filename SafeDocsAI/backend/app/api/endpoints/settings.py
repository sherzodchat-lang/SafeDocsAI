from datetime import datetime
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.api import deps
from app.domain_profiles import list_domain_profiles
from app.core.database import get_session
from app.shared.models import User
from app.shared.settings import RuntimeSettingsService

router = APIRouter()


class RuntimeSettingsResponse(BaseModel):
    model: str
    chat_model: str
    embedding_model: str
    enable_condense_query: bool
    retrieval_top_k: int = Field(ge=1, le=50)
    top_k: int = Field(ge=1, le=20)
    default_domain_profile: str
    available_models: list[str]
    available_chat_models: list[str]
    available_embedding_models: list[str]
    ollama_available: bool
    ollama_error: str | None = None
    available_domain_profiles: list[str]
    contextual_embedding_enabled: bool = False
    contextual_embedding_model: str = ""
    chat_model_num_ctx: int = 20000
    contextual_embedding_num_ctx: int = 8192
    reranker_enabled: bool = False
    reranker_model: str = "gemma4:e4b"


class RuntimeSettingsUpdate(BaseModel):
    model: str | None = None
    chat_model: str | None = None
    embedding_model: str | None = None
    enable_condense_query: bool | None = None
    retrieval_top_k: int | None = Field(default=None, ge=1, le=50)
    top_k: int | None = Field(default=None, ge=1, le=20)
    default_domain_profile: str | None = None
    contextual_embedding_enabled: bool | None = None
    contextual_embedding_model: str | None = None
    chat_model_num_ctx: int | None = None
    contextual_embedding_num_ctx: int | None = None
    reranker_enabled: bool | None = None
    reranker_model: str | None = None


class UserRoleItem(BaseModel):
    id: int
    username: str
    role: str
    created_at: datetime


class UserRoleUpdate(BaseModel):
    role: Literal["admin", "content_manager", "user"]


@router.get("/", response_model=RuntimeSettingsResponse)
async def get_runtime_settings(
    current_user: User = Depends(deps.get_current_active_superuser),
) -> Any:
    runtime_settings = RuntimeSettingsService.get_settings()
    model_catalog = RuntimeSettingsService.model_catalog()
    return RuntimeSettingsResponse(
        model=runtime_settings["model"],
        chat_model=runtime_settings["chat_model"],
        embedding_model=runtime_settings["embedding_model"],
        enable_condense_query=runtime_settings["enable_condense_query"],
        retrieval_top_k=runtime_settings["retrieval_top_k"],
        top_k=runtime_settings["top_k"],
        default_domain_profile=runtime_settings["default_domain_profile"],
        available_models=model_catalog["available_models"],
        available_chat_models=model_catalog["available_chat_models"],
        available_embedding_models=model_catalog["available_embedding_models"],
        ollama_available=model_catalog["ollama_available"],
        ollama_error=model_catalog["ollama_error"],
        available_domain_profiles=list_domain_profiles(),
        contextual_embedding_enabled=runtime_settings.get("contextual_embedding_enabled", False),
        contextual_embedding_model=runtime_settings.get("contextual_embedding_model", ""),
        chat_model_num_ctx=runtime_settings.get("chat_model_num_ctx", 20000),
        contextual_embedding_num_ctx=runtime_settings.get("contextual_embedding_num_ctx", 8192),
        reranker_enabled=runtime_settings.get("reranker_enabled", False),
        reranker_model=runtime_settings.get("reranker_model", "gemma4:e4b"),
    )


@router.put("/", response_model=RuntimeSettingsResponse)
async def update_runtime_settings(
    payload: RuntimeSettingsUpdate,
    current_user: User = Depends(deps.get_current_active_superuser),
) -> Any:
    try:
        updated = RuntimeSettingsService.update_settings(
            payload.model_dump(exclude_none=True)
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    model_catalog = RuntimeSettingsService.model_catalog()
    return RuntimeSettingsResponse(
        model=updated["model"],
        chat_model=updated["chat_model"],
        embedding_model=updated["embedding_model"],
        enable_condense_query=updated["enable_condense_query"],
        retrieval_top_k=updated["retrieval_top_k"],
        top_k=updated["top_k"],
        default_domain_profile=updated["default_domain_profile"],
        available_models=model_catalog["available_models"],
        available_chat_models=model_catalog["available_chat_models"],
        available_embedding_models=model_catalog["available_embedding_models"],
        ollama_available=model_catalog["ollama_available"],
        ollama_error=model_catalog["ollama_error"],
        available_domain_profiles=list_domain_profiles(),
        contextual_embedding_enabled=updated.get("contextual_embedding_enabled", False),
        contextual_embedding_model=updated.get("contextual_embedding_model", ""),
        chat_model_num_ctx=updated.get("chat_model_num_ctx", 20000),
        contextual_embedding_num_ctx=updated.get("contextual_embedding_num_ctx", 8192),
        reranker_enabled=updated.get("reranker_enabled", False),
        reranker_model=updated.get("reranker_model", "gemma4:e4b"),
    )


@router.get("/users", response_model=list[UserRoleItem])
async def list_users_for_role_management(
    current_user: User = Depends(deps.get_current_active_superuser),
    session: AsyncSession = Depends(get_session),
) -> Any:
    result = await session.exec(select(User).order_by(User.created_at.desc()))
    users = result.all()
    return [
        UserRoleItem(
            id=user.id,
            username=user.username,
            role=user.role,
            created_at=user.created_at,
        )
        for user in users
        if user.id is not None
    ]


@router.put("/users/{user_id}/role", response_model=UserRoleItem)
async def update_user_role(
    user_id: int,
    payload: UserRoleUpdate,
    current_user: User = Depends(deps.get_current_active_superuser),
    session: AsyncSession = Depends(get_session),
) -> Any:
    user = await session.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Avoid removing the last admin privileges from yourself by mistake.
    if user.id == current_user.id and payload.role != "admin":
        admins_result = await session.exec(select(User).where(User.role == "admin"))
        admins = admins_result.all()
        if len(admins) <= 1:
            raise HTTPException(
                status_code=400, detail="At least one admin must remain in the system"
            )

    user.role = payload.role
    session.add(user)
    await session.commit()
    await session.refresh(user)

    return UserRoleItem(
        id=user.id,
        username=user.username,
        role=user.role,
        created_at=user.created_at,
    )
