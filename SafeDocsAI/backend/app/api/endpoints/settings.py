import logging
from datetime import datetime
from typing import Any, Literal

from fastapi import APIRouter, Depends, Path
from pydantic import BaseModel, Field
from sqlalchemy import func, text
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.api import deps
from app.domain_profiles import list_domain_profiles
from app.core.database import get_session
from app.core.exceptions import ApiError, SettingsErrors
from app.shared.models import User
from app.shared.settings import RuntimeSettingsService

logger = logging.getLogger(__name__)

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
    # Векторы посчитаны прежней embedding-моделью: поиск идёт по коллекции,
    # которую ещё не заполнили. Флаг жил в файле настроек, но наружу не
    # выходил — интерфейс не мог даже показать, что индекс просрочен.
    reindex_required: bool = False


class RuntimeSettingsUpdate(BaseModel):
    # Неизвестный ключ в теле — отказ, а не тишина.
    #
    # Умолчание Pydantic v2 — extra="ignore": PUT {"topk": 7} (опечатка)
    # отвечал 200 OK с полным и корректным телом настроек, в котором ничего не
    # изменилось. Хуже отказа: клиент считает правку применённой и уходит, а
    # намерение потеряно без следа.
    #
    # Клиентов это не ломает: SettingsPage.jsx шлёт частичный патч из
    # перечисленных ниже полей плюс confirm_reindex, и все они объявлены —
    # включая confirm_reindex, иначе forbid отверг бы само подтверждение.
    #
    # Отказ приходит от валидации Pydantic: 422 с телом FastAPI
    # {"detail": [{"type": "extra_forbidden", "loc": ["body", "<ключ>"], ...}]}
    # и БЕЗ error_code — обработчика RequestValidationError в app/main.py нет.
    model_config = {"extra": "forbid"}

    model: str | None = None
    chat_model: str | None = None
    embedding_model: str | None = None
    enable_condense_query: bool | None = None
    retrieval_top_k: int | None = Field(default=None, ge=1, le=50)
    top_k: int | None = Field(default=None, ge=1, le=20)
    default_domain_profile: str | None = None
    contextual_embedding_enabled: bool | None = None
    contextual_embedding_model: str | None = None
    # Границы окна контекста намеренно НЕ ge/le схемы, в отличие от top_k:
    # Pydantic отвечает на них 422 без машинного кода и с английским текстом, а
    # объяснить админу нужно именно причину («столько KV-кэша на эту модель не
    # влезет, предел 32768»). Проверку держит RuntimeSettingsService и отвечает
    # settings.value_out_of_range — см. MIN_NUM_CTX/MAX_NUM_CTX.
    chat_model_num_ctx: int | None = None
    contextual_embedding_num_ctx: int | None = None
    reranker_enabled: bool | None = None
    reranker_model: str | None = None
    # Подтверждение смены embedding_model. Не настройка, а признак операции:
    # в файл не сохраняется и в ответе не возвращается. Без него запрос,
    # меняющий embedding_model, отклоняется (см. SettingsErrors.
    # REINDEX_CONFIRMATION_REQUIRED); на остальные поля не влияет никак.
    confirm_reindex: bool = False


class SettingsResetRequest(BaseModel):
    """Тело сброса настроек.

    Одно поле — и то же по смыслу, что в RuntimeSettingsUpdate: сброс
    возвращает embedding_model к умолчанию, то есть может увести поиск в
    другую коллекцию ChromaDB ровно так же, как ручная смена модели.
    """

    # По той же причине, что и в RuntimeSettingsUpdate: опечатка в единственном
    # ключе этого тела означала бы сброс БЕЗ подтверждения, отвергнутый с
    # непонятной админу мотивировкой.
    model_config = {"extra": "forbid"}

    confirm_reindex: bool = False


class UserRoleItem(BaseModel):
    id: int
    username: str
    role: str
    created_at: datetime


class UserRoleUpdate(BaseModel):
    role: Literal["admin", "content_manager", "user"]


# HTTP-статус по машинному коду. Разбор кодов — в app/core/exceptions.py;
# здесь только то, чего слой настроек знать не должен, — их отображение в HTTP.
#
# Умолчание 400: отказ на содержимом тела, повторять его как есть бессмысленно.
# Исключений два, и оба — не про тело:
#   * подтверждение переиндексации: тело валидно, не принят запрос из-за
#     состояния системы (в ChromaDB лежат векторы прежней модели), и клиент
#     повторяет ТОТ ЖЕ запрос, добавив confirm_reindex. 409 — тот же смысл, что
#     у остальных 409 раздела: «состояние сервера против»;
#   * недоступный каталог моделей: тело тоже валидно, а сверить модель не с чем,
#     потому что Ollama не ответила. 503 — «повторите позже», и повторять надо
#     ровно тот же запрос. Отвечать здесь 400 значило бы обвинять админа в
#     чужой аварии: модель может стоять на месте и уже работать.
_ERROR_STATUS = {
    SettingsErrors.REINDEX_CONFIRMATION_REQUIRED: 409,
    SettingsErrors.MODEL_CATALOG_UNAVAILABLE: 503,
}


def _as_api_error(exc: ValueError) -> ApiError:
    """ValueError из слоя настроек — в HTTP-ответ с машинным кодом.

    RuntimeSettingsService бросает SettingsError (наследник ValueError) с
    кодом. Голый ValueError оттуда прилететь тоже может — из чужого кода в
    глубине, — и остаётся 400 с общим кодом: без кода вовсе клиент показал бы
    английский detail в трёхъязычном интерфейсе, а это ровно то, что здесь
    чинилось.
    """
    error_code = getattr(exc, "error_code", None) or SettingsErrors.INVALID_VALUE
    return ApiError(_ERROR_STATUS.get(error_code, 400), error_code, str(exc))


def _settings_response(values: dict[str, Any]) -> RuntimeSettingsResponse:
    """Ответ раздела: сохранённые настройки плюс каталог моделей.

    Один сборщик на GET, PUT и сброс: три копии этого списка полей разъедутся
    на первой же новой настройке, и клиент получит от разных эндпоинтов разную
    форму одного и того же объекта.
    """
    model_catalog = RuntimeSettingsService.model_catalog()
    return RuntimeSettingsResponse(
        model=values["model"],
        chat_model=values["chat_model"],
        embedding_model=values["embedding_model"],
        enable_condense_query=values["enable_condense_query"],
        retrieval_top_k=values["retrieval_top_k"],
        top_k=values["top_k"],
        default_domain_profile=values["default_domain_profile"],
        available_models=model_catalog["available_models"],
        available_chat_models=model_catalog["available_chat_models"],
        available_embedding_models=model_catalog["available_embedding_models"],
        ollama_available=model_catalog["ollama_available"],
        ollama_error=model_catalog["ollama_error"],
        available_domain_profiles=list_domain_profiles(),
        contextual_embedding_enabled=values.get("contextual_embedding_enabled", False),
        contextual_embedding_model=values.get("contextual_embedding_model", ""),
        chat_model_num_ctx=values.get("chat_model_num_ctx", 20000),
        contextual_embedding_num_ctx=values.get("contextual_embedding_num_ctx", 8192),
        reranker_enabled=values.get("reranker_enabled", False),
        reranker_model=values.get("reranker_model", "gemma4:e4b"),
        reindex_required=bool(values.get("reindex_required", False)),
    )


@router.get("/", response_model=RuntimeSettingsResponse)
async def get_runtime_settings(
    current_user: User = Depends(deps.get_current_active_superuser),
) -> Any:
    return _settings_response(RuntimeSettingsService.get_settings())


@router.put("/", response_model=RuntimeSettingsResponse)
async def update_runtime_settings(
    payload: RuntimeSettingsUpdate,
    current_user: User = Depends(deps.get_current_active_superuser),
) -> Any:
    """Сохранить настройки.

    Смена embedding_model требует confirm_reindex=true в теле — отказ иначе
    приходит с кодом settings.reindex_confirmation_required (409). Проверку
    держит сам RuntimeSettingsService, а не этот обработчик: она обязана
    накрывать любой путь к настройкам, а не один эндпоинт.
    """
    try:
        updated = await RuntimeSettingsService.update_settings_locked(
            payload.model_dump(exclude_none=True)
        )
    except ValueError as exc:
        raise _as_api_error(exc) from exc
    return _settings_response(updated)


@router.post("/reset", response_model=RuntimeSettingsResponse)
async def reset_runtime_settings(
    payload: SettingsResetRequest | None = None,
    current_user: User = Depends(deps.get_current_active_superuser),
) -> Any:
    """Вернуть настройки к умолчаниям.

    Дороги назад не было вовсе: сброса в разделе нет, а любой удачный PUT
    оставляет после себя постоянный runtime_settings.json — то есть первая же
    сохранённая настройка становилась несмываемой.

    POST /reset, а не DELETE /: удаления ресурса здесь не происходит. Операция
    ставит новое состояние (умолчания) и возвращает его тем же телом, что GET
    и PUT, чтобы клиент обновился одним ответом. И у DELETE нет места для
    тела: подтверждение confirm_reindex пришлось бы тащить строкой запроса,
    а тело DELETE промежуточные узлы вправе выбросить.

    Подтверждение спрашивается по тому же правилу, что и при ручной смене
    модели, и только если сброс действительно меняет embedding_model.

    Только админ — как и остальной раздел.
    """
    try:
        restored = await RuntimeSettingsService.reset_settings(
            confirm_reindex=bool(payload.confirm_reindex) if payload else False
        )
    except ValueError as exc:
        raise _as_api_error(exc) from exc
    logger.info(
        "Runtime settings reset by actor_id=%s (%s)",
        current_user.id,
        current_user.username,
    )
    return _settings_response(restored)


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


# --- Смена роли ---------------------------------------------------------
#
# Машинные коды раздела переехали в реестр app/core/exceptions.py, в класс
# SettingsErrors рядом с SourceErrors/AuthErrors: интерфейс показывает
# пользователю свой перевод по коду, а не detail. Строки при переезде не
# менялись.

# Роли по возрастанию прав. Нужны ровно для одного вопроса: понижение это или
# повышение. Неизвестная роль (значение из старой схемы, ручная правка в БД)
# считается высшей: тогда любой перевод с неё считается понижением, и токены
# гасятся. Ошибиться безопаснее в эту сторону.
_ROLE_RANK = {"user": 0, "content_manager": 1, "admin": 2}
_UNKNOWN_ROLE_RANK = max(_ROLE_RANK.values()) + 1


def _role_rank(role: str) -> int:
    return _ROLE_RANK.get(role, _UNKNOWN_ROLE_RANK)


# Блокировка операции смены роли.
#
# Проверка «останется ли хоть один админ» и сам UPDATE обязаны идти под одной
# блокировкой. Без неё два запроса, понижающие двух разных админов, оба видят
# «админов двое», оба проходят проверку и оба коммитятся — админов не остаётся
# ни одного. Вернуть роль admin через API после этого нечем: эндпоинт ниже сам
# требует роль admin, создания пользователя в API нет, а регистрация жёстко
# ставит role="user". Восстановление — только мимо API (backend/create_admin.py,
# ADMIN_PROMOTE=1; см. DEPLOY.md).
#
# Взята advisory-блокировка транзакции, а не SELECT ... FOR UPDATE на строках
# админов (приём из deps.get_owned_notebook_for_update):
#
#   * защищается не строка, а счёт по таблице. FOR UPDATE запирает только те
#     строки, которые попали в выборку сейчас, и ничего не может сказать о
#     строках, которые в неё войдут, — например о повышении до admin,
#     идущем параллельно;
#   * FOR UPDATE несовместим с агрегатом: PostgreSQL отвергает
#     SELECT count(*) ... FOR UPDATE, и считать пришлось бы, вытаскивая все
#     строки админов в память ORM — ровно то, что делала прежняя проверка;
#   * блокировка транзакционная: снимается на commit и на rollback сама, в том
#     числе если обработчик отвалился с ошибкой.
#
# Ключ — OID таблицы "user". Advisory-блокировки общие на всю базу, а OID
# постоянен и различает одноимённые таблицы в разных схемах: блокировка
# попадает ровно на ту таблицу, которую защищает, и параллельные прогоны
# тестов (у каждого своя схема в общей базе, tests/dbfixtures.py) не
# выстраиваются в одну очередь.
_LOCK_ROLE_CHANGES = text(
    """SELECT pg_advisory_xact_lock('"user"'::regclass::oid::bigint)"""
)


async def _count_admins(session: AsyncSession) -> int:
    """Сколько в системе админов.

    COUNT(*), а не выборка строк: нужен счёт, а не пользователи, и цена его
    не должна расти вместе с таблицей.
    """
    result = await session.exec(
        select(func.count()).select_from(User).where(User.role == "admin")
    )
    return int(result.one())


@router.put("/users/{user_id}/role", response_model=UserRoleItem)
async def update_user_role(
    payload: UserRoleUpdate,
    # Диапазон id — общее правило API (deps.MAX_ID): значение вне int32
    # PostgreSQL не может существовать, а запрос за ним ронял бы asyncpg
    # (OverflowError) и превращал честный 404 в 500.
    user_id: int = Path(..., ge=1, le=deps.MAX_ID),
    current_user: User = Depends(deps.get_current_active_superuser),
    session: AsyncSession = Depends(get_session),
) -> Any:
    """Сменить роль пользователя.

    Единственный запрет: последнего админа снять нельзя — ни с себя, ни с
    кого-то ещё. Прежняя проверка стояла под условием «пользователь — это я»
    и понижение чужой роли не смотрела вовсе, поэтому два админа, понижающие
    друг друга, оставляли систему без администратора.

    Отказы различаются намеренно:
      * 400 — админ и правда последний. Повторять запрос бессмысленно, сначала
        нужно кого-то назначить;
      * 409 — админов было больше, но пока запрос ждал блокировки, их снял
        кто-то ещё. Список пользователей у клиента устарел, надо обновить его
        и решить заново.
    """
    # Счёт до блокировки — то состояние, из которого исходил вызывающий (список
    # пользователей он получил ещё раньше). Разница с числом под блокировкой и
    # отличает конфликт от честного отказа.
    admins_before_lock = await _count_admins(session)

    await session.exec(_LOCK_ROLE_CHANGES)

    # Строка читается уже под блокировкой и с populate_existing: current_user
    # приходит из этой же сессии (Depends кэширует get_session на запрос), и
    # без него при самопонижении вернулся бы объект из identity map — с ролью,
    # прочитанной до блокировки, то есть возможно устаревшей.
    result = await session.exec(
        select(User)
        .where(User.id == user_id)
        .with_for_update()
        .execution_options(populate_existing=True)
    )
    user = result.first()
    if not user:
        raise ApiError(404, SettingsErrors.USER_NOT_FOUND, "User not found")

    previous_role = user.role
    if previous_role == "admin" and payload.role != "admin":
        admins = await _count_admins(session)
        if admins <= 1:
            conflict = admins < admins_before_lock
            logger.warning(
                "Role change refused (%s): actor_id=%s (%s) target_id=%s (%s) "
                "%s -> %s, admins=%s (was %s)",
                "concurrent role change" if conflict else "last admin",
                current_user.id,
                current_user.username,
                user.id,
                user.username,
                previous_role,
                payload.role,
                admins,
                admins_before_lock,
            )
            if conflict:
                raise ApiError(
                    409,
                    SettingsErrors.ROLE_CHANGE_CONFLICT,
                    "Another admin was demoted while this request was waiting; "
                    "refresh the list and try again",
                )
            raise ApiError(
                400,
                SettingsErrors.LAST_ADMIN,
                "At least one admin must remain in the system",
            )

    user.role = payload.role
    # Понижение обесценивает всё, что выдано под прежней ролью. Правами это не
    # является — роль читается из БД на каждом запросе (deps.get_current_user), —
    # но без инкремента у разжалованного остаются рабочие refresh-токены ещё на
    # неделю, то есть сессия, которую никто не пересматривал. То же поколение
    # поднимают смена пароля и /auth/logout-all (app/api/endpoints/auth.py).
    if _role_rank(payload.role) < _role_rank(previous_role):
        user.token_version = (user.token_version or 0) + 1
    session.add(user)
    await session.commit()
    await session.refresh(user)

    # Назначение и снятие прав — самая привилегированная операция в системе, и
    # следа от неё не оставалось нигде: ни кто менял, ни кому, ни с чего на что.
    logger.info(
        "Role changed: actor_id=%s (%s) target_id=%s (%s) %s -> %s",
        current_user.id,
        current_user.username,
        user.id,
        user.username,
        previous_role,
        user.role,
    )

    return UserRoleItem(
        id=user.id,
        username=user.username,
        role=user.role,
        created_at=user.created_at,
    )
