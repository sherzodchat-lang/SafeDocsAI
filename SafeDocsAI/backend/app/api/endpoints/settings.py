import logging
from datetime import datetime
from typing import Any, Literal

from fastapi import APIRouter, Depends, Path, Query, Response
from pydantic import BaseModel, Field
from sqlalchemy import func, text
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.api import deps
from app.api.endpoints.documents import MAX_PAGE_SIZE, TOTAL_COUNT_HEADER
from app.domain_profiles import list_domain_profiles
from app.core.database import get_session
from app.core.exceptions import ApiError, SettingsErrors
from app.shared.models import User
from app.shared.settings import RuntimeSettingsService, setting_limits

logger = logging.getLogger(__name__)

router = APIRouter()

# Страница по умолчанию равна потолку намеренно — как в GET /notebooks/ и
# GET /notes/: клиент этого списка (SettingsPage.jsx) параметров не передаёт,
# и любой меньший размер молча урезал бы уже отдаваемый список — админ увидел
# бы часть пользователей и не узнал бы, что видит не всех. Потолок при этом
# закрывает главное: выгрузку всей таблицы user одним запросом.
DEFAULT_PAGE_SIZE = MAX_PAGE_SIZE


# Умолчания настроек — из ОДНОГО места, RuntimeSettingsService.DEFAULTS.
#
# Значения 20000, 8192, False и имена моделей были выписаны трижды: в DEFAULTS
# (настоящий источник — оттуда их берут и чтение, и сброс) и дважды здесь, в
# схеме ответа и в сборщике _settings_response. Совпадали они только пока их не
# правили: правка одного места разъезжается с двумя другими молча, без единой
# ошибки, и наружу выходит умолчание, с которым система не работает. Ровно этот
# класс дефекта раздел уже ловил — см. комментарий к выводу "model" из
# chat_model в RuntimeSettingsService.get_settings.
#
# Подстановка идёт по тому же ключу, под которым настройка лежит в файле
# настроек, поэтому опечатка в имени поля здесь — это KeyError, а не тихо
# разъехавшееся умолчание.
#
# Псевдонима вроде `_DEFAULTS = RuntimeSettingsService.DEFAULTS` здесь
# намеренно нет: он привязался бы к объекту словаря на импорте, и подмена
# самого атрибута класса (тесты, скрипт, будущая загрузка умолчаний из
# конфигурации) мимо него бы прошла. Помощник ниже читает атрибут в момент
# вызова.
def _default(field: str) -> Any:
    """Умолчание настройки. Источник один — RuntimeSettingsService.DEFAULTS."""
    return RuntimeSettingsService.DEFAULTS[field]


# Границы числовых полей — тоже из ОДНОГО места, SETTING_LIMITS (см.
# app/shared/settings/runtime_settings.py). Числа здесь были третьей копией: те
# же пары стоят в строгой проверке при записи (_require_int_in_range) и в
# снисходительном клампе при чтении (_clamp_on_read), и правка одной из трёх
# разводила запись с чтением молча.
#
# В отличие от _default здесь помощник вызывается ОДИН раз — на импорте, при
# разборе тела класса, — и иначе быть не может: ge/le обязаны попасть в
# статическую схему, то есть в OpenAPI, где они и служат документацией
# контракта. Поздней подстановки, как у умолчаний, тут нет и не нужно: границы
# описывают не состояние сервера, а сам контракт.
def _bounded(field: str) -> Any:
    """Field(ge, le) с границами настройки из SETTING_LIMITS."""
    limits = setting_limits(field)
    return Field(ge=limits.min, le=limits.max)


def _setting(values: dict[str, Any], field: str) -> Any:
    """Значение настройки из прочитанного файла, с откатом на её умолчание.

    Имя поля пишется один раз: в `values.get("x", DEFAULTS["x"])` разъехаться
    могли не только значения, но и сами ключи.
    """
    return values.get(field, _default(field))


class RuntimeSettingsResponse(BaseModel):
    model: str
    chat_model: str
    embedding_model: str
    enable_condense_query: bool
    retrieval_top_k: int = _bounded("retrieval_top_k")
    top_k: int = _bounded("top_k")
    default_domain_profile: str
    available_models: list[str]
    available_chat_models: list[str]
    available_embedding_models: list[str]
    ollama_available: bool
    ollama_error: str | None = None
    available_domain_profiles: list[str]
    # Умолчания полей взяты из RuntimeSettingsService.DEFAULTS, а не выписаны
    # литералами: контракт OpenAPI при этом не меняется — в схему попадают те
    # же значения, только теперь они гарантированно те же, что отдаёт сервер.
    contextual_embedding_enabled: bool = _default("contextual_embedding_enabled")
    contextual_embedding_model: str = _default("contextual_embedding_model")
    chat_model_num_ctx: int = _default("chat_model_num_ctx")
    contextual_embedding_num_ctx: int = _default("contextual_embedding_num_ctx")
    reranker_enabled: bool = _default("reranker_enabled")
    reranker_model: str = _default("reranker_model")
    # Векторы посчитаны прежней embedding-моделью: поиск идёт по коллекции,
    # которую ещё не заполнили. Флаг жил в файле настроек, но наружу не
    # выходил — интерфейс не мог даже показать, что индекс просрочен.
    reindex_required: bool = _default("reindex_required")


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
    # Границы числовых полей намеренно НЕ ge/le схемы: Pydantic отвечает на них
    # 422 без машинного кода и с английским текстом («Input should be less than
    # or equal to 20»), а интерфейс переведён на три языка и показывает свой
    # перевод по error_code. Проверку держит RuntimeSettingsService и отвечает
    # settings.value_out_of_range — см. _require_int_in_range; границы он берёт
    # из SETTING_LIMITS, то есть ровно те, что объявлены выше в схеме ответа.
    #
    # Держать здесь два образца политики нельзя: следующий, кто добавит поле,
    # выберет наугад.
    retrieval_top_k: int | None = None
    top_k: int | None = None
    default_domain_profile: str | None = None
    contextual_embedding_enabled: bool | None = None
    contextual_embedding_model: str | None = None
    # У окна контекста та же политика, и объяснить админу нужно именно причину
    # («столько KV-кэша на эту модель не влезет, вот предел») — см.
    # _require_num_ctx и SETTING_LIMITS (MIN_NUM_CTX/MAX_NUM_CTX).
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

    Умолчания подставляет _setting — из RuntimeSettingsService.DEFAULTS, а не
    литералами: второй копии этих значений здесь больше нет. Сработать откату
    вообще-то не на чем (все три вызывающих приходят из get_settings, а тот
    отдаёт полный набор ключей при любом состоянии диска — см. его docstring),
    но оставлен он затем, что этот сборщик — единственное, что стоит между
    настройками и ответом клиенту: падать здесь по KeyError значит погасить
    экран настроек целиком.
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
        contextual_embedding_enabled=_setting(values, "contextual_embedding_enabled"),
        contextual_embedding_model=_setting(values, "contextual_embedding_model"),
        chat_model_num_ctx=_setting(values, "chat_model_num_ctx"),
        contextual_embedding_num_ctx=_setting(values, "contextual_embedding_num_ctx"),
        reranker_enabled=_setting(values, "reranker_enabled"),
        reranker_model=_setting(values, "reranker_model"),
        reindex_required=bool(_setting(values, "reindex_required")),
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


@router.get(
    "/users",
    response_model=list[UserRoleItem],
    responses={
        200: {
            "headers": {
                TOTAL_COUNT_HEADER: {
                    "description": (
                        "Общее число пользователей в системе, без учёта "
                        "skip/limit."
                    ),
                    "schema": {"type": "integer"},
                }
            }
        }
    },
)
async def list_users_for_role_management(
    response: Response,
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=DEFAULT_PAGE_SIZE, ge=1, le=MAX_PAGE_SIZE),
    current_user: User = Depends(deps.get_current_active_superuser),
    session: AsyncSession = Depends(get_session),
) -> Any:
    """Страница пользователей для управления ролями, свежие сверху.

    Отдавалась вся таблица user одним запросом, без потолка вообще: экран
    админский, но растёт он вместе с числом регистраций, а интерфейс не мог
    даже показать общее число — в теле его нет, а тело и есть весь ответ.

    Пагинация ровно та же, что у GET /sources/, /notebooks/ и /notes/: skip,
    limit и общее число заголовком X-Total-Count. Тело осталось голым массивом
    UserRoleItem, поэтому клиент, не знающий о пагинации, ничего не заметил —
    менять форму ответа здесь нельзя, её читают как массив.

    Порядок — created_at DESC, id DESC. Второй ключ обязателен: без него
    пользователи, созданные в одну миллисекунду (а регистрации идут пачками, и
    тестовые пользователи заводятся скриптом), встают между запросами в разном
    порядке — одна и та же запись показывается на двух соседних страницах, а
    другая не показывается ни на одной.
    """
    page = (
        select(User)
        .order_by(User.created_at.desc(), User.id.desc())
        .offset(skip)
        .limit(limit)
    )
    result = await session.exec(page)
    users = result.all()

    # COUNT(*) отдельным запросом, а не len(users): нужно число ВСЕХ
    # пользователей, а не тех, что попали на страницу.
    total_result = await session.exec(select(func.count()).select_from(User))
    response.headers[TOTAL_COUNT_HEADER] = str(int(total_result.first() or 0))
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
