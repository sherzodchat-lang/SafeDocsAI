"""HTTP-слой раздела презентаций: шаблоны, заказ, опрос, скачивание, удаление.

Четыре решения этого файла, которые важнее остального кода.

**Порядок проверок на заказе — часть контракта, а не деталь реализации.**
Владение -> роль -> частота -> тело -> бизнес-условия. Каждый шаг стоит именно
там, где стоит, по своей причине:

  * владение первым и всегда 404, даже когда роль вызывающего заведомо не
    подходит. Иначе эндпоинт становится оракулом существования: по разнице
    403/404 чужие блокноты перебираются по id;
  * роль до частоты, потому что отказ по роли не должен расходовать лимит:
    иначе обычный пользователь, тыкающий в недоступную кнопку, выбивает себе
    же час тишины (тот же довод, что в tests/test_question_validation_db.py);
  * частота до разбора тела, потому что разбор — уже работа, а лимит защищает
    именно от вала запросов, в том числе заведомо негодных;
  * бизнес-условия последними: они делают запросы в БД, и платить за них на
    заведомо негодном теле незачем.

Порядок обеспечен механикой FastAPI, а не расстановкой строк в обработчике:
solve_dependencies выполняет зависимости в порядке параметров функции И ЦЕЛИКОМ
до валидации тела (fastapi/dependencies/utils.py). Поэтому владение, роль и
лимит — зависимости, а проверки полей живут в схеме.

**422 приходят НАШИМИ кодами.** В проекте есть обработчик RequestValidationError
(app/main.py), который отдаёт request.validation_failed без разбора по полям —
для отказа «поле не того типа» это правильно, а для «такого шаблона нет» уже
нет: пользователю нужно понять, что чинить. Поэтому проверки значений сделаны
валидаторами, бросающими ApiError с нужным кодом, — тем же приёмом, что
deps.QuestionStr. Границы при этом НЕ дублируются в Field(ge/le): такое
ограничение сработало бы раньше нашего валидатора и вернуло бы общий код.
В OpenAPI границы всё же попадают — через json_schema_extra, то есть как
документация, а не как вторая проверка.

**Инвариант «не больше одной активной колоды на блокнот» держит БАЗА.**
Предпроверка в обработчике (SELECT активных строк, потом INSERT) — это ровно то
место, которое гонится: два клика в одну секунду проходят её оба, а лимит
частоты считает десятки заказов в час, а не два в секунду. Поэтому рядом стоит
частичный уникальный индекс `uq_presentation_active_notebook` (создаётся в
app/core/database.py), и его нарушение переводится в ТОТ ЖЕ 409
presentation.generation_in_progress, что и предпроверка. Предпроверка при этом
остаётся: она отвечает без нарушения целостности и без отката транзакции, то
есть обычный повторный клик стоит дешевле.

**Пути к файлам берутся из реестра и из строки БД, но никогда из запроса.**
Превью шаблона отдаётся по КЛЮЧУ: путь ищется в реестре, который уже запер его
в своём каталоге — исходники шаблона в templates/presentations, нарисованное
Chrome превью в data/presentation_previews (templates.py: default_templates_dir
и default_preview_dir). Обхода каталога тут нет по построению, а не по
фильтрации ввода. Превью КОЛОДЫ отдаётся по id строки, и путь его картинки
выводится из presentation.file_path — то есть из того же места, откуда берётся
файл на скачивании, и защищён той же зависимостью владения.

**Превью колоды закрыто ровно как её скачивание.** На титульном слайде стоят
название темы и имя блокнота, поэтому картинка первой страницы — это выдержка
из чужой работы, а не безобидная иконка. Проверку делает та же зависимость
get_owned_presentation, что и у download; собственной проверки владения у
превью нет намеренно — вторая, разойдясь с первой, стала бы дырой в том месте,
где её никто не ищет.

**Тип и имя скачиваемого файла выводятся из ХРАНИМОГО ПУТИ, а не из константы.**
Колоды печатает headless Chrome, и новые файлы — .pdf; но на дисках лежат
колоды, собранные прежним рендерером в .pptx, и это пользовательские файлы,
которые никто не мигрирует. Константа «тип презентации» отдала бы им чужой
MIME, то есть сломала бы уже выданное обещание «скачается как было». Поэтому
расширение читается у файла, а таблица DOWNLOAD_MEDIA_TYPES переводит его в тип.
"""

import logging
import os
from datetime import datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Path, Query, Request, Response, status
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from pydantic import AfterValidator, BaseModel, Field, field_serializer
from sqlalchemy.exc import IntegrityError
from sqlmodel import func, select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.api import deps
from app.core.database import PRESENTATION_ACTIVE_INDEX
from app.api.endpoints.documents import (
    DEFAULT_PAGE_SIZE,
    MAX_PAGE_SIZE,
    TOTAL_COUNT_HEADER,
    serialize_utc,
)
from app.core.exceptions import ApiError, PresentationErrors
from app.core.rate_limit import RateLimiter, check_rate_limit
# Проверка браузера — из модуля рендера, своей копии здесь нет: «чем печатаем»
# знает он, а слой API только переводит его отказ в HTTP-код.
from app.modules.presentations.chromium import (
    RendererUnavailable,
    ensure_chromium_available,
)
from app.modules.presentations.constants import (
    DEFAULT_LANGUAGE,
    DESCRIPTION_MAX,
    SLIDE_COUNT_DEFAULT,
    SLIDE_COUNT_MAX,
    SLIDE_COUNT_MIN,
    STATUS_GENERATING,
    STATUS_QUEUED,
    STATUS_READY,
    SUPPORTED_LANGUAGES,
    normalize_language,
)
from app.modules.presentations.preview import (
    CARD_PREVIEW_MEDIA_TYPE,
    ensure_card_preview,
    remove_card_preview,
)
from app.modules.presentations.service import PresentationsService
from app.modules.presentations.templates import TemplateInfo, template_registry
# Санитизация имени и обрезка по БАЙТАМ — те же, что у источников. Своей копии
# здесь быть не должно: Starlette экранирует заголовок сам, но отображаемое имя
# всё равно надо чистить, и правило «как чистим» обязано быть одно на проект.
from app.services.source_service import SourceService
from app.shared.models import Document, Notebook, Presentation, User
from app.shared.settings import settings

router = APIRouter()
logger = logging.getLogger(__name__)

# Заказ презентации — самая дорогая операция раздела: она занимает GPU на
# минуты (замеры приёмки: 444 с на колоду из 15 слайдов) и держит очередь,
# общую на всю систему. Поэтому лимит куда жёстче, чем у загрузки источника
# (30 за 5 минут, documents.upload_limiter): десяти колод в час хватает и на
# работу, и на несколько неудачных попыток подряд, а скрипт, ставящий заказы в
# цикле, упирается в него на одиннадцатом.
#
# Ключ счётчика — АДРЕС клиента, а не пользователь: его выбирает общий
# check_rate_limit (get_client_id), и здесь взят он же, чтобы форма 429-ответа
# (Retry-After, X-RateLimit-*) осталась одна на весь API. Разница видна там,
# где два контент-менеджера сидят за одним NAT: десять заказов в час они делят.
# Ключ по пользователю потребовал бы принимать его в check_rate_limit — правку
# app/core/rate_limit.py, то есть общего механизма всех лимитов сразу; делать её
# ради одного эндпоинта в этом этапе не стали.
order_limiter = RateLimiter(requests=10, window=3600)

# Тип .pptx. Тот же, что MIME_BY_EXTENSION у источников по устройству, но для
# ДРУГОГО расширения: там .docx (wordprocessingml), здесь .pptx
# (presentationml). Ошибиться здесь легко, а последствие тихое — браузер
# предложит сохранить файл под чужим типом.
PPTX_MEDIA_TYPE = (
    "application/vnd.openxmlformats-officedocument.presentationml.presentation"
)

# Расширение хранимого файла -> тип, с которым он отдаётся. Таблица, а не одна
# константа, потому что на дисках одновременно живут колоды двух поколений:
# новые печатает Chrome в .pdf, старые собрал прежний рендерер в .pptx. Мигрировать
# старые нечем и незачем — это пользовательские файлы, которые кто-то уже ждёт в
# «Загрузках», — а один общий тип «презентация» назвал бы половину из них чужим
# именем, и открыть скачанное не удалось бы ни PowerPoint'у, ни просмотрщику PDF.
#
# Неизвестное расширение отдаётся octet-stream, а не угадывается: путь пишет наш
# же рендерер, и файл с расширением не из этого списка означает, что что-то
# пошло не по плану — пусть браузер спросит пользователя, а не сделает вид.
DOWNLOAD_MEDIA_TYPES = {
    ".pdf": "application/pdf",
    ".pptx": PPTX_MEDIA_TYPE,
}
FALLBACK_MEDIA_TYPE = "application/octet-stream"

# Тип картинки превью по расширению файла из манифеста. Неизвестное расширение
# отдаётся octet-stream, а не угадывается: манифест — наш артефакт, и появление
# в нём .svg (который браузер выполняет как документ со скриптами) должно
# требовать осознанной правки этого словаря.
PREVIEW_MEDIA_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
}

# Как называется скачиваемый файл. Имя блокнота подставляется целиком: именно
# по нему пользователь узнаёт колоду в папке «Загрузки». Расширение —
# подставляемое, и берётся оно у самого файла (см. _download_filename): имя
# «...презентация.pdf» на .pptx-файле было бы прямой ложью, и открыть такую
# «презентацию» не смог бы никто.
DOWNLOAD_NAME_TEMPLATE = "{notebook} — презентация{suffix}"
# Блокнота нет (строка пережила его в гонке) — имя без него, но осмысленное.
DOWNLOAD_NAME_FALLBACK = "презентация{suffix}"

# Статусы, при которых по блокноту нельзя заказать вторую колоду. Ровно те же,
# что стоят в предикате частичного уникального индекса
# (app/core/database.PRESENTATION_ACTIVE_STATUSES): предпроверка и база обязаны
# считать активным одно и то же множество, иначе одна из них начнёт пропускать
# то, что отвергает другая.
ACTIVE_STATUSES = (STATUS_QUEUED, STATUS_GENERATING)

# Текст отказа «уже генерируется» — ОДИН на обе точки отказа: предпроверку и
# нарушение уникальности в базе. Клиент не должен различать, кто именно
# отказал: для него это одно событие с одним действием («дождитесь или удалите
# текущий заказ»), и вторая формулировка означала бы вторую строку перевода об
# одном и том же.
GENERATION_IN_PROGRESS_DETAIL = (
    "A presentation for this notebook is already queued or being generated"
)


# --- Доступ ---------------------------------------------------------------


async def require_presentation_role(
    current_user: User = Depends(deps.get_current_user),
) -> User:
    """Заказывать презентации вправе content_manager и admin.

    Своя зависимость, а не deps.get_current_content_manager_or_admin: правило
    ролей то же, но код отказа нужен свой (см. PresentationErrors.
    ROLE_NOT_ALLOWED). Ставится ПОСЛЕ проверки владения — чужой блокнот обязан
    отвечать 404 независимо от роли вызывающего.

    Чтение, скачивание и удаление роль не проверяют вовсе: они живут на
    владении, как и весь раздел источников. Пользователь, у которого забрали
    роль, не должен терять доступ к уже заказанным им колодам — иначе они
    станут неудаляемым мусором на диске.
    """
    if current_user.role not in ("admin", "content_manager"):
        raise ApiError(
            status.HTTP_403_FORBIDDEN,
            PresentationErrors.ROLE_NOT_ALLOWED,
            "Only content managers and administrators can order presentations",
        )
    return current_user


async def enforce_order_rate_limit(request: Request) -> None:
    """Лимит частоты заказов. Зависимостью, а не строкой в обработчике.

    Внутри обработчика он сработал бы уже ПОСЛЕ разбора тела, то есть негодный
    запрос успевал бы получить 422 в обход лимита — и вал таких запросов
    ничего бы не стоил отправителю.
    """
    await check_rate_limit(request, order_limiter)


async def get_owned_presentation(
    presentation_id: int = Path(..., ge=1, le=deps.MAX_ID),
    session: AsyncSession = Depends(deps.get_session),
    current_user: User = Depends(deps.get_current_user),
) -> Presentation:
    """Презентация вызывающего или 404. Правило владения — общее (deps.user_owns)."""
    presentation = await session.get(Presentation, presentation_id)
    if not presentation or not deps.user_owns(presentation.owner_id, current_user):
        raise ApiError(404, PresentationErrors.NOT_FOUND, "Presentation not found")
    return presentation


async def get_owned_presentation_for_update(
    presentation_id: int = Path(..., ge=1, le=deps.MAX_ID),
    session: AsyncSession = Depends(deps.get_session),
    current_user: User = Depends(deps.get_current_user),
) -> Presentation:
    """То же, но строка берётся SELECT ... FOR UPDATE. Нужна удалению.

    Без блокировки статус читается устаревшим: воркер забирает очередной заказ
    в той же таблице (claim_next, FOR UPDATE SKIP LOCKED), и между нашим
    чтением 'queued' и DELETE он успевает перевести строку в 'generating' —
    удаление снесло бы строку из-под работающей генерации. С блокировкой
    расходятся оба случая честно: запертую нами 'queued' воркер пропустит, а
    уже начатую генерацию мы увидим именно как 'generating' и ответим 409.
    """
    result = await session.exec(
        select(Presentation)
        .where(Presentation.id == presentation_id)
        .with_for_update()
    )
    presentation = result.first()
    if not presentation or not deps.user_owns(presentation.owner_id, current_user):
        raise ApiError(404, PresentationErrors.NOT_FOUND, "Presentation not found")
    return presentation


# --- Проверка полей заказа ------------------------------------------------
#
# Все четыре валидатора бросают ApiError, а не ValueError: 422 от Pydantic
# приходит с общим кодом request.validation_failed, по которому пользователю
# нечего показать, кроме английского msg (см. docstring модуля).
#
# AfterValidator, а не BeforeValidator (в отличие от deps.QuestionStr): нам
# нужно значение УЖЕ приведённое к типу поля. С BeforeValidator строка "99" в
# slide_count проехала бы мимо проверки диапазона — валидатор пропустил бы
# не-int дальше, а Pydantic следом честно превратил бы её в 99. Ошибка типа
# при этом остаётся за Pydantic: «slide_count: "много"» — не наш случай, и
# отвечать на неё своим кодом было бы неправдой.


def _check_template_key(value: str) -> str:
    key = value.strip()
    if template_registry.get(key) is None:
        raise ApiError(
            422,
            PresentationErrors.UNSUPPORTED_TEMPLATE,
            f"Unknown presentation template {key!r}",
        )
    return key


def _check_language(value: str) -> str:
    try:
        # Тот же нормализатор, что у пайплайна: регистр и пробелы приводятся,
        # неизвестный код отвергается. Второй список языков здесь развёл бы
        # приём заказа с генерацией.
        return normalize_language(value)
    except ValueError as exc:
        raise ApiError(
            422,
            PresentationErrors.UNSUPPORTED_LANGUAGE,
            f"Unsupported language, expected one of {', '.join(SUPPORTED_LANGUAGES)}",
        ) from exc


def _check_slide_count(value: int) -> int:
    if not SLIDE_COUNT_MIN <= value <= SLIDE_COUNT_MAX:
        raise ApiError(
            422,
            PresentationErrors.VALUE_OUT_OF_RANGE,
            f"slide_count must be between {SLIDE_COUNT_MIN} and {SLIDE_COUNT_MAX}",
        )
    return value


def _check_description(value: str) -> str | None:
    """Описание: подрезанное, в пределах бюджета промпта; пустое допустимо.

    Пустая строка превращается в None, а не хранится как "": описание уезжает в
    промпт, и пустая строка там означала бы «пользователь ничего не просил» ровно
    так же, как её отсутствие, — но проверять пришлось бы два состояния.
    """
    trimmed = value.strip()
    if len(trimmed) > DESCRIPTION_MAX:
        raise ApiError(
            422,
            PresentationErrors.DESCRIPTION_TOO_LONG,
            f"Description must be at most {DESCRIPTION_MAX} characters",
        )
    return trimmed or None


class PresentationCreate(BaseModel):
    """Тело заказа.

    Умолчания есть у всех полей, кроме шаблона: оформление пользователь выбирает
    сам, и подставить его «по порядку из реестра» значит выдать колоду, которой
    он не заказывал.
    """

    template_key: Annotated[str, AfterValidator(_check_template_key)]
    language: Annotated[str, AfterValidator(_check_language)] = DEFAULT_LANGUAGE
    slide_count: Annotated[int, AfterValidator(_check_slide_count)] = Field(
        default=SLIDE_COUNT_DEFAULT,
        # json_schema_extra, а не ge/le: границы обязаны быть видны в
        # /openapi.json (по ним генератор клиента строит подсказку поля), но
        # ВТОРОЙ проверкой становиться не должны. Field(ge=...) сработал бы
        # раньше _check_slide_count и вернул бы общий request.validation_failed
        # вместо presentation.value_out_of_range.
        json_schema_extra={"minimum": SLIDE_COUNT_MIN, "maximum": SLIDE_COUNT_MAX},
    )
    description: Annotated[str, AfterValidator(_check_description)] | None = None


class PresentationResponse(BaseModel):
    """Заказ в ответах API — один и тот же объект в списке, на опросе и на 202.

    file_path наружу не отдаётся: это абсолютный путь на сервере, клиенту он не
    нужен, а в теле ответа он раскрывал бы устройство хранилища (то же решение,
    что у DocumentRead).

    queue_position приходит только у 'queued' и там всегда >= 1. Ноль означал бы
    «следующая на очереди», то есть ровно обратное «в очереди не стоит», поэтому
    вне очереди тут null (см. PresentationsService.queue_position).

    title и description соседствуют намеренно и разными полями: description
    пользователь пишет ДО генерации как пожелание, title формулирует модель
    ПОСЛЕ неё по найденному материалу. Свести их в одно поле значит подписать
    карточку то одним, то другим — а это разные утверждения о колоде.

    preview_url — путь того же API, как и у шаблона, и он есть ТОЛЬКО у готовой
    колоды: у очередной и упавшей файла нет, и адрес картинки, ведущий в 409,
    был бы обещанием, которого сервер не держит. Null здесь клиент читает как
    «картинки не будет», не делая лишнего запроса.
    """

    id: int
    notebook_id: int
    template_key: str
    language: str
    slide_count: int
    description: str | None = None
    title: str | None = None
    preview_url: str | None = None
    status: str
    progress: int
    queue_position: int | None = None
    error_code: str | None = None
    error_text: str | None = None
    file_size: int | None = None
    created_at: datetime
    updated_at: datetime

    @field_serializer("created_at", "updated_at")
    def _serialize_datetime(self, value: datetime) -> str:
        return serialize_utc(value)


class TemplateResponse(BaseModel):
    """Шаблон в списке выбора.

    name приходит на обоих языках сразу, а не на языке запроса: список
    показывается в форме заказа, где язык интерфейса пользователь переключает
    не перезагружая её.

    preview_url — путь того же API, а не файл: картинки шаблонов лежат вне
    каталога статики (они деплой-артефакт бэкенда), и отдаёт их эндпоинт ниже.
    """

    key: str
    name: dict[str, str]
    preview_url: str


def _presentation_preview_url(presentation_id: int) -> str:
    """Адрес картинки колоды — путь API, а не файл на диске.

    Ровно та же схема, что у превью шаблона (_preview_url ниже) и по той же
    причине: картинки лежат вне каталога статики, отдаёт их эндпоинт, и клиент
    обязан брать адрес из ответа, а не склеивать его сам.
    """
    return f"{settings.API_V1_STR}/presentations/{presentation_id}/preview"


def _to_response(
    presentation: Presentation, queue_position: int | None = None
) -> PresentationResponse:
    return PresentationResponse(
        id=presentation.id,
        notebook_id=presentation.notebook_id,
        template_key=presentation.template_key,
        language=presentation.language,
        slide_count=presentation.slide_count,
        description=presentation.description,
        title=presentation.title,
        preview_url=(
            _presentation_preview_url(presentation.id)
            if presentation.status == STATUS_READY
            else None
        ),
        status=presentation.status,
        progress=presentation.progress,
        queue_position=queue_position,
        error_code=presentation.error_code,
        error_text=presentation.error_text,
        file_size=presentation.file_size,
        created_at=presentation.created_at,
        updated_at=presentation.updated_at,
    )


# --- Шаблоны --------------------------------------------------------------
#
# Оба маршрута объявлены ДО /presentations/{presentation_id}: FastAPI выбирает
# первый подходящий по порядку объявления, и /presentations/templates иначе
# уехал бы в разбор идентификатора.
#
# Обработчики синхронные (def, а не async def) намеренно: FastAPI уводит такие
# в пул потоков, а первое обращение к реестру читает манифест и открывает все
# pptx, чтобы проверить layout'ы. В event loop это остановило бы процесс
# целиком — тот же довод, что у run_in_threadpool в рендере и в ChromaDB.


def _preview_url(key: str) -> str:
    return f"{settings.API_V1_STR}/presentations/templates/{key}/preview"


@router.get("/presentations/templates", response_model=list[TemplateResponse])
def list_presentation_templates(
    _current_user: User = Depends(require_presentation_role),
) -> Any:
    """Шаблоны, пригодные к сборке колоды, в порядке манифеста.

    Битые записи сюда не попадают вовсе (см. TemplateRegistry): предлагать
    выбрать шаблон, на котором генерация потом упадёт, нельзя. Пустой список —
    допустимый ответ: он означает, что комплект шаблонов не доехал до сервера,
    и разбираться с этим будет админ по ERROR в журнале.
    """
    return [
        TemplateResponse(
            key=info.key, name=dict(info.name), preview_url=_preview_url(info.key)
        )
        for info in template_registry.list()
    ]


@router.get("/presentations/templates/{template_key}/preview")
def get_presentation_template_preview(
    template_key: str = Path(..., min_length=1, max_length=64),
    _current_user: User = Depends(require_presentation_role),
) -> FileResponse:
    """Картинка превью по ключу шаблона.

    Путь к файлу берётся ИЗ РЕЕСТРА, а не из запроса, и это единственное, что
    здесь про безопасность: реестр отдаёт только пути, которые сам же запер в
    своих каталогах — исходники в templates/presentations, превью в
    data/presentation_previews. Ключ, которого в реестре нет, отвергается до
    всякой работы с файловой системой, поэтому «../../etc/passwd» здесь —
    просто неизвестный ключ.

    Два разных 404, потому что это две разные беды. Неизвестный ключ —
    unsupported_template: такого шаблона нет, и заказывать по нему нечего.
    Известный ключ без картинки — file_missing: шаблон рабочий и остаётся
    выбираемым, отсутствует только превью (его рисует Chrome, и без браузера
    рисовать некому — см. TemplateInfo.preview_file: Path | None).

    Именно 404, а не заглушка-картинка и не 500. Пятисотка была бы неправдой:
    сервер цел, а шаблон доступен для заказа — галерея из-за отсутствующей
    картинки падать не должна. Подставная картинка была бы неправдой другого
    рода: клиент показал бы её как превью дизайна, то есть соврал бы о том, как
    выглядит колода. 404 говорит ровно то, что есть, и клиент это уже умеет —
    галерея тянет превью XHR'ом и на любой отказ рисует «Превью недоступно»
    (PresentationTemplateGallery.jsx).
    """
    info: TemplateInfo | None = template_registry.get(template_key)
    if info is None:
        raise ApiError(
            404,
            PresentationErrors.UNSUPPORTED_TEMPLATE,
            f"Unknown presentation template {template_key!r}",
        )
    if info.preview_file is None:
        raise ApiError(
            404,
            PresentationErrors.FILE_MISSING,
            f"Presentation template {template_key!r} has no rendered preview",
        )
    media_type = PREVIEW_MEDIA_TYPES.get(
        info.preview_file.suffix.lower(), "application/octet-stream"
    )
    return FileResponse(info.preview_file, media_type=media_type)


# --- Заказ ----------------------------------------------------------------


@router.post(
    "/notebooks/{notebook_id}/presentations",
    response_model=PresentationResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def create_presentation(
    payload: PresentationCreate,
    # Порядок параметров = порядок проверок (см. docstring модуля): владение,
    # роль, частота. Переставить их местами здесь значит переставить порядок
    # ответов 404/403/429.
    notebook: Notebook = Depends(deps.get_owned_notebook),
    current_user: User = Depends(require_presentation_role),
    _rate_limited: None = Depends(enforce_order_rate_limit),
    session: AsyncSession = Depends(deps.get_session),
) -> Any:
    """Поставить заказ в очередь. 202: работа принята, но ещё не сделана.

    Ответ — тот же объект, что отдаёт опрос, вместе с местом в очереди: клиенту
    не нужен второй запрос, чтобы показать «вы третий».
    """
    # (0) Есть ли ЧЕМ печатать. Колоду печатает headless Chrome, и это внешний
    # бинарник: его может не быть на машине вовсе. Проверка стоит первой среди
    # бизнес-условий, потому что она единственная не ходит в базу (ответ
    # закэширован с проверки на старте, chromium_status) и потому что при
    # отсутствующем браузере остальные вопросы бессмысленны: даже блокнот с
    # источниками и свободной очередью напечатать нечем.
    #
    # Проверка своя, а не «пусть упадёт генерация»: без неё заказ встаёт в
    # очередь, доходит до рендера через минуты ожидания и падает
    # generation_failed — то есть пользователь платит временем за то, что было
    # известно в момент клика, а администратор получает жалобу «презентации
    # ломаются» вместо «браузер не установлен».
    #
    # ERROR в журнал с настоящей причиной (какой путь, что ответил бинарник)
    # пишет сам ensure_chromium_available; наружу причина не уходит: имена
    # каталогов сервера пользователю не нужны и ни на что не влияют.
    #
    # Через run_in_threadpool, хотя обычно это попадание в прогретый кэш и
    # стоит наносекунды. Прогревает кэш старт приложения — и ровно тогда,
    # когда старт этого не смог, проверка форкает процесс и ждёт ответа
    # бинарника, до двадцати секунд. То есть синхронный вызов блокировал бы
    # event loop именно в том случае, ради которого он здесь и стоит: браузер
    # сломан или висит. Дешёвая страховка от отказа, наступающего только в
    # плохой день, — тот же довод, что у рендера и клиента ChromaDB.
    try:
        await run_in_threadpool(ensure_chromium_available)
    except RendererUnavailable as exc:
        raise ApiError(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            PresentationErrors.RENDERER_UNAVAILABLE,
            "Presentation renderer is unavailable on the server",
        ) from exc

    # (1) Есть ли из чего собирать колоду. Проверка ДУБЛИРУЕТСЯ в пайплайне и
    # это не лишнее: там она про момент генерации (источники могли удалить,
    # пока задача стояла), а здесь — про момент заказа. Без неё пользователь
    # ждал бы минуты в очереди ради отказа, который виден сразу.
    #
    # Считаем по документам блокнота: владение блокнотом уже проверено выше, а
    # полное разрешение области поиска (resolve_notebook_scope) делает пайплайн
    # — оно ходит в чатовский модуль и здесь дало бы лишь тот же ответ на
    # вопрос «пусто или нет».
    indexed = await session.exec(
        select(func.count())
        .select_from(Document)
        .where(Document.notebook_id == notebook.id, Document.status == "indexed")
    )
    if not int(indexed.first() or 0):
        raise ApiError(
            409,
            PresentationErrors.NO_SOURCES,
            "Notebook has no indexed sources to build a presentation from",
        )

    # (2) Одна колода на блокнот за раз. Очередь берёт по одной задаче, и
    # второй заказ по тому же блокноту занял бы GPU, отложив чужие.
    active = await session.exec(
        select(Presentation.id)
        .where(Presentation.notebook_id == notebook.id)
        .where(Presentation.status.in_(ACTIVE_STATUSES))
        .limit(1)
    )
    if active.first() is not None:
        raise ApiError(
            409,
            PresentationErrors.GENERATION_IN_PROGRESS,
            GENERATION_IN_PROGRESS_DETAIL,
        )

    # Создание — сервисом, а не INSERT'ом по месту: он же будит воркер и
    # нормализует язык. Вторая точка постановки в очередь однажды забыла бы
    # одно из двух.
    #
    # (3) Та же проверка, но уже от БАЗЫ. Проверка (2) выше и вставка — два
    # разных запроса, и между ними помещается весь второй запрос: два клика в
    # одну секунду (или два процесса uvicorn) проходят предпроверку оба и
    # ставят по заказу. Ловит это частичный уникальный индекс
    # uq_presentation_active_notebook, и его нарушение — не 500, а ровно то же
    # событие, что и (2): клиент не должен различать, кто отказал.
    #
    # Чужие нарушения целостности (например, исчезнувший блокнот — внешний
    # ключ) сюда не попадают: они пробрасываются дальше и становятся честной
    # ошибкой сервера, а не ложным «уже генерируется».
    #
    # Идентификаторы для журнала снимаются ДО вставки: неудачная вставка
    # откатывает транзакцию сессии, а откат объявляет все её объекты
    # устаревшими — обращение к notebook.id после него полезло бы в базу за
    # перезагрузкой прямо из обработчика исключения (SQLAlchemy отвечает на это
    # PendingRollbackError).
    notebook_id, user_id = notebook.id, current_user.id
    try:
        presentation = await PresentationsService.create(
            session,
            notebook_id=notebook_id,
            owner_id=user_id,
            template_key=payload.template_key,
            language=payload.language,
            slide_count=payload.slide_count,
            description=payload.description,
        )
    except IntegrityError as exc:
        # Транзакция после нарушения уникальности оборвана: без явного отката
        # эта же сессия ответит InternalError на любой следующий запрос —
        # включая те, что сделает обработчик ошибки.
        await session.rollback()
        if PRESENTATION_ACTIVE_INDEX not in str(exc):
            raise
        logger.info(
            "Duplicate presentation order for notebook %s by user %s was "
            "rejected by %s",
            notebook_id,
            user_id,
            PRESENTATION_ACTIVE_INDEX,
        )
        raise ApiError(
            409,
            PresentationErrors.GENERATION_IN_PROGRESS,
            GENERATION_IN_PROGRESS_DETAIL,
        ) from exc

    queue_position = await PresentationsService.queue_position(session, presentation)
    logger.info(
        "Presentation %s queued for notebook %s by user %s "
        "(template=%s, language=%s, slides=%d, position=%s)",
        presentation.id,
        notebook.id,
        current_user.id,
        presentation.template_key,
        presentation.language,
        presentation.slide_count,
        queue_position,
    )
    return _to_response(presentation, queue_position)


# --- Чтение ---------------------------------------------------------------


@router.get(
    "/notebooks/{notebook_id}/presentations",
    response_model=list[PresentationResponse],
    responses={
        200: {
            "headers": {
                TOTAL_COUNT_HEADER: {
                    "description": (
                        "Общее число заказов этого блокнота без учёта skip/limit."
                    ),
                    "schema": {"type": "integer"},
                }
            }
        }
    },
)
async def list_notebook_presentations(
    response: Response,
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=DEFAULT_PAGE_SIZE, ge=1, le=MAX_PAGE_SIZE),
    notebook: Notebook = Depends(deps.get_owned_notebook),
    session: AsyncSession = Depends(deps.get_session),
) -> Any:
    """Страница заказов блокнота, свежие сверху.

    Границы и заголовок X-Total-Count — те же, что у GET /sources/ (значения
    берутся оттуда же, а не выписаны заново): списки соседствуют на одном
    экране и обязаны листаться одинаково.

    Сортировка created_at DESC, id DESC. Тай-брейк по id обязателен: заказы,
    созданные в одну миллисекунду (а на 202 это ровно то, что делает клиент,
    повторяя запрос), иначе прыгают между соседними страницами.
    """
    page = await session.exec(
        select(Presentation)
        .where(Presentation.notebook_id == notebook.id)
        .order_by(Presentation.created_at.desc(), Presentation.id.desc())
        .offset(skip)
        .limit(limit)
    )
    presentations = page.all()

    total = await session.exec(
        select(func.count())
        .select_from(Presentation)
        .where(Presentation.notebook_id == notebook.id)
    )
    response.headers[TOTAL_COUNT_HEADER] = str(int(total.first() or 0))

    # queue_position считается только для 'queued' (для остальных сервис
    # возвращает None, не заглядывая в БД), а очередных заказов у блокнота по
    # построению не больше одного — см. проверку generation_in_progress.
    return [
        _to_response(
            presentation,
            await PresentationsService.queue_position(session, presentation),
        )
        for presentation in presentations
    ]


@router.get("/presentations/{presentation_id}", response_model=PresentationResponse)
async def get_presentation(
    presentation: Presentation = Depends(get_owned_presentation),
    session: AsyncSession = Depends(deps.get_session),
) -> Any:
    """Один заказ — то, что клиент опрашивает до терминального статуса.

    Прогресс и updated_at здесь важнее самого статуса: по ним видно, что
    генерация жива, а не висит.
    """
    queue_position = await PresentationsService.queue_position(session, presentation)
    return _to_response(presentation, queue_position)


# --- Скачивание -----------------------------------------------------------


def _stored_suffix(path: str) -> str:
    """Расширение хранимого файла в нижнем регистре ('.pdf', '.pptx', '')."""
    return os.path.splitext(path)[1].lower()


def _download_filename(notebook: Notebook | None, suffix: str) -> str:
    """Человеческое имя файла: «{имя блокнота} — презентация.pdf».

    Расширение приходит параметром, а не вписано в шаблон: у файла, собранного
    прежним рендерером, оно .pptx, и колода обязана скачаться ровно тем, чем
    лежит. Имя и тип берутся из одного места — из пути на диске, — поэтому
    разойтись между собой они не могут.

    Чистится и обрезается теми же средствами, что имя источника
    (SourceService.sanitize_display_name -> truncate_name_bytes): имя блокнота
    пользователь пишет сам, оно может содержать слэши и управляющие символы, а
    в UTF-8 255 «символов» имени легко превращаются в 510 байт — больше, чем
    принимает NAME_MAX файловой системы на стороне скачивающего.

    Заголовок Content-Disposition кодирует Starlette (percent-encoding для
    не-ASCII), поэтому экранирование кавычек здесь не наша забота; а вот
    осмысленность и длина имени — наша.
    """
    name = (notebook.name or "").strip() if notebook is not None else ""
    template = DOWNLOAD_NAME_TEMPLATE if name else DOWNLOAD_NAME_FALLBACK
    raw = template.format(notebook=name, suffix=suffix)
    # Один вызов делает обе работы: sanitize_display_name внутри себя зовёт
    # truncate_name_bytes(..., MAX_NAME_BYTES) и сохраняет при обрезке
    # расширение. Дублировать обрезку по месту не нужно — это была бы вторая
    # граница на то же число.
    return SourceService.sanitize_display_name(raw)


@router.get("/presentations/{presentation_id}/download")
async def download_presentation(
    presentation: Presentation = Depends(get_owned_presentation),
    session: AsyncSession = Depends(deps.get_session),
) -> FileResponse:
    """Готовый файл колоды: .pdf у новых, .pptx у собранных прежним рендерером.

    Три исхода вместо одного 404, потому что действия пользователя разные:
    чужая или несуществующая строка — 404 (перебором id ничего не узнать),
    ещё не готовая — 409 (продолжай опрос), готовая без файла — 404
    file_missing (закажи заново).
    """
    if presentation.status != STATUS_READY:
        raise ApiError(
            409,
            PresentationErrors.NOT_READY,
            f"Presentation is not ready yet (status={presentation.status})",
        )
    path = presentation.file_path
    if not path or not os.path.exists(path):
        # Строка обещает файл, которого нет. В журнал — WARNING: это
        # рассинхронизация БД и диска, и знать о ней надо администратору, а не
        # только пользователю, увидевшему отказ.
        logger.warning(
            "Presentation %s is ready but its file is missing: %s",
            presentation.id,
            path or "<empty path>",
        )
        raise ApiError(
            404, PresentationErrors.FILE_MISSING, "Presentation file not found on disk"
        )
    notebook = await session.get(Notebook, presentation.notebook_id)
    # Тип — по расширению ФАЙЛА, а не по тому, что рендерер печатает сегодня:
    # колоды, собранные до перехода на HTML->PDF, лежат в .pptx и обязаны
    # скачиваться ровно как раньше.
    suffix = _stored_suffix(path)
    return FileResponse(
        path,
        media_type=DOWNLOAD_MEDIA_TYPES.get(suffix, FALLBACK_MEDIA_TYPE),
        filename=_download_filename(notebook, suffix),
    )


# --- Превью колоды --------------------------------------------------------


@router.get("/presentations/{presentation_id}/preview")
async def get_presentation_preview(
    presentation: Presentation = Depends(get_owned_presentation),
) -> FileResponse:
    """Первая страница готовой колоды картинкой — превью её карточки в сетке.

    Права — той же зависимостью, что у скачивания (get_owned_presentation), и
    это здесь не формальность: титульный слайд несёт название темы и имя
    блокнота, то есть по превью читается содержимое чужой работы. Второй
    проверки владения быть не должно — разойдясь с первой, она стала бы дырой
    ровно в том месте, где её никто не ищет.

    Три исхода те же, что у скачивания, и по тем же причинам: чужая или
    несуществующая строка — 404 ещё в зависимости, не готовая — 409 (продолжай
    опрос), готовая без пригодного файла — 404 file_missing.

    Рисование ленивое и кэшируется на диск рядом с колодой; почему именно так,
    а не шагом конвейера — в docstring modules/presentations/preview.py. Через
    run_in_threadpool, потому что декодирование страницы блокирует поток на
    сотни миллисекунд, а сетка запрашивает картинки пачкой.
    """
    if presentation.status != STATUS_READY:
        raise ApiError(
            409,
            PresentationErrors.NOT_READY,
            f"Presentation is not ready yet (status={presentation.status})",
        )
    path = presentation.file_path
    if not path or not os.path.exists(path):
        logger.warning(
            "Presentation %s is ready but its file is missing: %s",
            presentation.id,
            path or "<empty path>",
        )
        raise ApiError(
            404, PresentationErrors.FILE_MISSING, "Presentation file not found on disk"
        )

    preview_path = await run_in_threadpool(ensure_card_preview, path)
    if preview_path is None:
        # Файл есть, картинки из него не вышло: .pptx прежнего рендерера или
        # повреждённый PDF. Причину уже записал в журнал сам рендер превью;
        # наружу — тот же код, что у отсутствующего файла (см. FILE_MISSING).
        raise ApiError(
            404,
            PresentationErrors.FILE_MISSING,
            "Presentation preview cannot be rendered from the stored file",
        )
    return FileResponse(preview_path, media_type=CARD_PREVIEW_MEDIA_TYPE)


# --- Удаление -------------------------------------------------------------


@router.delete("/presentations/{presentation_id}")
async def delete_presentation(
    presentation: Presentation = Depends(get_owned_presentation_for_update),
    session: AsyncSession = Depends(deps.get_session),
) -> dict[str, Any]:
    """Удалить заказ и его файл.

    Порядок тот же, что у удаления блокнота: сначала строка и commit, и только
    ПОТОМ os.remove. Обратный порядок при откате транзакции оставил бы строку
    'ready' с путём в никуда — то есть отказ на скачивании уже после «готово».
    Худшее последствие этого порядка — осиротевший файл на диске: он невидим,
    безвреден и находится сверкой каталога с колонкой file_path.

    Сбой os.remove не превращается в 500 по той же причине, что и там: строки
    уже нет, работа сделана, и врать клиенту про несделанное нельзя. Путь
    остаётся в журнале, чтобы файл можно было убрать руками.
    """
    if presentation.status == STATUS_GENERATING:
        # Прервать генерацию нечем: она идёт в другом процессе, канала отмены у
        # очереди нет, а воркер всё равно допишет строку своим исходом. Отказ
        # временный — пайплайн конечен, а хвост убитого процесса очередь сама
        # вернёт в 'queued' (requeue_stuck).
        raise ApiError(
            409,
            PresentationErrors.GENERATION_IN_PROGRESS,
            "Presentation is being generated, try again later",
        )

    presentation_id = presentation.id
    # Путь читается ДО удаления: после commit он не хранится больше нигде, и
    # файл стал бы неотличим от чужого. Фильтра по статусу намеренно нет —
    # заполненная колонка file_path и есть запись о существовании файла.
    file_path = presentation.file_path

    await session.delete(presentation)
    await session.commit()

    if file_path:
        try:
            os.remove(file_path)
        except FileNotFoundError:
            # Файла и так нет — ровно то, чего мы добивались. Проверка
            # os.path.exists перед удалением делала бы то же самое, но с окном
            # гонки между проверкой и вызовом.
            pass
        except OSError as exc:
            logger.warning(
                "Could not remove file %s of deleted presentation %s: %s",
                file_path,
                presentation_id,
                exc,
            )
        # Кэш превью лежит рядом с колодой и вычисляется из её пути — поэтому
        # убирается здесь же, а не отдельным обходом каталога. Оставленный, он
        # был бы файлом, о котором не помнит уже ни одна строка в базе.
        remove_card_preview(file_path)

    return {"detail": "Presentation deleted", "id": presentation_id}
