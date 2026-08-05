"""HTTP-слой раздела тем: распределение, сведения о модели, переразметка.

Три решения этого файла, которые важнее остального кода.

**Раздел без модели — пустой, а не сломанный.** GET /topics отвечает пустым
массивом, когда обученной модели нет: тема — украшение документа, и её
отсутствие не является отказом. Единственная точка, где «модели нет»
рассказывается прямо, — GET /topics/model: туда за состоянием и ходят, и
пустой объект там означал бы «модель есть, но про неё ничего не известно».

**Владение проверяется тем же способом, что в источниках.** Распределение по
блокноту сначала проверяет владение (assert_owns_notebook — тот же 404 и на
чужой блокнот, и на несуществующий, поэтому оракулом для перебора id раздел не
становится), а выборка сужается _owner_filter'ом из раздела источников. Именно
импортом, а не своей копией того же выражения: разъехавшись, вторая копия
показала бы одному пользователю темы чужих документов, и заметили бы это по
числам, а не по ошибке.

**Переразметка — за админом и не больше одной сразу.** Она проходит по всем
документам системы, а не по своим: обычному пользователю такая кнопка не
положена (403), а вторая одновременная задача не ускорит первую и удвоит
нагрузку на ChromaDB (409). «Не больше одной» держит частичный уникальный
индекс в БД, а предпроверка в обработчике остаётся: она отвечает без нарушения
целостности и без отката транзакции, то есть обычный повторный клик стоит
дешевле.
"""

import logging
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, Query, status
from pydantic import BaseModel, field_serializer
from sqlalchemy.exc import IntegrityError
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.api import deps
# _owner_filter и serialize_utc — из раздела источников, а не своей копией:
# правило «админ видит всё, остальные своё» и формат даты обязаны быть одни на
# весь API. Даты тем показываются на одном экране с датами источников.
from app.api.endpoints.documents import _owner_filter, serialize_utc
from app.core.database import (
    TOPIC_REASSIGN_ACTIVE_STATUSES,
    TOPIC_REASSIGN_ACTIVE_INDEX,
    TOPIC_REASSIGN_JOB_TYPE,
)
from app.core.exceptions import ApiError, TopicErrors
from app.modules.topics.service import TopicsService, queue_wakeup
from app.shared.models import Job, User

router = APIRouter()
logger = logging.getLogger(__name__)

MODEL_MISSING_DETAIL = (
    "Topic model is not trained yet: no artifact has been registered. "
    "Train it (backend/cluster_topics.py) and restart the backend."
)
REASSIGN_IN_PROGRESS_DETAIL = (
    "Topic reassignment is already queued or running; wait for it to finish"
)


class TopicRead(BaseModel):
    """Одна тема в распределении.

    share — доля от размеченных АКТИВНОЙ моделью документов выборки, а не от
    всех документов вообще: сумма долей должна давать единицу, иначе диаграмма
    на клиенте не сойдётся. Документы без темы в знаменатель не входят и
    видны как разница между суммой document_count и числом источников.
    """

    cluster_index: int
    label: str
    document_count: int
    share: float


class TopicMetrics(BaseModel):
    """Метрики защиты. Каждая может быть null.

    Ноль вместо отсутствующего числа поставить нельзя: нулевой ARI — это
    осмысленный результат («совпадение с темами на уровне случайного»), и
    подменять им «не считали» значит показать на экране оценку, которой никто
    не получал.
    """

    ari_topic: float | None = None
    purity: float | None = None
    silhouette: float | None = None


class TopicModelRead(BaseModel):
    trained_at: datetime
    k: int
    embedding_model: str
    # Строкой, а не объектом: клиенту это подпись под моделью («что сделали с
    # вектором перед сравнением»), а не данные для расчёта. Сами вычитаемые
    # векторы живут в артефакте и наружу не уходят — их там мегабайты.
    transform: str
    metrics: TopicMetrics
    cluster_count: int

    @field_serializer("trained_at")
    def _serialize_trained_at(self, value: datetime) -> str:
        # Тем же сериализатором, что даты источников: в колонке лежит UTC без
        # смещения, и без явного Z браузер в UTC+5 показал бы дату обучения на
        # пять часов раньше настоящей.
        return serialize_utc(value)


class ReassignAccepted(BaseModel):
    """Ответ на постановку переразметки.

    Отдаётся id задачи, а не «ок»: прогресс переразметки виден в таблице job, и
    без id найти там свою строку нечем.
    """

    job_id: int
    status: str
    model_version: int


@router.get("/topics", response_model=list[TopicRead])
async def read_topics(
    notebook_id: int | None = Query(default=None, ge=1, le=deps.MAX_ID),
    session: AsyncSession = Depends(deps.get_session),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    """Распределение документов по темам.

    С notebook_id — внутри блокнота, без него — по всем доступным вызывающему
    документам. Владение при заданном notebook_id проверяется ДО выборки, тем
    же assert_owns_notebook, что в GET /sources/: иначе чужой блокнот отвечал
    бы пустым распределением, и клиент не мог бы отличить «блокнота нет» от «в
    блокноте нет размеченных документов».

    Модели нет — пустой массив, а не 404. Раздел тем в этом состоянии не
    сломан, он пуст: показывать пользователю ошибку там, где нечего показывать,
    значит требовать от него действия, которого он совершить не может (обучение
    модели — работа администратора и делается вне продукта).
    """
    if notebook_id is not None:
        await deps.assert_owns_notebook(notebook_id, session, current_user)
    model = await TopicsService.active_model(session)
    if model is None:
        return []
    rows = await TopicsService.distribution(
        session,
        model,
        notebook_id=notebook_id,
        owner_id=_owner_filter(current_user),
    )
    return [
        TopicRead(
            cluster_index=row.cluster_index,
            label=row.label,
            document_count=row.document_count,
            share=row.share,
        )
        for row in rows
    ]


@router.get("/topics/model", response_model=TopicModelRead)
async def read_topic_model(
    session: AsyncSession = Depends(deps.get_session),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    """Чем сейчас размечаются документы.

    Читает РЕЕСТР, а не файл. Так ответ не зависит от того, лежит ли артефакт
    на месте прямо сейчас, и — главное — называет ту версию, которой размечены
    документы, а не ту, которую последней положили на диск. Регистрация новой
    версии происходит на старте приложения и при переразметке: чтение, которое
    само меняет базу, гонялось бы с соседним процессом uvicorn и превращало
    случайный запрос пользователя в миграцию.
    """
    model = await TopicsService.active_model(session)
    if model is None:
        raise ApiError(404, TopicErrors.MODEL_MISSING, MODEL_MISSING_DETAIL)
    metrics = TopicsService.metrics_of(model)
    return TopicModelRead(
        trained_at=model.trained_at,
        k=model.k,
        embedding_model=model.embedding_model,
        transform=model.transform,
        metrics=TopicMetrics(**metrics),
        cluster_count=model.cluster_count,
    )


@router.post(
    "/topics/reassign",
    response_model=ReassignAccepted,
    status_code=status.HTTP_202_ACCEPTED,
)
async def reassign_topics(
    session: AsyncSession = Depends(deps.get_session),
    current_user: User = Depends(deps.get_current_active_superuser),
) -> Any:
    """Переразметить все проиндексированные документы активной моделью.

    202, а не 200: работа только поставлена в очередь. Она идёт по всем
    документам системы и занимает минуты — держать на ней HTTP-запрос значило
    бы, что переразметка живёт ровно столько, сколько браузер согласен ждать.

    Реестр сводится с диском прямо здесь: администратор нажимает эту кнопку
    ИМЕННО после переобучения, и требовать от него ещё и перезапуска бэкенда
    ради регистрации нового артефакта — лишний шаг, о котором он узнает по
    неизменившимся темам.
    """
    model = await TopicsService.sync_active_model(session)
    if model is None:
        raise ApiError(404, TopicErrors.MODEL_MISSING, MODEL_MISSING_DETAIL)

    # (1) Предпроверка: отвечает без нарушения целостности и без отката
    # транзакции, поэтому обычный повторный клик стоит дешевле.
    active = await session.exec(
        select(Job.id)
        .where(Job.job_type == TOPIC_REASSIGN_JOB_TYPE)
        .where(Job.status.in_(TOPIC_REASSIGN_ACTIVE_STATUSES))
        .limit(1)
    )
    if active.first() is not None:
        raise ApiError(
            409, TopicErrors.REASSIGN_IN_PROGRESS, REASSIGN_IN_PROGRESS_DETAIL
        )

    # (2) Та же проверка, но уже от БАЗЫ: между (1) и вставкой помещается весь
    # второй запрос, и два клика в одну секунду (или два процесса uvicorn)
    # проходят предпроверку оба. Ловит это частичный уникальный индекс
    # uq_topic_reassign_active, и его нарушение — не 500, а ровно то же
    # событие, что и (1): клиент не должен различать, кто отказал.
    #
    # Чужие нарушения целостности пробрасываются дальше и становятся честной
    # ошибкой сервера, а не ложным «уже идёт».
    from app.modules.jobs.service import JobsService

    user_id = current_user.id
    try:
        job = await JobsService.enqueue(
            session,
            TOPIC_REASSIGN_JOB_TYPE,
            {"model_version": model.version},
            created_by=user_id,
        )
    except IntegrityError as exc:
        # Транзакция после нарушения уникальности оборвана: без явного отката
        # эта же сессия ответит InternalError на любой следующий запрос.
        await session.rollback()
        if TOPIC_REASSIGN_ACTIVE_INDEX not in str(exc):
            raise
        logger.info(
            "Повторная переразметка тем от пользователя %s отклонена индексом %s",
            user_id,
            TOPIC_REASSIGN_ACTIVE_INDEX,
        )
        raise ApiError(
            409, TopicErrors.REASSIGN_IN_PROGRESS, REASSIGN_IN_PROGRESS_DETAIL
        ) from exc

    # Воркер тем слушает своё событие: JobsService.enqueue будит очередь
    # индексации, а она про этот тип задач ничего не знает.
    queue_wakeup().set()
    logger.info(
        "Переразметка тем %s поставлена в очередь пользователем %s (модель версии %s)",
        job.id,
        user_id,
        model.version,
    )
    return ReassignAccepted(
        job_id=job.id, status=job.status, model_version=model.version
    )
