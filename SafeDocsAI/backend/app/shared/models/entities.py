from datetime import datetime, timezone
from typing import Optional

from sqlmodel import Field, SQLModel


def utcnow() -> datetime:
    """Текущее время UTC без tzinfo.

    Колонки created_at/updated_at имеют тип TIMESTAMP WITHOUT TIME ZONE, и всё
    уже накопленное в них — UTC. Пишем naive-значение, чтобы не смешивать в
    одной колонке aware и naive; смещение проставляется на сериализации
    (см. as_utc). Заодно уходит устаревший datetime.utcnow().
    """
    return datetime.now(timezone.utc).replace(tzinfo=None)


def as_utc(value: datetime) -> datetime:
    """Пометить значение из БД как UTC.

    Naive-datetime без смещения JS трактует как локальное время: для UTC+5
    дата в интерфейсе уезжала на пять часов назад.
    """
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


class NotebookBase(SQLModel):
    name: str = Field(index=True)
    description: Optional[str] = None
    domain_profile: str = Field(default="tax", index=True)


class Notebook(NotebookBase, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    # Владелец обязателен. Блокнот без владельца был legacy-состоянием
    # (колонка появилась позже первых блокнотов) и вынуждал каждую проверку
    # владения помнить про особый случай «ничей — значит админский». Старые
    # строки бэкфиллит init_db, там же на колонку ставится NOT NULL.
    owner_id: int = Field(foreign_key="user.id")
    created_at: datetime = Field(default_factory=utcnow)


class UserBase(SQLModel):
    username: str = Field(index=True, unique=True)
    role: str = Field(default="user")


class User(UserBase, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    password_hash: str
    # Поколение токенов. Значение попадает в каждый выданный токен и
    # сверяется на каждом запросе: инкремент разом обесценивает все ранее
    # выданные токены пользователя (смена пароля, понижение роли, «выйти
    # со всех устройств») без смены общего SECRET_KEY.
    token_version: int = Field(default=0)
    created_at: datetime = Field(default_factory=utcnow)


class RefreshToken(SQLModel, table=True):
    """Выданный refresh-токен. В БД, а не в памяти процесса: ротация должна
    переживать перезапуск и работать одинаково при нескольких воркерах.

    Сам токен не хранится — только его jti, поэтому дамп таблицы не даёт
    возможности войти.
    """

    id: Optional[int] = Field(default=None, primary_key=True)
    jti: str = Field(index=True, unique=True)
    user_id: int = Field(foreign_key="user.id", index=True)
    issued_at: datetime = Field(default_factory=utcnow)
    expires_at: datetime = Field(index=True)
    # Проставляется при использовании или при отзыве: непустое значение
    # означает, что токен больше не принимается.
    revoked_at: Optional[datetime] = None
    # jti токена, выданного взамен. Предъявление уже заменённого токена —
    # признак того, что копия утекла.
    replaced_by: Optional[str] = None


class DocumentBase(SQLModel):
    name: str = Field(index=True)
    path: str
    size: int
    language: str = Field(default="ru")
    status: str = Field(default="pending")
    # Причина статуса 'error'. Без неё пользователь видит только «ошибка»
    # и не может отличить битый файл от упавшего Ollama.
    error_text: Optional[str] = None
    # Тот же отказ машинным кодом (app.core.exceptions.SourceErrors): error_text
    # написан на одном языке, а интерфейс переведён на три.
    error_code: Optional[str] = None


class Document(DocumentBase, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    notebook_id: Optional[int] = Field(
        default=None, foreign_key="notebook.id", nullable=True, index=True
    )
    # Владелец хранится на документе, а не выводится из блокнота: notebook_id
    # nullable, и у документа, загруженного вне блокнота, иначе не было бы
    # владельца вообще.
    #
    # Владелец обязателен. Документ без владельца был legacy-состоянием
    # (колонка появилась позже первых документов) и вынуждал каждую проверку
    # владения помнить про особый случай «ничей — значит админский». Старые
    # строки бэкфиллит init_db (документ наследует владельца своего блокнота,
    # иначе достаётся старейшему админу), там же на колонку ставится NOT NULL.
    owner_id: int = Field(foreign_key="user.id", index=True)
    created_at: datetime = Field(default_factory=utcnow)


class ChunkBase(SQLModel):
    text: str
    page: int
    chunk_index: Optional[int] = None
    section: Optional[str] = None
    embedding_id: Optional[str] = None


class Chunk(ChunkBase, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    doc_id: int = Field(foreign_key="document.id", ondelete="CASCADE", index=True)


class LogBase(SQLModel):
    question: str
    answer: str
    sources: Optional[str] = None
    time_ms: int
    rating: Optional[str] = None


class Log(LogBase, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: Optional[int] = Field(foreign_key="user.id", nullable=True)
    notebook_id: Optional[int] = Field(
        default=None, foreign_key="notebook.id", nullable=True, index=True
    )
    domain_profile: Optional[str] = Field(default=None, index=True)
    created_at: datetime = Field(default_factory=utcnow)


class NoteBase(SQLModel):
    title: str = Field(index=True)
    body: str = ""
    kind: str = Field(default="manual")
    status: str = Field(default="active")


class Note(NoteBase, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    notebook_id: int = Field(foreign_key="notebook.id", index=True)
    created_by: Optional[int] = Field(
        default=None, foreign_key="user.id", nullable=True
    )
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)


class InsightBase(SQLModel):
    title: str = Field(index=True)
    body: str = ""
    insight_type: str = Field(default="summary")
    evidence_json: Optional[str] = None


class Insight(InsightBase, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    notebook_id: int = Field(foreign_key="notebook.id", index=True)
    note_id: Optional[int] = Field(default=None, foreign_key="note.id", nullable=True)
    created_by: Optional[int] = Field(
        default=None, foreign_key="user.id", nullable=True
    )
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)


class PresentationBase(SQLModel):
    # Ключ шаблона оформления (app/modules/presentations/templates.py). Строка,
    # а не enum: набор шаблонов пополняется файлами в каталоге, а не миграцией.
    template_key: str
    # Код языка колоды. Каноническое значение — 'ru' или 'tj'
    # (см. app/modules/presentations/constants.py: ISO был бы tg, но проект
    # везде обозначает таджикский как tj).
    language: str = Field(default="ru")
    # ИТОГОВОЕ число слайдов в файле, включая титульный и «Источники»
    # (см. RENDERER_ADDED_SLIDES в modules/presentations/llm_schemas.py).
    slide_count: int
    description: Optional[str] = None
    # queued -> generating -> ready | error. Индекс по колонке нужен не сам по
    # себе (выборку очереди покрывает составной ix_presentation_queue из
    # init_db), а для «сколько презентаций в работе» без сканирования таблицы.
    status: str = Field(default="queued", index=True)
    progress: int = Field(default=0)
    # Причина отказа: машинный код для интерфейса (PresentationErrors) и текст
    # для человека — ровно та же пара, что у document.error_code/error_text.
    error_code: Optional[str] = None
    error_text: Optional[str] = None
    # Заполняются только у status='ready': путь к файлу колоды на диске и его
    # размер. Расширение из пути не выводится и здесь не закреплено: новые
    # колоды печатает Chrome в .pdf, а собранные прежним рендерером .pptx
    # остаются на диске и обязаны скачиваться ровно как прежде.
    file_path: Optional[str] = None
    file_size: Optional[int] = None


class Presentation(PresentationBase, table=True):
    """Заказ на генерацию презентации и её результат в одной строке.

    Своя таблица, а не запись в job. Job сцеплен с семантикой индексации
    (source_id, attempt_count, аренда через started_at, бюджет попыток), его
    внешние ключи уже приходилось чинить, и путь удаления блокнота вокруг него
    закалён. Самодостаточная строка ничего из этого не трогает и читается
    целиком одним SELECT — включая то, что показывается пользователю.

    updated_at обязан меняться при КАЖДОЙ смене статуса и на каждом шаге
    прогресса: по нему клиент понимает, что генерация жива, а не висит. В
    проекте уже был дефект «updated_at заявлен, но никогда не меняется»
    (note/insight), поэтому здесь его двигает единственная точка записи —
    PresentationsService (app/modules/presentations/service.py), а не
    вызывающий код.
    """

    id: Optional[int] = Field(default=None, primary_key=True)
    notebook_id: int = Field(foreign_key="notebook.id", index=True)
    # Владелец обязателен С РОЖДЕНИЯ таблицы: колонка заводится вместе с ней, и
    # legacy-строк без владельца здесь не будет никогда. Именно из-за поздней
    # колонки notebook.owner_id и document.owner_id проверки владения годами
    # помнили особый случай «ничей — значит админский», а init_db до сих пор
    # возит бэкфилл. Второй раз этот путь не проходим.
    owner_id: int = Field(foreign_key="user.id", index=True)
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)


class JobBase(SQLModel):
    job_type: str = Field(index=True)
    status: str = Field(default="queued", index=True)
    payload_json: Optional[str] = None
    result_json: Optional[str] = None
    error_text: Optional[str] = None
    progress: int = Field(default=0)
    attempt_count: int = Field(default=0)


class Job(JobBase, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    source_id: Optional[int] = Field(
        default=None, foreign_key="document.id", nullable=True, index=True
    )
    notebook_id: Optional[int] = Field(
        default=None, foreign_key="notebook.id", nullable=True, index=True
    )
    created_by: Optional[int] = Field(
        default=None, foreign_key="user.id", nullable=True
    )
    created_at: datetime = Field(default_factory=utcnow)
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
