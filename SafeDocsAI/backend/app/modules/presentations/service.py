"""Очередь презентаций и сам пайплайн генерации.

Две половины, у которых разная природа отказа, поэтому они и лежат рядом, но
не смешаны:

* PresentationsService — операции над строкой presentation. Все они пишут
  updated_at, и это не аккуратность, а условие работы клиента: по updated_at он
  отличает живую генерацию от зависшей. В проекте уже был дефект «updated_at
  заявлен, но никогда не меняется», поэтому единственная точка записи — здесь,
  а не в вызывающем коде, который однажды забудет.

* generate_presentation — сам пайплайн: перечитать источники, спланировать,
  собрать слайды, отрисовать, записать файл. Он НЕ решает, что делать с
  отказом: любая беда поднимается как PresentationGenerationError с готовым
  машинным кодом, а записывает её воркер (worker.py). Так таймаут, отмена и
  обычная ошибка обрабатываются одной веткой, а не тремя похожими.

Ретривал переиспользуется чатовский целиком (run_retrieval, load_chunk_texts,
resolve_notebook_scope): вторая реализация поиска по тем же документам
разошлась бы с первой на первой же правке ранжирования.
"""

import asyncio
import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Callable

from fastapi.concurrency import run_in_threadpool
from sqlalchemy import text
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.core.database import session_context
from app.core.exceptions import ApiError, ExternalServiceError, PresentationErrors
from app.modules.chat.service import (
    load_chunk_texts,
    resolve_notebook_scope,
    run_retrieval,
)
# Вырезание путей на сервере из текста ошибки. Живёт в разделе источников,
# потому что там его завели первым; заводить вторую копию ради презентаций
# значило бы, что однажды одна из них перестанет ловить новый вид пути.
from app.modules.documents.service import redact_server_paths
from app.modules.presentations.constants import (
    MAX_ERROR_TEXT,
    PLAN_RETRIEVAL_TOP_K,
    PRESENTATION_FILE_SUFFIX,
    PRESENTATION_NUM_CTX,
    PRESENTATION_STORAGE_DIR,
    SLIDE_COUNT_MAX,
    SLIDE_COUNT_MIN,
    SLIDE_RETRIEVAL_CANDIDATE_POOL,
    SLIDE_RETRIEVAL_TOP_K,
    STATUS_ERROR,
    STATUS_GENERATING,
    STATUS_QUEUED,
    STATUS_READY,
    normalize_language,
)
from app.modules.presentations.llm_schemas import (
    RENDERER_ADDED_SLIDES,
    LlmResponseError,
    PresentationPlan,
    PresentationSlide,
    validate_plan,
    validate_slide,
)
from app.modules.presentations.prompts import (
    build_context_block,
    build_plan_messages,
    build_retry_messages,
    build_slide_messages,
    build_written_digest,
)
from app.modules.presentations.renderer import RenderedSource, render_presentation
from app.modules.rag.model_manager import ModelManager
from app.services.profile_resolver import resolve_profile
from app.services.rag_service import RAGService
from app.services.runtime_settings_service import RuntimeSettingsService
from app.shared.models import Document, Log, Notebook, Presentation, User

logger = logging.getLogger(__name__)

# Мгновенная побудка воркера после постановки задачи в том же процессе. При
# uvicorn --workers 2 соседний процесс события не увидит и подберёт задачу
# следующим опросом — очередь живёт в БД, а не в памяти. Тот же приём, что у
# очереди индексации.
_QUEUE_WAKEUP = asyncio.Event()


def queue_wakeup() -> asyncio.Event:
    return _QUEUE_WAKEUP


class PresentationGenerationError(Exception):
    """Отказ пайплайна с уже выбранным машинным кодом.

    Код выбирается там, где известна причина, а не там, где ловят исключение:
    «ретривал не дал фрагментов» и «модель дважды вернула мусор» различимы
    только внутри шага, а снаружи выглядят одинаково.
    """

    def __init__(self, error_code: str, message: str) -> None:
        super().__init__(message)
        self.error_code = error_code


# --- Операции над строкой -------------------------------------------------


class PresentationsService:
    @staticmethod
    async def create(
        session: AsyncSession,
        *,
        notebook_id: int,
        owner_id: int,
        template_key: str,
        language: str,
        slide_count: int,
        description: str | None = None,
    ) -> Presentation:
        """Поставить заказ в очередь.

        Проверку прав и границ значений делает вызывающий (этап 2, HTTP-слой):
        здесь она превратилась бы во вторую, расходящуюся с первой.
        """
        presentation = Presentation(
            notebook_id=notebook_id,
            owner_id=owner_id,
            template_key=template_key,
            language=normalize_language(language),
            slide_count=slide_count,
            description=description or None,
            status=STATUS_QUEUED,
            progress=0,
        )
        session.add(presentation)
        await session.commit()
        await session.refresh(presentation)
        _QUEUE_WAKEUP.set()
        return presentation

    @staticmethod
    async def claim_next(session: AsyncSession) -> int | None:
        """Атомарно забрать самую раннюю очередную задачу.

        Захват целиком внутри одного UPDATE с FOR UPDATE SKIP LOCKED: отдельные
        SELECT и UPDATE отдали бы одну задачу обоим процессам uvicorn, потому
        что между ними второй успевает прочитать ту же строку.

        progress сбрасывается в ноль: строка могла вернуться в очередь после
        перезапуска, и показывать пользователю прогресс прошлой попытки —
        значит обещать, что работа продолжится с того места, где встала.
        """
        result = await session.execute(
            text(
                """
                UPDATE presentation
                SET status = :generating,
                    progress = 0,
                    error_code = NULL,
                    error_text = NULL,
                    updated_at = timezone('utc', now())
                WHERE id = (
                    SELECT id FROM presentation
                    WHERE status = :queued
                    ORDER BY created_at, id
                    LIMIT 1
                    FOR UPDATE SKIP LOCKED
                )
                RETURNING id
                """
            ),
            {"queued": STATUS_QUEUED, "generating": STATUS_GENERATING},
        )
        row = result.first()
        await session.commit()
        return None if row is None else int(row[0])

    @staticmethod
    async def queue_position(
        session: AsyncSession, presentation: Presentation
    ) -> int | None:
        """Место в очереди: сколько задач стоит строго раньше, плюс одна.

        Нигде не хранится и считается на каждый запрос статуса. Хранимая
        позиция была бы неверна с первой же соседней задачи, которую забрали
        или отменили, — и никто бы этого не заметил, потому что число
        правдоподобное.

        Не в очереди — позиции нет (None), а не 0: ноль читается как «следующая
        на очереди» и означал бы ровно обратное.
        """
        if presentation.status != STATUS_QUEUED:
            return None
        result = await session.execute(
            text(
                """
                SELECT COUNT(*) FROM presentation
                WHERE status = :queued
                  AND (created_at, id) < (:created_at, :id)
                """
            ),
            {
                "queued": STATUS_QUEUED,
                "created_at": presentation.created_at,
                "id": presentation.id,
            },
        )
        return int(result.scalar_one()) + 1

    @staticmethod
    async def set_progress(session: AsyncSession, presentation_id: int, progress: int) -> None:
        """Прогресс генерации; заодно двигает updated_at.

        Обновление ограничено строкой в работе: если задачу успели вернуть в
        очередь (перезапуск) или отменить, запоздавший прогресс не должен
        воскрешать её статус.
        """
        await session.execute(
            text(
                """
                UPDATE presentation
                SET progress = :progress,
                    updated_at = timezone('utc', now())
                WHERE id = :id AND status = :generating
                """
            ),
            {
                "id": presentation_id,
                "progress": max(0, min(100, int(progress))),
                "generating": STATUS_GENERATING,
            },
        )
        await session.commit()

    @staticmethod
    async def mark_ready(
        session: AsyncSession, presentation_id: int, *, file_path: str, file_size: int
    ) -> None:
        await session.execute(
            text(
                """
                UPDATE presentation
                SET status = :ready,
                    progress = 100,
                    error_code = NULL,
                    error_text = NULL,
                    file_path = :file_path,
                    file_size = :file_size,
                    updated_at = timezone('utc', now())
                WHERE id = :id
                """
            ),
            {
                "id": presentation_id,
                "ready": STATUS_READY,
                "file_path": file_path,
                "file_size": file_size,
            },
        )
        await session.commit()

    @staticmethod
    async def mark_error(
        session: AsyncSession,
        presentation_id: int,
        *,
        error_code: str,
        error_text: str,
    ) -> None:
        await session.execute(
            text(
                """
                UPDATE presentation
                SET status = :error,
                    error_code = :error_code,
                    error_text = :error_text,
                    updated_at = timezone('utc', now())
                WHERE id = :id
                """
            ),
            {
                "id": presentation_id,
                "error": STATUS_ERROR,
                "error_code": error_code,
                "error_text": (error_text or "")[:MAX_ERROR_TEXT],
            },
        )
        await session.commit()

    @staticmethod
    async def requeue(session: AsyncSession, presentation_id: int) -> None:
        """Вернуть прерванную задачу в очередь (штатная остановка сервера)."""
        await session.execute(
            text(
                """
                UPDATE presentation
                SET status = :queued,
                    progress = 0,
                    updated_at = timezone('utc', now())
                WHERE id = :id AND status = :generating
                """
            ),
            {
                "id": presentation_id,
                "queued": STATUS_QUEUED,
                "generating": STATUS_GENERATING,
            },
        )
        await session.commit()
        _QUEUE_WAKEUP.set()

    @staticmethod
    async def requeue_stuck(session: AsyncSession) -> list[int]:
        """Хвосты убитого процесса: 'generating' -> 'queued'.

        Генерация идёт в памяти процесса и целиком: пережить его смерть она не
        может, а частично собранной колоды не существует — файл появляется
        только в самом конце. Поэтому единственное осмысленное состояние
        задачи, оставшейся в 'generating' после перезапуска, — снова 'queued':
        так рестарт не теряет заказ и не оставляет вечно «генерирующуюся»
        строку, на которую пользователь смотрит до конца времён.

        Оговорка честная: под uvicorn --workers 2 старт одного процесса
        приходится на работу другого, и безусловный возврат отберёт у соседа
        живую задачу — она будет сгенерирована дважды, но не потеряна и не
        задвоена в результате (второй проход перезапишет ту же строку и тот же
        файл). Аренда, как у очереди индексации (JobsService.reap_stale), эту
        оговорку снимает, но требует heartbeat'а; заводить его до появления
        второй одновременной генерации преждевременно.
        """
        result = await session.execute(
            text(
                """
                UPDATE presentation
                SET status = :queued,
                    progress = 0,
                    updated_at = timezone('utc', now())
                WHERE status = :generating
                RETURNING id
                """
            ),
            {"queued": STATUS_QUEUED, "generating": STATUS_GENERATING},
        )
        rows = [int(row[0]) for row in result.all()]
        await session.commit()
        if rows:
            _QUEUE_WAKEUP.set()
        return rows


# --- Вспомогательное для пайплайна ---------------------------------------


@dataclass
class GenerationResult:
    """Что пайплайн успел сделать — для журнала блокнота.

    Возвращается, а не пишется на месте, потому что запись в журнал обязана
    случиться и при отказе, а итоговый статус знает только воркер. Пустой
    результат (сорванная генерация) он тоже умеет записать.
    """

    domain_profile: str = ""
    sources: list[RenderedSource] = field(default_factory=list)
    slides: int = 0


def select_slide_chunks(
    candidates: list[dict[str, Any]],
    *,
    used_chunk_ids: set[str],
    top_k: int = SLIDE_RETRIEVAL_TOP_K,
) -> list[dict[str, Any]]:
    """Финальная выборка из пула: предпочесть ещё не цитированные фрагменты.

    Мера 3 правила «не добивать». На небольшом корпусе ретривал отдаёт разным
    секциям почти одинаковые списки, и слайды повторяют друг друга даже при
    честном плане. Отбор из ПУЛА (SLIDE_RETRIEVAL_CANDIDATE_POOL кандидатов,
    отранжированных как в чате) даёт возможность подвинуть уже отработанное,
    не трогая сам ретривал.

    Исключение намеренно МЯГКОЕ:

    * лучший кандидат пула проходит всегда, даже если его уже цитировали.
      Фрагмент, центральный сразу для двух секций, не должен пропасть со
      второго слайда только потому, что он был на первом: слайд без своего
      главного источника — хуже, чем слайд с повтором;
    * дальше идут ещё не цитированные, в порядке ранжирования;
    * и только если новых не хватило до top_k — уже использованные, тоже по
      рангу.

    Порядок в промпте остаётся ранговым: перетасовка выдачи сбивает модели
    представление о том, что важнее.
    """
    if len(candidates) <= top_k:
        return list(candidates)

    fresh: list[int] = []
    stale: list[int] = []
    for index, item in enumerate(candidates[1:], start=1):
        chunk_id = str(item.get("chunk_id") or "")
        (stale if chunk_id in used_chunk_ids else fresh).append(index)

    chosen = {0, *fresh[: top_k - 1]}
    if len(chosen) < top_k:
        chosen.update(stale[: top_k - len(chosen)])
    return [item for index, item in enumerate(candidates) if index in chosen]


async def retrieve_for_query(
    *,
    rag_service: RAGService,
    session: AsyncSession,
    profile: Any,
    language: str,
    search_query: str,
    allowed_doc_ids: set[int] | None,
    notebook_id: int | None,
    final_top_k: int,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Ретривал одной строки запроса плюс тексты найденных чанков.

    final_top_k здесь — размер ВОЗВРАЩАЕМОГО пула, а не число чанков в промпте:
    отбор финальной пятёрки делает select_slide_chunks, и чтобы ему было из
    чего выбирать, ранжирование не должно резать список раньше времени.
    """
    chunks = await run_retrieval(
        rag_service=rag_service,
        session=session,
        profile=profile,
        language=language,
        search_query=search_query,
        allowed_doc_ids=allowed_doc_ids,
        retrieval_top_k=SLIDE_RETRIEVAL_CANDIDATE_POOL,
        final_top_k=final_top_k,
        original_query=search_query,
        notebook_id=notebook_id,
    )
    chunk_texts = await load_chunk_texts(session, chunks)
    return chunks, chunk_texts


async def call_with_one_retry(
    *,
    model_manager: ModelManager,
    model: str,
    messages: list[dict[str, str]],
    validate: Callable[[str], Any],
    label: str,
) -> Any:
    """Вызов модели, разбор и валидация; при провале — ОДНА повторная попытка.

    Повтор получает исходный ответ и текст ошибки валидатора: без них модель
    повторяет ту же ошибку, и вторая попытка тратит время впустую. Второй
    провал — честный отказ наружу, а не «починка» ответа руками: подставить
    недостающий буллет или выбросить лишнюю цитату означало бы выдать за
    проверенный результат то, чего модель не говорила.
    """
    current_messages = messages
    last_error: LlmResponseError | None = None
    for attempt in (1, 2):
        raw = await model_manager.chat(
            model=model,
            messages=current_messages,
            num_ctx=PRESENTATION_NUM_CTX,
        )
        try:
            return validate(raw)
        except LlmResponseError as exc:
            last_error = exc
            logger.warning(
                "Presentation %s: attempt %d rejected: %s",
                label,
                attempt,
                exc.error_text,
            )
            current_messages = build_retry_messages(messages, raw, exc.error_text)
    raise PresentationGenerationError(
        PresentationErrors.GENERATION_FAILED,
        f"{label}: модель дважды вернула ответ, не прошедший проверку "
        f"({last_error.error_text if last_error else 'unknown'})",
    )


def _collect_slide_sources(
    slide: PresentationSlide, chunks: list[dict[str, Any]]
) -> dict[int, tuple[str, list[int]]]:
    """Имена документов и страницы по цитатам одного слайда."""
    by_chunk = {str(item.get("chunk_id") or ""): item for item in chunks}
    collected: dict[int, tuple[str, list[int]]] = {}
    for citation in slide.citations:
        metadata = (by_chunk.get(citation.chunk_id) or {}).get("metadata") or {}
        name = str(metadata.get("doc_name") or "").strip()
        page = metadata.get("page")
        current_name, pages = collected.get(citation.source_id, ("", []))
        if page is not None:
            try:
                pages = [*pages, int(page)]
            except (TypeError, ValueError):
                pass
        collected[citation.source_id] = (current_name or name, pages)
    return collected


def _merge_sources(
    total: dict[int, tuple[str, list[int]]], part: dict[int, tuple[str, list[int]]]
) -> None:
    for source_id, (name, pages) in part.items():
        current_name, current_pages = total.get(source_id, ("", []))
        total[source_id] = (current_name or name, [*current_pages, *pages])


async def _load_indexed_sources(
    session: AsyncSession, *, notebook: Notebook, owner: User
) -> set[int]:
    """Проиндексированные источники блокнота, видимые владельцу заказа.

    Область поиска берётся у чата (resolve_notebook_scope), а не собирается
    заново: правило «чей документ видно» должно быть одно на всю систему.
    Сверх него остаётся только фильтр по статусу — незаконченная индексация в
    ретривал всё равно не попадёт, а пустой блокнот надо отличить от блокнота,
    который прямо сейчас индексируется.
    """
    _, allowed_doc_ids = await resolve_notebook_scope(
        notebook_id=notebook.id, session=session, current_user=owner
    )
    result = await session.exec(
        select(Document.id).where(
            Document.notebook_id == notebook.id, Document.status == "indexed"
        )
    )
    indexed = {doc_id for doc_id in result.all() if doc_id is not None}
    if allowed_doc_ids is None:
        return indexed
    return indexed & allowed_doc_ids


def _presentation_paths(presentation_id: int) -> tuple[str, str]:
    """Итоговый путь файла и временный рядом с ним.

    Временный лежит В ТОМ ЖЕ каталоге намеренно: os.replace атомарен только в
    пределах файловой системы, а /tmp на стенде бывает отдельным томом.
    """
    os.makedirs(PRESENTATION_STORAGE_DIR, exist_ok=True)
    final_path = os.path.join(
        PRESENTATION_STORAGE_DIR,
        f"presentation_{presentation_id}{PRESENTATION_FILE_SUFFIX}",
    )
    temp_path = f"{final_path}.tmp-{uuid.uuid4().hex}"
    return final_path, temp_path


async def write_journal_entry(
    *,
    presentation: Presentation,
    status: str,
    error_code: str | None = None,
    error_text: str | None = None,
    result: GenerationResult,
    elapsed_ms: int,
) -> None:
    """Строка в журнал блокнота: кто, шаблон, язык, слайды, итоговый статус.

    Пишется и на успех, и на отказ: журнал отвечает на вопрос «что тут
    происходило», и генерация, закончившаяся ошибкой, — ровно то событие, ради
    которого в него и заглядывают.

    Автор — владелец заказа, а не None: безавторская запись в этом журнале
    означает потерянного автора (см. require_log_author в modules/chat/
    service.py), а здесь автор известен всегда — owner_id колонка NOT NULL.

    Отказ записи журнала не отменяет результат генерации: файл уже собран, и
    ронять из-за журнала готовую колоду нельзя. Поэтому исключение только
    логируется — тем же приёмом, что persist_chat_log_short_lived.
    """
    try:
        async with session_context() as session:
            session.add(
                Log(
                    question=(
                        f"[презентация #{presentation.id}] "
                        f"шаблон {presentation.template_key}, "
                        f"язык {presentation.language}, "
                        f"слайдов {presentation.slide_count}"
                        + (
                            f", описание: {presentation.description}"
                            if presentation.description
                            else ""
                        )
                    )[:2000],
                    answer=(
                        f"Статус: {status}"
                        if not error_code
                        else f"Статус: {status} ({error_code}): {error_text or ''}"
                    )[:2000],
                    sources=_journal_sources(result.sources),
                    time_ms=elapsed_ms,
                    user_id=presentation.owner_id,
                    notebook_id=presentation.notebook_id,
                    domain_profile=result.domain_profile or None,
                )
            )
            await session.commit()
    except Exception:
        logger.exception(
            "Failed to write the journal entry for presentation %s", presentation.id
        )


def _journal_sources(sources: list[RenderedSource]) -> str | None:
    if not sources:
        return None
    return json.dumps(
        [
            {
                "doc_id": source.source_id,
                "doc_name": source.name,
                "pages": sorted(set(source.pages)),
            }
            for source in sources
        ],
        ensure_ascii=False,
    )


# --- Пайплайн -------------------------------------------------------------


async def generate_presentation(presentation_id: int) -> GenerationResult:
    """Собрать колоду для уже захваченной задачи.

    Ожидает, что строка уже переведена в 'generating' (claim_next). Успех
    коммитит сама; любой отказ поднимает PresentationGenerationError, а
    записывает его воркер. Возвращённый GenerationResult нужен воркеру для
    записи в журнал блокнота.
    """
    started = perf_counter()
    result = GenerationResult()
    model_manager = ModelManager()
    rag_service = RAGService()
    runtime_settings = RuntimeSettingsService.get_settings()
    model = runtime_settings.get("chat_model") or runtime_settings.get("model")

    async with session_context() as session:
        presentation = await session.get(Presentation, presentation_id)
        if presentation is None:
            # Заказ отменили или удалили вместе с блокнотом, пока он ждал
            # очереди. Работать не над чем и записывать некуда.
            logger.info("Presentation %s disappeared before generation", presentation_id)
            return result

        language = normalize_language(presentation.language)
        if not SLIDE_COUNT_MIN <= presentation.slide_count <= SLIDE_COUNT_MAX:
            # Границы проверяет HTTP-слой, но строку могли завести и мимо него
            # (скриптом, чинящим данные). Без проверки план-вызов ушёл бы в
            # Ollama с числом секций, которого не бывает, и отказ пришёл бы
            # минутой позже под видом «модель вернула не то».
            raise PresentationGenerationError(
                PresentationErrors.GENERATION_FAILED,
                f"Число слайдов {presentation.slide_count} вне допустимых границ "
                f"{SLIDE_COUNT_MIN}..{SLIDE_COUNT_MAX}",
            )
        notebook = await session.get(Notebook, presentation.notebook_id)
        owner = await session.get(User, presentation.owner_id)
        if owner is None:
            raise PresentationGenerationError(
                PresentationErrors.GENERATION_FAILED,
                "Владелец заказа не найден: определить область поиска не по чему",
            )

        # (а) Источники перечитываются ЗДЕСЬ, а не при постановке в очередь:
        # пока задача стояла, документы могли удалить или блокнот мог исчезнуть
        # целиком.
        if notebook is None:
            raise PresentationGenerationError(
                PresentationErrors.NO_SOURCES,
                "Блокнот удалён, пока задача стояла в очереди",
            )
        try:
            allowed_doc_ids = await _load_indexed_sources(
                session, notebook=notebook, owner=owner
            )
        except ApiError as exc:
            # resolve_notebook_scope отвечает 404 на блокнот, которого больше
            # нет или который больше не принадлежит владельцу заказа.
            raise PresentationGenerationError(
                PresentationErrors.NO_SOURCES, f"Блокнот недоступен: {exc.detail}"
            ) from exc
        if not allowed_doc_ids:
            raise PresentationGenerationError(
                PresentationErrors.NO_SOURCES,
                "В блокноте не осталось ни одного проиндексированного источника",
            )

        profile = resolve_profile(notebook=notebook)
        result.domain_profile = profile.name
        description = presentation.description or ""
        notebook_name = notebook.name or ""

        # (б) Обзорная выборка и план.
        overview_chunks, overview_texts = await retrieve_for_query(
            rag_service=rag_service,
            session=session,
            profile=profile,
            language=language,
            search_query=description or notebook_name,
            allowed_doc_ids=allowed_doc_ids,
            notebook_id=notebook.id,
            final_top_k=PLAN_RETRIEVAL_TOP_K,
        )
        overview_block, overview_allowed = build_context_block(
            overview_chunks, overview_texts
        )
        if not overview_allowed:
            # Источники в базе есть, а поиск не отдал ни одного фрагмента:
            # сломан индекс, а не блокнот. Не no_sources — совет «добавьте
            # источник» отправил бы пользователя чинить не то.
            raise PresentationGenerationError(
                PresentationErrors.GENERATION_FAILED,
                "Поиск не вернул ни одного фрагмента по источникам блокнота",
            )

        plan: PresentationPlan = await call_with_one_retry(
            model_manager=model_manager,
            model=model,
            messages=build_plan_messages(
                notebook_name=notebook_name,
                description=description,
                language=language,
                slide_count=presentation.slide_count,
                context_block=overview_block,
            ),
            validate=lambda raw: validate_plan(
                raw, slide_count=presentation.slide_count
            ),
            label=f"#{presentation_id} plan",
        )

        # (в) Слайды. Прогресс: 90% делятся поровну между секциями, последние
        # 10% остаются рендеру и записи файла — они не мгновенные, и полоса,
        # застывшая на 100% до появления файла, врала бы.
        slides: list[PresentationSlide] = []
        previous_bullets: list[list[str]] = []
        used_chunk_ids: set[str] = set()
        all_sources: dict[int, tuple[str, list[int]]] = {}
        section_count = len(plan.sections)

        for index, section in enumerate(plan.sections):
            pool, chunk_texts = await retrieve_for_query(
                rag_service=rag_service,
                session=session,
                profile=profile,
                language=language,
                search_query=section.search_query,
                allowed_doc_ids=allowed_doc_ids,
                notebook_id=notebook.id,
                final_top_k=SLIDE_RETRIEVAL_CANDIDATE_POOL,
            )
            selected = select_slide_chunks(pool, used_chunk_ids=used_chunk_ids)
            context_block, allowed_citations = build_context_block(
                selected, chunk_texts
            )
            if not allowed_citations:
                # Запрос секции промахнулся, хотя материал в блокноте есть.
                # Обзорная выборка — честный запасной вариант: это те же
                # документы блокнота, только найденные общим запросом.
                logger.warning(
                    "Presentation %s: section %r retrieved nothing, "
                    "falling back to the overview excerpts",
                    presentation_id,
                    section.heading,
                )
                selected = select_slide_chunks(
                    overview_chunks, used_chunk_ids=used_chunk_ids
                )
                context_block, allowed_citations = build_context_block(
                    selected, overview_texts
                )
            if not allowed_citations:
                raise PresentationGenerationError(
                    PresentationErrors.GENERATION_FAILED,
                    f"Секция «{section.heading}»: поиск не вернул фрагментов",
                )

            slide: PresentationSlide = await call_with_one_retry(
                model_manager=model_manager,
                model=model,
                messages=build_slide_messages(
                    heading=section.heading,
                    description=description,
                    language=language,
                    context_block=context_block,
                    allowed_citations=allowed_citations,
                    # (мера 2) Дайджест уже написанного: без него слайд-вызов
                    # физически не может не повторяться — он не видит
                    # предыдущих.
                    digest=build_written_digest(previous_bullets),
                ),
                validate=lambda raw: validate_slide(
                    raw, allowed_citations=allowed_citations
                ),
                label=f"#{presentation_id} slide {index + 1}",
            )

            slides.append(slide)
            previous_bullets.append(list(slide.bullets))
            used_chunk_ids.update(citation.chunk_id for citation in slide.citations)
            _merge_sources(all_sources, _collect_slide_sources(slide, selected))

            await PresentationsService.set_progress(
                session, presentation_id, 90 * (index + 1) // section_count
            )

        result.slides = len(slides)
        result.sources = [
            RenderedSource(source_id=source_id, name=name or f"#{source_id}", pages=pages)
            for source_id, (name, pages) in sorted(all_sources.items())
        ]

        # (г) и (д): рендер в отдельном потоке, файл — атомарно, и только
        # ПОТОМ commit со status='ready'.
        #
        # Порядок «файл, потом commit» зеркален удалению, где побочные эффекты
        # идут после commit, и по той же причине: строка не должна обещать
        # файл, которого нет. Обратный порядок оставил бы после падения между
        # шагами статус 'ready' с путём в никуда — а это отказ на скачивании,
        # который пользователь увидит уже после «готово». Здесь же худшее
        # последствие — осиротевший файл на диске без строки, невидимый и
        # безвредный.
        file_path, file_size = await _render_to_file(
            presentation_id=presentation_id,
            title=plan.title,
            slides=slides,
            sources=result.sources,
            language=language,
            template_key=presentation.template_key,
            notebook_name=notebook_name,
            created_at=presentation.created_at,
        )
        await PresentationsService.mark_ready(
            session, presentation_id, file_path=file_path, file_size=file_size
        )
        logger.info(
            "Presentation %s is ready: %s slides, %s bytes, %.1fs",
            presentation_id,
            len(slides) + RENDERER_ADDED_SLIDES,
            file_size,
            perf_counter() - started,
        )
    # (ж) Запись в журнал блокнота делает воркер: она обязана случиться и при
    # отказе, а итоговый статус знает только он.
    return result


async def _render_to_file(
    *,
    presentation_id: int,
    title: str,
    slides: list[PresentationSlide],
    sources: list[RenderedSource],
    language: str,
    template_key: str,
    notebook_name: str,
    created_at: Any,
) -> tuple[str, int]:
    """Отрисовать колоду во временный файл и атомарно поставить его на место.

    python-pptx синхронный: он распаковывает и пакует zip, и в event loop это
    блокировка всего процесса — ровно тот дефект, который в проекте уже чинили
    для ChromaDB. Отсюда run_in_threadpool.

    Временный файл подчищается при любом исходе: без этого каждая неудачная
    генерация оставляла бы на диске мусор, который никто не ищет.
    """
    final_path, temp_path = _presentation_paths(presentation_id)
    try:
        await run_in_threadpool(
            render_presentation,
            title=title,
            slides=slides,
            sources=sources,
            language=language,
            template_key=template_key,
            notebook_name=notebook_name,
            created_at=created_at,
            output_path=temp_path,
        )
        file_size = os.path.getsize(temp_path)
        os.replace(temp_path, final_path)
    except PresentationGenerationError:
        raise
    except Exception as exc:
        raise PresentationGenerationError(
            PresentationErrors.GENERATION_FAILED,
            f"Не удалось собрать файл презентации: {type(exc).__name__}: {exc}",
        ) from exc
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:  # pragma: no cover - каталог мог исчезнуть целиком
                logger.warning("Failed to remove temporary file %s", temp_path)
    return final_path, file_size


def error_code_for(exc: BaseException) -> str:
    """Машинный код по исключению, дошедшему до воркера.

    Отдельной функцией, а не цепочкой except в воркере: список кодов раздела
    короткий и обязан быть виден целиком в одном месте, иначе следующая ветка
    отказа получит generation_failed просто потому, что её забыли.
    """
    if isinstance(exc, PresentationGenerationError):
        return exc.error_code
    if isinstance(exc, asyncio.TimeoutError):
        return PresentationErrors.GENERATION_TIMEOUT
    if isinstance(exc, ExternalServiceError) and exc.service == "Ollama":
        return PresentationErrors.OLLAMA_UNAVAILABLE
    return PresentationErrors.GENERATION_FAILED


def error_text_for(exc: BaseException) -> str:
    """Причина отказа для пользователя: без трейсбека и не длиннее предела.

    Пути на сервере вырезаются тем же средством, что и в ошибках индексации
    (redact_server_paths): error_text уходит клиенту вместе со статусом, а
    OSError из рендера охотно подставляет туда полный путь к временному файлу —
    ровно тот дефект, который в разделе источников уже чинили. Полный текст
    остаётся в журнале.
    """
    if isinstance(exc, PresentationGenerationError):
        message = str(exc)
    elif isinstance(exc, asyncio.TimeoutError):
        message = "Генерация не уложилась в отведённое время"
    elif isinstance(exc, ExternalServiceError):
        message = exc.message
    else:
        message = f"{type(exc).__name__}: {exc}"
    return redact_server_paths(message)[:MAX_ERROR_TEXT]
