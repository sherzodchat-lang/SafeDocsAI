"""Фоновый воркер индексации.

HTTP-обработчик upload только кладёт задачу в таблицу job и отвечает; всю
дорогую работу (извлечение текста, OCR, эмбеддинги) делает этот цикл. Очередь
живёт в БД, а не в памяти, поэтому переживает перезапуск процесса, а захват
задачи атомарен и корректен при uvicorn --workers 2.
"""

import asyncio
import contextlib
import logging

from app.core.database import session_context
from app.core.exceptions import SourceErrors
from app.models.models import Document
from app.modules.documents.service import (
    DocumentIndexingError,
    DocumentModuleService,
    _INDEXING_SEMAPHORE,
)
from app.modules.jobs.service import (
    HEARTBEAT_SECONDS,
    JOB_INDEX_DOCUMENT,
    JobsService,
    queue_wakeup,
)

logger = logging.getLogger(__name__)

POLL_INTERVAL_SECONDS = 2.0
RECONCILE_INTERVAL_SECONDS = 30.0
ERROR_BACKOFF_SECONDS = 5.0
STOP_TIMEOUT_SECONDS = 30.0

# error_text уходит в API, а traceback туда не нужен
_MAX_ERROR_TEXT = 500

_INTERRUPTED = "Индексация прервана остановкой сервера"


class IndexingWorker:
    def __init__(self, poll_interval: float = POLL_INTERVAL_SECONDS) -> None:
        self._poll_interval = poll_interval
        self._task: asyncio.Task | None = None

    # -- жизненный цикл --------------------------------------------------
    def start(self) -> None:
        if self._task is not None and not self._task.done():
            return
        self._task = asyncio.create_task(self._run(), name="indexing-worker")
        logger.info("Indexing worker started")

    async def stop(self) -> None:
        task, self._task = self._task, None
        if task is None:
            return
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError, asyncio.TimeoutError):
            await asyncio.wait_for(task, timeout=STOP_TIMEOUT_SECONDS)
        logger.info("Indexing worker stopped")

    async def reconcile(self) -> None:
        """Разобрать хвосты предыдущего процесса.

        Отдельного «режима старта» нет намеренно: при --workers 2 второй
        процесс стартует, когда первый уже может что-то индексировать, и
        безусловный возврат всех 'running' в очередь отобрал бы у него
        живую задачу. Признак смерти — только протухшая аренда.
        """
        try:
            async with session_context() as session:
                jobs = await JobsService.reap_stale(session)
                docs = await DocumentModuleService.reconcile_stuck_documents(session)
        except Exception as exc:
            # Недоступная на старте БД не должна валить приложение: цикл
            # воркера повторит согласование через RECONCILE_INTERVAL_SECONDS.
            logger.warning("Job reconciliation skipped: %s", exc)
            return
        if jobs["requeued"] or jobs["failed"] or docs["requeued"] or docs["failed"]:
            logger.info(
                "Reconciled after restart: jobs requeued=%s failed=%s, "
                "documents requeued=%s failed=%s",
                len(jobs["requeued"]), len(jobs["failed"]),
                docs["requeued"], docs["failed"],
            )

    # -- цикл ------------------------------------------------------------
    async def _run(self) -> None:
        wakeup = queue_wakeup()
        loop = asyncio.get_running_loop()
        # Отсчёт от старта: reconcile() уже вызван снаружи, до запуска цикла.
        last_reconcile = loop.time()
        while True:
            try:
                if loop.time() - last_reconcile >= RECONCILE_INTERVAL_SECONDS:
                    last_reconcile = loop.time()
                    await self.reconcile()
                claimed = await self._claim_and_process()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Indexing worker iteration failed")
                await asyncio.sleep(ERROR_BACKOFF_SECONDS)
                continue
            if claimed:
                continue
            wakeup.clear()
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(wakeup.wait(), timeout=self._poll_interval)

    async def _claim_and_process(self) -> bool:
        # Семафор берём до захвата задачи: иначе задача уже числилась бы
        # 'running', пока воркер стоит в очереди за семафором.
        async with _INDEXING_SEMAPHORE:
            async with session_context() as session:
                job = await JobsService.claim_next(session, JOB_INDEX_DOCUMENT)
                if job is None:
                    return False
                job_id, doc_id = job.id, job.source_id
            await self._process(job_id, doc_id)
        return True

    async def _process(self, job_id: int, doc_id: int | None) -> None:
        heartbeat = asyncio.create_task(self._heartbeat(job_id))
        settled = False
        try:
            async with session_context() as session:
                doc = (
                    await session.get(Document, doc_id) if doc_id is not None else None
                )
                if doc is None:
                    await JobsService.finish(
                        session, job_id, error_text="Документ удалён до индексации"
                    )
                    settled = True
                    return

                # 'indexing' ставит воркер в момент реального старта: раньше
                # это делал upload, и пять параллельных загрузок сразу давали
                # пять «индексируемых» документов, четыре из которых стояли.
                doc.status = "indexing"
                doc.error_text = None
                doc.error_code = None
                session.add(doc)
                await session.commit()

                result = await DocumentModuleService.index_document_job(
                    session=session,
                    doc_id=doc_id,
                    progress_cb=lambda value: self._report_progress(job_id, value),
                )

                doc = await session.get(Document, doc_id)
                if doc is not None:
                    doc.status = "indexed"
                    doc.error_text = None
                    doc.error_code = None
                    session.add(doc)
                    await session.commit()
                await JobsService.finish(session, job_id, result=result)
                settled = True
        except asyncio.CancelledError:
            # CancelledError наследуется от BaseException, и обычный
            # except Exception его не видит. Без этой ветки штатный SIGTERM
            # оставлял бы документ в 'indexing' навсегда.
            settled = True
            await asyncio.shield(self._release(job_id, doc_id))
            raise
        except Exception as exc:
            settled = True
            logger.warning("Indexing job %s failed: %s", job_id, exc, exc_info=True)
            await self._fail(job_id, doc_id, exc)
        finally:
            heartbeat.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await heartbeat
            if not settled:
                # Страховка на случай BaseException мимо обеих веток
                # (KeyboardInterrupt, SystemExit): задача и документ не имеют
                # права остаться в 'running'/'indexing'.
                await asyncio.shield(self._release(job_id, doc_id))

    # -- терминальные состояния ------------------------------------------
    async def _release(self, job_id: int, doc_id: int | None) -> None:
        """Вернуть прерванную задачу в очередь, документ — в ожидание."""
        try:
            # Своя сессия: та, в которой шла работа, уже отменена или в
            # незавершённой транзакции.
            async with session_context() as session:
                status = await JobsService.requeue(
                    session, job_id, error_text=_INTERRUPTED
                )
                doc = (
                    await session.get(Document, doc_id) if doc_id is not None else None
                )
                if doc is not None and doc.status == "indexing":
                    if status == "queued":
                        doc.status = "pending"
                    else:
                        doc.status = "error"
                        doc.error_text = _INTERRUPTED
                        doc.error_code = SourceErrors.INDEXING_INTERRUPTED
                    session.add(doc)
                    await session.commit()
        except Exception:
            # Последний рубеж — reconcile() следующего запуска.
            logger.exception("Failed to release interrupted job %s", job_id)

    async def _fail(self, job_id: int, doc_id: int | None, exc: BaseException) -> None:
        if isinstance(exc, DocumentIndexingError):
            message = str(exc)
        else:
            message = f"{type(exc).__name__}: {exc}"
        message = message[:_MAX_ERROR_TEXT]
        # Код известен только для ожидаемых отказов; всё остальное для клиента
        # неразличимо и переводится одной строкой «ошибка индексации».
        error_code = getattr(exc, "error_code", SourceErrors.INDEXING_FAILED)
        try:
            async with session_context() as session:
                doc = (
                    await session.get(Document, doc_id) if doc_id is not None else None
                )
                if doc is not None:
                    doc.status = "error"
                    doc.error_text = message
                    doc.error_code = error_code
                    session.add(doc)
                    await session.commit()
                await JobsService.finish(session, job_id, error_text=message)
        except Exception:
            logger.exception("Failed to record failure of job %s", job_id)

    # -- телеметрия ------------------------------------------------------
    async def _heartbeat(self, job_id: int) -> None:
        """Продлевает аренду задачи: по ней reconcile() отличает живого
        воркера от мёртвого."""
        while True:
            await asyncio.sleep(HEARTBEAT_SECONDS)
            try:
                async with session_context() as session:
                    await JobsService.heartbeat(session, job_id)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.debug("Heartbeat for job %s failed: %s", job_id, exc)

    async def _report_progress(self, job_id: int, progress: int) -> None:
        try:
            async with session_context() as session:
                await JobsService.heartbeat(session, job_id, progress=progress)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.debug("Progress update for job %s failed: %s", job_id, exc)
