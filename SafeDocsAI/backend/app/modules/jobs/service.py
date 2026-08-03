import asyncio
import json
import logging
from typing import Any

from sqlalchemy import text
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.models.models import Job

logger = logging.getLogger(__name__)

JOB_INDEX_DOCUMENT = "index_document"

# Отложенное удаление векторов из ChromaDB. Ставится, когда строки chunk из
# PostgreSQL уже удалены, а ChromaDB в этот момент недоступна: id векторов
# больше нигде не хранятся, и без такой задачи вычистить их было бы нечем.
# payload: {"chunk_ids": [...], "notebook_id": <id или null>}.
JOB_CLEANUP_EMBEDDINGS = "cleanup_embeddings"

# Прерванная задача возвращается в очередь не бесконечно: документ, который
# стабильно роняет воркер, иначе занимал бы очередь после каждого рестарта.
MAX_ATTEMPTS = 3

# У очистки векторов бюджет попыток отдельный и намного больше: она идемпотентна
# (удаление несуществующих id — no-op), а причина отказа обычно временная —
# лежащая ChromaDB. Сдаться после трёх попыток значило бы оставить висячие
# векторы навсегда. Между попытками воркер выдерживает паузу, см. worker.py.
CLEANUP_MAX_ATTEMPTS = 240

# Аренда задачи. Воркер обновляет started_at каждые HEARTBEAT_SECONDS, поэтому
# 'running' со старым started_at принадлежит процессу, которого уже нет.
# Живой воркер под тем же условием не пострадает: его heartbeat моложе аренды.
HEARTBEAT_SECONDS = 15
LEASE_SECONDS = 60

# Мгновенная побудка воркера после enqueue в том же процессе. При
# uvicorn --workers 2 соседний процесс события не увидит и подберёт задачу
# следующим опросом — очередь живёт в БД, а не в памяти.
_QUEUE_WAKEUP = asyncio.Event()


def queue_wakeup() -> asyncio.Event:
    return _QUEUE_WAKEUP


class JobsService:
    @staticmethod
    async def enqueue(
        session: AsyncSession,
        job_type: str,
        payload: dict[str, Any] | None = None,
        *,
        source_id: int | None = None,
        notebook_id: int | None = None,
        created_by: int | None = None,
    ) -> Job:
        job = Job(
            job_type=job_type,
            status="queued",
            payload_json=json.dumps(payload or {}, ensure_ascii=False),
            source_id=source_id,
            notebook_id=notebook_id,
            created_by=created_by,
        )
        session.add(job)
        await session.commit()
        await session.refresh(job)
        _QUEUE_WAKEUP.set()
        return job

    @staticmethod
    async def list_jobs(session: AsyncSession, limit: int = 100) -> list[Job]:
        result = await session.exec(
            select(Job).order_by(Job.created_at.desc()).limit(limit)
        )
        return result.all()

    @staticmethod
    async def claim_next(
        session: AsyncSession, job_type: str = JOB_INDEX_DOCUMENT
    ) -> Job | None:
        """Атомарно забрать одну задачу из очереди.

        docker-compose поднимает uvicorn --workers 2, а значит воркеров тоже
        два. Отдельные SELECT и UPDATE отдали бы одну задачу обоим процессам:
        между ними другой воркер успевает прочитать ту же строку. Захват
        целиком внутри одного UPDATE с FOR UPDATE SKIP LOCKED этого не
        допускает — второй процесс просто пропустит заблокированную строку.
        """
        result = await session.execute(
            text(
                """
                UPDATE job
                SET status = 'running',
                    started_at = timezone('utc', now()),
                    finished_at = NULL,
                    error_text = NULL,
                    progress = 0,
                    attempt_count = attempt_count + 1
                WHERE id = (
                    SELECT id FROM job
                    WHERE status = 'queued' AND job_type = :job_type
                    ORDER BY created_at, id
                    LIMIT 1
                    FOR UPDATE SKIP LOCKED
                )
                RETURNING id
                """
            ),
            {"job_type": job_type},
        )
        row = result.first()
        await session.commit()
        if row is None:
            return None
        return await session.get(Job, row[0])

    @staticmethod
    async def heartbeat(
        session: AsyncSession, job_id: int, *, progress: int | None = None
    ) -> None:
        """Продлить аренду задачи и (опционально) обновить прогресс."""
        await session.execute(
            text(
                """
                UPDATE job
                SET started_at = timezone('utc', now()),
                    progress = COALESCE(CAST(:progress AS INTEGER), progress)
                WHERE id = :job_id AND status = 'running'
                """
            ),
            {"job_id": job_id, "progress": progress},
        )
        await session.commit()

    @staticmethod
    async def finish(
        session: AsyncSession,
        job_id: int,
        *,
        result: dict[str, Any] | None = None,
        error_text: str | None = None,
    ) -> None:
        """Терминальный статус задачи по id, без привязки к ORM-объекту.

        Ошибку пишем из отдельной сессии, поэтому объект Job из сессии,
        в которой упала транзакция, здесь непригоден.
        """
        await session.execute(
            text(
                """
                UPDATE job
                SET status = CASE
                        WHEN CAST(:error_text AS TEXT) IS NULL THEN 'completed'
                        ELSE 'failed'
                    END,
                    result_json = CAST(:result_json AS TEXT),
                    error_text = CAST(:error_text AS TEXT),
                    progress = 100,
                    finished_at = timezone('utc', now())
                WHERE id = :job_id
                """
            ),
            {
                "job_id": job_id,
                "result_json": (
                    json.dumps(result, ensure_ascii=False) if result is not None else None
                ),
                "error_text": error_text,
            },
        )
        await session.commit()

    @staticmethod
    async def requeue(
        session: AsyncSession,
        job_id: int,
        *,
        error_text: str | None = None,
        max_attempts: int = MAX_ATTEMPTS,
    ) -> str | None:
        """Вернуть прерванную задачу в очередь.

        Возвращает итоговый статус ('queued'/'failed') или None, если задача
        уже не в работе. После max_attempts попыток задача считается
        безнадёжной и закрывается как failed; бюджет попыток задаётся вызовом,
        потому что у разных типов задач цена отказа разная
        (см. CLEANUP_MAX_ATTEMPTS).
        """
        result = await session.execute(
            text(
                """
                UPDATE job
                SET status = CASE
                        WHEN attempt_count >= :max_attempts THEN 'failed'
                        ELSE 'queued'
                    END,
                    error_text = CAST(:error_text AS TEXT),
                    started_at = NULL,
                    finished_at = CASE
                        WHEN attempt_count >= :max_attempts
                        THEN timezone('utc', now())
                    END
                WHERE id = :job_id AND status = 'running'
                RETURNING status
                """
            ),
            {
                "job_id": job_id,
                "error_text": error_text,
                "max_attempts": max_attempts,
            },
        )
        row = result.first()
        await session.commit()
        if row is not None and row[0] == "queued":
            _QUEUE_WAKEUP.set()
        return row[0] if row is not None else None

    @staticmethod
    async def reap_stale(
        session: AsyncSession, *, lease_seconds: float = LEASE_SECONDS
    ) -> dict[str, list[int]]:
        """Освободить задачи, чей воркер умер, не закрыв их.

        Отличать мёртвого воркера от живого по одному лишь статусу нельзя
        (при --workers 2 старт одного процесса пришёлся бы на работу другого),
        поэтому признак — протухший heartbeat в started_at.

        Бюджет попыток зависит от типа задачи: у очистки векторов он свой,
        иначе задача, пережившая несколько отказов ChromaDB, закрывалась бы
        как failed при первом же падении воркера.
        """
        result = await session.execute(
            text(
                """
                UPDATE job
                SET status = CASE
                        WHEN attempt_count >= CASE
                            WHEN job_type = CAST(:cleanup_type AS TEXT)
                            THEN CAST(:cleanup_max_attempts AS INTEGER)
                            ELSE CAST(:max_attempts AS INTEGER)
                        END THEN 'failed'
                        ELSE 'queued'
                    END,
                    error_text = CAST(:error_text AS TEXT),
                    started_at = NULL,
                    finished_at = CASE
                        WHEN attempt_count >= CASE
                            WHEN job_type = CAST(:cleanup_type AS TEXT)
                            THEN CAST(:cleanup_max_attempts AS INTEGER)
                            ELSE CAST(:max_attempts AS INTEGER)
                        END
                        THEN timezone('utc', now())
                    END
                WHERE status = 'running'
                  AND (
                      started_at IS NULL
                      OR started_at < timezone('utc', now())
                                      - make_interval(secs => CAST(:lease AS DOUBLE PRECISION))
                  )
                RETURNING id, status
                """
            ),
            {
                "error_text": "Обработка прервана: воркер не отвечает",
                "max_attempts": MAX_ATTEMPTS,
                "cleanup_type": JOB_CLEANUP_EMBEDDINGS,
                "cleanup_max_attempts": CLEANUP_MAX_ATTEMPTS,
                "lease": float(lease_seconds),
            },
        )
        rows = result.all()
        await session.commit()
        requeued = [row[0] for row in rows if row[1] == "queued"]
        failed = [row[0] for row in rows if row[1] == "failed"]
        if requeued:
            _QUEUE_WAKEUP.set()
        return {"requeued": requeued, "failed": failed}

    @staticmethod
    async def mark_running(session: AsyncSession, job: Job) -> Job:
        job.status = "running"
        job.attempt_count += 1
        session.add(job)
        await session.commit()
        await session.refresh(job)
        return job

    @staticmethod
    async def mark_finished(
        session: AsyncSession,
        job: Job,
        *,
        result: dict[str, Any] | None = None,
        error_text: str | None = None,
    ) -> Job:
        job.status = "failed" if error_text else "completed"
        job.result_json = (
            json.dumps(result or {}, ensure_ascii=False) if result is not None else None
        )
        job.error_text = error_text
        job.progress = 100
        session.add(job)
        await session.commit()
        await session.refresh(job)
        return job
