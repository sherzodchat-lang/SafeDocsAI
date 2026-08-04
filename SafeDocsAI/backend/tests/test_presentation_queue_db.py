"""Очередь генерации презентаций на настоящем PostgreSQL.

Половина смысла очереди — в самом PostgreSQL: FOR UPDATE SKIP LOCKED, порядок
(created_at, id) и подсчёт «строго раньше» на sqlite или на моках проверить
нечем. Поэтому схему создаёт код проекта (init_db) в отдельной базе — со всеми
внешними ключами и индексами, включая ix_presentation_queue, ради которого
выборка очереди и написана так, как написана.

Ollama и ChromaDB здесь не поднимаются: сам пайплайн подменён — проверяется
цикл вокруг него, а не то, что он рисует.
"""

import asyncio
import os
import sys
import unittest
from contextlib import asynccontextmanager
from datetime import timedelta
from unittest.mock import AsyncMock, patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dbfixtures import DatabaseBackedTestCase  # noqa: E402

from app.core.exceptions import PresentationErrors  # noqa: E402
from app.modules.presentations.constants import (  # noqa: E402
    SLIDE_COUNT_MAX,
    SLIDE_COUNT_MIN,
    STATUS_ERROR,
    STATUS_GENERATING,
    STATUS_QUEUED,
    STATUS_READY,
    presentation_job_timeout,
)
from app.modules.presentations.service import (  # noqa: E402
    GenerationResult,
    PresentationsService,
)
from app.modules.presentations.worker import PresentationWorker  # noqa: E402
from app.shared.models import Notebook, Presentation, User, utcnow  # noqa: E402
from app.shared.settings.config import settings as app_settings  # noqa: E402


# Воркер не берёт задачи, пока не задана embedding-модель (без неё имя
# коллекции ChromaDB не вывести, а презентация без поиска — пересказ пустоты).
# Здесь проверяется не это, поэтому модель задана как на рабочем стенде.
EMBEDDING_MODEL = "qwen3-embedding:8b"


class PresentationQueueTestCase(DatabaseBackedTestCase):
    """Блокнот, владелец и воркер с подставленными тестовыми сессиями."""

    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()

        self.user: User = await self.make_user("owner", "user")
        self.as_user(self.user)
        self._notebooks = 0

        env_patcher = patch.object(
            app_settings, "OLLAMA_MODEL_EMBEDDING", EMBEDDING_MODEL
        )
        env_patcher.start()
        self.addCleanup(env_patcher.stop)

        # Воркер и сервис живут вне запроса и берут сессии сами.
        for target in (
            "app.modules.presentations.worker.session_context",
            "app.modules.presentations.service.session_context",
        ):
            patcher = patch(target, self._worker_session)
            patcher.start()
            self.addCleanup(patcher.stop)

        self.worker = PresentationWorker(poll_interval=0.05)
        self.addAsyncCleanup(self.worker.stop)

    @asynccontextmanager
    async def _worker_session(self):
        async with self.session_factory() as session:
            try:
                yield session
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()

    # --- данные ---

    async def make_notebook(self, name: str | None = None) -> Notebook:
        self._notebooks += 1
        return await self.seed(
            Notebook(
                name=name or f"Блокнот {self._notebooks}",
                domain_profile="tax",
                owner_id=self.user.id,
            )
        )

    async def make_presentation(self, *, created_at=None, **overrides) -> Presentation:
        """Заказ в очереди. По умолчанию — в СВОЁМ блокноте.

        Собственный блокнот на каждый заказ не украшение фикстуры, а следствие
        инварианта базы: частичный уникальный индекс
        uq_presentation_active_notebook не допускает двух активных
        ('queued'/'generating') заказов по одному блокноту. Очередь при этом
        общая на систему, и всё, что проверяется здесь — порядок, позиция,
        захват, — от того, чьи это блокноты, не зависит: ждущие заказы в
        рабочей базе принадлежат разным блокнотам ровно по той же причине.
        """
        fields = {
            "notebook_id": (await self.make_notebook()).id,
            "owner_id": self.user.id,
            "template_key": "classic",
            "language": "ru",
            "slide_count": 5,
            "status": STATUS_QUEUED,
        }
        fields.update(overrides)
        if created_at is not None:
            fields["created_at"] = created_at
            fields["updated_at"] = created_at
        return await self.seed(Presentation(**fields))

    async def claim(self) -> int | None:
        async with self.session_factory() as session:
            return await PresentationsService.claim_next(session)

    async def reload(self, presentation_id: int) -> Presentation:
        row = await self.get_row(Presentation, presentation_id)
        self.assertIsNotNone(row)
        return row

    async def wait_until(self, condition, timeout: float = 5.0) -> None:
        """Дождаться состояния в базе, не завися от скорости машины."""
        deadline = asyncio.get_running_loop().time() + timeout
        while asyncio.get_running_loop().time() < deadline:
            if await condition():
                return
            await asyncio.sleep(0.02)
        self.fail("Условие не выполнилось за отведённое время")


class ClaimTests(PresentationQueueTestCase):
    async def test_claim_takes_the_earliest_and_flips_the_status(self):
        now = utcnow()
        first = await self.make_presentation(created_at=now - timedelta(minutes=2))
        second = await self.make_presentation(created_at=now - timedelta(minutes=1))

        self.assertEqual(await self.claim(), first.id)
        self.assertEqual(await self.claim(), second.id)
        self.assertIsNone(await self.claim())

        claimed = await self.reload(first.id)
        self.assertEqual(claimed.status, STATUS_GENERATING)
        self.assertEqual(claimed.progress, 0)

    async def test_claim_ignores_rows_that_are_not_queued(self):
        now = utcnow()
        await self.make_presentation(
            created_at=now - timedelta(minutes=3), status=STATUS_READY
        )
        await self.make_presentation(
            created_at=now - timedelta(minutes=2), status=STATUS_ERROR
        )
        queued = await self.make_presentation(created_at=now - timedelta(minutes=1))

        self.assertEqual(await self.claim(), queued.id)
        self.assertIsNone(await self.claim())

    async def test_claim_clears_the_error_of_a_previous_attempt(self):
        """Повторно поставленный заказ не должен показывать старую ошибку."""
        presentation = await self.make_presentation(
            status=STATUS_QUEUED,
            error_code=PresentationErrors.GENERATION_FAILED,
            error_text="сломалось в прошлый раз",
        )
        await self.claim()

        claimed = await self.reload(presentation.id)
        self.assertIsNone(claimed.error_code)
        self.assertIsNone(claimed.error_text)


class QueuePositionTests(PresentationQueueTestCase):
    async def test_position_counts_only_earlier_queued_rows(self):
        now = utcnow()
        rows = [
            await self.make_presentation(created_at=now - timedelta(minutes=minutes))
            for minutes in (3, 2, 1)
        ]

        async with self.session_factory() as session:
            positions = [
                await PresentationsService.queue_position(session, row) for row in rows
            ]
        self.assertEqual(positions, [1, 2, 3])

    async def test_position_shifts_as_the_queue_moves(self):
        now = utcnow()
        first = await self.make_presentation(created_at=now - timedelta(minutes=2))
        second = await self.make_presentation(created_at=now - timedelta(minutes=1))

        await self.claim()

        async with self.session_factory() as session:
            taken = await PresentationsService.queue_position(
                session, await self.reload(first.id)
            )
            waiting = await PresentationsService.queue_position(
                session, await self.reload(second.id)
            )
        # Взятая в работу из очереди выбывает — позиции у неё больше нет.
        self.assertIsNone(taken)
        self.assertEqual(waiting, 1)

    async def test_rows_created_in_the_same_moment_are_ordered_by_id(self):
        moment = utcnow()
        first = await self.make_presentation(created_at=moment)
        second = await self.make_presentation(created_at=moment)

        async with self.session_factory() as session:
            self.assertEqual(
                await PresentationsService.queue_position(session, first), 1
            )
            self.assertEqual(
                await PresentationsService.queue_position(session, second), 2
            )


class UpdatedAtTests(PresentationQueueTestCase):
    async def test_every_status_change_moves_updated_at(self):
        """Дефект «updated_at заявлен, но никогда не меняется» — не здесь."""
        presentation = await self.make_presentation()
        stamps = [presentation.updated_at]

        await self.claim()
        stamps.append((await self.reload(presentation.id)).updated_at)

        async with self.session_factory() as session:
            await PresentationsService.set_progress(session, presentation.id, 45)
        stamps.append((await self.reload(presentation.id)).updated_at)

        async with self.session_factory() as session:
            await PresentationsService.mark_ready(
                session, presentation.id, file_path="/tmp/x.pptx", file_size=10
            )
        stamps.append((await self.reload(presentation.id)).updated_at)

        for earlier, later in zip(stamps, stamps[1:]):
            self.assertLess(earlier, later)

    async def test_error_also_moves_updated_at(self):
        presentation = await self.make_presentation()
        await self.claim()
        before = (await self.reload(presentation.id)).updated_at

        async with self.session_factory() as session:
            await PresentationsService.mark_error(
                session,
                presentation.id,
                error_code=PresentationErrors.GENERATION_FAILED,
                error_text="x",
            )

        after = await self.reload(presentation.id)
        self.assertEqual(after.status, STATUS_ERROR)
        self.assertLess(before, after.updated_at)

    async def test_progress_does_not_resurrect_a_requeued_row(self):
        """Запоздавший прогресс не должен возвращать строку в работу."""
        presentation = await self.make_presentation()
        await self.claim()
        async with self.session_factory() as session:
            await PresentationsService.requeue(session, presentation.id)
            await PresentationsService.set_progress(session, presentation.id, 70)

        row = await self.reload(presentation.id)
        self.assertEqual(row.status, STATUS_QUEUED)
        self.assertEqual(row.progress, 0)


class RestartRecoveryTests(PresentationQueueTestCase):
    async def test_generating_rows_return_to_the_queue_on_start(self):
        stuck = await self.make_presentation(status=STATUS_GENERATING, progress=60)
        untouched = await self.make_presentation(status=STATUS_READY, progress=100)

        with self.assertLogs("app.modules.presentations.worker", level="INFO") as logs:
            await self.worker.recover()

        recovered = await self.reload(stuck.id)
        self.assertEqual(recovered.status, STATUS_QUEUED)
        # Прогресс прошлой попытки обещал бы продолжение с того же места.
        self.assertEqual(recovered.progress, 0)
        self.assertLess(stuck.updated_at, recovered.updated_at)
        self.assertEqual((await self.reload(untouched.id)).status, STATUS_READY)
        self.assertTrue(any(str(stuck.id) in line for line in logs.output))

    async def test_recovered_row_is_claimed_again(self):
        stuck = await self.make_presentation(status=STATUS_GENERATING, progress=60)
        await self.worker.recover()
        self.assertEqual(await self.claim(), stuck.id)

    async def test_recovery_survives_an_unavailable_database(self):
        """Недоступная на старте БД не должна валить приложение."""
        with patch.object(
            PresentationsService, "requeue_stuck", side_effect=RuntimeError("db is down")
        ):
            with self.assertLogs("app.modules.presentations.worker", level="WARNING"):
                await self.worker.recover()


class WorkerLoopTests(PresentationQueueTestCase):
    """Цикл вокруг пайплайна: таймаут, устойчивость к падению джобы, FIFO."""

    async def test_job_ceiling_marks_the_row_with_the_timeout_code(self):
        """Потолок джобы — страховка от зависания МЕЖДУ стадиями.

        Стадии ограничены сами (LLM_CALL_TIMEOUT, см.
        tests/test_presentation_pipeline_db.py), и до этого потолка доходит
        только беда, которой отдельный wait_for не видит: бесконечный цикл,
        зависший ретривал, повисшее соединение с базой. Пайплайн здесь подменён
        целиком — именно такую беду он и изображает.
        """
        presentation = await self.make_presentation()

        async def never_finishes(_presentation_id, **_kwargs):
            await asyncio.sleep(30)

        with patch(
            "app.modules.presentations.worker.generate_presentation", never_finishes
        ), patch.object(
            self.worker, "_job_ceiling", AsyncMock(return_value=0.05)
        ):
            self.assertTrue(await self.worker._claim_and_process())

        row = await self.reload(presentation.id)
        self.assertEqual(row.status, STATUS_ERROR)
        self.assertEqual(row.error_code, PresentationErrors.GENERATION_TIMEOUT)
        self.assertTrue(row.error_text)

    async def test_job_ceiling_is_derived_from_the_order_and_logged_at_start(self):
        """Потолок не назначен числом, а посчитан из slide_count заказа.

        И посчитан он ДО первого await пайплайна: заказ на пять слайдов и заказ
        на пятнадцать не имеют права ждать одинаково, а узнать, из чего взялся
        потолок, должно быть можно из лога, а не из чтения constants.py.
        """
        presentation = await self.make_presentation(slide_count=SLIDE_COUNT_MIN)

        with self.assertLogs("app.modules.presentations.worker", level="INFO") as logs:
            ceiling = await self.worker._job_ceiling(presentation.id)

        self.assertEqual(ceiling, presentation_job_timeout(SLIDE_COUNT_MIN))
        self.assertLess(ceiling, presentation_job_timeout(SLIDE_COUNT_MAX))
        start_line = next(
            line for line in logs.output if "потолок джобы" in line
        )
        self.assertIn(str(SLIDE_COUNT_MIN), start_line)
        self.assertIn(f"{ceiling:.0f}", start_line)

    async def test_unreadable_order_falls_back_to_the_longest_deck(self):
        """Ошибка в длинную сторону: недоступная база не укорачивает потолок."""
        with patch(
            "app.modules.presentations.worker.session_context",
            side_effect=RuntimeError("db is down"),
        ):
            with self.assertLogs(
                "app.modules.presentations.worker", level="WARNING"
            ):
                ceiling = await self.worker._job_ceiling(1)

        self.assertEqual(ceiling, presentation_job_timeout(SLIDE_COUNT_MAX))

    async def test_call_statistics_reach_the_log_of_every_job(self):
        """p50/p90 по вызовам модели пишутся и у джобы, которая упала.

        Ради этой строки CallTimings и заводит воркер, а не пайплайн: у
        сорвавшейся джобы результата нет, а вызовы — есть, и смотрят в лог
        именно на них.
        """
        presentation = await self.make_presentation()

        async def records_calls(_presentation_id, *, timings=None):
            timings.record(stage="план презентации", attempt=1, seconds=10.0)
            for index, seconds in enumerate((20.0, 30.0, 40.0)):
                timings.record(
                    stage=f"слайд {index + 1} из 3", attempt=1, seconds=seconds
                )
            raise RuntimeError("модель выдала мусор")

        with patch(
            "app.modules.presentations.worker.generate_presentation", records_calls
        ):
            with self.assertLogs(
                "app.modules.presentations.worker", level="INFO"
            ) as logs:
                self.assertTrue(await self.worker._claim_and_process())

        stats_line = next(line for line in logs.output if "p50" in line)
        self.assertIn("вызовов модели 4", stats_line)
        self.assertIn("p50 20.0с", stats_line)
        self.assertIn("p90 40.0с", stats_line)
        self.assertIn(str(presentation.id), stats_line)
        self.assertEqual(
            (await self.reload(presentation.id)).status, STATUS_ERROR
        )

    async def test_two_orders_run_one_at_a_time_in_the_order_they_were_placed(self):
        """FIFO и ПО ОДНОЙ — ради этого очередь и существует.

        Порядок захвата проверен на самом claim_next (ClaimTests), но там нет
        цикла: между двумя захватами лежит вся обработка джобы, и наложение
        возможно ровно здесь — в `_run`, который после успешного захвата сразу
        уходит на следующую итерацию (`if claimed: continue`). Две генерации
        одновременно — это две модели в памяти одной видеокарты: либо OOM, либо
        обе задачи ползут вдвое дольше, а позиция в очереди начинает врать.

        Соседний тест про упавшую джобу тоже смотрит на порядок, но там первая
        задача падает, то есть до наложения дело не доходит по построению.
        Здесь обе успешны, и время работы каждой заведомо перекрывает опрос
        очереди (poll_interval 0.05 с) — если бы цикл брал вторую, не дождавшись
        первой, он успел бы это сделать.
        """
        now = utcnow()
        first = await self.make_presentation(created_at=now - timedelta(minutes=2))
        second = await self.make_presentation(created_at=now - timedelta(minutes=1))

        events: list[str] = []
        running = 0
        peak = 0

        async def slow(presentation_id, **_kwargs):
            nonlocal running, peak
            running += 1
            peak = max(peak, running)
            events.append(f"start {presentation_id}")
            await asyncio.sleep(0.1)
            async with self.session_factory() as session:
                await PresentationsService.mark_ready(
                    session, presentation_id, file_path="/tmp/x.pptx", file_size=1
                )
            events.append(f"done {presentation_id}")
            running -= 1
            return GenerationResult()

        with patch("app.modules.presentations.worker.generate_presentation", slow):
            self.worker.start()
            await self.wait_until(lambda: self._both_settled(first.id, second.id))
            await self.worker.stop()

        self.assertEqual(peak, 1, "воркер взял вторую задачу, не отпустив первую")
        self.assertEqual(
            events,
            [
                f"start {first.id}",
                f"done {first.id}",
                f"start {second.id}",
                f"done {second.id}",
            ],
        )
        for presentation in (first, second):
            self.assertEqual((await self.reload(presentation.id)).status, STATUS_READY)

    async def test_exception_inside_a_job_does_not_kill_the_worker(self):
        """Упавшая генерация — строка со status='error', а не мёртвая очередь.

        Перезапустить воркер до рестарта сервера некому, поэтому первый же
        неудачный заказ отменил бы функцию для всех остальных.
        """
        now = utcnow()
        broken = await self.make_presentation(created_at=now - timedelta(minutes=2))
        healthy = await self.make_presentation(created_at=now - timedelta(minutes=1))
        seen: list[int] = []

        async def flaky(presentation_id, **_kwargs):
            seen.append(presentation_id)
            if presentation_id == broken.id:
                raise RuntimeError("модель выдала мусор")
            async with self.session_factory() as session:
                await PresentationsService.mark_ready(
                    session, presentation_id, file_path="/tmp/x.pptx", file_size=1
                )
            return GenerationResult()

        with patch("app.modules.presentations.worker.generate_presentation", flaky):
            self.worker.start()
            await self.wait_until(
                lambda: self._both_settled(broken.id, healthy.id)
            )

        # Цикл жив и продолжает опрашивать очередь.
        self.assertFalse(self.worker._task.done())
        await self.worker.stop()

        failed = await self.reload(broken.id)
        self.assertEqual(failed.status, STATUS_ERROR)
        self.assertEqual(failed.error_code, PresentationErrors.GENERATION_FAILED)
        self.assertIn("модель выдала мусор", failed.error_text)
        self.assertEqual((await self.reload(healthy.id)).status, STATUS_READY)
        # FIFO: упавшая задача была первой по created_at.
        self.assertEqual(seen[:2], [broken.id, healthy.id])

    async def _both_settled(self, *ids: int) -> bool:
        for presentation_id in ids:
            row = await self.get_row(Presentation, presentation_id)
            if row is None or row.status in (STATUS_QUEUED, STATUS_GENERATING):
                return False
        return True

    async def test_queue_is_not_touched_without_an_embedding_model(self):
        """Без embedding-модели заказ ждёт, а не падает в ошибку."""
        presentation = await self.make_presentation()
        with patch.object(app_settings, "OLLAMA_MODEL_EMBEDDING", ""), patch(
            "app.shared.settings.runtime_settings.RuntimeSettingsService."
            "embedding_model",
            return_value="",
        ):
            with self.assertLogs("app.modules.presentations.worker", level="ERROR"):
                self.assertFalse(await self.worker._claim_and_process())

        self.assertEqual((await self.reload(presentation.id)).status, STATUS_QUEUED)

    async def test_interrupted_job_returns_to_the_queue(self):
        """Остановка сервера — не вина заказа: он снова встаёт в очередь."""
        presentation = await self.make_presentation()
        started = asyncio.Event()

        async def never_finishes(_presentation_id, **_kwargs):
            started.set()
            await asyncio.sleep(30)

        with patch(
            "app.modules.presentations.worker.generate_presentation", never_finishes
        ):
            self.worker.start()
            await asyncio.wait_for(started.wait(), timeout=5)
            await self.worker.stop()

        row = await self.reload(presentation.id)
        self.assertEqual(row.status, STATUS_QUEUED)
        self.assertEqual(row.progress, 0)


if __name__ == "__main__":
    unittest.main()
