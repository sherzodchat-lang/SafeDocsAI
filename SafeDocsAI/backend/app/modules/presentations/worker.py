"""Фоновый воркер генерации презентаций.

Отдельная корутина в том же lifespan, что и воркер индексации, со своим циклом
и своей очередью. Общего у них только устройство: очередь живёт в БД, а не в
памяти процесса, поэтому переживает перезапуск, а захват задачи атомарен и
корректен при uvicorn --workers 2.

Почему не задача в таблице job. Job сцеплен с семантикой индексации, его
внешние ключи уже приходилось чинить, а путь удаления блокнота вокруг него
закалён; самодостаточная строка presentation ничего из этого не трогает.
Почему не тот же цикл. Генерация занимает минуты и держит GPU; поставленная в
одну очередь с индексацией, она задержала бы загрузку документов ровно на своё
время, а очередь индексации — самое чувствительное место продукта.

Главное свойство цикла: ИСКЛЮЧЕНИЕ ВНУТРИ ДЖОБЫ ЕГО НЕ РОНЯЕТ. Упавшая
генерация — это строка со status='error', а не остановленная очередь: воркер
некому перезапустить до следующего рестарта сервера, и первый же неудачный
заказ отменил бы функцию для всех остальных.
"""

import asyncio
import contextlib
import logging
from time import perf_counter

from app.core.database import session_context
from app.core.exceptions import PresentationErrors, SettingsErrors
from app.modules.presentations.constants import (
    ERROR_BACKOFF_SECONDS,
    LLM_CALL_ATTEMPTS,
    LLM_CALL_TIMEOUT,
    LLM_CALL_WATCHDOG_TIMEOUT,
    LLM_RETRY_PAUSE_AFTER_TIMEOUT,
    POLL_INTERVAL_SECONDS,
    SLIDE_COUNT_MAX,
    STATUS_ERROR,
    STATUS_READY,
    STOP_TIMEOUT_SECONDS,
    presentation_job_timeout,
)
from app.modules.presentations.llm_schemas import content_section_count
from app.modules.presentations.service import (
    CallTimings,
    GenerationResult,
    PresentationsService,
    error_code_for,
    error_text_for,
    generate_presentation,
    queue_wakeup,
    write_journal_entry,
)
from app.shared.models import Presentation

logger = logging.getLogger(__name__)

# Как часто напоминать в журнале, что очередь стоит из-за незаданной
# embedding-модели. Значение и довод — те же, что у воркера индексации.
UNSET_MODEL_LOG_INTERVAL_SECONDS = 300.0

_INTERRUPTED = "Генерация прервана остановкой сервера"


class PresentationWorker:
    def __init__(self, poll_interval: float = POLL_INTERVAL_SECONDS) -> None:
        self._poll_interval = poll_interval
        self._task: asyncio.Task | None = None
        self._unset_model_logged_at = float("-inf")

    # -- жизненный цикл --------------------------------------------------
    def start(self) -> None:
        if self._task is not None and not self._task.done():
            return
        self._task = asyncio.create_task(self._run(), name="presentation-worker")
        logger.info("Presentation worker started")

    async def stop(self) -> None:
        task, self._task = self._task, None
        if task is None:
            return
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError, asyncio.TimeoutError):
            await asyncio.wait_for(task, timeout=STOP_TIMEOUT_SECONDS)
        logger.info("Presentation worker stopped")

    async def recover(self) -> None:
        """Согласовать раздел с диском: вернуть прерванное в очередь и
        восстановить заголовки колод, у которых их нет.

        Генерация не имеет промежуточного состояния на диске: файл появляется
        целиком и в самом конце. Поэтому 'generating', переживший перезапуск, —
        это не работа, которую можно продолжить, а работа, которую надо начать
        заново.

        Заголовки — вторая половина того же согласования, и стоит она здесь по
        той же причине, по какой здесь стоит первая: обе приводят строки БД в
        соответствие с тем, что реально лежит на диске, и обе обязаны случиться
        ДО того, как раздел начнёт отвечать пользователю. Колонка title
        появилась позже самой таблицы, и у колод, собранных раньше, значение
        лежит только в файле (см. PresentationsService.backfill_titles).

        Отсутствие БД на старте не должно валить приложение: следующая итерация
        цикла всё равно ничего не захватит, а согласование повторится при
        следующем запуске.
        """
        try:
            async with session_context() as session:
                requeued = await PresentationsService.requeue_stuck(session)
                restored = await PresentationsService.backfill_titles(session)
        except Exception as exc:
            logger.warning("Presentation reconciliation skipped: %s", exc)
            return
        if requeued:
            logger.info(
                "Presentations returned to the queue after restart: %s", requeued
            )
        if restored:
            logger.info(
                "Restored titles of %s presentations from their PDF files", restored
            )

    # -- цикл ------------------------------------------------------------
    async def _run(self) -> None:
        wakeup = queue_wakeup()
        while True:
            try:
                claimed = await self._claim_and_process()
            except asyncio.CancelledError:
                raise
            except Exception:
                # Сюда доходит только беда САМОГО цикла: недоступная БД,
                # неожиданный отказ захвата. Ошибки конкретной генерации
                # разбираются внутри _process и до этой ветки не поднимаются.
                logger.exception("Presentation worker iteration failed")
                await asyncio.sleep(ERROR_BACKOFF_SECONDS)
                continue
            if claimed:
                continue
            wakeup.clear()
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(wakeup.wait(), timeout=self._poll_interval)

    def _embedding_model_is_set(self) -> bool:
        """Есть ли чем искать. Иначе задачи не берём вовсе.

        Ровно то же решение, что у воркера индексации
        (IndexingWorker._embedding_model_is_set), и по той же причине: имя
        коллекции ChromaDB выводится из embedding-модели, а без поиска
        презентация — это пересказ пустоты.

        Захватить и провалить нельзя: у раздела нет и не должно быть кода
        «модель не выбрана» — причина не в заказе и не в блокноте, а в одной
        ненастроенной строке, которую админ исправит за минуту. Заказ остаётся
        'queued' с честной позицией в очереди и уедет в работу сам, как только
        модель выберут. Это тот же ответ по смыслу, что 503 на HTTP: «запрос
        верный, повторим позже».
        """
        from app.shared.settings.runtime_settings import RuntimeSettingsService

        if RuntimeSettingsService.embedding_model():
            return True
        now = asyncio.get_running_loop().time()
        if now - self._unset_model_logged_at >= UNSET_MODEL_LOG_INTERVAL_SECONDS:
            self._unset_model_logged_at = now
            logger.error(
                "Presentation queue is paused: embedding model is not set (%s). "
                "Orders stay 'queued'; pick a model in the admin panel or set "
                "OLLAMA_MODEL_EMBEDDING.",
                SettingsErrors.EMBEDDING_MODEL_UNSET,
            )
        return False

    async def _claim_and_process(self) -> bool:
        if not self._embedding_model_is_set():
            return False
        async with session_context() as session:
            presentation_id = await PresentationsService.claim_next(session)
        if presentation_id is None:
            return False
        await self._process(presentation_id)
        return True

    async def _process(self, presentation_id: int) -> None:
        """Одна генерация целиком, со всеми возможными исходами.

        Таймаут живёт на уровне ВЫЗОВА (LLM_CALL_TIMEOUT вокруг каждой стадии в
        service.py), а не здесь. Потолок джобы остался, но он теперь выведенный
        и служит страховкой от беды, которой отдельный wait_for не видит по
        построению, — от бесконечного цикла между стадиями.
        """
        started = perf_counter()
        timings = CallTimings()
        ceiling = await self._job_ceiling(presentation_id)
        try:
            result = await asyncio.wait_for(
                generate_presentation(presentation_id, timings=timings),
                timeout=ceiling,
            )
        except asyncio.CancelledError:
            # CancelledError наследуется от BaseException, и except Exception
            # его не видит. Без этой ветки штатный SIGTERM оставлял бы заказ в
            # 'generating' навсегда.
            await asyncio.shield(self._release(presentation_id))
            raise
        except Exception as exc:
            await self._fail(presentation_id, exc, started, ceiling=ceiling)
            return
        finally:
            # Строка статистики пишется при ЛЮБОМ исходе: у джобы, снятой на
            # десятом слайде, вызовов ровно десять, и это те самые вызовы, ради
            # которых в лог и смотрят.
            self._log_call_stats(presentation_id, timings, ceiling)
        await self._record_journal(
            presentation_id, status=STATUS_READY, result=result, started=started
        )

    async def _job_ceiling(self, presentation_id: int) -> float:
        """Потолок джобы — считается из slide_count и пишется в лог при старте.

        Число слайдов читается отдельным запросом, а не приходит из захвата:
        claim_next возвращает только id, а потолок нужен ДО первого await
        пайплайна. Запрос дешёвый и ровно один на джобу.

        Недоступная в этот момент база не должна решать судьбу заказа: берём
        потолок самой длинной колоды. Это ошибка в длинную сторону, а
        единственная альтернатива — ошибка в короткую, то есть таймаут по
        значению, к заказу отношения не имеющему.
        """
        slide_count = SLIDE_COUNT_MAX
        try:
            async with session_context() as session:
                presentation = await session.get(Presentation, presentation_id)
            if presentation is not None:
                slide_count = presentation.slide_count
        except Exception as exc:
            logger.warning(
                "Presentation %s: число слайдов не прочитано (%s), "
                "потолок джобы берём по самой длинной колоде (%s слайдов)",
                presentation_id,
                exc,
                SLIDE_COUNT_MAX,
            )
        ceiling = presentation_job_timeout(slide_count)
        calls = 1 + max(1, content_section_count(slide_count))
        # Формула ПЕЧАТАЕТСЯ СЛОВАМИ, поэтому обязана сходиться с числом рядом:
        # строка, в которой арифметика не бьётся, хуже отсутствия строки —
        # по ней принимают решения о потолках, а она врёт с уверенным видом.
        # Разъехалась она дважды, и оба раза молча:
        #
        #   * член вызовов стоял на LLM_CALL_TIMEOUT, тогда как настоящая
        #     внешняя граница вызова — LLM_CALL_WATCHDOG_TIMEOUT (страховка
        #     wait_for срабатывает позже клиента, и в худшем случае стадия
        #     занимает именно её);
        #   * паузы повторов в формулу вошли (LLM_RETRY_PAUSE_AFTER_TIMEOUT), а
        #     в строку — нет.
        #
        # Член рендера остаётся на LLM_CALL_TIMEOUT: это ВНЕШНЯЯ граница стадии
        # печати (внутри неё браузер ограничен меньшим RENDER_PRINT_TIMEOUT),
        # и в потолок джобы входит именно она.
        logger.info(
            "Presentation %s: старт, слайдов %s -> вызовов модели %s "
            "× попыток %s × %sс + паузы повторов %s × %s × %sс "
            "+ рендер %sс = потолок джобы %.0fс",
            presentation_id,
            slide_count,
            calls,
            LLM_CALL_ATTEMPTS,
            LLM_CALL_WATCHDOG_TIMEOUT,
            calls,
            LLM_CALL_ATTEMPTS - 1,
            LLM_RETRY_PAUSE_AFTER_TIMEOUT,
            LLM_CALL_TIMEOUT,
            ceiling,
        )
        return ceiling

    def _log_call_stats(
        self, presentation_id: int, timings: CallTimings, ceiling: float
    ) -> None:
        """Одна строка о том, сколько шли вызовы модели и рендер этой джобы.

        Рендер стоит в ней отдельным полем, потому что с переходом на печать
        браузером он перестал быть мгновенным: это внешний процесс, который
        грузит шрифты, верстает страницу и пишет PDF. Без своего поля его время
        либо растворилось бы в общем «суммарно», либо стало бы известно только
        по разнице между суммой вызовов и длительностью джобы — то есть никак.

        Смысл тот же, что у стартовой строки про embedding-коллекцию
        (`embedding_model=X -> коллекция Y, векторов N`, chroma_gateway.py):
        связать поведение системы с настройкой, которую меняют из админ-панели.
        LLM_CALL_TIMEOUT калибруется по замерам конкретной чат-модели, а
        чат-модель меняют мышкой — и без этой строки следующая смена
        обнаружится не в логе, а по покрасневшим заказам.
        """
        logger.info(
            "Presentation %s: %s; потолок вызова %sс, потолок джобы %.0fс",
            presentation_id,
            timings.summary(),
            LLM_CALL_TIMEOUT,
            ceiling,
        )

    # -- терминальные состояния ------------------------------------------
    async def _release(self, presentation_id: int) -> None:
        """Прерванный заказ возвращается в очередь, а не падает в ошибку.

        Остановка сервера — не вина заказа: при следующем старте он честно
        отработает с начала.
        """
        try:
            async with session_context() as session:
                await PresentationsService.requeue(session, presentation_id)
            logger.info("%s: presentation %s", _INTERRUPTED, presentation_id)
        except Exception:
            # Последний рубеж — recover() следующего запуска.
            logger.exception(
                "Failed to release interrupted presentation %s", presentation_id
            )

    async def _fail(
        self,
        presentation_id: int,
        exc: BaseException,
        started: float,
        *,
        ceiling: float,
    ) -> None:
        error_code = error_code_for(exc)
        error_text = error_text_for(exc)
        # Таймаут — единственный отказ, у которого в логе нет собственного
        # трейсбека с местом падения: там просто отменённая корутина. Зато
        # теперь у него есть имя стадии — его несёт error_text.
        if error_code == PresentationErrors.GENERATION_TIMEOUT:
            logger.warning(
                "Presentation %s timed out: %s (потолок вызова %sс, "
                "потолок джобы %.0fс)",
                presentation_id,
                error_text,
                LLM_CALL_TIMEOUT,
                ceiling,
            )
        else:
            logger.warning(
                "Presentation %s failed (%s): %s",
                presentation_id,
                error_code,
                error_text,
                exc_info=True,
            )
        try:
            async with session_context() as session:
                await PresentationsService.mark_error(
                    session,
                    presentation_id,
                    error_code=error_code,
                    error_text=error_text,
                )
        except Exception:
            logger.exception(
                "Failed to record the failure of presentation %s", presentation_id
            )
        await self._record_journal(
            presentation_id,
            status=STATUS_ERROR,
            result=GenerationResult(),
            started=started,
            error_code=error_code,
            error_text=error_text,
        )

    async def _record_journal(
        self,
        presentation_id: int,
        *,
        status: str,
        result: GenerationResult,
        started: float,
        error_code: str | None = None,
        error_text: str | None = None,
    ) -> None:
        """Событие в журнал блокнота — и на успех, и на отказ."""
        try:
            async with session_context() as session:
                presentation = await session.get(Presentation, presentation_id)
        except Exception:
            logger.exception(
                "Failed to load presentation %s for the journal entry", presentation_id
            )
            return
        if presentation is None:
            return
        await write_journal_entry(
            presentation=presentation,
            status=status,
            error_code=error_code,
            error_text=error_text,
            result=result,
            elapsed_ms=int((perf_counter() - started) * 1000),
        )
