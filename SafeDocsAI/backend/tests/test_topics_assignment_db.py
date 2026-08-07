"""Назначение тем: хвост индексации и фоновая переразметка.

Что закрепляем.

  * **Тема назначается сама, в конце индексации.** Отдельного действия
    пользователя для этого нет и быть не должно: он загрузил документ, а не
    заказал кластеризацию.
  * **ОТКАЗ В ТЕМЕ НЕ ЯВЛЯЕТСЯ ОТКАЗОМ ИНДЕКСАЦИИ.** Это главная проверка
    файла. Необученная модель, недоступная ChromaDB, чужая embedding-модель,
    любая неожиданная беда — документ обязан остаться 'indexed' и работающим.
    Тема украшение, поиск по документу основная функция, и обменивать вторую
    на первую нельзя ни при каких обстоятельствах.
  * **Переразметка идёт пачками, с прогрессом и по образцу чужой очереди.**
    Задача живёт в таблице job, захватывается тем же claim_next с
    FOR UPDATE SKIP LOCKED, а недостающие векторы у части документов не
    останавливают работу и называются машинным кодом в результате.
  * **Назначение привязано к версии модели.** Переразметка новой версией
    переписывает и номер, и подпись, и версию — все три вместе.

Настоящая база нужна: проверяется именно то, что попало в строки. ChromaDB и
Ollama подменены — здесь проверяется цикл вокруг векторов, а не то, как они
считаются.
"""

import json
import os
import sys
import unittest
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import AsyncMock, patch

import numpy as np
from sqlalchemy import text

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dbfixtures import DatabaseBackedTestCase  # noqa: E402
from topicfixtures import (  # noqa: E402
    ARTIFACT_EMBEDDING_MODEL,
    LABELS,
    LABELS_RU,
    LABELS_TG,
    document_vector_for,
    write_language_artifact,
)

from app.core.database import TOPIC_REASSIGN_JOB_TYPE  # noqa: E402
from app.core.exceptions import TopicErrors  # noqa: E402
from app.modules.jobs.service import JOB_INDEX_DOCUMENT, JobsService  # noqa: E402
from app.modules.jobs.worker import IndexingWorker  # noqa: E402
from app.modules.documents.service import DocumentModuleService  # noqa: E402
from app.modules.topics.service import (  # noqa: E402
    TOPIC_MODEL_PATH_ENV,
    TopicEmbeddingUnavailable,
    TopicsService,
    TopicsWorker,
    forget_cached_artifacts,
)
from app.shared.models import Chunk, Document, Job, Notebook, User  # noqa: E402
from app.shared.settings.runtime_settings import RuntimeSettingsService  # noqa: E402


class TopicAssignmentTestCase(DatabaseBackedTestCase):
    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()
        self.addCleanup(forget_cached_artifacts)

        self.artifact = Path(self._tmpdir.name) / "topic_model.npz"
        env = patch.dict(os.environ, {TOPIC_MODEL_PATH_ENV: str(self.artifact)})
        env.start()
        self.addCleanup(env.stop)

        # Система обязана считать векторы той же моделью, что и обучение:
        # иначе назначение отказывается работать (и здесь это отдельная
        # проверка). На стенде значение приходит из runtime_settings.json, и
        # тест не должен от него зависеть.
        model_patch = patch.object(
            RuntimeSettingsService, "embedding_model", return_value=ARTIFACT_EMBEDDING_MODEL
        )
        model_patch.start()
        self.addCleanup(model_patch.stop)

        # Векторы: что «лежит в ChromaDB» задаёт тест. Подменяется имя в модуле
        # тем, потому что вызов идёт через него (run_in_threadpool разрешает
        # имя в момент вызова).
        self.vectors: dict[str, list[float]] = {}
        vectors_patch = patch(
            "app.modules.topics.service.fetch_chunk_vectors", self._fetch_vectors
        )
        vectors_patch.start()
        self.addCleanup(vectors_patch.stop)

        self.owner = await self.make_user("owner", "user")
        self.notebook = await self.seed(Notebook(name="Блокнот", owner_id=self.owner.id))

    def _fetch_vectors(self, chunk_ids):
        return {
            str(chunk_id): np.asarray(self.vectors[str(chunk_id)], dtype=np.float64)
            for chunk_id in chunk_ids
            if str(chunk_id) in self.vectors
        }

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

    async def register_model(self):
        write_language_artifact(self.artifact)
        forget_cached_artifacts()
        async with self.session_factory() as session:
            return await TopicsService.sync_active_model(session)

    async def make_indexed_document(
        self, *, language: str = "ru", cluster: int = 1, name: str = "источник.txt",
        with_vectors: bool = True,
    ) -> Document:
        """Документ с двумя фрагментами и векторами вокруг центра темы.

        Два фрагмента, а не один: вектор документа — это среднее его
        фрагментов, и на одном значении усреднение проверить нечем.
        """
        document = await self.seed(
            Document(
                name=name,
                path=self.make_file(name),
                size=10,
                language=language,
                status="indexed",
                owner_id=self.owner.id,
                notebook_id=self.notebook.id,
            )
        )
        centre = np.asarray(document_vector_for(language, cluster))
        wobble = np.zeros_like(centre)
        wobble[-1] = 0.05
        chunks = await self.seed(
            Chunk(text="раз", page=1, chunk_index=0, doc_id=document.id),
            Chunk(text="два", page=1, chunk_index=1, doc_id=document.id),
        )
        if with_vectors:
            for chunk, shift in zip(chunks, (wobble, -wobble)):
                self.vectors[str(chunk.id)] = list(centre + shift)
        return document


class AssignmentAfterIndexingTests(TopicAssignmentTestCase):
    async def test_a_new_document_gets_its_topic(self):
        model = await self.register_model()
        document = await self.make_indexed_document(cluster=2)

        async with self.session_factory() as session:
            assigned = await TopicsService.assign_after_indexing(session, document.id)
        self.assertTrue(assigned)

        stored = await self.get_row(Document, document.id)
        self.assertEqual(stored.topic_cluster_index, 2)
        self.assertEqual(stored.topic_label, LABELS[2])
        # Все подписи пишутся В МОМЕНТ назначения и той версией модели, что его
        # сделала. Собирать перевод на лету по номеру кластера нельзя: у
        # документа, размеченного прошлой версией, тот же номер означает другую
        # тему, и рядом с исторической английской подписью встало бы чужое
        # название.
        self.assertEqual(stored.topic_label_ru, LABELS_RU[2])
        self.assertEqual(stored.topic_label_tg, LABELS_TG[2])
        self.assertEqual(stored.topic_model_version, model.version)

    async def test_a_model_without_translations_leaves_the_columns_empty(self):
        """Отсутствие перевода — не повод записать устойчивое имя дважды.

        Пустая колонка означает «показывай topic_label», и клиент так и
        делает. Записанная сюда английская подпись выглядела бы переводом, и
        отличить настоящий заголовок от подделки стало бы нечем.
        """
        write_language_artifact(self.artifact, localized=False)
        forget_cached_artifacts()
        async with self.session_factory() as session:
            await TopicsService.sync_active_model(session)
        document = await self.make_indexed_document(cluster=2)

        async with self.session_factory() as session:
            await TopicsService.assign_after_indexing(session, document.id)

        stored = await self.get_row(Document, document.id)
        self.assertEqual(stored.topic_label, LABELS[2])
        self.assertIsNone(stored.topic_label_ru)
        self.assertIsNone(stored.topic_label_tg)

    async def test_the_language_shift_is_undone_before_the_comparison(self):
        """Тот же документ на трёх языках обязан попасть в ОДНУ тему.

        Без преобразования он попал бы в кластер своего языка — и это не
        ошибка, а тихо неверный ответ (см. tests/test_topics_artifact.py).
        """
        await self.register_model()
        for language in ("en", "ru", "tg"):
            document = await self.make_indexed_document(
                language=language, cluster=1, name=f"{language}.txt"
            )
            async with self.session_factory() as session:
                await TopicsService.assign_after_indexing(session, document.id)
            stored = await self.get_row(Document, document.id)
            with self.subTest(language=language):
                self.assertEqual(stored.topic_cluster_index, 1)

    async def test_without_a_model_the_document_simply_has_no_topic(self):
        document = await self.make_indexed_document()
        async with self.session_factory() as session:
            self.assertFalse(await TopicsService.assign_after_indexing(session, document.id))
        stored = await self.get_row(Document, document.id)
        self.assertIsNone(stored.topic_cluster_index)
        self.assertIsNone(stored.topic_label)
        self.assertEqual(stored.status, "indexed")

    async def test_a_broken_vector_store_does_not_raise(self):
        """Наружу из назначения не выходит НИЧЕГО: его зовёт воркер индексации."""
        await self.register_model()
        document = await self.make_indexed_document()
        with patch(
            "app.modules.topics.service.fetch_chunk_vectors",
            side_effect=TopicEmbeddingUnavailable("ChromaDB недоступна"),
        ):
            async with self.session_factory() as session:
                self.assertFalse(
                    await TopicsService.assign_after_indexing(session, document.id)
                )
        self.assertIsNone((await self.get_row(Document, document.id)).topic_cluster_index)

    async def test_an_unexpected_failure_does_not_raise_either(self):
        """Ловится даже то, чего никто не ждёт: цена промаха — упавшая загрузка."""
        await self.register_model()
        document = await self.make_indexed_document()
        with patch(
            "app.modules.topics.service.fetch_chunk_vectors",
            side_effect=RuntimeError("что-то совсем неожиданное"),
        ):
            async with self.session_factory() as session:
                self.assertFalse(
                    await TopicsService.assign_after_indexing(session, document.id)
                )

    async def test_a_different_embedding_model_stops_the_assignment(self):
        """Вектор чужой модели даёт правдоподобный, но неверный кластер."""
        await self.register_model()
        document = await self.make_indexed_document()
        with patch.object(
            RuntimeSettingsService, "embedding_model", return_value="совсем-другая:7b"
        ):
            async with self.session_factory() as session:
                self.assertFalse(
                    await TopicsService.assign_after_indexing(session, document.id)
                )
        self.assertIsNone((await self.get_row(Document, document.id)).topic_cluster_index)

    async def test_a_document_without_vectors_keeps_its_previous_topic(self):
        """Неудачная попытка не стирает историю разметки."""
        model = await self.register_model()
        document = await self.make_indexed_document(with_vectors=False)
        async with self.session_factory() as session:
            await session.execute(
                text(
                    "UPDATE document SET topic_cluster_index = 0, topic_label = :label, "
                    "topic_model_version = :version WHERE id = :id"
                ),
                {"label": LABELS[0], "version": model.version - 1, "id": document.id},
            )
            await session.commit()

        async with self.session_factory() as session:
            self.assertFalse(await TopicsService.assign_after_indexing(session, document.id))
        stored = await self.get_row(Document, document.id)
        self.assertEqual(stored.topic_label, LABELS[0])
        self.assertEqual(stored.topic_model_version, model.version - 1)


class IndexingIsNotHostageToTopicsTests(TopicAssignmentTestCase):
    """Хвост настоящего воркера индексации, а не только вызов сервиса.

    Проверяется именно проводка: сервис можно сколько угодно делать
    незаваливающимся, но если воркер зовёт его не в том месте (или не зовёт
    вовсе), тема не появится, а падение в назначении утащит за собой документ.
    """

    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()
        session_patch = patch(
            "app.modules.jobs.worker.session_context", self._worker_session
        )
        session_patch.start()
        self.addCleanup(session_patch.stop)
        # Настоящая индексация здесь не нужна: файл разбирает, режет и считает
        # эмбеддинги совсем другой код, и проверяется он своими тестами.
        index_patch = patch.object(
            DocumentModuleService,
            "index_document_job",
            AsyncMock(return_value={"chunks": 2}),
        )
        index_patch.start()
        self.addCleanup(index_patch.stop)
        self.worker = IndexingWorker(poll_interval=0.05)

    async def _run_indexing(self, document: Document) -> Job:
        async with self.session_factory() as session:
            job = await JobsService.enqueue(
                session, JOB_INDEX_DOCUMENT, source_id=document.id
            )
            job_id = job.id
        await self.worker._process(job_id, document.id)
        return await self.get_row(Job, job_id)

    async def test_indexing_ends_with_a_topic_on_the_document(self):
        await self.register_model()
        document = await self.make_indexed_document(cluster=0)
        async with self.session_factory() as session:
            await session.execute(
                text("UPDATE document SET status = 'pending' WHERE id = :id"),
                {"id": document.id},
            )
            await session.commit()

        job = await self._run_indexing(document)

        self.assertEqual(job.status, "completed")
        stored = await self.get_row(Document, document.id)
        self.assertEqual(stored.status, "indexed")
        self.assertEqual(stored.topic_cluster_index, 0)
        self.assertEqual(stored.topic_label, LABELS[0])

    async def test_an_untrained_model_leaves_indexing_untouched(self):
        document = await self.make_indexed_document()
        job = await self._run_indexing(document)

        self.assertEqual(job.status, "completed")
        stored = await self.get_row(Document, document.id)
        self.assertEqual(stored.status, "indexed")
        self.assertIsNone(stored.error_code)
        self.assertIsNone(stored.topic_cluster_index)

    async def test_a_failing_topic_layer_does_not_fail_the_document(self):
        """Проводка обязана выдержать даже исключение из самого сервиса.

        Сервис обещает не бросать наружу ничего, но обещание держится кодом
        соседнего раздела. Цена его нарушения — успешно проиндексированный
        документ со статусом 'error', то есть потерянный для пользователя файл.
        Поэтому воркер не полагается на обещание, и проверяется здесь именно
        это: заведомо невозможная беда в разделе тем не трогает ни документ, ни
        задачу.
        """
        await self.register_model()
        document = await self.make_indexed_document()
        with patch.object(
            TopicsService,
            "assign_after_indexing",
            AsyncMock(side_effect=RuntimeError("раздел тем сломался целиком")),
        ):
            job = await self._run_indexing(document)

        self.assertEqual(job.status, "completed")
        stored = await self.get_row(Document, document.id)
        self.assertEqual(stored.status, "indexed")
        self.assertIsNone(stored.error_code)


class ReassignmentWorkerTests(TopicAssignmentTestCase):
    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()
        for target in (
            "app.modules.topics.service.session_context",
        ):
            session_patch = patch(target, self._worker_session)
            session_patch.start()
            self.addCleanup(session_patch.stop)
        self.worker = TopicsWorker(poll_interval=0.05)
        self.addAsyncCleanup(self.worker.stop)

    async def queue_reassign(self) -> int:
        async with self.session_factory() as session:
            job = await JobsService.enqueue(session, TOPIC_REASSIGN_JOB_TYPE)
            return job.id

    async def test_every_indexed_document_is_relabelled(self):
        model = await self.register_model()
        first = await self.make_indexed_document(cluster=0, name="первый.txt")
        second = await self.make_indexed_document(
            language="tg", cluster=2, name="второй.txt"
        )
        job_id = await self.queue_reassign()

        self.assertTrue(await self.worker.claim_and_process())

        job = await self.get_row(Job, job_id)
        self.assertEqual(job.status, "completed")
        result = json.loads(job.result_json)
        self.assertEqual(result["assigned"], 2)
        self.assertEqual(result["skipped"], 0)
        self.assertEqual(result["model_version"], model.version)

        self.assertEqual((await self.get_row(Document, first.id)).topic_cluster_index, 0)
        self.assertEqual((await self.get_row(Document, second.id)).topic_cluster_index, 2)

    async def test_reassignment_moves_documents_onto_the_new_version(self):
        """Ради этого версия и хранится: старые назначения не выдаются за новые."""
        first_model = await self.register_model()
        document = await self.make_indexed_document(cluster=1)
        async with self.session_factory() as session:
            await TopicsService.assign_after_indexing(session, document.id)
        self.assertEqual(
            (await self.get_row(Document, document.id)).topic_model_version,
            first_model.version,
        )

        # Переобучение: тот же путь, другое содержимое.
        write_language_artifact(self.artifact)
        with open(self.artifact, "ab") as handle:
            handle.write(b"\0")
        forget_cached_artifacts()

        await self.queue_reassign()
        self.assertTrue(await self.worker.claim_and_process())

        stored = await self.get_row(Document, document.id)
        self.assertEqual(stored.topic_model_version, first_model.version + 1)
        self.assertEqual(stored.topic_label, LABELS[1])

    async def test_documents_without_vectors_are_skipped_with_a_named_reason(self):
        """«Пропущено 1» без причины неотличимо от «модель считает их бестемными»."""
        await self.register_model()
        await self.make_indexed_document(cluster=0, name="есть.txt")
        await self.make_indexed_document(
            cluster=0, name="нет.txt", with_vectors=False
        )
        job_id = await self.queue_reassign()

        with patch("app.modules.topics.service.REASSIGN_BATCH", 1):
            self.assertTrue(await self.worker.claim_and_process())

        result = json.loads((await self.get_row(Job, job_id)).result_json)
        self.assertEqual(result["assigned"], 1)
        self.assertEqual(result["skipped"], 1)
        self.assertEqual(result["skipped_reason"], TopicErrors.EMBEDDING_UNAVAILABLE)

    async def test_progress_is_reported_while_the_job_runs(self):
        """Иначе длинная переразметка выглядит зависшей и теряет аренду задачи."""
        await self.register_model()
        for index in range(3):
            await self.make_indexed_document(cluster=index % 3, name=f"{index}.txt")
        job_id = await self.queue_reassign()

        seen: list[int | None] = []
        original = JobsService.heartbeat

        async def spy(session, job, *, progress=None):
            seen.append(progress)
            await original(session, job, progress=progress)

        with patch("app.modules.topics.service.REASSIGN_BATCH", 1):
            with patch.object(JobsService, "heartbeat", spy):
                await self.worker.claim_and_process()

        self.assertTrue(seen, "прогресс не обновлялся ни разу")
        self.assertEqual(seen, sorted(seen), "прогресс обязан только расти")
        self.assertEqual((await self.get_row(Job, job_id)).progress, 100)

    async def test_a_job_without_a_model_fails_instead_of_pretending(self):
        job_id = await self.queue_reassign()
        self.assertTrue(await self.worker.claim_and_process())
        job = await self.get_row(Job, job_id)
        self.assertEqual(job.status, "failed")
        self.assertIn("TopicModelUnusable", job.error_text)

    async def test_a_broken_vector_store_fails_the_job_instead_of_reporting_success(self):
        """Иначе задача бодро отчиталась бы «выполнено» с нулём разметки.

        Пропуск ОТДЕЛЬНОГО документа — рабочая ситуация и считается отдельно, а
        отказ, общий для всех, обязан остановить работу: продолжать после него
        значит тысячу раз получить один и тот же отказ.
        """
        await self.register_model()
        await self.make_indexed_document()
        job_id = await self.queue_reassign()

        with patch(
            "app.modules.topics.service.fetch_chunk_vectors",
            side_effect=TopicEmbeddingUnavailable("ChromaDB недоступна"),
        ):
            self.assertTrue(await self.worker.claim_and_process())

        job = await self.get_row(Job, job_id)
        self.assertEqual(job.status, "failed")
        # У job нет колонки error_code, а причина обязана быть различима
        # машинно: «недоступны векторы» и «модель не пригодна» лечатся разным.
        self.assertTrue(
            job.error_text.startswith(TopicErrors.EMBEDDING_UNAVAILABLE), job.error_text
        )

    async def test_a_retrained_artifact_is_noticed_without_a_restart(self):
        """Модель кладёт на диск офлайн-скрипт, ни о чём не спрашивая бэкенд.

        Пока новая версия не зарегистрирована, назначение отказывается работать
        (артефакт не совпадает с зарегистрированным), и новые документы молча
        остаются без тем. Починка «перезагрузите сервер» здесь неприемлема.
        """
        first = await self.register_model()
        write_language_artifact(self.artifact)
        with open(self.artifact, "ab") as handle:
            handle.write(b"\0")
        forget_cached_artifacts()

        await self.worker.sync_model_if_due()

        async with self.session_factory() as session:
            active = await TopicsService.active_model(session)
        self.assertEqual(active.version, first.version + 1)

    async def test_the_disk_is_not_re_read_on_every_iteration(self):
        """Сверка дешёвая, но не бесплатная, а переобучают раз в дни."""
        await self.register_model()
        await self.worker.sync_model_if_due()
        with patch.object(TopicsService, "sync_active_model", AsyncMock()) as spy:
            await self.worker.sync_model_if_due()
        spy.assert_not_called()

    async def test_an_empty_queue_is_not_a_claim(self):
        self.assertFalse(await self.worker.claim_and_process())


class UnclearDocumentsTests(TopicAssignmentTestCase):
    """Документ, который модель отказалась относить к теме.

    Отказ — законченный ответ, а не сбой, и путать его с пропуском нельзя:
    пропуск лечится переиндексацией, отказ лечить нечем. Поэтому он и считается
    отдельным числом, и в базе выглядит иначе — версия модели проставлена, а
    номера кластера нет.
    """

    async def register_model_with_threshold(self, threshold: float = 0.30):
        write_language_artifact(self.artifact)
        # Порог дописывается в уже собранный артефакт, чтобы остальные фикстуры
        # остались нетронутыми: их читают полтора десятка других проверок.
        with np.load(self.artifact, allow_pickle=False) as archive:
            payload = {name: archive[name] for name in archive.files}
        meta = json.loads(str(payload["meta"]))
        meta["params"] = {**meta.get("params", {}), "margin_threshold": threshold}
        payload["meta"] = np.array(json.dumps(meta, ensure_ascii=False))
        np.savez_compressed(self.artifact, **payload)
        forget_cached_artifacts()
        async with self.session_factory() as session:
            return await TopicsService.sync_active_model(session)

    async def make_document_between_two_topics(self, language: str = "ru") -> Document:
        """Документ ровно посередине между двумя центрами тем.

        Середина, а не «где-то рядом»: только там запас заведомо близок к нулю,
        и проверка не зависит от того, как далеко разошлись центроиды фикстуры.
        """
        document = await self.seed(
            Document(
                name="посередине.txt",
                path=self.make_file("посередине.txt"),
                size=10,
                language=language,
                status="indexed",
                owner_id=self.owner.id,
                notebook_id=self.notebook.id,
            )
        )
        first = np.asarray(document_vector_for(language, 0))
        second = np.asarray(document_vector_for(language, 1))
        chunks = await self.seed(
            Chunk(text="раз", page=1, chunk_index=0, doc_id=document.id),
            Chunk(text="два", page=1, chunk_index=1, doc_id=document.id),
        )
        # Среднее двух фрагментов — это середина между темами: вектор документа
        # считается именно усреднением (document_vector).
        for chunk, vector in zip(chunks, (first, second)):
            self.vectors[str(chunk.id)] = list(vector)
        return document

    async def test_the_document_in_the_middle_gets_no_topic_but_gets_the_version(self):
        model = await self.register_model_with_threshold()
        document = await self.make_document_between_two_topics()
        async with self.session_factory() as session:
            outcome = await TopicsService.assign_documents(session, model, [document.id])
        self.assertEqual(outcome.unclear, 1)
        self.assertEqual(outcome.assigned, 0)
        self.assertEqual(outcome.skipped, 0)

        async with self.session_factory() as session:
            stored = await session.get(Document, document.id)
            # Версия есть, номера нет — это и есть «модель посмотрела и не
            # решилась», в отличие от «ещё не размечен» (версии тоже нет).
            self.assertEqual(stored.topic_model_version, model.version)
            self.assertIsNone(stored.topic_cluster_index)
            self.assertIsNone(stored.topic_label)
            self.assertIsNone(stored.topic_label_ru)
            self.assertIsNone(stored.topic_label_tg)

    async def test_a_confident_document_is_still_assigned(self):
        """Порог не должен отказывать всем подряд: документ у центра темы
        получает её как раньше. Без этой проверки предыдущая была бы зелёной и
        при пороге, отвергающем весь корпус."""
        model = await self.register_model_with_threshold()
        document = await self.make_indexed_document(cluster=1)
        async with self.session_factory() as session:
            outcome = await TopicsService.assign_documents(session, model, [document.id])
        self.assertEqual((outcome.assigned, outcome.unclear), (1, 0))
        async with self.session_factory() as session:
            stored = await session.get(Document, document.id)
        self.assertEqual(stored.topic_cluster_index, 1)
        self.assertEqual(stored.topic_label, LABELS[1])

    async def test_without_a_threshold_the_same_document_gets_a_topic(self):
        """Краснота предыдущей проверки: без порога середина между темами
        по-прежнему получает тему — то есть отказ вызывает именно порог, а не
        что-то ещё в этом документе."""
        model = await self.register_model()
        document = await self.make_document_between_two_topics()
        async with self.session_factory() as session:
            outcome = await TopicsService.assign_documents(session, model, [document.id])
        self.assertEqual((outcome.assigned, outcome.unclear), (1, 0))
        async with self.session_factory() as session:
            stored = await session.get(Document, document.id)
        self.assertIsNotNone(stored.topic_cluster_index)

    async def test_an_unclear_document_falls_out_of_the_distribution(self):
        """Распределение считает только строки с номером кластера, поэтому
        отказавшийся документ честно не попадает ни в одну тему — и сумма по
        темам не сходится с числом документов. Так и должно быть."""
        model = await self.register_model_with_threshold()
        middle = await self.make_document_between_two_topics()
        clear = await self.make_indexed_document(cluster=2, name="ясный.txt")
        async with self.session_factory() as session:
            await TopicsService.assign_documents(session, model, [middle.id, clear.id])
            rows = await TopicsService.distribution(session, model)
        self.assertEqual(sum(row.document_count for row in rows), 1)

    async def test_the_counts_of_unclear_and_skipped_do_not_mix(self):
        """Пропуск чинится переиндексацией, отказ чинить нечем. Одно число на
        двоих отправило бы администратора искать поломку там, где её нет."""
        model = await self.register_model_with_threshold()
        middle = await self.make_document_between_two_topics()
        without = await self.make_indexed_document(name="без-векторов.txt", with_vectors=False)
        async with self.session_factory() as session:
            outcome = await TopicsService.assign_documents(session, model, [middle.id, without.id])
        self.assertEqual((outcome.assigned, outcome.skipped, outcome.unclear), (0, 1, 1))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
