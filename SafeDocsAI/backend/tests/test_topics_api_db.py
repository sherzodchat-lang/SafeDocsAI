"""HTTP-контракт раздела тем — на настоящем PostgreSQL.

Что закрепляем.

  * **Раздел без модели пуст, а не сломан.** GET /topics отвечает пустым
    массивом, и только GET /topics/model говорит прямо: 404 topic.model_missing.
    Ошибка там, где показывать нечего, требовала бы от пользователя действия,
    которого он совершить не может: модель обучают вне продукта.
  * **Распределение считает ТОЛЬКО доступные документы.** Правило то же, что у
    источников (_owner_filter), и проверяется оно здесь на числах: чужие
    документы не должны попадать даже в счётчик — по нему они восстанавливаются
    не хуже, чем по списку.
  * **Назначения прошлой версии в распределение не попадают.** После
    переобучения номер 3 у старой модели и номер 3 у новой — разные темы, и
    сумма по ним была бы числом без смысла. Пока переразметка не прошла,
    распределение честно показывает нули.
  * **Переразметка — за админом и не больше одной сразу.** 403 обычному
    пользователю, 409 на вторую, причём 409 обязан приходить и от предпроверки,
    и от БАЗЫ: между предпроверкой и вставкой помещается весь второй запрос.

Настоящая база нужна почти всем проверкам: распределение — это GROUP BY с
фильтрами владения, а «не больше одной переразметки» — частичный уникальный
индекс, которого на моках не существует. Ollama и ChromaDB не поднимаются:
HTTP-слой до них не доходит.
"""

import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dbfixtures import DatabaseBackedTestCase  # noqa: E402
from topicfixtures import (  # noqa: E402
    ARTIFACT_EMBEDDING_MODEL,
    LABELS,
    LABELS_RU,
    LABELS_TG,
    METRICS,
    write_language_artifact,
)

from app.core.database import (  # noqa: E402
    TOPIC_REASSIGN_ACTIVE_INDEX,
    TOPIC_REASSIGN_JOB_TYPE,
)
from app.core.exceptions import SourceErrors, TopicErrors  # noqa: E402
from app.modules.topics.service import (  # noqa: E402
    TOPIC_MODEL_PATH_ENV,
    TopicsService,
    forget_cached_artifacts,
)
from app.shared.models import Document, Job, Notebook, TopicModelVersion, User  # noqa: E402

TOPICS_URL = "/api/v1/topics"
MODEL_URL = "/api/v1/topics/model"
REASSIGN_URL = "/api/v1/topics/reassign"


class TopicsApiTestCase(DatabaseBackedTestCase):
    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()
        self.addCleanup(forget_cached_artifacts)

        # Путь к артефакту подменяется ВСЕГДА, даже там, где модель не нужна:
        # иначе тест «модели нет» зарегистрировал бы настоящую обученную модель
        # стенда и проверял бы совсем не то.
        self.artifact = Path(self._tmpdir.name) / "topic_model.npz"
        env = patch.dict(os.environ, {TOPIC_MODEL_PATH_ENV: str(self.artifact)})
        env.start()
        self.addCleanup(env.stop)

        self.owner = await self.make_user("owner", "user")
        self.other = await self.make_user("other", "user")
        self.admin = await self.make_user("root", "admin")
        self.as_user(self.owner)

    async def register_model(self) -> TopicModelVersion:
        write_language_artifact(self.artifact)
        forget_cached_artifacts()
        async with self.session_factory() as session:
            model = await TopicsService.sync_active_model(session)
        self.assertIsNotNone(model)
        return model

    async def make_notebook(self, owner: User, name: str = "Блокнот") -> Notebook:
        return await self.seed(Notebook(name=name, owner_id=owner.id))

    async def make_document(
        self,
        owner: User,
        notebook: Notebook,
        *,
        cluster: int | None = None,
        version: int | None = None,
        name: str = "источник.txt",
    ) -> Document:
        return await self.seed(
            Document(
                name=name,
                path=self.make_file(f"{owner.username}-{name}"),
                size=10,
                status="indexed",
                owner_id=owner.id,
                notebook_id=notebook.id,
                topic_cluster_index=cluster,
                topic_label=None if cluster is None else LABELS[cluster],
                topic_label_ru=None if cluster is None else LABELS_RU[cluster],
                topic_label_tg=None if cluster is None else LABELS_TG[cluster],
                topic_model_version=version,
            )
        )


class WithoutAModelTests(TopicsApiTestCase):
    async def test_topic_list_is_empty_rather_than_broken(self):
        response = await self.client.get(TOPICS_URL)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), [])

    async def test_model_endpoint_says_so_with_a_machine_code(self):
        response = await self.client.get(MODEL_URL)
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["error_code"], TopicErrors.MODEL_MISSING)

    async def test_reassignment_has_nothing_to_do(self):
        self.as_user(self.admin)
        response = await self.client.post(REASSIGN_URL)
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["error_code"], TopicErrors.MODEL_MISSING)
        self.assertEqual(await self.all_rows(Job), [])


class ModelDescriptionTests(TopicsApiTestCase):
    async def test_the_registry_answers_with_everything_the_section_shows(self):
        model = await self.register_model()

        response = await self.client.get(MODEL_URL)
        self.assertEqual(response.status_code, 200)
        body = response.json()

        self.assertEqual(body["k"], model.k)
        self.assertEqual(body["cluster_count"], len(LABELS))
        self.assertEqual(body["embedding_model"], ARTIFACT_EMBEDDING_MODEL)
        self.assertEqual(body["transform"], "group_mean_shift(language)")
        self.assertEqual(body["metrics"], METRICS)
        # Дата обучения — с явным UTC. Без него браузер в UTC+5 показал бы её
        # на пять часов раньше настоящей (тот же довод, что у дат источников).
        self.assertTrue(body["trained_at"].endswith("Z"), body["trained_at"])

    async def test_retraining_makes_a_new_version_and_deactivates_the_old_one(self):
        first = await self.register_model()
        # Тот же путь, другое содержимое: артефакт всегда перезаписывается на
        # месте, поэтому распознаётся он по sha256, а не по имени или дате.
        write_language_artifact(self.artifact)
        with open(self.artifact, "ab") as handle:
            handle.write(b"\0")
        forget_cached_artifacts()
        async with self.session_factory() as session:
            second = await TopicsService.sync_active_model(session)

        self.assertEqual(second.version, first.version + 1)
        rows = {row.version: row.is_active for row in await self.all_rows(TopicModelVersion)}
        self.assertEqual(rows, {first.version: False, second.version: True})

    async def test_the_same_artifact_does_not_bump_the_version(self):
        """Перезапуск и копирование файла не обесценивают назначения."""
        first = await self.register_model()
        async with self.session_factory() as session:
            again = await TopicsService.sync_active_model(session)
        self.assertEqual(again.version, first.version)
        self.assertEqual(len(await self.all_rows(TopicModelVersion)), 1)


class DistributionTests(TopicsApiTestCase):
    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()
        self.model = await self.register_model()
        self.mine = await self.make_notebook(self.owner, "Мой")
        self.foreign = await self.make_notebook(self.other, "Чужой")

        await self.make_document(self.owner, self.mine, cluster=0, version=self.model.version)
        await self.make_document(
            self.owner, self.mine, cluster=0, version=self.model.version, name="второй.txt"
        )
        await self.make_document(
            self.owner, self.mine, cluster=1, version=self.model.version, name="третий.txt"
        )
        await self.make_document(
            self.other, self.foreign, cluster=2, version=self.model.version, name="чужой.txt"
        )

    async def test_user_sees_only_the_documents_available_to_them(self):
        response = await self.client.get(TOPICS_URL)
        self.assertEqual(response.status_code, 200)
        counts = {row["cluster_index"]: row["document_count"] for row in response.json()}
        self.assertEqual(counts, {0: 2, 1: 1, 2: 0})

    async def test_admin_sees_every_document(self):
        self.as_user(self.admin)
        counts = {
            row["cluster_index"]: row["document_count"]
            for row in (await self.client.get(TOPICS_URL)).json()
        }
        self.assertEqual(counts, {0: 2, 1: 1, 2: 1})

    async def test_shares_are_computed_over_the_visible_documents(self):
        rows = (await self.client.get(TOPICS_URL)).json()
        shares = {row["cluster_index"]: row["share"] for row in rows}
        self.assertAlmostEqual(shares[0], 2 / 3)
        self.assertAlmostEqual(shares[1], 1 / 3)
        self.assertAlmostEqual(shares[2], 0.0)
        self.assertAlmostEqual(sum(shares.values()), 1.0)

    async def test_empty_clusters_stay_in_the_answer(self):
        """Иначе два блокнота дают два разных набора строк и не сопоставляются.

        Пустые кластеры остаются и по второй причине: полный список — это
        ответ администратору на вопрос «что модель вообще различает». Прячет
        их от обычного пользователя ЭКРАН, свернув под раскрытие, а не выдача:
        отфильтровав их здесь, мы отняли бы данные у всех сразу.
        """
        rows = (await self.client.get(TOPICS_URL)).json()
        self.assertEqual(len(rows), len(LABELS))
        self.assertEqual([row["label"] for row in rows][0], LABELS[0])
        self.assertEqual([row["document_count"] for row in rows][-1], 0)

    async def test_every_label_comes_back_and_the_client_chooses(self):
        """Переводы — показать пользователю, label — сослаться на тему.

        Интерфейс переведён на ru и tg, английских экранов в продукте нет, и
        одно только label означало бы английские подписи у всех. Решать за
        клиента, какую из подписей показать, API всё же не должен: подпись темы
        путешествует между экранами параметром фильтра.
        """
        rows = (await self.client.get(TOPICS_URL)).json()
        by_cluster = {row["cluster_index"]: row for row in rows}
        self.assertEqual(by_cluster[0]["label"], LABELS[0])
        self.assertEqual(by_cluster[0]["label_ru"], LABELS_RU[0])
        self.assertEqual(by_cluster[0]["label_tg"], LABELS_TG[0])

    async def test_a_model_without_translations_answers_null_not_english(self):
        """null — сигнал клиенту откатиться к label, а не «перевод пустой»."""
        write_language_artifact(self.artifact, localized=False)
        forget_cached_artifacts()
        async with self.session_factory() as session:
            await TopicsService.sync_active_model(session)

        rows = (await self.client.get(TOPICS_URL)).json()
        self.assertEqual({row["label_ru"] for row in rows}, {None})
        self.assertEqual({row["label_tg"] for row in rows}, {None})
        self.assertEqual({row["label"] for row in rows}, set(LABELS))

    async def test_the_biggest_topic_comes_first(self):
        rows = (await self.client.get(TOPICS_URL)).json()
        self.assertEqual([row["document_count"] for row in rows], [2, 1, 0])

    async def test_notebook_filter_narrows_the_count(self):
        rows = (await self.client.get(f"{TOPICS_URL}?notebook_id={self.mine.id}")).json()
        self.assertEqual({row["cluster_index"]: row["document_count"] for row in rows}, {0: 2, 1: 1, 2: 0})

    async def test_foreign_notebook_answers_404_and_not_an_empty_distribution(self):
        """Пустое распределение не отличить от «блокнота нет» — и это оракул."""
        response = await self.client.get(f"{TOPICS_URL}?notebook_id={self.foreign.id}")
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["error_code"], SourceErrors.NOTEBOOK_NOT_FOUND)

    async def test_assignments_of_an_older_model_are_not_counted(self):
        """Номер 3 старой модели и номер 3 новой — разные темы."""
        await self.make_document(
            self.owner, self.mine, cluster=2, version=self.model.version - 1, name="старый.txt"
        )
        counts = {
            row["cluster_index"]: row["document_count"]
            for row in (await self.client.get(TOPICS_URL)).json()
        }
        self.assertEqual(counts[2], 0)

    async def test_stored_label_survives_in_the_document_list(self):
        """Подпись приходит хранимая: пересборка по номеру переписала бы историю."""
        response = await self.client.get("/api/v1/sources/")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        labelled = [item for item in body if item["topic_label"]]
        self.assertTrue(labelled)
        self.assertEqual(
            {item["topic_label"] for item in labelled},
            {LABELS[0], LABELS[1]},
        )
        # Переводы тоже хранимые и приходят рядом с ключом: клиент показывает
        # их, а к topic_label откатывается, когда перевода нет.
        self.assertEqual(
            {item["topic_label_ru"] for item in labelled},
            {LABELS_RU[0], LABELS_RU[1]},
        )
        self.assertEqual(
            {item["topic_label_tg"] for item in labelled},
            {LABELS_TG[0], LABELS_TG[1]},
        )
        self.assertIn("topic_cluster_index", body[0])


class SourceListFilteredByTopicTests(TopicsApiTestCase):
    """Отбор по теме делает СЕРВЕР, а не клиент по загруженной странице.

    Клиентский отбор врёт на объёме: он видит ограниченное число страниц, и
    «источников по теме: 12» при сорока настоящих выглядит как правда. Здесь
    фильтр стоит рядом с остальными условиями, то есть попадает и в
    X-Total-Count.
    """

    SOURCES_URL = "/api/v1/sources/"

    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()
        self.model = await self.register_model()
        self.mine = await self.make_notebook(self.owner, "Мой")
        self.foreign = await self.make_notebook(self.other, "Чужой")

        self.zero = await self.make_document(
            self.owner, self.mine, cluster=0, version=self.model.version, name="ноль.txt"
        )
        self.one = await self.make_document(
            self.owner, self.mine, cluster=1, version=self.model.version, name="один.txt"
        )
        self.without_topic = await self.make_document(
            self.owner, self.mine, name="без-темы.txt"
        )
        self.alien = await self.make_document(
            self.other, self.foreign, cluster=0, version=self.model.version, name="чужой.txt"
        )

    async def _names(self, query: str) -> set[str]:
        response = await self.client.get(f"{self.SOURCES_URL}{query}")
        self.assertEqual(response.status_code, 200)
        return {item["name"] for item in response.json()}

    async def test_zero_is_a_topic_and_not_an_absent_filter(self):
        """`if topic:` молча выбросил бы нулевой кластер из фильтруемых."""
        self.assertEqual(await self._names("?topic=0"), {self.zero.name})

    async def test_the_filter_narrows_the_page_and_the_total_header(self):
        response = await self.client.get(f"{self.SOURCES_URL}?topic=1")
        self.assertEqual([item["name"] for item in response.json()], [self.one.name])
        self.assertEqual(response.headers["X-Total-Count"], "1")

    async def test_documents_of_other_owners_stay_invisible(self):
        """Счётчик восстанавливает чужие документы не хуже, чем список."""
        self.assertNotIn(self.alien.name, await self._names("?topic=0"))
        self.as_user(self.admin)
        self.assertEqual(
            await self._names("?topic=0"), {self.zero.name, self.alien.name}
        )

    async def test_the_filter_combines_with_the_notebook(self):
        self.assertEqual(
            await self._names(f"?topic=0&notebook_id={self.mine.id}"), {self.zero.name}
        )

    async def test_assignments_of_an_older_model_do_not_leak_in(self):
        """Третий кластер прошлой модели — другая тема с тем же номером."""
        await self.make_document(
            self.owner,
            self.mine,
            cluster=1,
            version=self.model.version - 1,
            name="прошлая-модель.txt",
        )
        self.assertEqual(await self._names("?topic=1"), {self.one.name})

    async def test_an_unknown_cluster_is_an_empty_page_and_not_a_refusal(self):
        """«В этой теме ничего нет» — обычный ответ, а не ошибка запроса."""
        response = await self.client.get(f"{self.SOURCES_URL}?topic=999")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), [])

    async def test_a_negative_cluster_is_refused_by_validation(self):
        response = await self.client.get(f"{self.SOURCES_URL}?topic=-1")
        self.assertEqual(response.status_code, 422)

    async def test_without_a_model_the_filtered_page_is_empty(self):
        """Иначе на «покажи тему N» пришёл бы вообще весь список."""
        async with self.session_factory() as session:
            await session.execute(text("DELETE FROM topicmodelversion"))
            await session.commit()
        response = await self.client.get(f"{self.SOURCES_URL}?topic=0")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), [])
        self.assertEqual(response.headers["X-Total-Count"], "0")

    async def test_without_the_parameter_nothing_changes(self):
        self.assertEqual(
            await self._names(""),
            {self.zero.name, self.one.name, self.without_topic.name},
        )


class ReassignmentTests(TopicsApiTestCase):
    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()
        self.model = await self.register_model()

    async def test_a_regular_user_cannot_start_it(self):
        """Переразметка идёт по всем документам системы, а не по своим."""
        response = await self.client.post(REASSIGN_URL)
        self.assertEqual(response.status_code, 403)
        self.assertEqual(await self.all_rows(Job), [])

    async def test_a_content_manager_cannot_start_it_either(self):
        self.as_user(await self.make_user("editor", "content_manager"))
        self.assertEqual((await self.client.post(REASSIGN_URL)).status_code, 403)

    async def test_admin_queues_a_background_job(self):
        self.as_user(self.admin)
        response = await self.client.post(REASSIGN_URL)
        self.assertEqual(response.status_code, 202)
        body = response.json()
        self.assertEqual(body["model_version"], self.model.version)
        self.assertEqual(body["status"], "queued")

        jobs = await self.all_rows(Job)
        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0].job_type, TOPIC_REASSIGN_JOB_TYPE)
        self.assertEqual(jobs[0].created_by, self.admin.id)
        self.assertEqual(
            json.loads(jobs[0].payload_json)["model_version"], self.model.version
        )

    async def test_the_second_request_is_refused_with_409(self):
        self.as_user(self.admin)
        self.assertEqual((await self.client.post(REASSIGN_URL)).status_code, 202)
        response = await self.client.post(REASSIGN_URL)
        self.assertEqual(response.status_code, 409)
        self.assertEqual(
            response.json()["error_code"], TopicErrors.REASSIGN_IN_PROGRESS
        )
        self.assertEqual(len(await self.all_rows(Job)), 1)

    async def test_a_running_job_also_blocks_a_new_one(self):
        """Захваченная воркером задача — это тоже «уже идёт»."""
        self.as_user(self.admin)
        await self.client.post(REASSIGN_URL)
        async with self.session_factory() as session:
            await session.execute(text("UPDATE job SET status = 'running'"))
            await session.commit()
        self.assertEqual((await self.client.post(REASSIGN_URL)).status_code, 409)

    async def test_a_finished_job_does_not_block_the_next_one(self):
        self.as_user(self.admin)
        await self.client.post(REASSIGN_URL)
        async with self.session_factory() as session:
            await session.execute(text("UPDATE job SET status = 'completed'"))
            await session.commit()
        self.assertEqual((await self.client.post(REASSIGN_URL)).status_code, 202)
        self.assertEqual(len(await self.all_rows(Job)), 2)

    async def test_the_invariant_is_held_by_the_database_itself(self):
        """Предпроверка гонится: между ней и вставкой помещается второй запрос.

        Поэтому вторая активная задача обязана отвергаться самой БД. Здесь
        предпроверка обойдена намеренно — вставкой мимо обработчика.
        """
        self.as_user(self.admin)
        await self.client.post(REASSIGN_URL)
        with self.assertRaises(IntegrityError) as caught:
            async with self.session_factory() as session:
                session.add(Job(job_type=TOPIC_REASSIGN_JOB_TYPE, status="queued"))
                await session.commit()
        self.assertIn(TOPIC_REASSIGN_ACTIVE_INDEX, str(caught.exception))

    async def test_it_registers_a_freshly_trained_artifact(self):
        """Кнопку жмут ИМЕННО после переобучения.

        Требовать сверх этого перезапуска бэкенда ради регистрации нового
        артефакта — лишний шаг, о котором администратор узнаёт по
        неизменившимся темам.
        """
        self.as_user(self.admin)
        write_language_artifact(self.artifact)
        with open(self.artifact, "ab") as handle:
            handle.write(b"\0")
        forget_cached_artifacts()

        response = await self.client.post(REASSIGN_URL)
        self.assertEqual(response.status_code, 202)
        self.assertEqual(response.json()["model_version"], self.model.version + 1)

    async def test_indexing_jobs_are_not_touched_by_the_invariant(self):
        """Уникальность частичная: очередь индексации остаётся многострочной."""
        async with self.session_factory() as session:
            session.add(Job(job_type="index_document", status="queued"))
            session.add(Job(job_type="index_document", status="queued"))
            await session.commit()
        self.assertEqual(len(await self.all_rows(Job)), 2)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
