"""Права на источники: модель владения вместо проверки роли.

Что закрепляем (см. блок «Доступ к источникам» в app/api/endpoints/documents.py
и правило владения в app/api/deps.py):

  * раздел живёт на владении, а не на роли. Любой аутентифицированный
    пользователь — в том числе с ролью `user` — работает с источниками внутри
    своих блокнотов: загружает, видит в списке, читает чанки и превью,
    привязывает к своему блокноту и удаляет свой документ;
  * чужой блокнот и чужой документ отдают 404, а не 403. Это осознанное
    решение: 403 подтверждал бы существование чужого ресурса;
  * ресурса без владельца больше не бывает вовсе: и notebook.owner_id, и
    document.owner_id объявлены NOT NULL, а старые строки забрал старейший
    админ (app/core/database.py, init_db). Бывшие «ничьи» источники видит
    только он — то же поведение, что и раньше, но держит его схема, а не
    особый случай в коде;
  * админ не потерял ничего, content_manager — тоже: раньше он был ограничен
    теми же рамками владения;
  * единственное исключение — POST /reindex: обслуживание всего индекса разом
    осталось за админом.

Почему настоящий PostgreSQL, а не FakeAsyncSession из tests/test_ownership.py.
Здесь есть загрузка, а она пишет document и job одной транзакцией, причём
job.source_id -> document.id объявлен настоящим внешним ключом. В словаре
вместо БД внешних ключей нет, и такой тест «зеленел» бы на любом порядке
записи. Тот же довод — у удаления документа вместе с его задачей.

Тяжёлые части не поднимаем: lifespan через ASGITransport не запускается,
поэтому фоновый воркер индексации не стартует, а сам обработчик upload только
кладёт задачу в очередь — извлечение текста и эмбеддинги делает воркер.
ChromaDB нужна лишь удалению документа и там замокана.
"""

import os
import sys
import time
import unittest
from unittest.mock import AsyncMock, patch

# Каталог с тестами в sys.path: unittest discover кладёт его туда сам, но при
# запуске `python -m unittest tests.test_source_access_db` этого не происходит.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dbfixtures import DatabaseBackedTestCase  # noqa: E402

import app.services.document_service as document_service_module  # noqa: E402
from app.api import deps  # noqa: E402
from app.api.endpoints.documents import upload_limiter  # noqa: E402
from app.core.exceptions import SourceErrors  # noqa: E402
from app.shared.models import Chunk, Document, Job, Notebook  # noqa: E402


SOURCES = "/api/v1/sources"


class SourceAccessTestCase(DatabaseBackedTestCase):
    """Два обычных пользователя, админ, content_manager и мигрированный документ."""

    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()

        self.user = await self.make_user("owner", "user")
        self.other = await self.make_user("stranger", "user")
        self.admin = await self.make_user("root", "admin")
        self.manager = await self.make_user("editor", "content_manager")

        self.notebook = await self._make_notebook("Свой блокнот", self.user.id)
        self.foreign_notebook = await self._make_notebook("Чужой блокнот", self.other.id)
        # Бывший «блокнот без владельца». Владелец блокнота теперь обязателен
        # (notebook.owner_id NOT NULL), а бэкфилл в init_db отдал такие
        # блокноты старейшему админу — здесь то же самое состояние. Проверки
        # ниже не изменились: обычному пользователю по-прежнему 404, админу —
        # полный доступ. В этом и был смысл миграции: особый случай ушёл из
        # кода в схему, поведение осталось прежним.
        self.migrated_notebook = await self._make_notebook(
            "Мигрированный блокнот", self.admin.id
        )

        self.document = await self._make_document(
            "own.txt", self.notebook.id, self.user.id, "SECRET-OWN"
        )
        self.foreign_document = await self._make_document(
            "foreign.txt", self.foreign_notebook.id, self.other.id, "SECRET-FOREIGN"
        )
        # Бывший «документ без владельца» — источник вне блокнота. Владелец
        # теперь обязателен и здесь (document.owner_id NOT NULL), а бэкфилл в
        # init_db отдал такие документы старейшему админу: наследовать не у
        # кого, блокнота нет. Проверки ниже не изменились — обычному
        # пользователю 404, админу полный доступ.
        self.migrated_document = await self._make_document(
            "migrated.txt", None, self.admin.id, "SECRET-MIGRATED"
        )

        self.own_chunk = await self.seed(
            Chunk(text="Фрагмент своего документа", page=1, chunk_index=0,
                  embedding_id="emb-own", doc_id=self.document.id)
        )
        self.foreign_chunk = await self.seed(
            Chunk(text="Фрагмент чужого документа", page=1, chunk_index=0,
                  embedding_id="emb-foreign", doc_id=self.foreign_document.id)
        )
        await self.seed(
            Chunk(text="Фрагмент мигрированного документа", page=1, chunk_index=0,
                  embedding_id="emb-migrated", doc_id=self.migrated_document.id)
        )
        # Задача у своего документа: её удаление вместе с документом упирается
        # в job_source_id_fkey, если порядок DELETE неверен.
        await self.seed(
            Job(job_type="index_document", status="completed",
                source_id=self.document.id, notebook_id=self.notebook.id,
                created_by=self.user.id)
        )

        # Загруженные файлы — во временный каталог, а не в data/uploads.
        uploads = os.path.join(self._tmpdir.name, "uploads")
        os.makedirs(uploads, exist_ok=True)
        upload_patcher = patch.object(document_service_module, "UPLOAD_DIR", uploads)
        upload_patcher.start()
        self.addCleanup(upload_patcher.stop)

        # Лимитер загрузки держит счётчики в памяти процесса (словарь
        # RateLimiter.clients, ключ — адрес клиента). Все тесты ходят с одного
        # адреса, поэтому без сброса тридцать первая загрузка в наборе получала
        # бы 429. Сбрасываем состояние, а не отключаем лимитер.
        upload_limiter.clients.clear()
        self.addCleanup(upload_limiter.clients.clear)

        self.as_user(self.user)

    # --- фикстуры ---

    async def _make_notebook(self, name: str, owner_id: int) -> Notebook:
        return await self.seed(
            Notebook(name=name, description=None, domain_profile="general",
                     owner_id=owner_id)
        )

    async def _make_document(
        self, name: str, notebook_id: int | None, owner_id: int | None, content: str
    ) -> Document:
        return await self.seed(
            Document(
                name=name,
                path=self.make_file(name, content),
                size=len(content),
                status="indexed",
                notebook_id=notebook_id,
                owner_id=owner_id,
            )
        )

    # --- обращения к API ---

    async def upload(self, name: str = "новый.txt", notebook_id: int | None = None):
        data = {} if notebook_id is None else {"notebook_id": str(notebook_id)}
        return await self.client.post(
            f"{SOURCES}/upload",
            data=data,
            files={"file": (name, b"tekst novogo istochnika", "text/plain")},
        )

    def assertNotForbidden(self, response, expected: int = 404) -> None:
        """404, а не 403: 403 подтвердил бы существование чужого ресурса."""
        self.assertNotEqual(
            response.status_code, 403,
            "ответ 403 подтверждает существование чужого ресурса; "
            "модель доступа требует 404",
        )
        self.assertEqual(response.status_code, expected, response.text)


# --- Пользователь с ролью `user` в своих блокнотах ----------------------


class RegularUserOwnSourcesTests(SourceAccessTestCase):
    """Роль `user` больше не отсекается: всё это раньше отвечало 403."""

    async def test_user_uploads_source_into_own_notebook(self):
        response = await self.upload("загруженный.txt", self.notebook.id)

        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["notebook_id"], self.notebook.id)
        self.assertEqual(body["status"], "pending")
        # owner_id и path наружу не отдаются
        self.assertNotIn("owner_id", body)
        self.assertNotIn("path", body)

        stored = await self.get_row(Document, body["id"])
        self.assertEqual(stored.owner_id, self.user.id)
        self.assertEqual(stored.notebook_id, self.notebook.id)
        self.assertTrue(os.path.exists(stored.path))

    async def test_upload_enqueues_indexing_job_for_the_new_source(self):
        response = await self.upload("синдексацией.txt", self.notebook.id)
        self.assertEqual(response.status_code, 200, response.text)

        jobs = await self.rows_where(Job, Job.source_id == response.json()["id"])
        self.assertEqual(len(jobs), 1, "задача индексации не поставлена")
        self.assertEqual(jobs[0].job_type, "index_document")
        self.assertEqual(jobs[0].status, "queued")
        self.assertEqual(jobs[0].created_by, self.user.id)

    async def test_upload_without_notebook_still_belongs_to_uploader(self):
        """Источник «вне блокнота» виден только загрузившему (и админу)."""
        response = await self.upload("вне-блокнота.txt", notebook_id=None)
        self.assertEqual(response.status_code, 200, response.text)

        stored = await self.get_row(Document, response.json()["id"])
        self.assertIsNone(stored.notebook_id)
        self.assertEqual(stored.owner_id, self.user.id)

    async def test_uploaded_source_appears_in_own_list(self):
        upload = await self.upload("виден-в-списке.txt", self.notebook.id)
        self.assertEqual(upload.status_code, 200, upload.text)
        new_id = upload.json()["id"]

        response = await self.client.get(
            f"{SOURCES}/", params={"notebook_id": self.notebook.id}
        )

        self.assertEqual(response.status_code, 200, response.text)
        ids = [item["id"] for item in response.json()]
        self.assertIn(new_id, ids)
        self.assertIn(self.document.id, ids)
        self.assertEqual(response.headers["X-Total-Count"], str(len(ids)))

    async def test_own_list_contains_no_foreign_or_migrated_sources(self):
        response = await self.client.get(f"{SOURCES}/")

        self.assertEqual(response.status_code, 200, response.text)
        ids = {item["id"] for item in response.json()}
        self.assertEqual(ids, {self.document.id})
        self.assertNotIn("SECRET-FOREIGN", response.text)

    async def test_user_reads_chunks_of_own_document(self):
        response = await self.client.get(f"{SOURCES}/{self.document.id}/chunks")

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(
            [item["id"] for item in response.json()], [self.own_chunk.id]
        )

    async def test_user_reads_chunk_context_of_own_document(self):
        response = await self.client.get(
            f"{SOURCES}/{self.document.id}/chunk/{self.own_chunk.id}/context"
        )

        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertEqual([item["chunk_id"] for item in payload], [self.own_chunk.id])
        self.assertTrue(payload[0]["highlight"])

    async def test_user_previews_own_document(self):
        response = await self.client.get(f"{SOURCES}/{self.document.id}/preview")

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.content, b"SECRET-OWN")

    async def test_user_attaches_own_source_to_own_notebook(self):
        upload = await self.upload("для-привязки.txt", notebook_id=None)
        self.assertEqual(upload.status_code, 200, upload.text)
        new_id = upload.json()["id"]

        response = await self.client.post(
            f"{SOURCES}/attach",
            json={"notebook_id": self.notebook.id, "source_ids": [new_id]},
        )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["updated_count"], 1)
        stored = await self.get_row(Document, new_id)
        self.assertEqual(stored.notebook_id, self.notebook.id)

    async def test_user_deletes_own_document(self):
        path = self.document.path
        with patch("app.modules.documents.service.RAGService") as rag_service:
            response = await self.client.delete(f"{SOURCES}/{self.document.id}")

        self.assertEqual(response.status_code, 200, response.text)
        rag_service.return_value.delete_documents.assert_called_once_with(
            [str(self.own_chunk.id)]
        )
        self.assertFalse(await self.exists(Document, self.document.id))
        self.assertEqual(
            await self.rows_where(Chunk, Chunk.doc_id == self.document.id), []
        )
        self.assertEqual(
            await self.rows_where(Job, Job.source_id == self.document.id), []
        )
        self.assertFalse(os.path.exists(path), "файл документа остался на диске")


class ContentManagerKeepsPreviousAccessTests(SourceAccessTestCase):
    """content_manager не потерял ничего: он и раньше жил в рамках владения."""

    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()
        self.manager_notebook = await self._make_notebook(
            "Блокнот редактора", self.manager.id
        )
        self.as_user(self.manager)

    async def test_manager_uploads_into_own_notebook(self):
        response = await self.upload("редакторский.txt", self.manager_notebook.id)
        self.assertEqual(response.status_code, 200, response.text)
        stored = await self.get_row(Document, response.json()["id"])
        self.assertEqual(stored.owner_id, self.manager.id)

    async def test_manager_cannot_reach_foreign_document(self):
        response = await self.client.get(
            f"{SOURCES}/{self.foreign_document.id}/preview"
        )
        self.assertNotForbidden(response)
        self.assertNotIn(b"SECRET-FOREIGN", response.content)

    async def test_manager_cannot_upload_into_foreign_notebook(self):
        response = await self.upload("чужой.txt", self.foreign_notebook.id)
        self.assertNotForbidden(response)

    async def test_manager_cannot_reach_migrated_document(self):
        response = await self.client.get(f"{SOURCES}/{self.migrated_document.id}/chunks")
        self.assertNotForbidden(response)


# --- Чужое и «ничьё» для роли `user` -----------------------------------


class RegularUserForeignAccessTests(SourceAccessTestCase):
    async def test_upload_into_foreign_notebook_returns_404_and_creates_nothing(self):
        before = len(await self.all_rows(Document))

        response = await self.upload("подкидыш.txt", self.foreign_notebook.id)

        self.assertNotForbidden(response)
        self.assertEqual(len(await self.all_rows(Document)), before,
                         "документ создан в чужом блокноте")

    async def test_upload_into_migrated_notebook_returns_404(self):
        response = await self.upload("подкидыш.txt", self.migrated_notebook.id)
        self.assertNotForbidden(response)

    async def test_upload_into_nonexistent_notebook_returns_404(self):
        response = await self.upload("подкидыш.txt", 10_000_019)
        self.assertNotForbidden(response)

    async def test_chunks_of_foreign_document_return_404(self):
        response = await self.client.get(
            f"{SOURCES}/{self.foreign_document.id}/chunks"
        )
        self.assertNotForbidden(response)
        self.assertNotIn("Фрагмент чужого документа", response.text)

    async def test_preview_of_foreign_document_returns_404(self):
        response = await self.client.get(
            f"{SOURCES}/{self.foreign_document.id}/preview"
        )
        self.assertNotForbidden(response)
        self.assertNotIn(b"SECRET-FOREIGN", response.content)

    async def test_chunk_context_of_foreign_document_returns_404(self):
        response = await self.client.get(
            f"{SOURCES}/{self.foreign_document.id}"
            f"/chunk/{self.foreign_chunk.id}/context"
        )
        self.assertNotForbidden(response)
        self.assertNotIn("Фрагмент чужого документа", response.text)

    async def test_foreign_chunk_id_through_own_document_returns_404(self):
        response = await self.client.get(
            f"{SOURCES}/{self.document.id}/chunk/{self.foreign_chunk.id}/context"
        )
        self.assertNotForbidden(response)
        self.assertNotIn("Фрагмент чужого документа", response.text)

    async def test_delete_of_foreign_document_returns_404_and_keeps_it(self):
        with patch("app.modules.documents.service.RAGService") as rag_service:
            response = await self.client.delete(
                f"{SOURCES}/{self.foreign_document.id}"
            )

        self.assertNotForbidden(response)
        rag_service.assert_not_called()
        self.assertTrue(await self.exists(Document, self.foreign_document.id))
        self.assertTrue(os.path.exists(self.foreign_document.path))

    async def test_attach_of_foreign_document_returns_404(self):
        response = await self.client.post(
            f"{SOURCES}/attach",
            json={"notebook_id": self.notebook.id,
                  "source_ids": [self.foreign_document.id]},
        )

        self.assertNotForbidden(response)
        stored = await self.get_row(Document, self.foreign_document.id)
        self.assertEqual(stored.notebook_id, self.foreign_notebook.id)

    async def test_attach_into_foreign_notebook_returns_404(self):
        response = await self.client.post(
            f"{SOURCES}/attach",
            json={"notebook_id": self.foreign_notebook.id,
                  "source_ids": [self.document.id]},
        )

        self.assertNotForbidden(response)
        stored = await self.get_row(Document, self.document.id)
        self.assertEqual(stored.notebook_id, self.notebook.id)

    async def test_mixed_attach_batch_changes_nothing(self):
        response = await self.client.post(
            f"{SOURCES}/attach",
            json={
                "notebook_id": self.notebook.id,
                "source_ids": [self.document.id, self.foreign_document.id],
            },
        )

        self.assertNotForbidden(response)
        stored = await self.get_row(Document, self.foreign_document.id)
        self.assertEqual(stored.notebook_id, self.foreign_notebook.id)

    async def test_list_filtered_by_foreign_notebook_returns_404(self):
        """Список источников чужого блокнота — 404, как у /notes/ и /insights/.

        Пустой список подтверждал бы, что блокнот существует и просто пуст, и
        оставлял бы один экран блокнота с разными ответами на соседние
        запросы. Подтверждением существования чужого блокнота 404 не является:
        тот же ответ приходит и на несуществующий id (тест ниже).
        """
        response = await self.client.get(
            f"{SOURCES}/", params={"notebook_id": self.foreign_notebook.id}
        )

        self.assertNotForbidden(response)
        self.assertEqual(
            response.json().get("error_code"), SourceErrors.NOTEBOOK_NOT_FOUND
        )
        self.assertNotIn("foreign.txt", response.text)

    async def test_foreign_and_missing_notebook_filters_are_indistinguishable(self):
        foreign = await self.client.get(
            f"{SOURCES}/", params={"notebook_id": self.foreign_notebook.id}
        )
        missing = await self.client.get(
            f"{SOURCES}/", params={"notebook_id": 10_000_019}
        )

        self.assertEqual(foreign.status_code, missing.status_code)
        self.assertEqual(foreign.json(), missing.json())

    async def test_list_filtered_by_migrated_notebook_returns_404(self):
        response = await self.client.get(
            f"{SOURCES}/", params={"notebook_id": self.migrated_notebook.id}
        )
        self.assertNotForbidden(response)

    async def test_own_empty_notebook_is_the_only_source_of_an_empty_list(self):
        """Пустой массив остался ответом ровно на один случай."""
        empty = await self._make_notebook("Свой пустой", self.user.id)

        response = await self.client.get(
            f"{SOURCES}/", params={"notebook_id": empty.id}
        )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json(), [])
        self.assertEqual(response.headers["X-Total-Count"], "0")

    async def test_foreign_and_missing_ids_are_indistinguishable(self):
        foreign = await self.client.get(
            f"{SOURCES}/{self.foreign_document.id}/chunks"
        )
        missing = await self.client.get(f"{SOURCES}/10000019/chunks")

        self.assertEqual(foreign.status_code, missing.status_code)
        self.assertEqual(foreign.json(), missing.json())


class MigratedResourcesAreAdminOnlyTests(SourceAccessTestCase):
    """Бывшее «ничьё» — по-прежнему только админу, но теперь через владение.

    Документа без владельца в базе больше не бывает: document.owner_id
    объявлен NOT NULL, а бэкфилл в init_db отдал такие строки старейшему
    админу. Проверки остались прежними — в этом и был смысл миграции:
    особый случай «owner_id IS NULL доступен только админу» ушёл из кода в
    схему, а наблюдаемое поведение не изменилось. Само ограничение проверяет
    tests/test_document_owner_migration_db.py.
    """

    async def test_regular_user_cannot_read_migrated_document(self):
        for path in ("chunks", "preview"):
            with self.subTest(path=path):
                response = await self.client.get(
                    f"{SOURCES}/{self.migrated_document.id}/{path}"
                )
                self.assertNotForbidden(response)
                self.assertNotIn(b"SECRET-MIGRATED", response.content)

    async def test_regular_user_cannot_delete_migrated_document(self):
        with patch("app.modules.documents.service.RAGService") as rag_service:
            response = await self.client.delete(
                f"{SOURCES}/{self.migrated_document.id}"
            )
        self.assertNotForbidden(response)
        rag_service.assert_not_called()
        self.assertTrue(await self.exists(Document, self.migrated_document.id))

    async def test_regular_user_cannot_attach_migrated_document(self):
        response = await self.client.post(
            f"{SOURCES}/attach",
            json={"notebook_id": self.notebook.id,
                  "source_ids": [self.migrated_document.id]},
        )
        self.assertNotForbidden(response)
        stored = await self.get_row(Document, self.migrated_document.id)
        self.assertIsNone(stored.notebook_id)

    async def test_admin_reads_migrated_document(self):
        self.as_user(self.admin)

        chunks = await self.client.get(f"{SOURCES}/{self.migrated_document.id}/chunks")
        preview = await self.client.get(f"{SOURCES}/{self.migrated_document.id}/preview")

        self.assertEqual(chunks.status_code, 200, chunks.text)
        self.assertEqual(len(chunks.json()), 1)
        self.assertEqual(preview.status_code, 200, preview.text)
        self.assertEqual(preview.content, b"SECRET-MIGRATED")

    async def test_admin_deletes_migrated_document(self):
        self.as_user(self.admin)
        with patch("app.modules.documents.service.RAGService"):
            response = await self.client.delete(
                f"{SOURCES}/{self.migrated_document.id}"
            )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertFalse(await self.exists(Document, self.migrated_document.id))


# --- Админ ---------------------------------------------------------------


class AdminKeepsFullAccessTests(SourceAccessTestCase):
    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()
        self.as_user(self.admin)

    async def test_admin_sees_every_document_in_list(self):
        response = await self.client.get(f"{SOURCES}/")

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(
            {item["id"] for item in response.json()},
            {self.document.id, self.foreign_document.id, self.migrated_document.id},
        )
        self.assertEqual(response.headers["X-Total-Count"], "3")

    async def test_admin_lists_sources_of_any_notebook(self):
        """Новая проверка владения в списке не должна задеть админа."""
        for notebook, expected in (
            (self.foreign_notebook, self.foreign_document.id),
            (self.notebook, self.document.id),
        ):
            with self.subTest(notebook=notebook.name):
                response = await self.client.get(
                    f"{SOURCES}/", params={"notebook_id": notebook.id}
                )
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(
                    [item["id"] for item in response.json()], [expected]
                )

    async def test_admin_uploads_into_foreign_notebook(self):
        response = await self.upload("админский.txt", self.foreign_notebook.id)

        self.assertEqual(response.status_code, 200, response.text)
        stored = await self.get_row(Document, response.json()["id"])
        self.assertEqual(stored.notebook_id, self.foreign_notebook.id)
        self.assertEqual(stored.owner_id, self.admin.id)

    async def test_admin_uploads_into_migrated_notebook(self):
        response = await self.upload("админский.txt", self.migrated_notebook.id)
        self.assertEqual(response.status_code, 200, response.text)

    async def test_admin_reads_and_previews_foreign_document(self):
        chunks = await self.client.get(f"{SOURCES}/{self.foreign_document.id}/chunks")
        preview = await self.client.get(
            f"{SOURCES}/{self.foreign_document.id}/preview"
        )

        self.assertEqual(chunks.status_code, 200, chunks.text)
        self.assertEqual(preview.content, b"SECRET-FOREIGN")

    async def test_admin_attaches_foreign_document_to_any_notebook(self):
        response = await self.client.post(
            f"{SOURCES}/attach",
            json={"notebook_id": self.migrated_notebook.id,
                  "source_ids": [self.foreign_document.id]},
        )

        self.assertEqual(response.status_code, 200, response.text)
        stored = await self.get_row(Document, self.foreign_document.id)
        self.assertEqual(stored.notebook_id, self.migrated_notebook.id)

    async def test_admin_deletes_foreign_document(self):
        with patch("app.modules.documents.service.RAGService"):
            response = await self.client.delete(
                f"{SOURCES}/{self.foreign_document.id}"
            )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertFalse(await self.exists(Document, self.foreign_document.id))


class ReindexStaysAdminOnlyTests(SourceAccessTestCase):
    """Единственное исключение из модели владения: обслуживание индекса.

    Сама переиндексация тут не выполняется — она читает файлы и считает
    эмбеддинги; проверяется только, кого до неё допускают.
    """

    def _patched_reindex(self):
        return patch(
            "app.api.endpoints.documents.DocumentModuleService.reindex_all_documents",
            new=AsyncMock(return_value={"status": "ok", "total_chunks": 0}),
        )

    async def test_regular_user_gets_403_on_reindex(self):
        with self._patched_reindex() as reindex:
            response = await self.client.post(f"{SOURCES}/reindex")
        self.assertEqual(response.status_code, 403, response.text)
        reindex.assert_not_awaited()

    async def test_content_manager_gets_403_on_reindex(self):
        self.as_user(self.manager)
        with self._patched_reindex() as reindex:
            response = await self.client.post(f"{SOURCES}/reindex")
        self.assertEqual(response.status_code, 403, response.text)
        reindex.assert_not_awaited()

    async def test_admin_may_reindex(self):
        self.as_user(self.admin)
        with self._patched_reindex() as reindex:
            response = await self.client.post(f"{SOURCES}/reindex")
        self.assertEqual(response.status_code, 200, response.text)
        reindex.assert_awaited_once()


# --- Ограничение частоты загрузок ---------------------------------------


class UploadRateLimitTests(SourceAccessTestCase):
    """Лимит на загрузку и изоляция его состояния между тестами.

    RateLimiter хранит отметки времени в словаре clients (ключ — адрес
    клиента), общем для всего процесса. Все тесты набора ходят с одного
    адреса, поэтому состояние сбрасывается в asyncSetUp: иначе 429 начал бы
    прилетать в середине прогона и зависел бы от порядка тестов.
    """

    async def test_limiter_state_starts_empty_in_every_test(self):
        self.assertEqual(upload_limiter.clients, {})
        response = await self.upload("первый.txt", self.notebook.id)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(len(upload_limiter.clients), 1)

    async def test_upload_is_rejected_after_limit_is_exhausted(self):
        first = await self.upload("первый.txt", self.notebook.id)
        self.assertEqual(first.status_code, 200, first.text)

        # Ключ берём из самого лимитера: так тест не зависит от того, каким
        # адресом представляется ASGI-транспорт.
        (client_id,) = upload_limiter.clients
        now = time.time()
        upload_limiter.clients[client_id] = [now] * upload_limiter.requests

        response = await self.upload("тридцать первый.txt", self.notebook.id)

        self.assertEqual(response.status_code, 429, response.text)
        self.assertIn("Retry-After", response.headers)
        self.assertEqual(
            response.headers["X-RateLimit-Limit"], str(upload_limiter.requests)
        )

    async def test_series_of_uploads_does_not_hit_the_limit(self):
        """Обычная серия загрузок в одном тесте лимит не выбирает."""
        for number in range(5):
            response = await self.upload(f"пачка-{number}.txt", self.notebook.id)
            self.assertEqual(response.status_code, 200, response.text)


# --- Осмысленность проверок ---------------------------------------------


class OwnershipCheckIsLoadBearingTests(SourceAccessTestCase):
    """Тесты выше не должны «зеленеть» сами по себе.

    Правило владения подменяется только в памяти теста (deps.user_owns), файлы
    app/** не трогаются. Если ответ 404 на чужой ресурс не зависит от этой
    проверки, значит проверки выше ничего не доказывают.
    """

    async def test_foreign_preview_is_hidden_only_by_user_owns(self):
        blocked = await self.client.get(
            f"{SOURCES}/{self.foreign_document.id}/preview"
        )
        self.assertEqual(blocked.status_code, 404)

        with patch.object(deps, "user_owns", return_value=True):
            leaked = await self.client.get(
                f"{SOURCES}/{self.foreign_document.id}/preview"
            )

        self.assertEqual(
            leaked.status_code, 200,
            "проверка владения ни на что не влияет — тесты 404 бессмысленны",
        )
        self.assertEqual(leaked.content, b"SECRET-FOREIGN")

    async def test_upload_into_foreign_notebook_is_blocked_only_by_user_owns(self):
        blocked = await self.upload("подкидыш.txt", self.foreign_notebook.id)
        self.assertEqual(blocked.status_code, 404)

        with patch.object(deps, "user_owns", return_value=True):
            allowed = await self.upload("подкидыш.txt", self.foreign_notebook.id)

        self.assertEqual(allowed.status_code, 200, allowed.text)

    async def test_list_is_filtered_by_owner_not_by_luck(self):
        """Список чужого не показывает именно из-за фильтра по владельцу."""
        response = await self.client.get(f"{SOURCES}/")
        self.assertEqual({item["id"] for item in response.json()},
                         {self.document.id})

        self.as_user(self.admin)
        response = await self.client.get(f"{SOURCES}/")
        self.assertIn(self.foreign_document.id,
                      {item["id"] for item in response.json()})


if __name__ == "__main__":
    unittest.main()
