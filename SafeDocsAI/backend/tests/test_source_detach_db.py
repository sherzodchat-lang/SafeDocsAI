"""Снятие источника с блокнота: POST /sources/detach.

Что закрепляем (см. detach_documents в app/api/endpoints/documents.py и
DocumentModuleService.detach_documents_from_notebook):

  * отвязка — не удаление: notebook_id обнуляется, а файл на диске, чанки,
    векторы в ChromaDB и назначенная тема остаются как были. До этой ручки
    единственным способом «убрать из блокнота» было DELETE /sources/{id},
    которое стирает документ насовсем, — подмена одного намерения другим;
  * форма запроса и отказы парны привязке: та же пара notebook_id +
    source_ids, те же 400/404 с теми же кодами. Чужой и несуществующий
    документ неотличимы, смешанная пачка не меняет ничего;
  * операция идемпотентна: «убрать из блокнота X» на документе, которого в X
    уже нет, — успех, а не отказ. Повторное нажатие и гонка двух вкладок не
    должны заканчиваться ошибкой на действии, чья цель и так достигнута.

Настоящий PostgreSQL — по той же причине, что и в test_source_access_db.py:
проверяется, что после отвязки строки чанков живы при настоящих внешних
ключах, а не в словаре, где ссылок нет вовсе.
"""

import os
import sys
import unittest
from unittest.mock import patch

# Каталог с тестами в sys.path: unittest discover кладёт его туда сам, но при
# запуске `python -m unittest tests.test_source_detach_db` этого не происходит.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dbfixtures import DatabaseBackedTestCase  # noqa: E402

from app.core.exceptions import SourceErrors  # noqa: E402
from app.shared.models import Chunk, Document, Notebook  # noqa: E402


SOURCES = "/api/v1/sources"


class DetachTestCase(DatabaseBackedTestCase):
    """Два пользователя, админ, по блокноту у каждого и второй — у владельца."""

    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()

        self.user = await self.make_user("owner", "user")
        self.other = await self.make_user("stranger", "user")
        self.admin = await self.make_user("root", "admin")

        self.notebook = await self.seed(
            Notebook(name="Свой блокнот", description=None,
                     domain_profile="general", owner_id=self.user.id)
        )
        # Второй свой блокнот нужен случаю «документ уехал в другой блокнот»:
        # оба блокнота принадлежат одному пользователю, поэтому 404 владения
        # тут невозможен и проверяется именно поведение отвязки.
        self.second_notebook = await self.seed(
            Notebook(name="Второй свой", description=None,
                     domain_profile="general", owner_id=self.user.id)
        )
        self.foreign_notebook = await self.seed(
            Notebook(name="Чужой блокнот", description=None,
                     domain_profile="general", owner_id=self.other.id)
        )

        # У документа заполнена тема: отвязка не имеет права её тронуть,
        # и без этих полей такая проверка была бы тривиально зелёной.
        self.document = await self._make_document(
            "own.txt", self.notebook.id, self.user.id,
            topic_label="Taxes", topic_cluster_index=3, topic_model_version=2,
        )
        self.loose_document = await self._make_document(
            "loose.txt", None, self.user.id
        )
        self.foreign_document = await self._make_document(
            "foreign.txt", self.foreign_notebook.id, self.other.id
        )

        self.own_chunk = await self.seed(
            Chunk(text="Фрагмент своего документа", page=1, chunk_index=0,
                  doc_id=self.document.id)
        )

        self.as_user(self.user)

    async def _make_document(
        self,
        name: str,
        notebook_id: int | None,
        owner_id: int,
        **topic_fields,
    ) -> Document:
        return await self.seed(
            Document(
                name=name,
                path=self.make_file(name, f"содержимое {name}"),
                size=10,
                status="indexed",
                notebook_id=notebook_id,
                owner_id=owner_id,
                **topic_fields,
            )
        )

    async def detach(self, notebook_id: int, source_ids: list[int]):
        return await self.client.post(
            f"{SOURCES}/detach",
            json={"notebook_id": notebook_id, "source_ids": source_ids},
        )

    def assertNotForbidden(self, response, expected: int = 404) -> None:
        """404, а не 403: 403 подтвердил бы существование чужого ресурса."""
        self.assertNotEqual(
            response.status_code, 403,
            "ответ 403 подтверждает существование чужого ресурса; "
            "модель доступа требует 404",
        )
        self.assertEqual(response.status_code, expected, response.text)


class DetachOwnSourcesTests(DetachTestCase):
    async def test_user_detaches_own_source_from_own_notebook(self):
        response = await self.detach(self.notebook.id, [self.document.id])

        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["updated_count"], 1)
        self.assertIsNone(body["documents"][0]["notebook_id"])
        # Форма ответа парна привязке: наружу уходит DocumentRead,
        # без owner_id и path.
        self.assertNotIn("owner_id", body["documents"][0])
        self.assertNotIn("path", body["documents"][0])

        stored = await self.get_row(Document, self.document.id)
        self.assertIsNone(stored.notebook_id)

    async def test_detach_touches_nothing_but_notebook_id(self):
        """Отвязка — не удаление: файл, чанки, векторы и тема остаются."""
        with patch("app.modules.documents.service.RAGService") as rag_service:
            response = await self.detach(self.notebook.id, [self.document.id])

        self.assertEqual(response.status_code, 200, response.text)
        # В ChromaDB ручка не ходит вовсе: векторы принадлежат документу,
        # а не блокноту, и после отвязки продолжают ему служить.
        rag_service.assert_not_called()

        stored = await self.get_row(Document, self.document.id)
        self.assertTrue(os.path.exists(stored.path), "файл документа удалён")
        self.assertEqual(stored.status, "indexed")
        self.assertEqual(stored.topic_label, "Taxes")
        self.assertEqual(stored.topic_cluster_index, 3)
        self.assertEqual(stored.topic_model_version, 2)
        self.assertEqual(
            [chunk.id for chunk in
             await self.rows_where(Chunk, Chunk.doc_id == self.document.id)],
            [self.own_chunk.id],
            "чанки документа исчезли при отвязке",
        )

    async def test_detach_of_already_detached_document_is_a_success(self):
        """Повторное нажатие и вторая вкладка получают согласие, не отказ.

        Цель запроса — «этого документа нет в блокноте X» — уже достигнута,
        и отказ заставил бы клиента показывать ошибку на исправном состоянии.
        """
        first = await self.detach(self.notebook.id, [self.document.id])
        self.assertEqual(first.status_code, 200, first.text)

        second = await self.detach(self.notebook.id, [self.document.id])

        self.assertEqual(second.status_code, 200, second.text)
        self.assertEqual(second.json()["updated_count"], 0)
        stored = await self.get_row(Document, self.document.id)
        self.assertIsNone(stored.notebook_id)

    async def test_detach_does_not_pull_document_out_of_another_notebook(self):
        """Запрос «убери из X» не трогает документ, уехавший в Y.

        Вкладка со вчерашним списком блокнота X не должна уметь снять
        документ с блокнота Y, которого пользователь в глаза не видел.
        """
        response = await self.detach(self.second_notebook.id, [self.document.id])

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["updated_count"], 0)
        stored = await self.get_row(Document, self.document.id)
        self.assertEqual(stored.notebook_id, self.notebook.id)

    async def test_updated_count_counts_only_actually_detached(self):
        response = await self.detach(
            self.notebook.id, [self.document.id, self.loose_document.id]
        )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["updated_count"], 1)
        self.assertIsNone(
            (await self.get_row(Document, self.document.id)).notebook_id
        )


class DetachRefusalsTests(DetachTestCase):
    async def test_empty_source_ids_return_400(self):
        response = await self.detach(self.notebook.id, [])

        self.assertEqual(response.status_code, 400, response.text)
        self.assertEqual(
            response.json().get("error_code"), SourceErrors.NO_IDS_PROVIDED
        )

    async def test_detach_of_foreign_document_returns_404(self):
        response = await self.detach(
            self.foreign_notebook.id, [self.foreign_document.id]
        )

        self.assertNotForbidden(response)
        stored = await self.get_row(Document, self.foreign_document.id)
        self.assertEqual(stored.notebook_id, self.foreign_notebook.id)

    async def test_detach_from_foreign_notebook_returns_404(self):
        response = await self.detach(self.foreign_notebook.id, [self.document.id])

        self.assertNotForbidden(response)
        self.assertEqual(
            response.json().get("error_code"), SourceErrors.NOTEBOOK_NOT_FOUND
        )

    async def test_mixed_batch_changes_nothing(self):
        response = await self.detach(
            self.notebook.id, [self.document.id, self.foreign_document.id]
        )

        self.assertNotForbidden(response)
        self.assertEqual(
            (await self.get_row(Document, self.document.id)).notebook_id,
            self.notebook.id,
            "часть смешанной пачки отвязана несмотря на отказ",
        )

    async def test_foreign_and_missing_documents_are_indistinguishable(self):
        foreign = await self.detach(self.notebook.id, [self.foreign_document.id])
        missing = await self.detach(self.notebook.id, [10_000_019])

        # Не только «одинаковы», но и «одинаково 404»: без этого сравнение
        # зеленело бы и на паре одинаковых 405 от отсутствующей ручки.
        self.assertNotForbidden(foreign)
        self.assertEqual(foreign.status_code, missing.status_code)
        # Тела сравниваются без detail: там перечислены сами id, а одинаковость
        # требуется от формы отказа — код и статус не выдают, существует ли
        # чужой документ.
        self.assertEqual(
            foreign.json().get("error_code"), missing.json().get("error_code")
        )


class AdminDetachTests(DetachTestCase):
    async def test_admin_detaches_foreign_document(self):
        self.as_user(self.admin)

        response = await self.detach(
            self.foreign_notebook.id, [self.foreign_document.id]
        )

        self.assertEqual(response.status_code, 200, response.text)
        stored = await self.get_row(Document, self.foreign_document.id)
        self.assertIsNone(stored.notebook_id)


if __name__ == "__main__":
    unittest.main()
