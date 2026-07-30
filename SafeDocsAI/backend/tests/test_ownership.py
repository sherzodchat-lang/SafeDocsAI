"""Регрессия на модель владения ресурсами (серия IDOR).

Единое правило живёт в app/api/deps.py: админ видит всё, остальные — только
свои ресурсы, а owner_id IS NULL доступен только админу. Наружу отдаётся 404,
а не 403, поэтому тесты проверяют именно 404: 403 подтвердил бы существование
чужого ресурса.

Тесты не поднимают ни Postgres, ни ChromaDB, ни Ollama: сессия БД заменяется
in-memory заглушкой (см. FakeAsyncSession), внешние сервисы — моками, как это
уже сделано в tests/test_services.py.
"""

import os
import tempfile
import unittest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from fastapi import HTTPException
from fastapi.testclient import TestClient
from sqlalchemy.sql import operators
from sqlalchemy.sql.elements import (
    BinaryExpression,
    BindParameter,
    BooleanClauseList,
    Null,
    UnaryExpression,
)

from app.api import deps
from app.core.database import get_session
from app.main import app
from app.modules.chat.service import resolve_notebook_scope
from app.shared.models import Chunk, Document, Log, Notebook, User


# --- Заглушка асинхронной сессии ----------------------------------------
#
# Достаточно интерпретировать те формы запросов, которые реально встречаются
# в проверяемых эндпоинтах: конъюнкция сравнений (=, IS, IN) плюс
# order_by/offset/limit. Всё остальное поднимает NotImplementedError, чтобы
# тест падал громко, а не «проходил» на молча пропущенном фильтре.


_COMPARATORS = {
    operators.eq: lambda left, right: left == right,
    operators.ne: lambda left, right: left != right,
    operators.lt: lambda left, right: left < right,
    operators.le: lambda left, right: left <= right,
    operators.gt: lambda left, right: left > right,
    operators.ge: lambda left, right: left >= right,
    operators.in_op: lambda left, right: left in right,
    operators.is_: lambda left, right: left is right,
    operators.is_not: lambda left, right: left is not right,
}

_NOT_IN_OP = getattr(operators, "not_in_op", None)
if _NOT_IN_OP is not None:  # pragma: no branch - зависит от версии SQLAlchemy
    _COMPARATORS[_NOT_IN_OP] = lambda left, right: left not in right


def _literal_value(expression):
    if isinstance(expression, BindParameter):
        return expression.value
    if isinstance(expression, Null):
        return None
    raise NotImplementedError(f"Unsupported literal: {expression!r}")


def _clause_matches(clause, obj) -> bool:
    if clause is None:
        return True
    if isinstance(clause, BooleanClauseList):
        matches = [_clause_matches(item, obj) for item in clause.clauses]
        if clause.operator is operators.and_:
            return all(matches)
        if clause.operator is operators.or_:
            return any(matches)
        raise NotImplementedError(f"Unsupported boolean operator: {clause.operator!r}")
    if isinstance(clause, BinaryExpression):
        comparator = _COMPARATORS.get(clause.operator)
        if comparator is None:
            raise NotImplementedError(f"Unsupported operator: {clause.operator!r}")
        attribute = getattr(clause.left, "key", None)
        if attribute is None:
            raise NotImplementedError(f"Unsupported left operand: {clause.left!r}")
        return comparator(getattr(obj, attribute), _literal_value(clause.right))
    raise NotImplementedError(f"Unsupported clause: {clause!r}")


def _sort_rows(rows, order_by_clauses):
    for clause in reversed(list(order_by_clauses)):
        descending = False
        column = clause
        if isinstance(clause, UnaryExpression):
            column = clause.element
            descending = clause.modifier is operators.desc_op
        attribute = getattr(column, "key", None)
        if attribute is None:
            raise NotImplementedError(f"Unsupported order_by: {clause!r}")
        rows.sort(key=lambda row: getattr(row, attribute), reverse=descending)
    return rows


class FakeResult:
    def __init__(self, rows):
        self._rows = list(rows)

    def all(self):
        return list(self._rows)

    def first(self):
        return self._rows[0] if self._rows else None

    def one_or_none(self):
        return self.first()

    def __iter__(self):
        return iter(self._rows)


class FakeAsyncSession:
    """Минимальная in-memory замена AsyncSession для проверок владения."""

    def __init__(self, rows=()):
        self.store: dict[type, dict[int, object]] = {}
        self.committed = 0
        for row in rows:
            self.seed(row)

    # -- наполнение -----------------------------------------------------
    def seed(self, obj):
        self.store.setdefault(type(obj), {})[obj.id] = obj
        return obj

    def rows(self, model):
        return list(self.store.get(model, {}).values())

    # -- API сессии -----------------------------------------------------
    async def get(self, model, primary_key):
        return self.store.get(model, {}).get(primary_key)

    async def exec(self, statement):
        descriptions = statement.column_descriptions
        if not descriptions:
            raise NotImplementedError("Statement without column descriptions")
        entities = {description["entity"] for description in descriptions}
        if len(entities) != 1:
            raise NotImplementedError("Multi-entity statements are not supported")
        entity = descriptions[0]["entity"]

        rows = [
            obj
            for obj in self.store.get(entity, {}).values()
            if _clause_matches(statement.whereclause, obj)
        ]
        _sort_rows(rows, statement._order_by_clauses)

        offset = statement._offset or 0
        if offset:
            rows = rows[offset:]
        if statement._limit is not None:
            rows = rows[: statement._limit]

        if descriptions[0]["expr"] is not entity:
            projected = []
            for row in rows:
                values = tuple(
                    getattr(row, description["name"]) for description in descriptions
                )
                projected.append(values[0] if len(values) == 1 else values)
            rows = projected
        return FakeResult(rows)

    def add(self, obj):
        if getattr(obj, "id", None) is None:
            bucket = self.store.setdefault(type(obj), {})
            obj.id = max(bucket, default=0) + 1
        self.seed(obj)

    async def delete(self, obj):
        self.store.get(type(obj), {}).pop(obj.id, None)

    async def commit(self):
        self.committed += 1

    async def refresh(self, obj):
        return obj

    async def flush(self):
        return None

    async def close(self):
        return None

    async def rollback(self):
        return None


# --- Общая фикстура -----------------------------------------------------

USER_A_ID = 1
USER_B_ID = 2
ADMIN_ID = 3

NOTEBOOK_A_ID = 10
NOTEBOOK_B_ID = 20
NOTEBOOK_ORPHAN_ID = 30

DOC_A_ID = 100
DOC_B_ID = 200
DOC_ORPHAN_ID = 300

CHUNK_A_ID = 1000
CHUNK_B_ID = 2000

LOG_A_ID = 400
LOG_B_ID = 500


def make_user(user_id: int, username: str, role: str) -> User:
    return User(id=user_id, username=username, role=role, password_hash="x")


class OwnershipFixture:
    """Два пользователя со своими ресурсами плюс админ и «ничьи» ресурсы."""

    def __init__(self, doc_a_path: str, doc_b_path: str):
        base = datetime(2026, 1, 1, 12, 0, 0)
        self.user_a = make_user(USER_A_ID, "alice", "content_manager")
        self.user_b = make_user(USER_B_ID, "bob", "content_manager")
        self.admin = make_user(ADMIN_ID, "root", "admin")

        self.notebook_a = Notebook(
            id=NOTEBOOK_A_ID, name="A notebook", domain_profile="general",
            owner_id=USER_A_ID, created_at=base,
        )
        self.notebook_b = Notebook(
            id=NOTEBOOK_B_ID, name="B notebook", domain_profile="general",
            owner_id=USER_B_ID, created_at=base + timedelta(minutes=1),
        )
        self.notebook_orphan = Notebook(
            id=NOTEBOOK_ORPHAN_ID, name="Legacy notebook", domain_profile="general",
            owner_id=None, created_at=base + timedelta(minutes=2),
        )

        self.doc_a = Document(
            id=DOC_A_ID, name="a-secret.txt", path=doc_a_path, size=11,
            language="ru", status="indexed",
            notebook_id=NOTEBOOK_A_ID, owner_id=USER_A_ID, created_at=base,
        )
        self.doc_b = Document(
            id=DOC_B_ID, name="b-secret.txt", path=doc_b_path, size=11,
            language="ru", status="indexed",
            notebook_id=NOTEBOOK_B_ID, owner_id=USER_B_ID, created_at=base,
        )
        self.doc_orphan = Document(
            id=DOC_ORPHAN_ID, name="legacy.txt", path=doc_a_path, size=11,
            language="ru", status="indexed",
            notebook_id=None, owner_id=None, created_at=base,
        )

        self.chunk_a = Chunk(
            id=CHUNK_A_ID, text="Секрет пользователя A", page=1,
            chunk_index=0, doc_id=DOC_A_ID,
        )
        self.chunk_b = Chunk(
            id=CHUNK_B_ID, text="Секрет пользователя B", page=1,
            chunk_index=0, doc_id=DOC_B_ID,
        )

        self.log_a = Log(
            id=LOG_A_ID, question="Вопрос A", answer="Ответ A", time_ms=1,
            user_id=USER_A_ID, notebook_id=NOTEBOOK_A_ID, created_at=base,
        )
        self.log_b = Log(
            id=LOG_B_ID, question="Вопрос B", answer="Ответ B", time_ms=1,
            user_id=USER_B_ID, notebook_id=NOTEBOOK_B_ID, created_at=base,
        )

        self.session = FakeAsyncSession(
            [
                self.user_a, self.user_b, self.admin,
                self.notebook_a, self.notebook_b, self.notebook_orphan,
                self.doc_a, self.doc_b, self.doc_orphan,
                self.chunk_a, self.chunk_b,
                self.log_a, self.log_b,
            ]
        )


class OwnershipApiTestCase(unittest.TestCase):
    """База для API-тестов: TestClient с подменёнными сессией и пользователем."""

    def setUp(self):
        self._tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp_dir.cleanup)

        doc_a_path = os.path.join(self._tmp_dir.name, "a-secret.txt")
        doc_b_path = os.path.join(self._tmp_dir.name, "b-secret.txt")
        with open(doc_a_path, "w", encoding="utf-8") as file_obj:
            file_obj.write("SECRET-A")
        with open(doc_b_path, "w", encoding="utf-8") as file_obj:
            file_obj.write("SECRET-B")

        self.fixture = OwnershipFixture(doc_a_path, doc_b_path)
        self.session = self.fixture.session
        self.current_user = self.fixture.user_a

        async def _override_session():
            yield self.session

        async def _override_current_user():
            return self.current_user

        app.dependency_overrides[get_session] = _override_session
        app.dependency_overrides[deps.get_session] = _override_session
        app.dependency_overrides[deps.get_current_user] = _override_current_user
        app.dependency_overrides[deps.get_current_user_short_lived] = (
            _override_current_user
        )
        self.addCleanup(app.dependency_overrides.clear)

        self.client = TestClient(app)
        self.addCleanup(self.client.close)

    def as_user(self, user: User) -> None:
        self.current_user = user


# --- Правило владения ---------------------------------------------------


class UserOwnsRuleTests(unittest.TestCase):
    def test_admin_owns_everything(self):
        admin = make_user(ADMIN_ID, "root", "admin")
        self.assertTrue(deps.user_owns(USER_B_ID, admin))
        self.assertTrue(deps.user_owns(ADMIN_ID, admin))

    def test_admin_owns_resources_without_owner(self):
        admin = make_user(ADMIN_ID, "root", "admin")
        self.assertTrue(deps.user_owns(None, admin))

    def test_regular_user_owns_only_own_resource(self):
        user = make_user(USER_A_ID, "alice", "user")
        self.assertTrue(deps.user_owns(USER_A_ID, user))
        self.assertFalse(deps.user_owns(USER_B_ID, user))

    def test_orphan_resource_is_not_owned_by_regular_user(self):
        """owner_id IS NULL — legacy-состояние, а не «общий ресурс»."""
        user = make_user(USER_A_ID, "alice", "user")
        self.assertFalse(deps.user_owns(None, user))
        manager = make_user(USER_A_ID, "alice", "content_manager")
        self.assertFalse(deps.user_owns(None, manager))


class OwnershipDependencyTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.fixture = OwnershipFixture("/nonexistent/a", "/nonexistent/b")
        self.session = self.fixture.session

    async def test_get_owned_document_hides_foreign_document_behind_404(self):
        with self.assertRaises(HTTPException) as ctx:
            await deps.get_owned_document(
                DOC_B_ID, self.session, self.fixture.user_a
            )
        self.assertEqual(ctx.exception.status_code, 404)

    async def test_get_owned_document_uses_same_404_for_missing_id(self):
        """Чужой и несуществующий id должны быть неотличимы снаружи."""
        with self.assertRaises(HTTPException) as foreign:
            await deps.get_owned_document(
                DOC_B_ID, self.session, self.fixture.user_a
            )
        with self.assertRaises(HTTPException) as missing:
            await deps.get_owned_document(999999, self.session, self.fixture.user_a)
        self.assertEqual(
            (foreign.exception.status_code, foreign.exception.detail),
            (missing.exception.status_code, missing.exception.detail),
        )

    async def test_get_owned_notebook_rejects_foreign_and_orphan(self):
        for notebook_id in (NOTEBOOK_B_ID, NOTEBOOK_ORPHAN_ID):
            with self.subTest(notebook_id=notebook_id):
                with self.assertRaises(HTTPException) as ctx:
                    await deps.get_owned_notebook(
                        notebook_id, self.session, self.fixture.user_a
                    )
                self.assertEqual(ctx.exception.status_code, 404)

    async def test_get_owned_notebook_allows_admin_everywhere(self):
        for notebook_id in (NOTEBOOK_A_ID, NOTEBOOK_B_ID, NOTEBOOK_ORPHAN_ID):
            with self.subTest(notebook_id=notebook_id):
                notebook = await deps.get_owned_notebook(
                    notebook_id, self.session, self.fixture.admin
                )
                self.assertEqual(notebook.id, notebook_id)

    async def test_assert_owns_notebook_is_noop_without_notebook_id(self):
        await deps.assert_owns_notebook(None, self.session, self.fixture.user_a)

    async def test_assert_owns_notebook_rejects_foreign_notebook(self):
        with self.assertRaises(HTTPException) as ctx:
            await deps.assert_owns_notebook(
                NOTEBOOK_B_ID, self.session, self.fixture.user_a
            )
        self.assertEqual(ctx.exception.status_code, 404)


# --- Источники ----------------------------------------------------------


class SourcePreviewOwnershipTests(OwnershipApiTestCase):
    def test_preview_of_foreign_document_returns_404(self):
        """До фикса любой авторизованный скачивал любой файл по id."""
        response = self.client.get(f"/api/v1/sources/{DOC_B_ID}/preview")
        self.assertEqual(response.status_code, 404)
        self.assertNotIn(b"SECRET-B", response.content)

    def test_preview_of_own_document_succeeds(self):
        response = self.client.get(f"/api/v1/sources/{DOC_A_ID}/preview")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, b"SECRET-A")

    def test_preview_of_orphan_document_is_admin_only(self):
        response = self.client.get(f"/api/v1/sources/{DOC_ORPHAN_ID}/preview")
        self.assertEqual(response.status_code, 404)

        self.as_user(self.fixture.admin)
        response = self.client.get(f"/api/v1/sources/{DOC_ORPHAN_ID}/preview")
        self.assertEqual(response.status_code, 200)

    def test_admin_can_preview_any_document(self):
        self.as_user(self.fixture.admin)
        response = self.client.get(f"/api/v1/sources/{DOC_B_ID}/preview")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, b"SECRET-B")


class SourceChunkContextOwnershipTests(OwnershipApiTestCase):
    def test_chunk_context_of_foreign_document_returns_404(self):
        response = self.client.get(
            f"/api/v1/sources/{DOC_B_ID}/chunk/{CHUNK_B_ID}/context"
        )
        self.assertEqual(response.status_code, 404)
        self.assertNotIn("Секрет пользователя B", response.text)

    def test_foreign_chunk_id_through_own_document_returns_404(self):
        """Чужой chunk_id не должен вытягиваться через свой doc id."""
        response = self.client.get(
            f"/api/v1/sources/{DOC_A_ID}/chunk/{CHUNK_B_ID}/context"
        )
        self.assertEqual(response.status_code, 404)
        self.assertNotIn("Секрет пользователя B", response.text)

    def test_chunk_context_of_own_document_succeeds(self):
        response = self.client.get(
            f"/api/v1/sources/{DOC_A_ID}/chunk/{CHUNK_A_ID}/context"
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual([item["chunk_id"] for item in payload], [CHUNK_A_ID])
        self.assertTrue(payload[0]["highlight"])

    def test_admin_can_read_any_chunk_context(self):
        self.as_user(self.fixture.admin)
        response = self.client.get(
            f"/api/v1/sources/{DOC_B_ID}/chunk/{CHUNK_B_ID}/context"
        )
        self.assertEqual(response.status_code, 200)


class SourceChunksAndDeleteOwnershipTests(OwnershipApiTestCase):
    def test_chunks_of_foreign_document_return_404(self):
        response = self.client.get(f"/api/v1/sources/{DOC_B_ID}/chunks")
        self.assertEqual(response.status_code, 404)
        self.assertNotIn("Секрет пользователя B", response.text)

    def test_chunks_of_orphan_document_are_admin_only(self):
        response = self.client.get(f"/api/v1/sources/{DOC_ORPHAN_ID}/chunks")
        self.assertEqual(response.status_code, 404)

        self.as_user(self.fixture.admin)
        response = self.client.get(f"/api/v1/sources/{DOC_ORPHAN_ID}/chunks")
        self.assertEqual(response.status_code, 200)

    def test_chunks_of_own_document_succeed(self):
        response = self.client.get(f"/api/v1/sources/{DOC_A_ID}/chunks")
        self.assertEqual(response.status_code, 200)
        self.assertEqual([item["id"] for item in response.json()], [CHUNK_A_ID])

    def test_delete_of_foreign_document_returns_404_and_keeps_document(self):
        with patch("app.modules.documents.service.RAGService") as rag_service:
            response = self.client.delete(f"/api/v1/sources/{DOC_B_ID}")
        self.assertEqual(response.status_code, 404)
        rag_service.assert_not_called()
        self.assertIsNotNone(self.session.store[Document].get(DOC_B_ID))
        self.assertTrue(os.path.exists(self.fixture.doc_b.path))

    def test_delete_of_orphan_document_is_admin_only(self):
        with patch("app.modules.documents.service.RAGService"):
            response = self.client.delete(f"/api/v1/sources/{DOC_ORPHAN_ID}")
        self.assertEqual(response.status_code, 404)
        self.assertIsNotNone(self.session.store[Document].get(DOC_ORPHAN_ID))

    def test_delete_of_own_document_succeeds(self):
        with patch("app.modules.documents.service.RAGService"):
            response = self.client.delete(f"/api/v1/sources/{DOC_A_ID}")
        self.assertEqual(response.status_code, 200)
        self.assertIsNone(self.session.store[Document].get(DOC_A_ID))


class SourceListOwnershipTests(OwnershipApiTestCase):
    def test_list_hides_documents_of_other_users(self):
        response = self.client.get("/api/v1/sources/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual([item["id"] for item in response.json()], [DOC_A_ID])

    def test_list_does_not_leak_owner_or_server_path(self):
        response = self.client.get("/api/v1/sources/")
        self.assertEqual(response.status_code, 200)
        for item in response.json():
            self.assertNotIn("path", item)
            self.assertNotIn("owner_id", item)

    def test_list_for_admin_contains_everything(self):
        self.as_user(self.fixture.admin)
        response = self.client.get("/api/v1/sources/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            sorted(item["id"] for item in response.json()),
            [DOC_A_ID, DOC_B_ID, DOC_ORPHAN_ID],
        )

    def test_list_filtered_by_foreign_notebook_is_empty(self):
        response = self.client.get(
            "/api/v1/sources/", params={"notebook_id": NOTEBOOK_B_ID}
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), [])


class SourceAttachOwnershipTests(OwnershipApiTestCase):
    def test_cannot_attach_foreign_document_to_own_notebook(self):
        response = self.client.post(
            "/api/v1/sources/attach",
            json={"notebook_id": NOTEBOOK_A_ID, "source_ids": [DOC_B_ID]},
        )
        self.assertEqual(response.status_code, 404)
        self.assertEqual(self.fixture.doc_b.notebook_id, NOTEBOOK_B_ID)

    def test_cannot_attach_own_document_to_foreign_notebook(self):
        response = self.client.post(
            "/api/v1/sources/attach",
            json={"notebook_id": NOTEBOOK_B_ID, "source_ids": [DOC_A_ID]},
        )
        self.assertEqual(response.status_code, 404)
        self.assertEqual(self.fixture.doc_a.notebook_id, NOTEBOOK_A_ID)

    def test_cannot_attach_orphan_document(self):
        response = self.client.post(
            "/api/v1/sources/attach",
            json={"notebook_id": NOTEBOOK_A_ID, "source_ids": [DOC_ORPHAN_ID]},
        )
        self.assertEqual(response.status_code, 404)
        self.assertIsNone(self.fixture.doc_orphan.notebook_id)

    def test_mixed_batch_with_foreign_document_changes_nothing(self):
        response = self.client.post(
            "/api/v1/sources/attach",
            json={"notebook_id": NOTEBOOK_A_ID, "source_ids": [DOC_A_ID, DOC_B_ID]},
        )
        self.assertEqual(response.status_code, 404)
        self.assertEqual(self.fixture.doc_b.notebook_id, NOTEBOOK_B_ID)

    def test_attach_own_document_to_own_notebook_succeeds(self):
        self.fixture.doc_a.notebook_id = None
        response = self.client.post(
            "/api/v1/sources/attach",
            json={"notebook_id": NOTEBOOK_A_ID, "source_ids": [DOC_A_ID]},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["updated_count"], 1)
        self.assertEqual(self.fixture.doc_a.notebook_id, NOTEBOOK_A_ID)


# --- Блокноты -----------------------------------------------------------


class NotebookOwnershipTests(OwnershipApiTestCase):
    def test_get_foreign_notebook_returns_404(self):
        response = self.client.get(f"/api/v1/notebooks/{NOTEBOOK_B_ID}")
        self.assertEqual(response.status_code, 404)
        self.assertNotIn("B notebook", response.text)

    def test_get_own_notebook_succeeds(self):
        response = self.client.get(f"/api/v1/notebooks/{NOTEBOOK_A_ID}")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["id"], NOTEBOOK_A_ID)

    def test_get_orphan_notebook_is_admin_only(self):
        response = self.client.get(f"/api/v1/notebooks/{NOTEBOOK_ORPHAN_ID}")
        self.assertEqual(response.status_code, 404)

        self.as_user(self.fixture.admin)
        response = self.client.get(f"/api/v1/notebooks/{NOTEBOOK_ORPHAN_ID}")
        self.assertEqual(response.status_code, 200)

    def test_delete_foreign_notebook_returns_404_and_keeps_it(self):
        response = self.client.delete(f"/api/v1/notebooks/{NOTEBOOK_B_ID}")
        self.assertEqual(response.status_code, 404)
        self.assertIsNotNone(self.session.store[Notebook].get(NOTEBOOK_B_ID))
        self.assertIsNotNone(self.session.store[Document].get(DOC_B_ID))

    def test_delete_orphan_notebook_returns_404_for_regular_user(self):
        response = self.client.delete(f"/api/v1/notebooks/{NOTEBOOK_ORPHAN_ID}")
        self.assertEqual(response.status_code, 404)
        self.assertIsNotNone(self.session.store[Notebook].get(NOTEBOOK_ORPHAN_ID))

    def test_delete_own_notebook_succeeds(self):
        with patch("app.modules.rag.service.RAGService"):
            response = self.client.delete(f"/api/v1/notebooks/{NOTEBOOK_A_ID}")
        self.assertEqual(response.status_code, 200)
        self.assertIsNone(self.session.store[Notebook].get(NOTEBOOK_A_ID))
        self.assertIsNotNone(self.session.store[Notebook].get(NOTEBOOK_B_ID))
        self.assertIsNotNone(self.session.store[Document].get(DOC_B_ID))

    def test_list_shows_only_own_notebooks(self):
        response = self.client.get("/api/v1/notebooks/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual([item["id"] for item in response.json()], [NOTEBOOK_A_ID])

    def test_list_for_admin_shows_all_notebooks(self):
        self.as_user(self.fixture.admin)
        response = self.client.get("/api/v1/notebooks/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            sorted(item["id"] for item in response.json()),
            [NOTEBOOK_A_ID, NOTEBOOK_B_ID, NOTEBOOK_ORPHAN_ID],
        )


# --- Логи ---------------------------------------------------------------


class LogRatingOwnershipTests(OwnershipApiTestCase):
    def test_rating_foreign_log_returns_404(self):
        response = self.client.post(
            f"/api/v1/logs/{LOG_B_ID}/rating", json={"rating": "up"}
        )
        self.assertEqual(response.status_code, 404)
        self.assertIsNone(self.fixture.log_b.rating)

    def test_rating_foreign_log_does_not_leak_question_or_answer(self):
        response = self.client.post(
            f"/api/v1/logs/{LOG_B_ID}/rating", json={"rating": "up"}
        )
        self.assertNotIn("Вопрос B", response.text)
        self.assertNotIn("Ответ B", response.text)

    def test_rating_own_log_succeeds(self):
        response = self.client.post(
            f"/api/v1/logs/{LOG_A_ID}/rating", json={"rating": "down"}
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.fixture.log_a.rating, "down")

    def test_rating_response_exposes_only_id_and_rating(self):
        """Полная модель Log вернула бы question/answer/sources наружу."""
        response = self.client.post(
            f"/api/v1/logs/{LOG_A_ID}/rating", json={"rating": "up"}
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(set(response.json()), {"id", "rating"})
        self.assertNotIn("Вопрос A", response.text)
        self.assertNotIn("Ответ A", response.text)

    def test_orphan_log_rating_is_admin_only(self):
        orphan_log = Log(
            id=600, question="Ничей вопрос", answer="Ничей ответ", time_ms=1,
            user_id=None, notebook_id=None, created_at=datetime(2026, 1, 1),
        )
        self.session.seed(orphan_log)

        response = self.client.post(
            "/api/v1/logs/600/rating", json={"rating": "up"}
        )
        self.assertEqual(response.status_code, 404)

        self.as_user(self.fixture.admin)
        response = self.client.post(
            "/api/v1/logs/600/rating", json={"rating": "up"}
        )
        self.assertEqual(response.status_code, 200)


# --- Чат ----------------------------------------------------------------


class ChatNotebookScopeTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.fixture = OwnershipFixture("/nonexistent/a", "/nonexistent/b")
        self.session = self.fixture.session

    async def test_foreign_notebook_id_raises_404(self):
        with self.assertRaises(HTTPException) as ctx:
            await resolve_notebook_scope(
                notebook_id=NOTEBOOK_B_ID,
                session=self.session,
                current_user=self.fixture.user_a,
            )
        self.assertEqual(ctx.exception.status_code, 404)

    async def test_orphan_notebook_id_raises_404_for_regular_user(self):
        with self.assertRaises(HTTPException) as ctx:
            await resolve_notebook_scope(
                notebook_id=NOTEBOOK_ORPHAN_ID,
                session=self.session,
                current_user=self.fixture.user_a,
            )
        self.assertEqual(ctx.exception.status_code, 404)

    async def test_own_notebook_scopes_retrieval_to_own_documents(self):
        notebook, allowed_doc_ids = await resolve_notebook_scope(
            notebook_id=NOTEBOOK_A_ID,
            session=self.session,
            current_user=self.fixture.user_a,
        )
        self.assertEqual(notebook.id, NOTEBOOK_A_ID)
        self.assertEqual(allowed_doc_ids, {DOC_A_ID})

    async def test_missing_notebook_id_still_scopes_to_own_documents(self):
        """Без блокнота поиск не должен падать на все документы системы."""
        notebook, allowed_doc_ids = await resolve_notebook_scope(
            notebook_id=None,
            session=self.session,
            current_user=self.fixture.user_a,
        )
        self.assertIsNone(notebook)
        self.assertEqual(allowed_doc_ids, {DOC_A_ID})

    async def test_admin_without_notebook_is_unrestricted(self):
        notebook, allowed_doc_ids = await resolve_notebook_scope(
            notebook_id=None,
            session=self.session,
            current_user=self.fixture.admin,
        )
        self.assertIsNone(notebook)
        self.assertIsNone(allowed_doc_ids)

    async def test_admin_can_scope_to_any_notebook(self):
        notebook, allowed_doc_ids = await resolve_notebook_scope(
            notebook_id=NOTEBOOK_B_ID,
            session=self.session,
            current_user=self.fixture.admin,
        )
        self.assertEqual(notebook.id, NOTEBOOK_B_ID)
        self.assertEqual(allowed_doc_ids, {DOC_B_ID})


class ChatEndpointOwnershipTests(OwnershipApiTestCase):
    def test_chat_with_foreign_notebook_returns_404(self):
        with patch("app.modules.chat.service.RAGService", MagicMock()):
            response = self.client.post(
                "/api/v1/chat/",
                json={"question": "Что в чужих документах?",
                      "notebook_id": NOTEBOOK_B_ID},
            )
        self.assertEqual(response.status_code, 404)

    def test_retrieve_with_foreign_notebook_returns_404(self):
        with patch("app.modules.chat.service.RAGService", MagicMock()):
            response = self.client.post(
                "/api/v1/chat/retrieve",
                json={"question": "Что в чужих документах?",
                      "notebook_id": NOTEBOOK_B_ID},
            )
        self.assertEqual(response.status_code, 404)

    def test_ask_with_foreign_notebook_returns_404(self):
        """/ask ходит через тот же resolve_notebook_scope."""
        with patch("app.modules.ask.service.RAGService", MagicMock()):
            response = self.client.post(
                "/api/v1/ask/",
                json={"question": "Что в чужих документах?",
                      "notebook_id": NOTEBOOK_B_ID},
            )
        self.assertEqual(response.status_code, 404)


if __name__ == "__main__":
    unittest.main()
