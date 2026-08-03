"""Цитата в ответе берётся из chunk.text, а не из текста, лежащего в индексе.

Что закрепляет этот файл. В ChromaDB кладётся не сам чанк, а обогащённый
текст: _build_embedding_text (app/modules/documents/service.py) приписывает к
нему «[имя документа | раздел | стр. N] », чтобы вектор нёс в себе контекст
документа. Для поиска это полезно, но выдача ChromaDB возвращается вместе с
этим префиксом, и слой отображения, бравший цитату из текста кандидата,
показывал пользователю служебную строку:

    [QA_taxcode_nb1 | СТАТЬЯ 3. НАЛОГ НА ДОБАВЛЕННУЮ СТОИМОСТЬ (НДС) | стр. 1]
    Стандартная ставка...

Правильный источник цитаты — таблица chunk: в ней лежит ровно тот текст,
который был в документе. Индекс при этом не трогается: обогащение там
намеренное, и переиндексация ничего бы не исправила.

Тест строит текст кандидата настоящим _build_embedding_text, а не строкой с
заранее прописанным префиксом. Поэтому он остаётся честным и после смены
формата обогащения: важно не «префикс выглядит так», а «в цитату уходит текст
из БД». Если цитата снова начнёт браться из item["text"], сравнение с
chunk.text не сойдётся — на всех трёх путях сразу (обычный чат, стриминг,
ask).

Отдельно проверяется цена правки: тексты достаются одним запросом на всю
выдачу. Счётчик обращений к таблице chunk ловит возврат к SELECT на каждую
цитату — N+1, который на глаз незаметен и вылезает только под нагрузкой.

Почему настоящий PostgreSQL. Проверяется связка «chunk_id из выдачи → строка в
таблице chunk», то есть выборка по первичному ключу и типы на её краях
(в выдаче id строковый, в базе — целый). Словарь вместо базы принял бы что
угодно и зеленел бы при любой ошибке в этом переходе; он же не заметил бы и
лишних запросов.

Ни ChromaDB, ни Ollama не поднимаются: поиск и генерация подменены, а всё
остальное — запросы к настоящей базе.
"""

import os
import sys
import unittest
from contextlib import asynccontextmanager, contextmanager
from unittest.mock import AsyncMock, patch

from sqlalchemy import event

# Каталог с тестами в sys.path: unittest discover кладёт его туда сам, но при
# запуске `python -m unittest tests.test_source_quote_db` этого не происходит.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dbfixtures import DatabaseBackedTestCase  # noqa: E402

from app.modules.ask.schemas import AskRequest  # noqa: E402
from app.modules.ask.service import handle_ask_request  # noqa: E402
from app.modules.chat.schemas import ChatRequest, RetrievalRequest  # noqa: E402
from app.modules.chat.service import (  # noqa: E402
    build_quote,
    chat_request,
    chat_request_stream,
    load_chunk_texts,
    retrieve_chunks,
)
from app.modules.documents.service import _build_embedding_text  # noqa: E402
from app.modules.rag import text_utils  # noqa: E402
from app.shared.models import Chunk, Document, Notebook  # noqa: E402


ANSWER = "Стандартная ставка налога составляет 14 процентов."
SECTION_JSON = '["СТАТЬЯ 3. НАЛОГ НА ДОБАВЛЕННУЮ СТОИМОСТЬ (НДС)"]'


class StubRAGService:
    """RAGService без ChromaDB и Ollama.

    Разбор запроса оставлен настоящим (те же функции text_utils, что и в
    рабочем классе): подмена касается только того, что ходит по сети.
    """

    normalize_query = staticmethod(text_utils.normalize_query)
    detect_language = staticmethod(text_utils.detect_language)
    is_prompt_injection_attempt = staticmethod(text_utils.is_prompt_injection_attempt)
    _detect_article_reference = staticmethod(text_utils.detect_article_reference)
    _sanitize_answer_text = staticmethod(text_utils.sanitize_answer_text)

    async def condense_query(self, query, chat_history, model=None):
        return query

    async def generate_answer(self, **kwargs):
        return ANSWER

    async def stream_answer(self, **kwargs):
        yield ANSWER


class QuoteSourceTestCase(DatabaseBackedTestCase):
    """Документ с чанками и выдача поиска, повторяющая содержимое индекса."""

    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()

        self.user = await self.make_user("owner", "user")
        self.as_user(self.user)

        self.notebook = await self.seed(
            Notebook(
                name="Налоги",
                description=None,
                domain_profile="tax",
                owner_id=self.user.id,
            )
        )
        self.document = await self.seed(
            Document(
                name="QA_taxcode_nb1.pdf",
                path=self.make_file("QA_taxcode_nb1.pdf"),
                size=42,
                notebook_id=self.notebook.id,
                owner_id=self.user.id,
            )
        )
        self.chunk = await self.make_chunk(
            "Стандартная ставка налога на добавленную стоимость составляет "
            "14 процентов от облагаемого оборота.",
            chunk_index=0,
        )

        for target in ("app.modules.chat.service", "app.modules.ask.service"):
            rag_patcher = patch(f"{target}.RAGService", StubRAGService)
            rag_patcher.start()
            self.addCleanup(rag_patcher.stop)

    # --- данные ---

    async def make_chunk(self, text: str, chunk_index: int) -> Chunk:
        return await self.seed(
            Chunk(
                text=text,
                page=1,
                chunk_index=chunk_index,
                section=SECTION_JSON,
                doc_id=self.document.id,
            )
        )

    def candidate(self, chunk: Chunk, rank: int = 1) -> dict:
        """Кандидат ровно в том виде, в каком его отдаёт векторный поиск.

        text — то, что лежит в ChromaDB: чанк, обогащённый при индексации.
        """
        return {
            "idx": chunk.chunk_index,
            "text": _build_embedding_text(
                chunk.text, self.document.name, chunk.page, chunk.section
            ),
            "metadata": {
                "doc_id": self.document.id,
                "doc_name": self.document.name,
                "page": chunk.page,
                "chunk_index": chunk.chunk_index,
                "section": chunk.section,
            },
            "chunk_id": str(chunk.id),
            "distance": 0.21,
            "retrieval_method": "vector",
            "rank": rank,
        }

    # --- вызовы обработчиков ---

    @contextmanager
    def patched_retrieval(self, candidates: list[dict]):
        with patch(
            "app.modules.chat.service.run_retrieval",
            AsyncMock(return_value=candidates),
        ), patch(
            "app.modules.ask.service.run_retrieval",
            AsyncMock(return_value=candidates),
        ):
            yield

    @asynccontextmanager
    async def _test_session(self):
        async with self.session_factory() as session:
            yield session

    async def run_chat(self, candidates: list[dict]):
        with self.patched_retrieval(candidates):
            async with self.session_factory() as session:
                return await chat_request(
                    chat_request=ChatRequest(question="Какая ставка НДС?"),
                    current_user=self.user,
                    session=session,
                )

    async def run_ask(self, candidates: list[dict]):
        with self.patched_retrieval(candidates):
            async with self.session_factory() as session:
                return await handle_ask_request(
                    ask_request=AskRequest(question="Какая ставка НДС?"),
                    current_user=self.user,
                    session=session,
                )

    async def run_stream(self, candidates: list[dict]) -> list[str]:
        # Стриминг открывает сессию сам (session_context), поэтому подменяется
        # именно она: подстановка через зависимости FastAPI сюда не достаёт.
        with self.patched_retrieval(candidates), patch(
            "app.modules.chat.service.session_context", self._test_session
        ):
            return [
                event_line
                async for event_line in chat_request_stream(
                    chat_request=ChatRequest(question="Какая ставка НДС?"),
                    current_user=self.user,
                )
            ]

    @contextmanager
    def chunk_select_counter(self):
        """Счётчик обращений к таблице chunk за время блока."""
        statements: list[str] = []

        def before(conn, cursor, statement, parameters, context, executemany):
            if "FROM chunk" in statement.replace('"', ""):
                statements.append(statement)

        engine = self.engine.sync_engine
        event.listen(engine, "before_cursor_execute", before)
        try:
            yield statements
        finally:
            event.remove(engine, "before_cursor_execute", before)


class QuoteComesFromDatabaseTests(QuoteSourceTestCase):
    """Цитата равна chunk.text на всех путях, отдающих источники."""

    async def test_chat_quote_is_chunk_text(self):
        response = await self.run_chat([self.candidate(self.chunk)])
        self.assertEqual(len(response.sources), 1)
        self.assertEqual(response.sources[0].quote, self.chunk.text)

    async def test_chat_quote_has_no_index_prefix(self):
        """Отдельная проверка ради понятного падения: префикс виден в отчёте."""
        response = await self.run_chat([self.candidate(self.chunk)])
        quote = response.sources[0].quote
        self.assertFalse(quote.startswith("["), quote)
        self.assertNotIn(self.document.name.removesuffix(".pdf"), quote)
        self.assertNotIn("стр. 1", quote)

    async def test_chat_stream_quote_is_chunk_text(self):
        events = await self.run_stream([self.candidate(self.chunk)])
        done = [line for line in events if line.startswith("event: done")]
        self.assertEqual(len(done), 1)
        self.assertIn(self.chunk.text, done[0])
        # Тот же кусок текста, но с префиксом, в ответе появиться не должен.
        self.assertNotIn("| стр. 1]", done[0])

    async def test_ask_quote_is_chunk_text(self):
        response = await self.run_ask([self.candidate(self.chunk)])
        self.assertEqual(len(response.citations), 1)
        self.assertEqual(response.citations[0].quote, self.chunk.text)

    async def test_retrieval_debug_quote_is_chunk_text(self):
        """Выдача /retrieval — тот же слой отображения, что и sources."""
        candidates = [self.candidate(self.chunk)]
        retrieval_result = {
            "vector_candidates": candidates,
            "lexical_candidates": [],
            "fused_candidates": candidates,
            "final_chunks": candidates,
        }
        with patch(
            "app.modules.chat.service.run_hybrid_retrieval",
            AsyncMock(return_value=retrieval_result),
        ):
            async with self.session_factory() as session:
                response = await retrieve_chunks(
                    retrieval_request=RetrievalRequest(question="Какая ставка НДС?"),
                    current_user=self.user,
                    session=session,
                )
        quotes = [
            item.quote
            for item in [
                *response.chunks,
                *response.vector_candidates,
                *response.fused_candidates,
            ]
        ]
        self.assertEqual(quotes, [self.chunk.text] * 3)

    async def test_long_chunk_is_truncated_from_database_text(self):
        """Обрезка до 240 символов считается по chunk.text, а не по индексу.

        Проверка не про длину как таковую: если бы цитата бралась из текста
        кандидата, в первые 240 символов попал бы префикс и хвост чанка
        оказался бы срезан раньше времени.
        """
        long_chunk = await self.make_chunk("Ставка. " * 60, chunk_index=1)
        response = await self.run_chat([self.candidate(long_chunk)])
        expected = long_chunk.text.strip()[:240].rstrip() + "..."
        self.assertEqual(response.sources[0].quote, expected)
        self.assertTrue(long_chunk.text.startswith(response.sources[0].quote[:200]))

    async def test_context_for_model_keeps_enriched_text(self):
        """Обогащение остаётся там, где оно полезно, — в контексте генерации.

        Правка касается только показа. Если бы заодно «почистили» контекст,
        модель потеряла бы указание на документ и раздел.
        """
        candidate = self.candidate(self.chunk)
        captured: dict = {}

        async def capture(_self, **kwargs):
            captured.update(kwargs)
            return ANSWER

        with patch.object(StubRAGService, "generate_answer", capture):
            await self.run_chat([candidate])
        self.assertIn(candidate["text"], captured["context"])


class MissingChunkTests(QuoteSourceTestCase):
    """Чанк удалён между поиском и отрисовкой.

    Решение: источник остаётся в ответе, цитаты у него нет. Ссылка по doc_id и
    странице продолжает работать, а текст из индекса как запасной вариант не
    подставляется — иначе служебный префикс вернулся бы в ответ молча и только
    на редком пути.
    """

    async def test_source_survives_without_quote(self):
        candidate = self.candidate(self.chunk)
        async with self.session_factory() as session:
            await session.delete(await session.get(Chunk, self.chunk.id))
            await session.commit()

        response = await self.run_chat([candidate])
        self.assertEqual(len(response.sources), 1)
        source = response.sources[0]
        self.assertIsNone(source.quote)
        self.assertEqual(source.doc_id, self.document.id)
        self.assertEqual(source.doc_name, self.document.name)
        self.assertEqual(source.chunk_id, candidate["chunk_id"])

    async def test_candidate_without_chunk_id_is_not_fatal(self):
        candidate = self.candidate(self.chunk)
        candidate["chunk_id"] = None
        response = await self.run_chat([candidate])
        self.assertIsNone(response.sources[0].quote)

    async def test_unknown_chunk_id_gives_no_quote(self):
        self.assertIsNone(build_quote("не-число", {}))
        self.assertIsNone(build_quote(None, {"1": "текст"}))


class ChunkTextsAreLoadedInOneQueryTests(QuoteSourceTestCase):
    """Тексты цитат достаются пакетом, как и в лексическом поиске."""

    async def test_single_query_for_many_candidates(self):
        chunks = [self.chunk]
        for index in range(1, 6):
            chunks.append(await self.make_chunk(f"Фрагмент {index}.", index))
        candidates = [
            self.candidate(chunk, rank=rank)
            for rank, chunk in enumerate(chunks, start=1)
        ]

        async with self.session_factory() as session:
            with self.chunk_select_counter() as statements:
                texts = await load_chunk_texts(session, candidates)

        self.assertEqual(len(statements), 1, statements)
        self.assertEqual(
            texts, {str(chunk.id): chunk.text for chunk in chunks}
        )

    async def test_chat_does_not_scale_queries_with_candidates(self):
        """Число обращений к chunk не зависит от размера выдачи."""
        extra = [await self.make_chunk(f"Фрагмент {i}.", i) for i in range(1, 6)]

        with self.chunk_select_counter() as one:
            await self.run_chat([self.candidate(self.chunk)])
        single_count = len(one)

        with self.chunk_select_counter() as many:
            await self.run_chat(
                [
                    self.candidate(chunk, rank=rank)
                    for rank, chunk in enumerate([self.chunk, *extra], start=1)
                ]
            )

        self.assertEqual(len(many), single_count, many)

    async def test_no_query_without_chunk_ids(self):
        async with self.session_factory() as session:
            with self.chunk_select_counter() as statements:
                texts = await load_chunk_texts(session, [])
        self.assertEqual(texts, {})
        self.assertEqual(statements, [])


if __name__ == "__main__":
    unittest.main()
