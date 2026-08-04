"""Настройки ретривала действуют сразу и на всех ручках, где они обещаны.

Здесь закрепляются два случая, в которых исправная настройка выглядела
сломанной именно для того, кто её проверяет.

1. Ключ кэша выдачи и reranker_enabled.

   Результат гибридного поиска лежит в _RETRIEVAL_CACHE пять минут. Ключ
   собирался из блокнота, набора доступных документов, лимитов, языка и
   запроса — но не из настроек. Админ переключал реранкер, повторял тот же
   вопрос, получал ответ из кэша, делал вывод «не работает» и возвращал
   настройку обратно. В ключ теперь входит хэш настроек, влияющих на выдачу
   (_RETRIEVAL_SETTING_KEYS), и заодно имя доменного профиля: его запрос может
   переопределить, а варианты запроса и rerank_results у профилей разные.

   Проверка идёт не на самом ключе, а на наблюдаемом следствии: считаются
   обращения к векторному поиску. Так тест ловит и обратную регрессию — если
   ключ начнёт зависеть от чего-то лишнего, кэш перестанет попадать на
   повторе одинакового запроса, и это тоже падение.

   Тот же кэш и порог релевантности (relevance_distance_threshold). Порог
   решает, какие фрагменты вообще считаются относящимися к вопросу, то есть
   меняет САМ СОСТАВ выдачи, — а правят его ровно тогда, когда ассистент
   ответил «данных нет», и проверяют повтором того же вопроса. Не попади он в
   ключ — повтор пришёл бы из кэша с прежним составом.

2. enable_condense_query и POST /chat/retrieve.

   Панель разбора запроса конденсировала запрос безусловно, то есть с
   выключенной настройкой показывала переписанный search_query — врала ровно
   тому инструменту, которым настройку проверяют.

   Отдельным тестом зафиксировано и решение по /ask: конденсации там нет и не
   добавляется. /ask — разовый запрос без истории диалога, а конденсация нужна
   для follow-up с местоимениями; на пустой истории condense_query возвращает
   исходный запрос, то есть вызов модели ушёл бы впустую. Тест держит это
   решение явным: если конденсацию туда однажды добавят «для симметрии», он
   упадёт и заставит перечитать обоснование.

Почему настоящий PostgreSQL. Лексическая половина гибридного поиска — это
запросы к таблице chunk, и в первом наборе она работает по-настоящему: кэш
проверяется на том же пути, что и в бою. Остальным тестам нужна связка
пользователь → блокнот → документ → журнал, то есть настоящие внешние ключи.

Ни ChromaDB, ни Ollama не поднимаются: векторный поиск, генерация и
cross-encoder подменены.
"""

import os
import sys
import types
import unittest
from contextlib import contextmanager
from unittest.mock import AsyncMock, patch

# Каталог с тестами в sys.path: unittest discover кладёт его туда сам, но при
# запуске `python -m unittest tests.test_retrieval_settings_db` этого не
# происходит.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dbfixtures import DatabaseBackedTestCase  # noqa: E402

from app.modules.ask.schemas import AskRequest  # noqa: E402
from app.modules.ask.service import handle_ask_request  # noqa: E402
from app.modules.chat.schemas import ChatRequest, RetrievalRequest  # noqa: E402
from app.modules.chat import service as chat_service  # noqa: E402
from app.modules.chat.service import (  # noqa: E402
    _RETRIEVAL_CACHE,
    _retrieval_cache_key,
    chat_request,
    retrieve_chunks,
    run_hybrid_retrieval,
)
from app.modules.rag import text_utils  # noqa: E402
from app.services.profile_resolver import resolve_profile  # noqa: E402
from app.shared.models import Chunk, Document, Log, Notebook  # noqa: E402
from app.shared.settings.runtime_settings import RuntimeSettingsService  # noqa: E402


QUESTION = "Какая ставка налога на добавленную стоимость?"
ANSWER = "Стандартная ставка налога составляет 14 процентов."
CONDENSED = "ставка налога на добавленную стоимость в кодексе"


def settings_with(**overrides) -> dict:
    """Полный набор настроек с точечными изменениями.

    Именно полный: get_settings() всегда отдаёт словарь со всеми ключами, и
    тест, подсовывающий обрезанный, проверял бы несуществующее состояние.
    """
    merged = dict(RuntimeSettingsService.DEFAULTS)
    merged["model"] = merged["chat_model"]
    merged.update(overrides)
    return merged


class CacheKeyCompositionTests(unittest.TestCase):
    """Состав ключа кэша: что обязано его менять, а что — нет."""

    def key(self, **overrides) -> str:
        return _retrieval_cache_key(
            1,
            QUESTION,
            {7, 8},
            limits=(20, 5),
            language="ru",
            runtime_settings=settings_with(**overrides),
            profile_name="tax",
        )

    def test_reranker_toggle_changes_key(self):
        self.assertNotEqual(
            self.key(reranker_enabled=False), self.key(reranker_enabled=True)
        )

    def test_embedding_model_changes_key(self):
        """Модель эмбеддингов выбирает коллекцию, то есть весь индекс."""
        self.assertNotEqual(
            self.key(embedding_model="nomic-embed-text"),
            self.key(embedding_model="bge-m3"),
        )

    def test_relevance_threshold_changes_key(self):
        """Порог решает состав выдачи, а не только её порядок."""
        self.assertNotEqual(
            self.key(relevance_distance_threshold=1.0),
            self.key(relevance_distance_threshold=1.3),
        )

    def test_reranker_model_changes_key(self):
        self.assertNotEqual(
            self.key(reranker_model="gemma4:e4b"), self.key(reranker_model="qwen3:4b")
        )

    def test_unrelated_setting_keeps_key(self):
        """Настройки, не влияющие на выдачу, кэш не сбрасывают.

        Модель чата участвует в конденсации, но её результат — сам запрос, а он
        в ключе уже есть строкой.
        """
        self.assertEqual(
            self.key(chat_model="gemma3n:e4b"), self.key(chat_model="llama3:8b")
        )

    def test_limits_are_not_duplicated_by_settings(self):
        """retrieval_top_k/top_k берутся из limits, а не из настроек.

        В limits лежат значения после resolve_retrieval_limits — с учётом
        переопределения из запроса, которого в настройках не видно. Если бы
        сырые настройки попали в ключ вторым слагаемым, один и тот же
        фактический поиск разъезжался бы по двум записям кэша.
        """
        self.assertEqual(
            self.key(retrieval_top_k=20, top_k=5), self.key(retrieval_top_k=40, top_k=3)
        )
        self.assertNotEqual(
            _retrieval_cache_key(1, QUESTION, {7, 8}, limits=(20, 5), language="ru"),
            _retrieval_cache_key(1, QUESTION, {7, 8}, limits=(40, 3), language="ru"),
        )

    def test_domain_profile_changes_key(self):
        base = dict(runtime_settings=settings_with(), limits=(20, 5), language="ru")
        self.assertNotEqual(
            _retrieval_cache_key(1, QUESTION, {7, 8}, profile_name="tax", **base),
            _retrieval_cache_key(1, QUESTION, {7, 8}, profile_name="legal", **base),
        )


class StubVectorSearch:
    """Векторная половина поиска без ChromaDB: считает обращения к себе."""

    def __init__(self, chunks: list[Chunk], document: Document):
        self.chunks = chunks
        self.document = document
        self.calls: list[str] = []

    normalize_query = staticmethod(text_utils.normalize_query)
    detect_language = staticmethod(text_utils.detect_language)

    def query_documents(self, query_text, n_results=10, where=None):
        self.calls.append(query_text)
        return {
            "documents": [[chunk.text for chunk in self.chunks]],
            "ids": [[str(chunk.id) for chunk in self.chunks]],
            "metadatas": [
                [
                    {
                        "doc_id": self.document.id,
                        "doc_name": self.document.name,
                        "page": chunk.page,
                        "chunk_index": chunk.chunk_index,
                    }
                    for chunk in self.chunks
                ]
            ],
            "distances": [[0.10 + 0.05 * index for index in range(len(self.chunks))]],
        }


async def _reversing_reranker(candidates, query, model, model_manager, top_k):
    """Cross-encoder-заглушка: порядок заведомо другой, чем без неё."""
    return list(reversed(candidates))[:top_k]


class RerankerToggleTests(DatabaseBackedTestCase):
    """Переключение реранкера видно сразу, повтор того же запроса — из кэша."""

    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()

        _RETRIEVAL_CACHE.clear()
        self.addCleanup(_RETRIEVAL_CACHE.clear)

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
                name="taxcode.pdf",
                path=self.make_file("taxcode.pdf"),
                size=42,
                notebook_id=self.notebook.id,
                owner_id=self.user.id,
            )
        )
        self.chunks = [
            await self.seed(
                Chunk(
                    text=text,
                    page=1,
                    chunk_index=index,
                    section=None,
                    doc_id=self.document.id,
                )
            )
            for index, text in enumerate(
                [
                    "Стандартная ставка налога на добавленную стоимость — 14 процентов.",
                    "Пониженная ставка налога применяется к отдельным операциям.",
                    "Ставка налога на добавленную стоимость при экспорте нулевая.",
                ]
            )
        ]
        self.profile = resolve_profile(notebook=self.notebook)
        self.search = StubVectorSearch(self.chunks, self.document)

    @contextmanager
    def _patched_reranker(self):
        # Модуль подменяется целиком: настоящий тянет torch и Qwen3-Reranker-4B.
        stub = types.ModuleType("app.modules.rag.reranker_service")
        stub.rerank_candidates = _reversing_reranker
        with patch.dict(
            sys.modules, {"app.modules.rag.reranker_service": stub}
        ):
            yield

    async def retrieve(self, **setting_overrides) -> dict:
        with patch.object(
            RuntimeSettingsService,
            "get_settings",
            return_value=settings_with(**setting_overrides),
        ), self._patched_reranker():
            async with self.session_factory() as session:
                return await run_hybrid_retrieval(
                    rag_service=self.search,
                    session=session,
                    profile=self.profile,
                    language="ru",
                    search_query=QUESTION,
                    original_query=QUESTION,
                    allowed_doc_ids={self.document.id},
                    retrieval_top_k=20,
                    final_top_k=2,
                    notebook_id=self.notebook.id,
                )

    @staticmethod
    def chunk_ids(result: dict) -> list[str]:
        return [item["chunk_id"] for item in result["final_chunks"]]

    async def test_repeat_of_same_question_comes_from_cache(self):
        """Опора для следующих тестов: кэш вообще работает."""
        first = await self.retrieve(reranker_enabled=False)
        self.assertGreater(len(self.search.calls), 0)
        calls_after_first = len(self.search.calls)

        second = await self.retrieve(reranker_enabled=False)
        self.assertEqual(len(self.search.calls), calls_after_first)
        self.assertIs(second, first)

    async def test_reranker_toggle_is_visible_immediately(self):
        without = await self.retrieve(reranker_enabled=False)
        calls_before_toggle = len(self.search.calls)

        with_reranker = await self.retrieve(reranker_enabled=True)

        self.assertGreater(
            len(self.search.calls),
            calls_before_toggle,
            "выдача взята из кэша: переключение реранкера не попало в ключ",
        )
        self.assertNotEqual(
            self.chunk_ids(with_reranker),
            self.chunk_ids(without),
            "реранкер отработал, но выдача не изменилась",
        )
        # И обратно: возвращать настройку админ тоже должен без пятиминутного
        # ожидания.
        restored = await self.retrieve(reranker_enabled=False)
        self.assertEqual(self.chunk_ids(restored), self.chunk_ids(without))

    async def test_embedding_model_change_is_visible_immediately(self):
        await self.retrieve(embedding_model="nomic-embed-text")
        calls_before = len(self.search.calls)

        await self.retrieve(embedding_model="bge-m3")

        self.assertGreater(
            len(self.search.calls),
            calls_before,
            "смена модели эмбеддингов меняет коллекцию, кэш обязан промахнуться",
        )

    async def test_saved_threshold_reaches_the_relevance_filter(self):
        """Порог берётся из настроек В МОМЕНТ ЗАПРОСА, а не с импорта модуля.

        Раньше он приезжал в rerank_retrieval_candidates умолчанием параметра —
        модульной константой RELEVANCE_DISTANCE_THRESHOLD, замороженной при
        импорте: правка в админ-панели не действовала бы до перезапуска
        процесса. Смотрим на то, с чем фильтр реально зовут.
        """
        captured: list[float] = []
        original = chat_service.rerank_retrieval_candidates

        def spy(candidates, **kwargs):
            captured.append(kwargs["distance_threshold"])
            return original(candidates, **kwargs)

        with patch.object(chat_service, "rerank_retrieval_candidates", spy):
            await self.retrieve(relevance_distance_threshold=1.25)

        self.assertEqual(captured, [1.25])

    async def test_threshold_change_is_visible_immediately(self):
        """Настройку правят ровно после «ответа нет» и проверяют тем же вопросом.

        Попади повтор в кэш — админ увидел бы прежний ответ и вернул порог
        обратно, решив, что настройка не работает. Ровно этот сценарий раздел
        уже проходил с reranker_enabled.
        """
        await self.retrieve(relevance_distance_threshold=1.0)
        calls_before = len(self.search.calls)

        await self.retrieve(relevance_distance_threshold=1.3)

        self.assertGreater(
            len(self.search.calls),
            calls_before,
            "выдача взята из кэша: порог релевантности не попал в ключ",
        )

    async def test_other_profile_does_not_reuse_cached_result(self):
        await self.retrieve()
        calls_before = len(self.search.calls)

        self.profile = resolve_profile(requested="legal")
        await self.retrieve()

        self.assertGreater(
            len(self.search.calls),
            calls_before,
            "профиль запроса меняет варианты запроса и rerank_results",
        )


class StubRAGService:
    """RAGService без ChromaDB и Ollama, с журналом обращений к конденсации."""

    condense_calls: list[tuple[str, list]] = []
    generate_calls: list[dict] = []

    normalize_query = staticmethod(text_utils.normalize_query)
    detect_language = staticmethod(text_utils.detect_language)
    is_prompt_injection_attempt = staticmethod(text_utils.is_prompt_injection_attempt)
    _detect_article_reference = staticmethod(text_utils.detect_article_reference)
    _sanitize_answer_text = staticmethod(text_utils.sanitize_answer_text)

    async def condense_query(self, query, chat_history, model=None):
        StubRAGService.condense_calls.append((query, list(chat_history)))
        return CONDENSED

    async def generate_answer(self, **kwargs):
        StubRAGService.generate_calls.append(kwargs)
        return ANSWER


class CondenseFlagTestCase(DatabaseBackedTestCase):
    """Пользователь с историей диалога: конденсации есть что переписывать."""

    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()

        StubRAGService.condense_calls = []
        StubRAGService.generate_calls = []

        self.user = await self.make_user("owner", "user")
        self.as_user(self.user)
        self.document = await self.seed(
            Document(
                name="taxcode.pdf",
                path=self.make_file("taxcode.pdf"),
                size=42,
                notebook_id=None,
                owner_id=self.user.id,
            )
        )
        self.chunk = await self.seed(
            Chunk(
                text="Стандартная ставка налога — 14 процентов.",
                page=1,
                chunk_index=0,
                section=None,
                doc_id=self.document.id,
            )
        )
        # Без истории condense_query возвращает запрос как есть, и тест на
        # выключенный флаг зеленел бы сам собой.
        await self.seed(
            Log(
                question="Что такое НДС?",
                answer="Налог на добавленную стоимость.",
                sources="[]",
                time_ms=1,
                user_id=self.user.id,
                notebook_id=None,
                domain_profile="tax",
            )
        )

        for target in ("app.modules.chat.service", "app.modules.ask.service"):
            rag_patcher = patch(f"{target}.RAGService", StubRAGService)
            rag_patcher.start()
            self.addCleanup(rag_patcher.stop)

    @contextmanager
    def condensation(self, enabled: bool):
        with patch.object(
            RuntimeSettingsService,
            "get_settings",
            return_value=settings_with(enable_condense_query=enabled),
        ):
            yield

    def candidate(self) -> dict:
        return {
            "idx": 0,
            "text": self.chunk.text,
            "metadata": {
                "doc_id": self.document.id,
                "doc_name": self.document.name,
                "page": 1,
                "chunk_index": 0,
            },
            "chunk_id": str(self.chunk.id),
            "distance": 0.2,
            "retrieval_method": "vector",
            "rank": 1,
        }

    async def run_retrieve_endpoint(self, enabled: bool):
        """POST /chat/retrieve: сам поиск подменён, важен только search_query."""
        candidates = [self.candidate()]
        hybrid = AsyncMock(
            return_value={
                "vector_candidates": candidates,
                "lexical_candidates": [],
                "fused_candidates": candidates,
                "final_chunks": candidates,
            }
        )
        with self.condensation(enabled), patch(
            "app.modules.chat.service.run_hybrid_retrieval", hybrid
        ):
            async with self.session_factory() as session:
                response = await retrieve_chunks(
                    retrieval_request=RetrievalRequest(question=QUESTION),
                    current_user=self.user,
                    session=session,
                )
        return response, hybrid.await_args.kwargs["search_query"]


class RetrieveEndpointRespectsFlagTests(CondenseFlagTestCase):
    """Панель разбора запроса показывает то, что происходит на самом деле."""

    async def test_condensation_runs_when_enabled(self):
        response, search_query = await self.run_retrieve_endpoint(True)
        self.assertEqual(len(StubRAGService.condense_calls), 1)
        self.assertEqual(search_query, CONDENSED)
        self.assertEqual(response.search_query, CONDENSED)

    async def test_condensation_skipped_when_disabled(self):
        response, search_query = await self.run_retrieve_endpoint(False)
        self.assertEqual(StubRAGService.condense_calls, [])
        expected = text_utils.normalize_query(QUESTION)
        self.assertEqual(search_query, expected)
        self.assertEqual(response.search_query, expected)

    async def test_history_reaches_condensation(self):
        """Проверка честности предыдущих тестов: история не пустая.

        Иначе настоящий condense_query вернул бы запрос без изменений и
        «выключено» было бы неотличимо от «включено» по любому признаку.
        """
        await self.run_retrieve_endpoint(True)
        _, chat_history = StubRAGService.condense_calls[0]
        self.assertTrue(chat_history)


class ChatRespectsFlagTests(CondenseFlagTestCase):
    """Та же настройка на обычном чате — чтобы поведение ручек не разошлось."""

    async def run_chat(self, enabled: bool):
        with self.condensation(enabled), patch(
            "app.modules.chat.service.run_retrieval",
            AsyncMock(return_value=[self.candidate()]),
        ):
            async with self.session_factory() as session:
                return await chat_request(
                    chat_request=ChatRequest(question=QUESTION),
                    current_user=self.user,
                    session=session,
                )

    async def test_condensation_runs_when_enabled(self):
        await self.run_chat(True)
        self.assertEqual(len(StubRAGService.condense_calls), 1)

    async def test_condensation_skipped_when_disabled(self):
        await self.run_chat(False)
        self.assertEqual(StubRAGService.condense_calls, [])


class AskDoesNotCondenseTests(CondenseFlagTestCase):
    """Решение по /ask, зафиксированное тестом.

    Конденсация туда не добавляется: /ask отвечает на разовый вопрос, историю
    диалога не читает и передаёт в генерацию chat_history=[]. На пустой истории
    condense_query возвращает исходный запрос, поэтому вызов модели ради неё
    был бы работой ради симметрии.
    """

    async def run_ask(self, enabled: bool):
        with self.condensation(enabled), patch(
            "app.modules.ask.service.run_retrieval",
            AsyncMock(return_value=[self.candidate()]),
        ):
            async with self.session_factory() as session:
                return await handle_ask_request(
                    ask_request=AskRequest(question=QUESTION),
                    current_user=self.user,
                    session=session,
                )

    async def test_ask_never_condenses(self):
        for enabled in (True, False):
            with self.subTest(enable_condense_query=enabled):
                StubRAGService.condense_calls = []
                await self.run_ask(enabled)
                self.assertEqual(StubRAGService.condense_calls, [])

    async def test_ask_generates_without_chat_history(self):
        """Обоснование решения, а не только его следствие."""
        await self.run_ask(True)
        self.assertEqual(StubRAGService.generate_calls[-1]["chat_history"], [])


if __name__ == "__main__":
    unittest.main()
