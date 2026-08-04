"""Нормализация ответа модели и отбор фрагментов под слайд.

Проверяется то, что решает исход генерации ещё до базы и до Ollama:

* канонизация chunk_id (число -> строка) и то, что проверка на подмножество
  выданных фрагментов выполняется уже ПОСЛЕ неё;
* дедупликация цитат внутри слайда — в нормализаторе схемы, а не у
  потребителей;
* нижняя граница в два буллета (мера 1 правила «не добивать»);
* дайджест уже написанного (мера 2) и его потолок;
* отбор финальной пятёрки, предпочитающий ещё не цитированные фрагменты
  (мера 3), включая его мягкость.

Ни базы, ни модели здесь не нужно: всё перечисленное — чистые функции.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.modules.presentations.constants import (  # noqa: E402
    DIGEST_MAX_CHARS,
    SLIDE_RETRIEVAL_TOP_K,
)
from app.modules.presentations.llm_schemas import (  # noqa: E402
    LlmResponseError,
    PresentationSlide,
    SlideCitation,
    validate_slide,
)
from app.modules.presentations.prompts import build_written_digest  # noqa: E402
from app.modules.presentations.service import select_slide_chunks  # noqa: E402


def slide_payload(**overrides):
    payload = {
        "heading": "Ставки НДС",
        "bullets": ["Первый факт", "Второй факт"],
        "citations": [{"source_id": 7, "chunk_id": 45}],
    }
    payload.update(overrides)
    return payload


class ChunkIdCanonicalizationTests(unittest.TestCase):
    """Число или строка на входе — строка на выходе, всегда."""

    def test_integer_becomes_string(self):
        citation = SlideCitation.model_validate({"source_id": 7, "chunk_id": 45})
        self.assertEqual(citation.chunk_id, "45")

    def test_string_stays_untouched(self):
        citation = SlideCitation.model_validate({"source_id": 7, "chunk_id": "45"})
        self.assertEqual(citation.chunk_id, "45")

    def test_boolean_is_not_a_chunk_id(self):
        # bool — подкласс int, и без отдельной ветки True превратился бы в "True".
        with self.assertRaises(Exception):
            SlideCitation.model_validate({"source_id": 7, "chunk_id": True})

    def test_float_is_rejected(self):
        # 45.0 -> "45.0" не совпало бы ни с одним выданным идентификатором,
        # то есть приведение молча испортило бы ссылку.
        with self.assertRaises(Exception):
            SlideCitation.model_validate({"source_id": 7, "chunk_id": 45.0})

    def test_subset_check_runs_after_canonicalization(self):
        """Числовая цитата на выданный фрагмент проходит проверку.

        Ключи allowed_citations — строки, ответ модели — числа. Если бы
        проверка стояла до канонизации, законный ответ отвергался бы целиком.
        """
        slide = validate_slide(
            '{"heading": "h", "bullets": ["a", "b"], '
            '"citations": [{"source_id": 7, "chunk_id": 45}]}',
            allowed_citations={"45": 7},
        )
        self.assertEqual([c.chunk_id for c in slide.citations], ["45"])

    def test_subset_check_still_rejects_a_foreign_chunk(self):
        with self.assertRaises(LlmResponseError):
            validate_slide(
                '{"heading": "h", "bullets": ["a", "b"], '
                '"citations": [{"source_id": 7, "chunk_id": 999}]}',
                allowed_citations={"45": 7},
            )


class CitationDeduplicationTests(unittest.TestCase):
    def test_duplicates_collapse_preserving_first_order(self):
        slide = PresentationSlide.model_validate(
            slide_payload(
                citations=[
                    {"source_id": 7, "chunk_id": 45},
                    {"source_id": 8, "chunk_id": "46"},
                    {"source_id": 7, "chunk_id": "45"},
                    {"source_id": 8, "chunk_id": 46},
                ]
            )
        )
        self.assertEqual(
            [(c.source_id, c.chunk_id) for c in slide.citations],
            [(7, "45"), (8, "46")],
        )

    def test_same_chunk_with_a_different_source_is_not_a_duplicate(self):
        """Противоречие обязано дойти до проверки, а не схлопнуться в дубль."""
        with self.assertRaises(LlmResponseError):
            validate_slide(
                '{"heading": "h", "bullets": ["a", "b"], "citations": ['
                '{"source_id": 7, "chunk_id": 45}, {"source_id": 8, "chunk_id": 45}]}',
                allowed_citations={"45": 7},
            )


class SlideBulletBoundsTests(unittest.TestCase):
    def test_two_bullets_are_enough(self):
        slide = PresentationSlide.model_validate(slide_payload(bullets=["a", "b"]))
        self.assertEqual(len(slide.bullets), 2)

    def test_single_bullet_is_still_rejected(self):
        with self.assertRaises(Exception):
            PresentationSlide.model_validate(slide_payload(bullets=["a"]))

    def test_six_bullets_are_rejected(self):
        with self.assertRaises(Exception):
            PresentationSlide.model_validate(
                slide_payload(bullets=["a", "b", "c", "d", "e", "f"])
            )


class WrittenDigestTests(unittest.TestCase):
    def test_digest_holds_only_bullet_texts(self):
        digest = build_written_digest([["Первый", "Второй"], ["Третий"]])
        self.assertEqual(digest, "- Первый\n- Второй\n- Третий")

    def test_empty_history_gives_empty_digest(self):
        self.assertEqual(build_written_digest([]), "")
        self.assertEqual(build_written_digest([[], ["   "]]), "")

    def test_angle_brackets_are_escaped(self):
        # Дайджест уезжает в промпт как данные, а не как разметка.
        self.assertEqual(build_written_digest([["<b>факт</b>"]]), "- &lt;b&gt;факт&lt;/b&gt;")

    def test_cap_drops_the_oldest_bullets(self):
        bullets = [[f"{index:04d} " + "я" * 195] for index in range(200)]
        digest = build_written_digest(bullets)
        self.assertLessEqual(len(digest), DIGEST_MAX_CHARS)
        # Последний слайд обязан остаться, самый первый — уйти.
        self.assertIn("0199", digest)
        self.assertNotIn("0000", digest)


class SlideChunkSelectionTests(unittest.TestCase):
    """Мера 3: финальная пятёрка предпочитает ещё не цитированные фрагменты."""

    @staticmethod
    def pool(ids):
        return [{"chunk_id": str(chunk_id), "metadata": {"doc_id": 1}} for chunk_id in ids]

    def ids(self, selected):
        return [item["chunk_id"] for item in selected]

    def test_small_pool_is_returned_as_is(self):
        pool = self.pool([1, 2, 3])
        self.assertEqual(self.ids(select_slide_chunks(pool, used_chunk_ids={"1"})), ["1", "2", "3"])

    def test_used_chunks_are_pushed_out_by_fresh_ones(self):
        pool = self.pool(range(1, 11))
        selected = select_slide_chunks(pool, used_chunk_ids={"2", "3", "4", "5"})
        self.assertEqual(len(selected), SLIDE_RETRIEVAL_TOP_K)
        self.assertEqual(self.ids(selected), ["1", "6", "7", "8", "9"])

    def test_top_candidate_survives_even_if_already_cited(self):
        """Мягкость исключения: центральный для двух секций фрагмент остаётся."""
        pool = self.pool(range(1, 11))
        selected = select_slide_chunks(pool, used_chunk_ids={"1"})
        self.assertIn("1", self.ids(selected))
        self.assertEqual(self.ids(selected), ["1", "2", "3", "4", "5"])

    def test_falls_back_to_used_chunks_when_fresh_ones_run_out(self):
        pool = self.pool(range(1, 11))
        used = {str(index) for index in range(1, 9)}
        selected = select_slide_chunks(pool, used_chunk_ids=used)
        self.assertEqual(len(selected), SLIDE_RETRIEVAL_TOP_K)
        # Свежие (9, 10) обязаны войти, добор — по рангу из использованных.
        self.assertIn("9", self.ids(selected))
        self.assertIn("10", self.ids(selected))
        self.assertEqual(self.ids(selected), ["1", "2", "3", "9", "10"])

    def test_selection_keeps_ranking_order(self):
        pool = self.pool(range(1, 21))
        selected = select_slide_chunks(pool, used_chunk_ids={"2", "3"})
        self.assertEqual(self.ids(selected), sorted(self.ids(selected), key=int))


if __name__ == "__main__":
    unittest.main()
