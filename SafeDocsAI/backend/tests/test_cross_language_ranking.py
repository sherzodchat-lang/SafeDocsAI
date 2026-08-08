"""Качество финального ранжирования: кросс-языковые пары и дубликаты.

Эти тесты охраняют «воду», а не трубы: живой замер показал, что русский
вопрос про отпуск по Трудовому кодексу получал в топ-5 пять русских
энциклопедических чанков (лексический шум), хотя векторный поиск находил
правильные таджикские чанки с запасом по порогу. Каждый тест здесь падал
на прежней версии _score_retrieval_candidate / rerank_retrieval_candidates.
"""

import unittest

from app.modules.chat.service import rerank_retrieval_candidates


def _vector_candidate(
    text: str,
    distance: float,
    doc_id: int,
    chunk_index: int,
    doc_name: str = "labor_code_2016.pdf",
    rank: int = 1,
) -> dict:
    return {
        "idx": chunk_index,
        "text": text,
        "metadata": {
            "doc_id": doc_id,
            "doc_name": doc_name,
            "chunk_index": chunk_index,
            "page": 1,
        },
        "chunk_id": str(doc_id * 1000 + chunk_index),
        "distance": distance,
        "rank": rank,
        "vector_rank": rank,
        "retrieval_method": "vector",
    }


def _lexical_candidate(
    text: str,
    doc_id: int,
    chunk_index: int,
    doc_name: str = "ru_economy.txt",
    rank: int = 1,
) -> dict:
    return {
        "idx": chunk_index,
        "text": text,
        "metadata": {
            "doc_id": doc_id,
            "doc_name": doc_name,
            "chunk_index": chunk_index,
            "page": 1,
        },
        "chunk_id": str(doc_id * 1000 + chunk_index),
        "distance": None,
        "rank": rank,
        "lexical_rank": rank,
        "retrieval_method": "lexical",
    }


# Воспроизведение живого случая: русский вопрос, ответ — в таджикском кодексе.
RU_QUERY = "какова продолжительность ежегодного оплачиваемого отпуска по трудовому кодексу"

TJ_ANSWER_TEXT = (
    "Рухсатии меҳнатии асосии ҳарсолаи камтарин аз 24 рӯзи тақвимӣ "
    "иборат мебошад ва тибқи шартномаи меҳнатӣ дода мешавад."
)

RU_NOISE_TEXTS = [
    "Минимальная оплата труда составляет 400 сомони в месяц, а ежегодного "
    "роста экономики Таджикистана ожидают на уровне семи процентов.",
    "По трудовому законодательству ежегодного прироста населения Таджикистана "
    "ожидает министерство, продолжительность жизни растёт.",
    "Кодексу поведения следуют участники ежегодного саммита, продолжительность "
    "встречи составила два дня.",
]


class CrossLanguageRankingTests(unittest.TestCase):
    def test_vector_hit_in_other_language_beats_lexical_noise(self):
        """Таджикский чанк, найденный вектором, обязан обойти русский шум.

        До правки лексическое пересечение весом 0.3 у разноязычной пары было
        нулевым по построению, и русские чанки с distance=None выигрывали."""
        candidates = [
            _vector_candidate(TJ_ANSWER_TEXT, distance=0.73, doc_id=87, chunk_index=1),
            *[
                _lexical_candidate(text, doc_id=48, chunk_index=i + 1, rank=i + 1)
                for i, text in enumerate(RU_NOISE_TEXTS)
            ],
        ]
        result = rerank_retrieval_candidates(
            candidates, query_text=RU_QUERY, final_top_k=3
        )
        self.assertEqual(result[0]["text"], TJ_ANSWER_TEXT)

    def test_same_language_ranking_still_prefers_overlap(self):
        """Регресс-щит: одноязычный случай не должен пострадать от правки."""
        query = "рухсатии меҳнатии ҳарсола чанд рӯз аст"
        relevant = _vector_candidate(
            TJ_ANSWER_TEXT, distance=0.4, doc_id=87, chunk_index=1, rank=1
        )
        weaker = _vector_candidate(
            "Шартномаи меҳнатӣ байни корманд ва корфармо баста мешавад.",
            distance=0.9,
            doc_id=87,
            chunk_index=2,
            rank=2,
        )
        result = rerank_retrieval_candidates(
            [weaker, relevant], query_text=query, final_top_k=2
        )
        self.assertEqual(result[0]["text"], TJ_ANSWER_TEXT)

    def test_missing_distance_scores_no_higher_than_found_distance(self):
        """Кандидат без расстояния — не выше реально найденного вектором.

        Тексты различаются служебными словами (ниже порога дедупликации), но
        несут одинаковое пересечение с запросом; до правки None получал
        нейтральные 0.5 против 0.4 у честно найденного с d=1.5 и вставал выше.
        """
        query = "ставка налога на прибыль"
        found = _vector_candidate(
            "Ставка налога на прибыль составляет восемнадцать процентов.",
            distance=1.5,
            doc_id=1,
            chunk_index=1,
            doc_name="tax.txt",
        )
        lexical_only = _lexical_candidate(
            "Ставка налога на прибыль установлена законом республики в размере "
            "восемнадцати процентов согласно действующей редакции кодекса.",
            doc_id=2,
            chunk_index=7,
            doc_name="tax_copy.txt",
        )
        result = rerank_retrieval_candidates(
            [lexical_only, found], query_text=query, final_top_k=2
        )
        by_id = {item["chunk_id"]: item["rerank_score"] for item in result}
        self.assertLessEqual(by_id[lexical_only["chunk_id"]], by_id[found["chunk_id"]])
        self.assertEqual(result[0]["chunk_id"], found["chunk_id"])


class DuplicateTextInFinalSelectionTests(unittest.TestCase):
    def test_same_text_from_two_documents_takes_one_slot(self):
        """Один закон, загруженный как PDF и как DOCX, — одна цитата в выдаче.

        candidate_identity различает их по doc_id, поэтому до правки обе копии
        занимали два слота из final_top_k."""
        pdf_copy = _vector_candidate(
            TJ_ANSWER_TEXT, distance=0.4, doc_id=87, chunk_index=1,
            doc_name="labor_code_2016.pdf",
        )
        docx_copy = _vector_candidate(
            TJ_ANSWER_TEXT, distance=0.41, doc_id=77, chunk_index=5,
            doc_name="labor_code_2016.docx", rank=2,
        )
        # У копий полное пересечение с запросом и лучшее расстояние — они
        # обязаны стоять выше distinct при любой версии скоринга, иначе тест
        # проверяет не дедупликацию, а порядок.
        distinct = _vector_candidate(
            "Рӯзҳои иди ғайрикорӣ ба ин ҳисоб дохил намегарданд.",
            distance=0.6,
            doc_id=87,
            chunk_index=9,
            rank=3,
        )
        result = rerank_retrieval_candidates(
            [pdf_copy, docx_copy, distinct],
            query_text="рухсатии меҳнатии ҳарсола чанд рӯз",
            final_top_k=2,
        )
        texts = [item["text"] for item in result]
        self.assertEqual(len(result), 2)
        self.assertEqual(len(set(texts)), 2, "дословная копия заняла второй слот")

    def test_boundary_shifted_copy_is_still_deduplicated(self):
        """Копии из pdf/docx расходятся границами нарезки, но не содержанием:
        совпадение множества слов выше порога — тоже дубликат."""
        base_words = TJ_ANSWER_TEXT
        shifted = TJ_ANSWER_TEXT + " Моддаи"  # чужой хвост от соседнего чанка
        first = _vector_candidate(base_words, 0.4, doc_id=87, chunk_index=1)
        second = _vector_candidate(
            shifted, 0.45, doc_id=77, chunk_index=2,
            doc_name="labor_code_2016.docx", rank=2,
        )
        result = rerank_retrieval_candidates(
            [first, second], query_text="давомнокии рухсатии меҳнатӣ", final_top_k=2
        )
        self.assertEqual(len(result), 1)


if __name__ == "__main__":
    unittest.main()
