from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DomainProfile:
    name: str
    assistant_name: str = "SafeDocsAI"

    def no_data_answer(self, language: str) -> str:
        if language == "tj":
            return "Маълумот дар манбаъҳои интихобшуда мавҷуд нест / Ответ не найден в выбранных источниках"
        return "Ответ не найден в выбранных источниках / Маълумот дар манбаъҳои интихобшуда мавҷуд нест"

    def greeting(self, language: str) -> str:
        if language == "tj":
            return f"Салом! Ман {self.assistant_name}, ёрдамчии шумо барои кор бо дониш ва манбаъҳо. Савол диҳед."
        return f"Здравствуйте! Я {self.assistant_name}, ваш помощник по работе со знаниями и источниками. Задавайте вопросы."

    def prompt_injection_message(self, language: str) -> str:
        if language == "tj":
            return "Дархост рад шуд: савол дорои дастурҳои хатарнок аст. Лутфан саволи худро дар бораи манбаъҳои интихобшуда нависед."
        return "Запрос отклонен: обнаружена попытка обойти системные правила. Сформулируйте вопрос по выбранным источникам."

    def search_queries(self, query_text: str, language: str) -> list[str]:
        return [query_text]

    def rerank_results(
        self, query_text: str, results: dict[str, Any]
    ) -> dict[str, Any]:
        return results

    def answer_rules(self, language: str) -> str:
        return (
            "Answer the user question using only the provided context. "
            "If the context is insufficient, return the exact no-data message."
        )
