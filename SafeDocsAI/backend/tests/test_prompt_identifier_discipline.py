"""Служебные идентификаторы запрещены в тексте, который увидит человек.

Дефект наблюдался дважды и в обоих случаях выглядел одинаково: модель писала
имя файла или маркер «(source_id: 35, chunk_id: 45)» прямо в текст. В чате на
таджикском имя ещё и портилось — «(payom20лади .txt)» вместо payom2024.txt, —
потому что модель имя не копирует, а порождает. Точное имя всё это время
лежало в структурном поле sources, куда попадает из базы.

Проверяемо здесь ровно одно: правило действительно попало в промпт, который
уходит модели, и попало во ВСЕ промпты сразу, а не в один. Заставить модель
правило соблюсти тест не может — модель здесь не участвует, — поэтому эффект
проверяется живым прогоном, а не этим файлом.

Вторая половина файла — про то, что запрет ничего не сломал: структурные
цитаты презентации по-прежнему валидируются, а ответ с именем файла внутри
текста по-прежнему проходит разбор (постобработки, вычищающей имена, нет и не
заводится: она резала бы и законные упоминания документов).
"""

import os
import sys
import unittest
from unittest.mock import AsyncMock, patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.modules.presentations.constants import LANGUAGE_TJ  # noqa: E402
from app.modules.presentations.llm_schemas import (  # noqa: E402
    LAYOUT_BULLETS,
    SLIDE_ADAPTER,
    validate_slide,
)
from app.modules.presentations.prompts import (  # noqa: E402
    build_plan_messages,
    build_slide_messages,
)
from app.modules.rag.generation_service import GenerationService  # noqa: E402
from app.modules.rag.text_utils import sanitize_answer_text  # noqa: E402

# Фактический ответ стенда: каждый абзац кончался вставкой имени файла, и в
# одном месте имя оказалось испорчено.
BROKEN_TAJIK_ANSWER = (
    "Дар соли 2025 барои вилояти Суғд ва Бадахшон низ нақшаҳои роҳсозӣ "
    "мавҷуданд (payom20лади .txt)"
)


def chat_system_prompt() -> str:
    """Системный промпт чата и /ask — у них общий сборщик."""
    messages, _ = GenerationService._build_answer_messages(
        query="Дар соли 2025 чӣ нақшаҳо ҳастанд?",
        context=["[payom2024 | Роҳсозӣ | стр. 4] Нақшаҳои роҳсозӣ."],
        language=LANGUAGE_TJ,
        context_metadata=[{"doc_name": "payom2024.txt", "page": 4}],
    )
    return messages[0]["content"]


def plan_system_prompt() -> str:
    return build_plan_messages(
        notebook_name="Послания",
        description="Про дороги",
        language=LANGUAGE_TJ,
        slide_count=10,
        context_block="",
    )[0]["content"]


def slide_system_prompt() -> str:
    return build_slide_messages(
        heading="Роҳсозӣ",
        # Раскладку слайд-вызову назначает план, и без неё промпт не собрать.
        layout=LAYOUT_BULLETS,
        description="Про дороги",
        language=LANGUAGE_TJ,
        context_block="",
        allowed_citations={"45": 7},
    )[0]["content"]


# Каждый промпт, который порождает видимый пользователю текст. Список ведётся
# руками намеренно: он и есть то, что проверяется, — правило, добавленное в
# один промпт из трёх, чинит один экран из трёх.
VISIBLE_TEXT_PROMPTS = {
    "chat/ask": chat_system_prompt,
    "presentation plan": plan_system_prompt,
    "presentation slide": slide_system_prompt,
}


class ServiceIdentifiersAreForbiddenEverywhereTests(unittest.TestCase):
    def test_every_system_prompt_states_the_ban(self):
        for name, build in VISIBLE_TEXT_PROMPTS.items():
            with self.subTest(prompt=name):
                prompt = build()
                self.assertIn("NEVER write file names, source_id, chunk_id", prompt)

    def test_every_system_prompt_explains_that_tags_are_markup(self):
        """Одного запрета мало: маркеры модель видит частью текста.

        Убрать их из подачи нельзя — по source_id/chunk_id собираются цитаты
        презентации, а имя файла отделяет послание одного года от другого, —
        поэтому промпт обязан объяснить, что это разметка, а не текст.
        """
        for name, build in VISIBLE_TEXT_PROMPTS.items():
            with self.subTest(prompt=name):
                self.assertIn("service markup", build())


class ChatPromptNoLongerDemandsCitationsInTextTests(unittest.TestCase):
    """Причина, а не симптом: раньше промпт САМ требовал писать имя файла."""

    def test_the_old_demand_is_gone(self):
        prompt = chat_system_prompt()
        self.assertNotIn("cite the source file name", prompt)

    def test_the_old_example_is_gone(self):
        # Образец «(payom2005.txt)» показывал модели готовую форму вставки —
        # ровно ту, что она потом порождала с ошибкой.
        self.assertNotIn("payom", chat_system_prompt())

    def test_the_prompt_points_at_the_structured_field(self):
        self.assertIn("structured field", chat_system_prompt())


class PresentationCitationsSurviveTests(unittest.TestCase):
    """Запрет касается текста, а не поля citations."""

    def test_slide_prompt_still_lists_the_allowed_chunk_ids(self):
        prompt = build_slide_messages(
            heading="Роҳсозӣ",
            layout=LAYOUT_BULLETS,
            description="",
            language=LANGUAGE_TJ,
            context_block="",
            allowed_citations={"45": 7, "46": 7},
        )[0]["content"]
        self.assertIn("citations", prompt)
        self.assertIn("45, 46", prompt)

    def test_structured_citation_still_validates(self):
        slide = validate_slide(
            '{"layout": "bullets", "heading": "h", "bullets": ["a", "b"], '
            '"citations": [{"source_id": 7, "chunk_id": 45}]}',
            allowed_citations={"45": 7},
        )
        self.assertEqual([c.chunk_id for c in slide.citations], ["45"])

    def test_foreign_chunk_is_still_rejected(self):
        from app.modules.presentations.llm_schemas import LlmResponseError

        with self.assertRaises(LlmResponseError):
            validate_slide(
                '{"layout": "bullets", "heading": "h", "bullets": ["a", "b"], '
                '"citations": [{"source_id": 7, "chunk_id": 999}]}',
                allowed_citations={"45": 7},
            )


class DisobedientAnswerStillParsesTests(unittest.TestCase):
    """Модель может правило нарушить — разбор от этого падать не должен.

    Фиксация текущего поведения: имя файла из текста не вырезается. Чистка
    регулярным выражением сняла бы и законное упоминание документа в ответе на
    вопрос «в каком документе это сказано», а испорченное имя всё равно не
    поймала бы — оно на то и испорченное, что ни с чем не совпадает.
    """

    def test_chat_answer_with_a_file_name_passes_through_unchanged(self):
        self.assertEqual(sanitize_answer_text(BROKEN_TAJIK_ANSWER), BROKEN_TAJIK_ANSWER)

    def test_slide_with_an_identifier_in_a_bullet_still_validates(self):
        slide = SLIDE_ADAPTER.validate_python(
            {
                "layout": LAYOUT_BULLETS,
                "heading": "Роҳсозӣ",
                "bullets": ["Нақшаҳо (source_id: 35, chunk_id: 45)", "Дуюм далел"],
                "citations": [{"source_id": 7, "chunk_id": 45}],
            }
        )
        self.assertEqual(len(slide.bullets), 2)


class GeneratedAnswerKeepsTheModelTextTests(unittest.IsolatedAsyncioTestCase):
    async def test_generate_answer_returns_what_the_model_said(self):
        service = GenerationService()
        with patch.object(
            service.model_manager,
            "chat",
            new=AsyncMock(return_value=BROKEN_TAJIK_ANSWER),
        ):
            answer = await service.generate_answer(
                query="Дар соли 2025 чӣ нақшаҳо ҳастанд?",
                context=["Нақшаҳои роҳсозӣ."],
                language=LANGUAGE_TJ,
                context_metadata=[{"doc_name": "payom2024.txt"}],
            )
        self.assertEqual(answer, BROKEN_TAJIK_ANSWER)

    async def test_the_ban_reaches_the_model(self):
        """То единственное, что тест здесь может: правило дошло до вызова."""
        service = GenerationService()
        chat = AsyncMock(return_value="ok")
        with patch.object(service.model_manager, "chat", new=chat):
            await service.generate_answer(
                query="Савол",
                context=["Матн"],
                language=LANGUAGE_TJ,
                context_metadata=[{"doc_name": "payom2024.txt"}],
            )
        system_message = chat.await_args.kwargs["messages"][0]
        self.assertEqual(system_message["role"], "system")
        self.assertIn(
            "NEVER write file names, source_id, chunk_id", system_message["content"]
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
