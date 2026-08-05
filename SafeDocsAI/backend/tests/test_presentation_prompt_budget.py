"""Бюджет промпта презентации сходится — и сходится с ФАКТИЧЕСКОЙ настройкой.

DESCRIPTION_MAX и DIGEST_MAX_CHARS не выбраны, а посчитаны: в constants.py над
ними лежит арифметика на окно в 12 000 токенов. Пока эта арифметика живёт
только в комментарии, она сторожится единственным способом — вниманием того,
кто правит соседнюю константу. Ровно так уже разъезжались границы заказа
(tests/test_presentation_limits_single_source.py): человек заметил, прогон нет.

Здесь расчёт сделан исполняемым. Промпты собираются НАСТОЯЩИМИ сборщиками
(build_plan_messages, build_slide_messages) на предельных входах, к ним
добавляется зарезервированное место под ответ модели — дважды, потому что
повторная попытка получает и исходный промпт, и отвергнутый ответ, и место под
новый, — и сумма сверяется с окном.

Про «фактический chat_model_num_ctx» — честная оговорка.
=========================================================
Задание звучало как «посчитать бюджет из фактического chat_model_num_ctx».
Буквально так проверять нечего, и вот почему: пайплайн презентаций
chat_model_num_ctx НЕ ЧИТАЕТ. Он передаёт в Ollama своё PRESENTATION_NUM_CTX
(service.py, call_with_one_retry -> model_manager.chat(num_ctx=...)), то есть
окно, в котором промпт реально живёт, равно PRESENTATION_NUM_CTX, а не
настройке. Проверка «сумма частей ≤ 20 000 токенов» была бы зелёной всегда, в
том числе если бы PRESENTATION_NUM_CTX уронили до 4 000, — то есть сторожила
бы не то.

Поэтому сторожатся две связи вместо одной:

  1) сумма частей промпта ≤ окно, в котором промпт РЕАЛЬНО живёт
     (PRESENTATION_NUM_CTX). Это и есть расчёт из комментария, сделанный
     исполняемым;
  2) PRESENTATION_NUM_CTX ≤ фактический chat_model_num_ctx, прочитанный тем же
     RuntimeSettingsService.get_settings(), которым его читает приложение.
     Направление важно: сегодня 12 000 ≤ 20 000, и это «безопасная сторона» —
     пайплайн просит у модели меньше, чем считает безопасным само
     развёртывание. Опустят настройку под 12 000 (модель побольше, видеопамяти
     поменьше) — и пайплайн начнёт просить окно, которого развёртывание уже не
     обещает, а расчёт бюджета будет посчитан по окну, которого нет. Сегодня
     эту связь не сторожит ничто; после этого файла — сторожит.
"""

from __future__ import annotations

import unittest

from app.api.deps import TITLE_MAX_LENGTH
from app.modules.presentations.constants import (
    CHARS_PER_TOKEN,
    DESCRIPTION_MAX,
    DIGEST_MAX_CHARS,
    PLAN_RETRIEVAL_TOP_K,
    PRESENTATION_NUM_CTX,
    SLIDE_COUNT_MAX,
    SLIDE_RETRIEVAL_TOP_K,
)
from app.modules.presentations.llm_schemas import (
    PLAN_TITLE_MAX_CHARS,
    SECTION_HEADING_MAX_CHARS,
    SECTION_SEARCH_QUERY_MAX_CHARS,
    SLIDE_BULLETS_MAX,
    SLIDE_BULLET_MAX_CHARS,
    SLIDE_COMPARE_BULLETS_MAX,
    SLIDE_COMPARE_BULLET_MAX_CHARS,
    SLIDE_COMPARE_HEADING_MAX_CHARS,
    SLIDE_HEADING_MAX_CHARS,
    SLIDE_LAYOUTS,
    SLIDE_METRIC_CAPTION_MAX_CHARS,
    SLIDE_METRIC_NOTE_MAX_CHARS,
    SLIDE_METRIC_VALUE_MAX_CHARS,
    SLIDE_QUOTE_ATTRIBUTION_MAX_CHARS,
    SLIDE_QUOTE_TEXT_MAX_CHARS,
    SLIDE_STEPS_MAX,
    SLIDE_STEP_TEXT_MAX_CHARS,
    SLIDE_STEP_TITLE_MAX_CHARS,
    content_section_count,
)
from app.modules.presentations.prompts import (
    build_context_block,
    build_plan_messages,
    build_retry_messages,
    build_slide_messages,
    build_written_digest,
)
from app.services.hybrid_chunker import HybridChunker
from app.shared.settings.runtime_settings import RuntimeSettingsService

# Имя блокнота уезжает в план-промпт целиком, поэтому в худший случай входит
# его предел — тот самый, которым его режет схема запроса.
NOTEBOOK_NAME_MAX = TITLE_MAX_LENGTH

# Синтаксис JSON вокруг полей ответа: скобки, кавычки, запятые, имена ключей.
# Округлено вверх — место под ответ закладывается щедро намеренно, ошибка
# здесь в короткую сторону означает обрезанный ответ модели.
#
# У секции плана к ним добавилась раскладка: имя из закрытого списка плюс ключ
# "layout" с кавычками и двоеточием. Считается по САМОМУ ДЛИННОМУ имени —
# короткое занизило бы худший случай ровно тем способом, против которого этот
# файл и написан.
PLAN_SECTION_LAYOUT_CHARS = max(len(name) for name in SLIDE_LAYOUTS) + 14
PLAN_SECTION_SYNTAX_CHARS = 40
PLAN_ENVELOPE_SYNTAX_CHARS = 60
SLIDE_SYNTAX_CHARS = 200
# Вложенный объект ответа (колонка сравнения, шаг процесса) стоит своих скобок,
# кавычек и имён ключей сверх общей обвязки слайда.
SLIDE_NESTED_SYNTAX_CHARS = 40

# Претензия валидатора, которую получает повторная попытка (build_retry_messages
# добавляет к ней ещё и обвязку — она входит в измеряемый текст).
VALIDATOR_COMPLAINT_CHARS = 220


def chunk_text_of_max_size() -> str:
    """Текст чанка предельного размера — по настройке самого чанкера.

    Не «2 880 знаков», а max_tokens × CHARS_PER_TOKEN: подняли верхнюю границу
    чанка — и бюджет промпта обязан покраснеть здесь, а не молча обрезаться
    где-то в Ollama. Умолчание чанкера спрашивается у самого чанкера — им
    индексируются все документы проекта.
    """
    return "я" * int(HybridChunker().max_tokens * CHARS_PER_TOKEN)


def worst_case_context_block(chunk_count: int) -> tuple[str, dict[str, int]]:
    text = chunk_text_of_max_size()
    chunks = [
        {
            "chunk_id": f"{index:08d}",
            "metadata": {
                "doc_id": index + 1,
                # Имя файла уезжает в промпт целиком: длинное имя — часть
                # худшего случая, а не экзотика.
                "doc_name": "документ" * 8 + ".pdf",
                "page": 100 + index,
            },
        }
        for index in range(chunk_count)
    ]
    texts = {chunk["chunk_id"]: text for chunk in chunks}
    return build_context_block(chunks, texts)


def messages_chars(messages: list[dict[str, str]]) -> int:
    return sum(len(message["content"]) for message in messages)


def tokens(chars: float) -> float:
    return chars / CHARS_PER_TOKEN


class PromptBudgetFitsTheWindowTests(unittest.TestCase):
    """Сумма заложенных частей не превышает окна, в котором промпт живёт."""

    def window_tokens(self) -> int:
        return PRESENTATION_NUM_CTX

    def plan_answer_chars(self) -> int:
        sections = content_section_count(SLIDE_COUNT_MAX)
        return (
            PLAN_TITLE_MAX_CHARS
            + PLAN_ENVELOPE_SYNTAX_CHARS
            + sections
            * (
                SECTION_HEADING_MAX_CHARS
                + SECTION_SEARCH_QUERY_MAX_CHARS
                + PLAN_SECTION_LAYOUT_CHARS
                + PLAN_SECTION_SYNTAX_CHARS
            )
        )

    def slide_answer_chars_by_layout(self) -> dict[str, int]:
        """Предельный ответ КАЖДОЙ раскладки — по её собственным полям.

        Считать бюджет по одной раскладке нельзя с тех пор, как их пять:
        самый длинный ответ даёт не bullets (1 080 знаков полей), а steps
        (1 180 плюс обвязка пяти вложенных объектов). Проверка по bullets
        занизила бы худший случай молча — то есть ровно тем способом, против
        которого этот файл и написан.
        """
        common = SLIDE_HEADING_MAX_CHARS + SLIDE_SYNTAX_CHARS
        return {
            "bullets": common + SLIDE_BULLETS_MAX * SLIDE_BULLET_MAX_CHARS,
            "compare": common
            + 2
            * (
                SLIDE_COMPARE_HEADING_MAX_CHARS
                + SLIDE_COMPARE_BULLETS_MAX * SLIDE_COMPARE_BULLET_MAX_CHARS
                + SLIDE_NESTED_SYNTAX_CHARS
            ),
            "metric": common
            + SLIDE_METRIC_VALUE_MAX_CHARS
            + SLIDE_METRIC_CAPTION_MAX_CHARS
            + SLIDE_METRIC_NOTE_MAX_CHARS,
            "steps": common
            + SLIDE_STEPS_MAX
            * (
                SLIDE_STEP_TITLE_MAX_CHARS
                + SLIDE_STEP_TEXT_MAX_CHARS
                + SLIDE_NESTED_SYNTAX_CHARS
            ),
            "quote": common
            + SLIDE_QUOTE_TEXT_MAX_CHARS
            + SLIDE_QUOTE_ATTRIBUTION_MAX_CHARS,
        }

    def slide_answer_chars(self, layout: str) -> int:
        """Предельный ответ на слайд-вызов с НАЗНАЧЕННОЙ раскладкой.

        Ответов у такого вызова ровно два вида: назначенная раскладка и bullets
        — единственная законная замена, когда материал назначенную не держит
        (правило 5 слайд-промпта). Место закладывается под больший из двух:
        слайду, назначенному metric, модель имеет право ответить пятью
        полноширинными буллетами, и они длиннее величины с подписью.
        """
        by_layout = self.slide_answer_chars_by_layout()
        # Ни одна раскладка не должна выпасть из расчёта: добавили шестую в
        # схему, забыли здесь — и бюджет снова считается не по худшему случаю.
        assert set(by_layout) == set(SLIDE_LAYOUTS), (
            "расчёт бюджета не знает про раскладки "
            f"{sorted(set(SLIDE_LAYOUTS) - set(by_layout))}"
        )
        return max(by_layout[layout], by_layout["bullets"])

    def test_plan_call_fits(self) -> None:
        """План-вызов на предельном заказе: промпт + ответ дважды + претензия."""
        context_block, _ = worst_case_context_block(PLAN_RETRIEVAL_TOP_K)
        messages = build_plan_messages(
            notebook_name="Б" * NOTEBOOK_NAME_MAX,
            description="о" * DESCRIPTION_MAX,
            language="ru",
            slide_count=SLIDE_COUNT_MAX,
            context_block=context_block,
        )
        # Повтор получает исходный промпт целиком, отвергнутый ответ и
        # претензию — и сверх того ему нужно место под НОВЫЙ ответ.
        retry = build_retry_messages(
            messages,
            "x" * self.plan_answer_chars(),
            "e" * VALIDATOR_COMPLAINT_CHARS,
        )
        needed = tokens(messages_chars(retry) + self.plan_answer_chars())

        self.assertLessEqual(
            needed,
            self.window_tokens(),
            f"план-вызов не влезает в окно: {needed:.0f} токенов из "
            f"{self.window_tokens()}. Пересчитайте DESCRIPTION_MAX или "
            "PLAN_RETRIEVAL_TOP_K — расчёт в constants.py устарел",
        )

    def test_slide_call_fits(self) -> None:
        """Слайд-вызов: чанки, описание, ПОЛНЫЙ дайджест и ответ дважды.

        Проверяется КАЖДАЯ назначаемая раскладка, а не одна: с переносом выбора
        в план системный промпт слайда зависит от назначения — в нём описана
        назначенная форма и bullets как замена. Значит, и длина его разная, и
        худший случай — максимум по всем пяти назначениям, а не по одному.
        """
        context_block, allowed = worst_case_context_block(SLIDE_RETRIEVAL_TOP_K)
        # Дайджест предельного размера: буллеты предельной длины, пока
        # build_written_digest не начнёт их выбрасывать.
        bullets = [
            ["ф" * SLIDE_BULLET_MAX_CHARS] * SLIDE_BULLETS_MAX for _ in range(40)
        ]
        digest = build_written_digest(bullets)
        self.assertGreater(
            len(digest),
            DIGEST_MAX_CHARS * 0.9,
            "дайджест не дорос до своего потолка — проверка выродилась",
        )

        for layout in SLIDE_LAYOUTS:
            with self.subTest(layout=layout):
                messages = build_slide_messages(
                    heading="З" * SECTION_HEADING_MAX_CHARS,
                    layout=layout,
                    description="о" * DESCRIPTION_MAX,
                    language="ru",
                    context_block=context_block,
                    allowed_citations=allowed,
                    digest=digest,
                )
                answer_chars = self.slide_answer_chars(layout)
                retry = build_retry_messages(
                    messages,
                    "x" * answer_chars,
                    "e" * VALIDATOR_COMPLAINT_CHARS,
                )
                needed = tokens(messages_chars(retry) + answer_chars)

                self.assertLessEqual(
                    needed,
                    self.window_tokens(),
                    f"слайд-вызов ({layout}) не влезает в окно: {needed:.0f} "
                    f"токенов из {self.window_tokens()}. Пересчитайте "
                    "DIGEST_MAX_CHARS, DESCRIPTION_MAX или "
                    "SLIDE_RETRIEVAL_TOP_K — расчёт в constants.py устарел",
                )


class WindowMatchesTheActualSettingTests(unittest.TestCase):
    """Связь посчитанного бюджета с настройкой, которую правят из админ-панели."""

    def actual_chat_num_ctx(self) -> int:
        # Тот же путь чтения, которым настройку читает приложение: файл
        # runtime_settings.json этого развёртывания, дополненный умолчаниями.
        return int(RuntimeSettingsService.get_settings()["chat_model_num_ctx"])

    def test_pipeline_window_is_not_wider_than_the_configured_one(self) -> None:
        """Пайплайн не имеет права просить окно шире настроенного.

        Сегодня это безопасная сторона: 12 000 ≤ 20 000, бюджет консервативнее
        реальности. Но связь между настройкой и посчитанными числами не
        сторожило ничто, а настройка правится мышкой. Опустят её ниже
        PRESENTATION_NUM_CTX — и весь расчёт DESCRIPTION_MAX/DIGEST_MAX_CHARS
        окажется сделан по окну, которого в развёртывании больше нет.
        """
        configured = self.actual_chat_num_ctx()
        self.assertLessEqual(
            PRESENTATION_NUM_CTX,
            configured,
            f"chat_model_num_ctx = {configured} меньше PRESENTATION_NUM_CTX = "
            f"{PRESENTATION_NUM_CTX}: бюджет промпта презентаций посчитан по "
            "окну, которого развёртывание больше не обещает. Пересчитайте "
            "DESCRIPTION_MAX и DIGEST_MAX_CHARS в "
            "app/modules/presentations/constants.py или верните настройку",
        )

    def test_the_setting_is_readable_at_all(self) -> None:
        # Если ключ переименуют, проверка выше молча выродится в сравнение с
        # умолчанием — а это ровно тот класс дефекта, против которого она.
        self.assertIn("chat_model_num_ctx", RuntimeSettingsService.get_settings())
        self.assertIn("chat_model_num_ctx", RuntimeSettingsService.DEFAULTS)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
