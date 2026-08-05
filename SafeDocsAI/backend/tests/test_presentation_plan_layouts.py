"""Раскладку выбирает ПЛАН, а слайд её исполняет.

Пять раскладок появились волной раньше, и на стенде из восьми содержательных
слайдов нестандартную форму получил один. Причина структурная: выбирал форму
слайд-вызов, а он видит свою секцию и не видит колоды. Из такой позиции список
выигрывает всегда — он подходит любому материалу, — и никакими словами промпта
это не лечится: секция, уже сформулированная как перечисление, сравнением не
станет.

Поэтому выбор переехал в план: он единственный видит дайджест корпуса, описание
заказа и всю длину колоды сразу. Проверяется здесь вся эта дорога:

1. Секция плана НЕСЁТ раскладку, любую из пяти, и поле обязательное. Молчаливое
   умолчание в bullets вернуло бы нас ровно туда, откуда волна началась, и
   сделало бы это незаметно — колода собралась бы, просто снова однообразной.
2. Отказ ГОВОРИТ, чего не хватило. Текст отказа — единственная подсказка
   повторной попытки (build_retry_messages), и «response does not match the
   required schema» отправляет модель гадать заново по всему плану.
3. Назначенное ДОЕЗЖАЕТ до слайд-промпта и требуется в нём как исполнение, а не
   как выбор. Обрыв здесь бесшумный: колода соберётся, просто раскладку снова
   будет выбирать тот, кто не видит колоды.
4. Пределы плана (число секций из SLIDE_COUNT_MIN/MAX) от нового поля не
   поехали.
5. Предел серии одинаковых раскладок проверяет СХЕМА, а не текст промпта.
   Просьба «не больше трёх подряд» стояла в план-промпте, и живой прогон отдал
   семь списков подряд: правило, которое некому проверить, — не механизм.
   Проверять его обязана машина, а отказ — говорить, что именно переразметить,
   и называть честный запасной выход (quote), иначе требование разнообразия
   превращается в требование выдумать сравнение.
"""

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.modules.presentations.constants import (  # noqa: E402
    LANGUAGE_RU,
    LAYOUT_BULLETS,
    LAYOUT_COMPARE,
    LAYOUT_METRIC,
    LAYOUT_QUOTE,
    LAYOUT_STEPS,
    PLAN_LAYOUT_RUN_MAX,
    SLIDE_COUNT_MAX,
    SLIDE_COUNT_MIN,
    SLIDE_LAYOUTS,
)
from app.modules.presentations.llm_schemas import (  # noqa: E402
    PLAN_LAYOUT_RUN_MAX as SCHEMA_LAYOUT_RUN_MAX,  # noqa: E402
)
from app.modules.presentations.llm_schemas import (  # noqa: E402
    RENDERER_ADDED_SLIDES,
    LlmResponseError,
    content_section_count,
    validate_plan,
)
from app.modules.presentations.prompts import (  # noqa: E402
    build_plan_messages,
    build_slide_messages,
)

SLIDE_COUNT = SLIDE_COUNT_MIN
SECTIONS = content_section_count(SLIDE_COUNT)


def section(index: int, layout: str = LAYOUT_BULLETS) -> dict:
    return {
        "heading": f"Секция {index}",
        "search_query": f"запрос {index}",
        "layout": layout,
    }


def plan_json(layouts: list[str], *, title: str = "Налоговые льготы") -> str:
    sections = [section(index, layout) for index, layout in enumerate(layouts, start=1)]
    return json.dumps({"title": title, "sections": sections}, ensure_ascii=False)


def validate(layouts: list[str], *, slide_count: int = SLIDE_COUNT):
    return validate_plan(plan_json(layouts), slide_count=slide_count)


def rejection(test: unittest.TestCase, payload: dict, *, slide_count: int) -> str:
    """Текст отказа — тот самый, что уедет в повторный промпт."""
    with test.assertRaises(LlmResponseError) as ctx:
        validate_plan(json.dumps(payload, ensure_ascii=False), slide_count=slide_count)
    return ctx.exception.error_text


class EveryLayoutPassesInAPlanSection(unittest.TestCase):
    """Все пять раскладок законны в плане — иначе половина их недостижима.

    Раскладка, которую не принимает схема плана, для колоды не существует
    вовсе: слайд-вызов её больше не выбирает, и назначить её некому.
    """

    def test_each_layout_is_accepted(self):
        for layout in SLIDE_LAYOUTS:
            with self.subTest(layout=layout):
                plan = validate([layout] * SECTIONS)
                self.assertEqual(
                    [item.layout for item in plan.sections], [layout] * SECTIONS
                )

    def test_a_mixed_plan_keeps_the_order_of_its_sections(self):
        # Раскладка привязана к СВОЕЙ секции: перепутанный порядок отдал бы
        # секции цифры форму сравнения, и заметить это можно было бы только
        # глазами в готовом PDF.
        layouts = [LAYOUT_COMPARE, LAYOUT_METRIC, LAYOUT_QUOTE][:SECTIONS]
        plan = validate(layouts)
        self.assertEqual([item.layout for item in plan.sections], layouts)
        self.assertEqual(
            [item.heading for item in plan.sections],
            [f"Секция {index}" for index in range(1, SECTIONS + 1)],
        )


class LayoutIsMandatoryInThePlan(unittest.TestCase):
    """Нет раскладки — нет плана. Без умолчания и без догадок."""

    def missing_layout_payload(self) -> dict:
        sections = [section(index) for index in range(1, SECTIONS + 1)]
        del sections[1]["layout"]
        return {"title": "Налоговые льготы", "sections": sections}

    def test_a_section_without_a_layout_is_rejected(self):
        with self.assertRaises(LlmResponseError):
            validate_plan(
                json.dumps(self.missing_layout_payload(), ensure_ascii=False),
                slide_count=SLIDE_COUNT,
            )

    def test_the_rejection_names_the_place_and_the_allowed_values(self):
        """Отказ обязан быть исполнимым, а не просто справедливым.

        «sections.1.layout: Field required» называет место, но не говорит, чем
        его заполнить, — а забывшей поле модели перечень нужен больше всех:
        сама она о нём и не вспомнила.
        """
        text = rejection(
            self, self.missing_layout_payload(), slide_count=SLIDE_COUNT
        )
        self.assertIn("sections.1.layout", text)
        for layout in SLIDE_LAYOUTS:
            with self.subTest(layout=layout):
                self.assertIn(repr(layout), text)

    def test_an_unknown_layout_is_rejected_with_the_closed_list(self):
        sections = [section(index) for index in range(1, SECTIONS + 1)]
        sections[0]["layout"] = "table"
        text = rejection(
            self,
            {"title": "Налоговые льготы", "sections": sections},
            slide_count=SLIDE_COUNT,
        )
        self.assertIn("sections.0.layout", text)
        # Список закрыт: шестую раскладку рисовать нечем, и узнать об этом
        # модель обязана из отказа, а не из пустого слайда.
        self.assertIn(repr(LAYOUT_METRIC), text)

    def test_a_blank_layout_is_not_a_layout(self):
        sections = [section(index) for index in range(1, SECTIONS + 1)]
        sections[0]["layout"] = ""
        with self.assertRaises(LlmResponseError):
            validate_plan(
                json.dumps({"title": "Т", "sections": sections}, ensure_ascii=False),
                slide_count=SLIDE_COUNT,
            )


class ThePlanLimitsStillHold(unittest.TestCase):
    """Новое поле не сдвинуло границы заказа.

    Число секций выводится из slide_count и проверяется контекстом валидации.
    Раскладка к этому счёту отношения не имеет — и проверка обязана остаться
    такой же строгой на обоих концах продуктового диапазона.
    """

    def test_both_ends_of_the_order_range_validate(self):
        for slide_count in (SLIDE_COUNT_MIN, SLIDE_COUNT_MAX):
            with self.subTest(slide_count=slide_count):
                sections = content_section_count(slide_count)
                # Раскладки берутся ЧЕРЕДОВАНИЕМ, а не сплошным списком: на
                # предельном заказе сплошной список — это нарушение предела
                # серии, и проверка числа секций падала бы на чужом правиле.
                plan = validate(
                    [SLIDE_LAYOUTS[index % len(SLIDE_LAYOUTS)] for index in range(sections)],
                    slide_count=slide_count,
                )
                self.assertEqual(len(plan.sections), sections)

    def test_a_section_too_few_is_still_rejected(self):
        text = rejection(
            self,
            {
                "title": "Т",
                "sections": [section(index) for index in range(1, SECTIONS)],
            },
            slide_count=SLIDE_COUNT,
        )
        self.assertIn(f"exactly {SECTIONS}", text)

    def test_a_section_too_many_is_still_rejected(self):
        # Лишняя секция размечена ДРУГОЙ раскладкой нарочно: сплошной список из
        # четырёх нарушил бы заодно и предел серии, и проверка счёта секций
        # прошла бы на чужом отказе, ничего про счёт не доказав.
        text = rejection(
            self,
            {
                "title": "Т",
                "sections": [
                    section(index, LAYOUT_BULLETS if index <= SECTIONS else LAYOUT_QUOTE)
                    for index in range(1, SECTIONS + 2)
                ],
            },
            slide_count=SLIDE_COUNT,
        )
        self.assertIn(f"exactly {SECTIONS}", text)


class TheRunLimitIsCheckedByTheSchema(unittest.TestCase):
    """Предел серии одинаковых раскладок проверяет МАШИНА, а не промпт.

    Правило «не больше PLAN_LAYOUT_RUN_MAX подряд» стояло только в тексте
    план-промпта, и живой прогон показал, чего это стоит: из восьми
    содержательных секций нестандартную раскладку получила одна — семь списков
    подряд при заявленном пределе три. Счётчик расхождений при этом показал, что
    слайды исполнили назначенное, то есть однообразие пришло из плана. Просьба,
    которую нечем проверить, — не механизм.

    Отвергнутый план уходит в уже существующий повтор (call_with_one_retry,
    LLM_CALL_ATTEMPTS = 2), и текст отказа — единственная подсказка второй
    попытки: он обязан назвать раскладку, длину серии и предел.
    """

    def deck(self, layouts: list[str]):
        """План на столько слайдов, сколько в нём секций, — заказ под разметку.

        Число секций тут вспомогательное: проверяется предел серии, и упереться
        в чужую проверку (счёт секций) на полпути было бы обидно.
        """
        return validate(layouts, slide_count=len(layouts) + 2)

    def deck_rejection(self, layouts: list[str]) -> str:
        with self.assertRaises(LlmResponseError) as ctx:
            self.deck(layouts)
        return ctx.exception.error_text

    def test_a_run_exactly_at_the_limit_passes(self):
        # Ровно предел — это ЧЕСТНАЯ разметка, а не нарушение: материал,
        # который и правда не держит ни сравнений, ни цифр, ни порядка, имеет
        # право на три списка подряд.
        layouts = [LAYOUT_BULLETS] * PLAN_LAYOUT_RUN_MAX + [LAYOUT_QUOTE]
        plan = self.deck(layouts)
        self.assertEqual([item.layout for item in plan.sections], layouts)

    def test_one_section_over_the_limit_is_rejected(self):
        with self.assertRaises(LlmResponseError):
            self.deck([LAYOUT_BULLETS] * (PLAN_LAYOUT_RUN_MAX + 1))

    def test_the_rejection_names_the_layout_the_length_and_the_limit(self):
        """«Validation failed» бесполезен: на нём повтор выродится в ту же разметку.

        Все три факта обязаны быть в тексте — какая раскладка, сколько её
        подряд и какой предел, — иначе модель не знает ни что чинить, ни до
        какого числа.
        """
        over = PLAN_LAYOUT_RUN_MAX + 1
        text = self.deck_rejection([LAYOUT_BULLETS] * over)
        self.assertIn(repr(LAYOUT_BULLETS), text)
        self.assertIn(f"{over} in a row", text)
        self.assertIn(f"more than {PLAN_LAYOUT_RUN_MAX} sections in a row", text)
        # И место: секции названы номерами, по которым модель найдёт их в своём
        # же ответе.
        self.assertIn(f"sections.0-{over - 1}", text)

    def test_the_rejection_offers_the_honest_way_out(self):
        """Отказ не имеет права толкать на враньё.

        Требование разнообразия без названного запасного выхода читается как
        «выдумай сравнение», а выдуманное сравнение хуже однообразной колоды.
        Честный выход ровно один: дословная формулировка есть в любом
        документе, в отличие от второй стороны или центрального числа.
        """
        text = self.deck_rejection([LAYOUT_BULLETS] * (PLAN_LAYOUT_RUN_MAX + 1))
        self.assertIn(repr(LAYOUT_QUOTE), text)
        self.assertIn("unlike a second side or a central number", text)

    def test_every_layout_is_limited_the_same_way(self):
        # Предел — про ОДНООБРАЗИЕ, а не про списки. Четыре сравнения подряд
        # ровно так же означают, что план не смотрел в материал.
        for layout in SLIDE_LAYOUTS:
            with self.subTest(layout=layout):
                text = self.deck_rejection([layout] * (PLAN_LAYOUT_RUN_MAX + 1))
                self.assertIn(repr(layout), text)

    def test_a_run_broken_by_another_layout_passes(self):
        """Серия считается ПОДРЯД, а не по всей колоде.

        Иначе предел превратился бы в квоту «не больше трёх списков на колоду»,
        то есть в прямое требование выдумать форму для четвёртой секции.
        Комбинация «три списка, цитата, три списка» честна и обязана проходить.
        """
        layouts = (
            [LAYOUT_BULLETS] * PLAN_LAYOUT_RUN_MAX
            + [LAYOUT_QUOTE]
            + [LAYOUT_BULLETS] * PLAN_LAYOUT_RUN_MAX
        )
        plan = self.deck(layouts)
        self.assertEqual([item.layout for item in plan.sections], layouts)

    def test_the_shortest_deck_of_the_catalogue_still_validates(self):
        """Колода минимальной длины не должна стать невозможной.

        При SLIDE_COUNT_MIN содержательных секций всего
        content_section_count(SLIDE_COUNT_MIN), и если их не больше предела, то
        сплошной список — законный план. Предел, который запрещает самый
        короткий заказ, запрещает не лень, а продукт.
        """
        sections = content_section_count(SLIDE_COUNT_MIN)
        self.assertLessEqual(
            sections,
            PLAN_LAYOUT_RUN_MAX,
            "предел серии стал строже самой короткой колоды: заказ на "
            f"{SLIDE_COUNT_MIN} слайдов больше нельзя разметить одними списками",
        )
        plan = validate([LAYOUT_BULLETS] * sections, slide_count=SLIDE_COUNT_MIN)
        self.assertEqual(len(plan.sections), sections)

    def test_all_violating_runs_are_named_not_just_the_first(self):
        """Попытка после отказа ровно одна (LLM_CALL_ATTEMPTS = 2).

        План, исправленный в одном месте из двух, сгорит вместе с заказом,
        поэтому претензия обязана перечислить все нарушенные серии сразу.
        """
        layouts = (
            [LAYOUT_BULLETS] * (PLAN_LAYOUT_RUN_MAX + 1)
            + [LAYOUT_QUOTE]
            + [LAYOUT_STEPS] * (PLAN_LAYOUT_RUN_MAX + 1)
        )
        text = self.deck_rejection(layouts)
        self.assertIn(repr(LAYOUT_BULLETS), text)
        self.assertIn(repr(LAYOUT_STEPS), text)

    def test_only_content_sections_are_counted(self):
        """Титул и «Источники» раскладок не имеют — и в план не попадают.

        Их дорисовывает рендерер (RENDERER_ADDED_SLIDES), поэтому в sections
        лежат ТОЛЬКО содержательные секции, и исключать из счёта нечего.
        Сторож на случай, если в план однажды заведут служебную секцию: тогда
        предел начнёт считать её частью серии, и это придётся заметить здесь, а
        не по странной колоде.
        """
        sections = content_section_count(SLIDE_COUNT_MAX)
        self.assertEqual(sections + RENDERER_ADDED_SLIDES, SLIDE_COUNT_MAX)
        # Ровно предел на каждую серию по всей длине предельного заказа —
        # разметка, законная и целиком содержательная.
        layouts = [
            SLIDE_LAYOUTS[(index // PLAN_LAYOUT_RUN_MAX) % len(SLIDE_LAYOUTS)]
            for index in range(sections)
        ]
        plan = validate(layouts, slide_count=SLIDE_COUNT_MAX)
        self.assertEqual(len(plan.sections), sections)


class ThePlanPromptAsksForTheMarkup(unittest.TestCase):
    """План-промпт обязан объяснить, чем размечать и по какому признаку.

    Схема умеет только отвергнуть чужое значение; ВЫБОР раскладки под материал
    ей не выразить, и он целиком держится на этом тексте.
    """

    def plan_prompt(self, **overrides) -> str:
        params = {
            "notebook_name": "Налоги",
            "description": "Обзор льгот",
            "language": LANGUAGE_RU,
            "slide_count": SLIDE_COUNT,
            "context_block": "",
        }
        params.update(overrides)
        return build_plan_messages(**params)[0]["content"]

    def test_the_schema_line_names_the_field(self):
        # Модель пишет поля в том порядке, в каком их назвали. Раскладки нет в
        # образце схемы — её не будет и в ответе, а весь план сгорит в повторе.
        self.assertIn('"layout": string', self.plan_prompt())

    def test_every_layout_of_the_contract_is_offered_with_its_case(self):
        prompt = self.plan_prompt()
        for layout in SLIDE_LAYOUTS:
            with self.subTest(layout=layout):
                self.assertIn(f"   {layout} — ", prompt)

    def test_the_rule_says_the_layout_follows_the_material(self):
        """Главная половина правила: форма следует содержанию.

        Без неё требование разнообразия превращается в квоту, а квота — в
        выдуманную вторую сторону сравнения, то есть во враньё про документ.
        """
        prompt = self.plan_prompt()
        self.assertIn("never the turn", prompt)
        self.assertIn("Do not cycle through", prompt)

    def test_the_rule_offers_quote_as_the_honest_way_to_break_a_run(self):
        """Выход «размечай всё списками» промпт предлагать больше не имеет права.

        Раньше он предлагал: «если материала нет — все секции bullets, это
        корректный план». С тех пор как предел серии ПРОВЕРЯЕТСЯ схемой, этот
        совет прямо противоречил бы проверке — модель, ему поверившая, получала
        бы отказ за исполнение правила. Заменить его можно только тем, что можно
        взять, ничего не выдумав: дословная формулировка есть в любом документе.
        """
        prompt = self.plan_prompt()
        self.assertIn(f"break the run with {LAYOUT_QUOTE}", prompt)
        self.assertIn("a wording that matters literally is in every document", prompt)
        # И честность второй половины правила остаётся: выдумывать материал под
        # форму запрещено прямым текстом.
        self.assertIn("Never make up a comparison or a number", prompt)
        self.assertIn("is a correct plan, not a failure", prompt)

    def test_the_prompt_says_the_run_limit_is_checked(self):
        # Валидатор — страховка, а не замена: повтор стоит целого вызова
        # модели, и предупреждённая модель чаще попадает с первой попытки.
        self.assertIn("This rule is CHECKED", self.plan_prompt())

    def test_the_run_limit_reaches_the_prompt_as_a_number(self):
        # Требование «не больше N одинаковых подряд» осмысленно только здесь:
        # план видит колоду целиком. Число берётся из константы — выписанное
        # руками, оно разъехалось бы с ней молча.
        self.assertIn(
            f"more than {PLAN_LAYOUT_RUN_MAX} sections in a row", self.plan_prompt()
        )

    def test_the_prompt_and_the_validator_read_the_same_constant(self):
        """Сторож на расхождение: два числа вместо одного — уже было.

        Предел серии жил в constants.py как просьба, схема его не знала, и
        разошлись они ровно так, как и должны были. Теперь число объявлено в
        схеме (llm_schemas.PLAN_LAYOUT_RUN_MAX) и переэкспортировано через
        constants.py; проверяется не совпадение значений, а ТОЖДЕСТВО объекта —
        совпадающие значения ничего не доказывают, пока их два.
        """
        self.assertIs(PLAN_LAYOUT_RUN_MAX, SCHEMA_LAYOUT_RUN_MAX)
        # И то же самое число доезжает до текста промпта. Сдвинутый предел
        # обязан сдвинуть промпт — иначе модель просят об одном, а отвергают за
        # другое.
        prompt = self.plan_prompt()
        self.assertIn(f"more than {SCHEMA_LAYOUT_RUN_MAX} sections in a row", prompt)
        self.assertNotIn(
            f"more than {SCHEMA_LAYOUT_RUN_MAX + 1} sections in a row", prompt
        )

    def test_the_prompt_explains_why_the_choice_is_here(self):
        # Причина в промпте не для красоты: она объясняет модели, почему нельзя
        # отложить решение «на потом» — потом никто колоды не увидит.
        prompt = self.plan_prompt()
        self.assertIn("sees the whole material at once", prompt)
        self.assertIn("will see this section and nothing else", prompt)


class TheSlidePromptExecutesTheAssignment(unittest.TestCase):
    """Слайд-вызов получает раскладку и требует именно её."""

    def slide_prompt(self, layout: str) -> str:
        return build_slide_messages(
            heading="Ставки НДС",
            layout=layout,
            description="",
            language=LANGUAGE_RU,
            context_block="",
            allowed_citations={"45": 7},
        )[0]["content"]

    def test_the_assigned_layout_is_named_as_already_decided(self):
        for layout in SLIDE_LAYOUTS:
            with self.subTest(layout=layout):
                prompt = self.slide_prompt(layout)
                self.assertIn(f'assigned this section its layout: "{layout}"', prompt)
                self.assertIn("it is decided already", prompt)

    def test_the_other_layouts_are_not_offered_as_a_menu(self):
        """Меню из пяти форм в вызове, которому выбирать нечего, — приглашение
        выбрать заново, то есть ровно то поведение, которое волна убирает.

        Исключение одно — bullets: это единственная законная замена, когда
        материал назначенную форму не держит, и адрес правила «не добивать».
        """
        for layout in SLIDE_LAYOUTS:
            prompt = self.slide_prompt(layout)
            for other in SLIDE_LAYOUTS:
                if other in (layout, LAYOUT_BULLETS):
                    continue
                with self.subTest(assigned=layout, other=other):
                    self.assertNotIn(f'"layout": "{other}"', prompt)

    def test_the_only_allowed_deviation_is_bullets(self):
        prompt = self.slide_prompt(LAYOUT_STEPS)
        self.assertIn(f"answer in the {LAYOUT_BULLETS} layout instead", prompt)
        self.assertIn("ONLY other layout you may use", prompt)
        # И замена не должна выглядеть удобнее исполнения: выдумывать материал
        # под форму запрещено тем же правилом.
        self.assertIn("Never invent material to fit the layout", prompt)

    def test_a_bullets_slide_is_not_offered_a_way_out(self):
        # Назначен список — заменять его нечем, и второй раз описывать его
        # форму незачем.
        prompt = self.slide_prompt(LAYOUT_BULLETS)
        self.assertIn("no other layout is allowed", prompt)
        self.assertEqual(prompt.count(f'"layout": "{LAYOUT_BULLETS}"'), 1)

    def test_the_history_of_used_layouts_is_gone(self):
        """Списка «какие раскладки уже были» больше нет, и это не упущение.

        Он подсказывал разнообразие тому, кто раскладку ВЫБИРАЛ. Исполнителю
        назначенного он говорил бы прямо противоположное правилу 4 — «возьми
        ту, которой ещё не было», — то есть спорил бы с планом. Разнообразие
        требуется там, где видно колоду: в план-промпте.
        """
        messages = build_slide_messages(
            heading="Ставки НДС",
            layout=LAYOUT_METRIC,
            description="",
            language=LANGUAGE_RU,
            context_block="",
            allowed_citations={"45": 7},
        )
        for message in messages:
            with self.subTest(role=message["role"]):
                self.assertNotIn("layouts_already_used", message["content"])

    def test_an_unknown_layout_breaks_the_build_instead_of_reaching_the_prompt(self):
        """Граница, а не недоверие вызывающему.

        Раскладка приезжает из провалидированного плана. Пропустив чужое
        значение молча, сборщик отдал бы промпт, требующий форму, которую сам
        же и не описал, — и слайд сгорел бы в повторной попытке с невнятным
        отказом.
        """
        with self.assertRaises(KeyError):
            build_slide_messages(
                heading="Ставки НДС",
                layout="table",
                description="",
                language=LANGUAGE_RU,
                context_block="",
                allowed_citations={"45": 7},
            )


if __name__ == "__main__":
    unittest.main()
