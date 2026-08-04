"""Раскладки в рендере: структура из схемы -> поля контекста -> ветка шаблона.

Раскладка проходит через три руки, и здесь проверяется стык между ними. Схема
решает, ЧТО модели позволено написать (это проверяет
tests/test_presentation_layouts.py); рендерер решает, какие поля из написанного
получит шаблон; шаблон решает, как их нарисовать. Разъезжаются такие стыки
молча: слайд собирается, колода печатается, а содержимого на листе нет.

Что закрепляется здесь.

  * ПОЛНОТА. Раскладок в рендерере ровно столько же, сколько в схеме
    (SLIDE_LAYOUTS), и на каждую есть фикстура. Раскладка, добавленная в схему и
    забытая в таблице рендерера, доехала бы до печати и упала на ней — уже после
    того, как модель отработала минуты.
  * СОСТАВ ПОЛЕЙ. У слайда есть layout, четыре общих ключа и поля СВОЕЙ
    раскладки — и ничего больше. Лишнее поле не безобидно: окружение шаблонов
    строгое (StrictUndefined), и ветка, случайно прочитавшая чужой ключ, падает
    не на смоук-рендере при старте, а там, где этот ключ вдруг оказался.
  * НИЧЕГО НЕ ВЫДУМАНО. metric без note получает None, а не подставленный текст.
    Сноска, сочинённая рендерером, на слайде неотличима от сноски из документа —
    у неё тот же кегль, то же место и та же убедительность.
  * НИЧЕГО НЕ ОБРЕЗАНО. Граница длины у текста модели одна — схема. Слайд, все
    поля которого набиты до её пределов, доезжает до шаблона знак в знак:
    появись у рендерера свои числа, они разошлись бы со схемой на первой же её
    правке, и разошлись бы МОЛЧА — обрезанный текст выглядит как текст.
  * РАЗМЕТКА СОБИРАЕТСЯ. Контекст каждой раскладки прогоняется через ВСЕ четыре
    шаблона настоящей Jinja в строгом окружении, но без Chrome. Это дешёвая
    половина проверки печати: расхождение «контекст против шаблона» ловится за
    миллисекунды. Вторая половина — что из этого попало на ЛИСТ — живёт там, где
    есть браузер: tests/test_presentation_print.py и
    tests/test_presentation_pipeline_db.py.
"""

import logging
import os
import sys
import unittest
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.modules.presentations.llm_schemas import (  # noqa: E402
    LAYOUT_BULLETS,
    LAYOUT_COMPARE,
    LAYOUT_METRIC,
    LAYOUT_QUOTE,
    LAYOUT_STEPS,
    SLIDE_BULLETS_MAX,
    SLIDE_LAYOUTS,
    SLIDE_STEPS_MAX,
)
from app.modules.presentations.renderer import (  # noqa: E402
    _LAYOUT_FIELDS,
    build_render_context,
    render_html,
)
from render_fixtures import (  # noqa: E402
    FIXTURE_CREATED_AT,
    FIXTURE_LAYOUTS,
    TEMPLATE_KEYS,
    make_context,
    make_slide,
    make_slides,
    maxed_slide_payload,
    offline_registry,
    slide_payload,
    structure_texts,
)

# Ключи слайда в контексте — по раскладкам. Записаны здесь целиком и намеренно:
# это единственное место, где видно, что именно обещано шаблонам, и любое
# расхождение обязано ронять тест, а не колоду пользователя. Общие ключи
# повторены в каждой строке, потому что «общие плюс свои» проверяется как ОДИН
# набор: проверка на «есть ключ X» пережила бы исчезновение ключа Y.
COMMON_SLIDE_KEYS = {"index", "layout", "heading", "citations"}
RENDERER_LOGGER = "app.modules.presentations.renderer"

LAYOUT_SLIDE_KEYS = {
    LAYOUT_BULLETS: COMMON_SLIDE_KEYS | {"bullets"},
    LAYOUT_COMPARE: COMMON_SLIDE_KEYS | {"left", "right"},
    LAYOUT_METRIC: COMMON_SLIDE_KEYS | {"value", "caption", "note"},
    LAYOUT_STEPS: COMMON_SLIDE_KEYS | {"steps"},
    LAYOUT_QUOTE: COMMON_SLIDE_KEYS | {"text", "attribution"},
}


def only_slide(layout: str, **overrides) -> dict:
    """Контекст из одного слайда заданной раскладки — и сам этот слайд."""
    context = make_context(slides=make_slides(1, layout=layout), **overrides)
    return context["slides"][0]


class LayoutCoverageTests(unittest.TestCase):
    """Раскладок везде поровну: в схеме, в рендерере, в фикстурах."""

    def test_the_renderer_draws_every_layout_the_schema_allows(self):
        """Список закрыт схемой, а рисует его рендерер — состав обязан совпасть.

        Раскладка, известная схеме и неизвестная таблице рендерера, проходит
        валидацию и падает на печати: пользователь ждёт минуты работы модели и
        получает отказ. Обратное расхождение тише и оттого хуже — таблица умеет
        раскладку, которой модель попросить не может, и ветка шаблона под неё не
        выполняется никогда.
        """
        self.assertEqual(set(_LAYOUT_FIELDS), set(SLIDE_LAYOUTS))

    def test_the_fixtures_cover_every_layout(self):
        """Без фикстуры раскладка не проходит через рендер ни в одном тесте."""
        self.assertEqual(FIXTURE_LAYOUTS, SLIDE_LAYOUTS)


class SlideShapeTests(unittest.TestCase):
    def test_each_slide_carries_its_layout_and_only_its_fields(self):
        for layout in SLIDE_LAYOUTS:
            with self.subTest(layout=layout):
                slide = only_slide(layout)

                self.assertEqual(slide["layout"], layout)
                self.assertEqual(set(slide), LAYOUT_SLIDE_KEYS[layout])

    def test_slides_are_numbered_from_one_whatever_the_layout(self):
        built = make_context(slides=make_slides(3, layout=LAYOUT_QUOTE))

        self.assertEqual([slide["index"] for slide in built["slides"]], [1, 2, 3])

    def test_citations_stay_with_the_slide_and_are_not_split_by_column(self):
        """Метка «[1]» принадлежит слайду целиком, а не колонке сравнения.

        В схеме цитата относится к слайду; разложить её по сторонам можно было бы
        только угадав, какое из утверждений на неё опиралось. «[1]» под левой
        колонкой утверждает, что источник подтверждает именно её, — а этого никто
        не проверял.
        """
        slide = only_slide(LAYOUT_COMPARE)

        self.assertEqual(slide["citations"], ["[1]"])
        self.assertNotIn("citations", slide["left"])
        self.assertNotIn("citations", slide["right"])


class BulletsTests(unittest.TestCase):
    def test_the_bullets_arrive_in_order(self):
        slide = only_slide(LAYOUT_BULLETS)

        self.assertEqual(slide["bullets"], ["Первый факт слайда 1", "Второй факт"])


class CompareTests(unittest.TestCase):
    def test_both_columns_keep_their_heading_and_their_list(self):
        slide = only_slide(LAYOUT_COMPARE)

        self.assertEqual(
            slide["left"],
            {
                "heading": "Было 1",
                "bullets": ["Ставка 15 процентов", "Отчёт раз в квартал"],
            },
        )
        self.assertEqual(
            slide["right"],
            {
                "heading": "Стало 1",
                "bullets": ["Ставка 12 процентов", "Отчёт раз в год"],
            },
        )

    def test_the_sides_do_not_swap(self):
        """Слева — left.

        Проверяется отдельно, потому что перепутанные стороны не видно ни по
        числу полей, ни по набору ключей: «было» и «стало» местами — это уже
        другой слайд, и неверный он ровно наполовину.
        """
        slide = only_slide(LAYOUT_COMPARE)

        self.assertTrue(slide["left"]["heading"].startswith("Было"))
        self.assertTrue(slide["right"]["heading"].startswith("Стало"))


class MetricTests(unittest.TestCase):
    def test_value_caption_and_note_arrive_as_written(self):
        slide = only_slide(LAYOUT_METRIC)

        self.assertEqual(slide["value"], "11 процентов")
        self.assertEqual(slide["caption"], "Доля отказов в приёме документов 1")
        self.assertEqual(slide["note"], "По данным за 2021 год")

    def test_a_metric_without_a_note_gets_none_not_a_substitute(self):
        """Пусто у модели — пусто на слайде.

        Ключ note обязан быть в контексте (StrictUndefined не прощает
        отсутствующих), но нести он обязан именно пустоту: подставить «по данным
        документа» значит написать за модель то, чего в документах не нашлось.
        """
        payload = slide_payload(1, layout=LAYOUT_METRIC)
        payload.pop("note")

        built = make_context(slides=[make_slide(payload)])

        self.assertIn("note", built["slides"][0])
        self.assertIsNone(built["slides"][0]["note"])

    def test_a_blank_note_stays_empty(self):
        """Пробельное note схема приводит к None; рендерер его не воскрешает."""
        payload = slide_payload(1, layout=LAYOUT_METRIC)
        payload["note"] = "   "

        built = make_context(slides=[make_slide(payload)])

        self.assertIsNone(built["slides"][0]["note"])


class StepsTests(unittest.TestCase):
    def test_steps_keep_the_order_the_model_wrote_them_in(self):
        """Порядок и есть содержание раскладки: им процедура отличается от списка.

        Номера в контексте нет намеренно: его рисует шаблон по позиции в списке.
        Поле с номером было бы вторым источником истины о порядке и однажды
        разошлось бы с самим списком, поставив «шаг 2» третьей карточкой.
        """
        slide = only_slide(LAYOUT_STEPS)

        self.assertEqual(
            [step["title"] for step in slide["steps"]],
            ["Подать заявление 1", "Приложить документы 1", "Получить решение 1"],
        )
        self.assertEqual(set(slide["steps"][0]), {"title", "text"})
        self.assertEqual(
            slide["steps"][0]["text"],
            "Заявление подаётся в налоговый орган по месту учёта",
        )


class QuoteTests(unittest.TestCase):
    def test_the_quote_keeps_its_text_and_its_attribution(self):
        slide = only_slide(LAYOUT_QUOTE)

        self.assertEqual(
            slide["text"], "Цитата 1: льгота предоставляется на срок до пяти лет"
        )
        self.assertEqual(slide["attribution"], "Налоговый кодекс, статья 1")


class NoTruncationTests(unittest.TestCase):
    """Граница длины одна — схема. У рендерера своей нет и быть не должно."""

    def test_every_layout_reaches_the_template_character_for_character(self):
        """Что схема пропустила, шаблон получает целиком.

        Второй набор пределов внутри рендерера разошёлся бы со схемой на первой
        же её правке — и разошёлся бы молча: обрезанный текст выглядит как
        текст, и заметить пропажу можно, только сверив слайд с документом.
        """
        for layout in SLIDE_LAYOUTS:
            with self.subTest(layout=layout):
                payload = maxed_slide_payload(layout)

                built = make_context(slides=[make_slide(payload)])

                self.assertEqual(
                    structure_texts(built["slides"][0]),
                    structure_texts(payload),
                    f"{layout}: текст слайда доехал до шаблона не таким, каким "
                    f"его пропустила схема",
                )

    def test_the_longest_allowed_list_keeps_every_item(self):
        """Выброшенный «лишний» буллет — та же потеря, только не по знакам."""
        bullets = make_context(slides=[make_slide(maxed_slide_payload(LAYOUT_BULLETS))])
        self.assertEqual(len(bullets["slides"][0]["bullets"]), SLIDE_BULLETS_MAX)

        steps = make_context(slides=[make_slide(maxed_slide_payload(LAYOUT_STEPS))])
        self.assertEqual(len(steps["slides"][0]["steps"]), SLIDE_STEPS_MAX)


class DegradedSlideTests(unittest.TestCase):
    """Слайд, разошедшийся с таблицей раскладок, не уносит с собой колоду.

    Слайды здесь собраны в обход схемы — иначе их не собрать вовсе: схема ровно
    это и запрещает. Проверяется поведение рендера на стыке, который схема
    сегодня закрывает собой: разъедутся релизы — и в контекст попадёт то, чего
    сегодня попасть не может. Гарантия шаблону при этом принадлежит РЕНДЕРУ, а
    не схеме, поэтому и проверяется она здесь.

    Исход — деградация в список, а не отказ, и это разница между «один слайд
    потерял форму» и «пользователь не получил ничего после минут работы модели».
    Плата за деградацию — ERROR в журнале: в колоде такой слайд выглядит обычным
    списком, и другого места узнать о нём нет.
    """

    def context_of(self, slide) -> dict:
        return build_render_context(
            title="Налоговые льготы",
            slides=[slide],
            sources=[],
            language="ru",
            notebook_name="Налоги",
            created_at=FIXTURE_CREATED_AT,
        )

    def degraded(self, slide) -> dict:
        with self.assertLogs(RENDERER_LOGGER, level=logging.ERROR) as captured:
            built = self.context_of(slide)
        self.complaint = "\n".join(captured.output)
        return built["slides"][0]

    def test_an_unknown_layout_becomes_a_list_and_keeps_the_text(self):
        """Чужая раскладка — дефект сборки, а не повод отменить заказ."""
        slide = SimpleNamespace(
            layout="poster",
            heading="Заголовок",
            citations=[],
            headline="Главное за год",
            body=["Первый факт", "Второй факт"],
        )

        degraded = self.degraded(slide)

        self.assertEqual(degraded["layout"], LAYOUT_BULLETS)
        self.assertEqual(
            degraded["bullets"], ["Главное за год", "Первый факт", "Второй факт"]
        )
        # Заголовок и ссылки переживают деградацию: теряется форма, а не слайд.
        self.assertEqual(degraded["heading"], "Заголовок")
        self.assertIn("poster", self.complaint)

    def test_a_layout_without_its_field_degrades_too(self):
        """«None» посреди слайда — не отказ, а напечатанная пустота.

        Строгое окружение шаблонов ловит ОТСУТСТВУЮЩИЙ ключ, но определённую
        пустоту считает законным значением и печатает как есть. Поэтому неполная
        раскладка деградирует целиком, а уцелевший текст уходит в список.
        """
        slide = SimpleNamespace(
            layout=LAYOUT_METRIC, heading="Заголовок", citations=[], value="12 %"
        )

        degraded = self.degraded(slide)

        self.assertEqual(degraded["layout"], LAYOUT_BULLETS)
        self.assertEqual(degraded["bullets"], ["12 %"])
        self.assertIn("caption", self.complaint)

    def test_a_field_of_the_wrong_type_degrades_instead_of_exploding(self):
        """«Поля нет» — не единственный способ структуре не совпасть.

        Список, оказавшийся числом, даёт TypeError на ровном месте, и для колоды
        это то же самое: рисовать нечем. Разница только в том, что исключение
        отсюда стоило бы всей страницы, а деградация — одного слайда.
        """
        slide = SimpleNamespace(
            layout=LAYOUT_BULLETS, heading="Заголовок", citations=[], bullets=5
        )

        degraded = self.degraded(slide)

        self.assertEqual(degraded["layout"], LAYOUT_BULLETS)
        self.assertEqual(degraded["bullets"], [])
        self.assertIn("TypeError", self.complaint)

    def test_a_slide_without_a_layout_degrades_as_well(self):
        slide = SimpleNamespace(heading="Заголовок", citations=[], bullets=["Факт"])

        degraded = self.degraded(slide)

        self.assertEqual(degraded["layout"], LAYOUT_BULLETS)
        self.assertEqual(degraded["bullets"], ["Факт"])

    def test_a_slide_with_no_salvageable_text_still_leaves_a_list(self):
        """Спасать нечего — но ключ bullets всё равно на месте.

        Пустой список печатается пустым перечнем, а отсутствующий ключ роняет
        страницу целиком: разница между «слайд ни о чём» и «колоды нет».
        """
        slide = SimpleNamespace(layout="poster", heading="Заголовок", citations=[])

        degraded = self.degraded(slide)

        self.assertEqual(degraded["bullets"], [])

    def test_the_context_always_promises_a_known_layout_with_its_fields(self):
        """Главный инвариант: что бы ни пришло, шаблону есть что читать.

        Проверка идёт по НАБОРУ ключей, а не по одному: ветка шаблона читает
        поля своей раскладки все сразу, и слайд с value без caption уронит
        страницу так же надёжно, как слайд без обоих.
        """
        strangers = [
            SimpleNamespace(layout="poster", heading="З", citations=[]),
            SimpleNamespace(layout=None, heading="З", citations=[]),
            SimpleNamespace(layout=LAYOUT_STEPS, heading="З", citations=[]),
            SimpleNamespace(
                layout=LAYOUT_COMPARE,
                heading="З",
                citations=[],
                left=SimpleNamespace(heading="Было", bullets=["Факт"]),
            ),
        ]
        for slide in strangers:
            with self.subTest(layout=getattr(slide, "layout", None)):
                with self.assertLogs(RENDERER_LOGGER, level=logging.ERROR):
                    built = self.context_of(slide)

                degraded = built["slides"][0]
                self.assertIn(degraded["layout"], SLIDE_LAYOUTS)
                self.assertEqual(set(degraded), LAYOUT_SLIDE_KEYS[degraded["layout"]])

    def test_a_healthy_deck_stays_quiet(self):
        """Обратная сторона громкой деградации.

        Без неё проверка «жалуется, когда сломано» была бы совместима с
        рендерером, который жалуется всегда, — а такой ERROR перестают читать на
        второй неделе.
        """
        with self.assertNoLogs(RENDERER_LOGGER, level=logging.ERROR):
            for layout in SLIDE_LAYOUTS:
                make_context(slides=make_slides(1, layout=layout))


class LayoutMarkupTests(unittest.TestCase):
    """Контекст против шаблона: настоящая Jinja, строгое окружение, без Chrome.

    Дешёвая половина проверки печати. Расхождение «контекст против шаблона» —
    самая частая беда этого стыка, и ловится она за миллисекунды: ветка,
    прочитавшая поле, которого в её раскладке нет, роняет рендер на
    StrictUndefined. Что при этом попало на ЛИСТ, отвечают проверки с настоящим
    браузером; здесь браузер только замедлял бы.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = offline_registry()

    def test_the_text_of_every_layout_reaches_the_markup_of_every_template(self):
        """Не «собралось», а «содержимое на месте».

        Шаблон с пустой веткой раскладки собирается без единой ошибки:
        StrictUndefined ловит чтение несуществующего поля, но молчит про поле,
        которого не прочитали вовсе. Слайд-цитата без текста цитаты — ровно
        такой случай, и виден он только по разметке.
        """
        for key in TEMPLATE_KEYS:
            template = self.registry.get(key)
            self.assertIsNotNone(template, f"шаблон {key} не прошёл проверку реестра")
            for layout in SLIDE_LAYOUTS:
                with self.subTest(template=key, layout=layout):
                    context = make_context(slides=make_slides(1, layout=layout))

                    html = render_html(template, context)

                    for text in structure_texts(context["slides"][0]):
                        self.assertIn(
                            text,
                            html,
                            f"{key}/{layout}: текста {text!r} нет в разметке",
                        )

    def test_a_foreign_layout_does_not_take_the_whole_deck_down(self):
        """Один разошедшийся слайд не отменяет колоду — ни на одном шаблоне.

        Шаблоны ветвятся по layout и всё незнакомое отправляют в ветку bullets, а
        та читает slide.bullets. В строгом окружении отсутствие этого поля роняет
        не слайд, а всю страницу: «пропустить один слайд» Jinja не умеет.
        Проверяется поэтому не контекст (он проверен выше), а САМА сборка — и на
        каждом дизайне, потому что ветвление у каждого своё.

        Соседний здоровый слайд обязан напечататься целиком: деградация стоит
        ровно одного слайда, а не всего, что идёт после него.
        """
        stranger = SimpleNamespace(
            layout="poster",
            heading="Чужая раскладка",
            citations=[],
            body=["Уцелевший факт"],
        )
        with self.assertLogs(RENDERER_LOGGER, level=logging.ERROR):
            context = make_context(slides=[*make_slides(1), stranger])

        for key in TEMPLATE_KEYS:
            with self.subTest(template=key):
                html = render_html(self.registry.get(key), context)

                self.assertIn("Уцелевший факт", html)
                self.assertIn("Чужая раскладка", html)
                self.assertIn("Первый факт слайда 1", html)


if __name__ == "__main__":
    unittest.main()
