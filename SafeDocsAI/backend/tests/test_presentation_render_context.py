"""Контекст рендера: тот же провалидированный JSON -> словарь для шаблонов.

Ни браузера, ни базы: build_render_context — чистая функция, и почти всё, чем
рендер способен соврать, живёт именно в ней. Что здесь закрепляется.

  * КОНТРАКТ ПОЛЕЙ. Шаблоны читают контекст в строгом окружении
    (StrictUndefined), поэтому пропущенное поле — это не пустое место в колоде,
    а упавший заказ. Набор ключей проверяется целиком, а не по одному: тест на
    «есть ключ X» пережил бы удаление ключа Y.
  * СТРУКТУРА ПРИХОДИТ ИЗ СХЕМЫ. Слайды строятся через PresentationSlide, а не
    руками: инвариант раздела — «модель пишет структуру, код рисует», и фикстура
    в обход схемы проверяла бы случай, которого в бою не бывает.
  * tj -> tg. Внутренний код языка в HTML не попадает: lang="tj" невалиден по
    BCP-47, и браузер с ним теряет и переносы, и выбор начертания.
  * ОБРЕЗКА ИСТОЧНИКОВ ГРОМКАЯ. `overflow: hidden` режет молча — это тихая
    потеря данных. Поэтому лишние источники считаются, о них знает шаблон
    (sources_truncated) и знает журнал (WARNING). Проверяются ОБЕ стороны: при
    коротком списке ни того, ни другого быть не должно, иначе тест не отличил бы
    «работает» от «жалуется всегда».
  * ПОРОГ ВЫВЕДЕН ИЗ ЗАМЕРОВ. Отдельная проверка следит, что модель цены
    воспроизводит замеренную таблицу ёмкостей ровно; разъехавшись с ней, она
    перестала бы быть выведенной и стала бы просто числом.
"""

import logging
import os
import sys
import unittest
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.modules.presentations.constants import SOURCES_MORE  # noqa: E402
from app.modules.presentations.llm_schemas import (  # noqa: E402
    PresentationSlide,
    validate_slide,
)
from app.modules.presentations.renderer import (  # noqa: E402
    SOURCE_FIT_BUDGET,
    SOURCE_FIT_MEASURED,
    SOURCE_NAME_MAX_CHARS,
    RenderedSource,
    build_render_context,
    fit_sources,
    format_date,
    format_pages,
    source_cost,
)
from render_fixtures import make_slides, make_sources  # noqa: E402

RENDERER_LOGGER = "app.modules.presentations.renderer"

# Контракт контекста. Зафиксирован и читается всеми четырьмя шаблонами; список
# записан здесь целиком намеренно — это единственное место, где видно, что
# именно обещано шаблонам, и любое расхождение обязано ронять тест, а не колоду
# пользователя.
CONTEXT_KEYS = {
    "title",
    "notebook_name",
    "generated_on",
    "language",
    "html_lang",
    "slides",
    "sources",
    "sources_truncated",
    "strings",
}

CREATED_AT = datetime(2026, 8, 4, 9, 15, tzinfo=timezone.utc)


def context(**overrides):
    params = {
        "title": "Налоговые льготы",
        "slides": make_slides(3),
        "sources": make_sources(2),
        "language": "ru",
        "notebook_name": "Налоги",
        "created_at": CREATED_AT,
    }
    params.update(overrides)
    return build_render_context(**params)


class ContextContractTests(unittest.TestCase):
    def test_the_context_has_exactly_the_promised_keys(self):
        self.assertEqual(set(context()), CONTEXT_KEYS)

    def test_strings_carry_the_two_captions_templates_read(self):
        strings = context()["strings"]

        self.assertEqual(strings["sources_heading"], "Источники")
        # Плейсхолдер обязан быть ровно {count}: шаблон подставляет число через
        # str.format(count=...), и чужое имя даёт KeyError на рендере — то есть
        # падение заказа, у которого источники не поместились.
        self.assertIn("{count}", strings["sources_more"])
        self.assertEqual(strings["sources_more"].format(count=7).count("7"), 1)

    def test_slides_are_numbered_from_one_in_plan_order(self):
        slides = context(slides=make_slides(4))["slides"]

        self.assertEqual([slide["index"] for slide in slides], [1, 2, 3, 4])
        self.assertEqual(slides[0]["heading"], "Заголовок 1")
        self.assertEqual(slides[0]["bullets"], ["Первый факт слайда 1", "Второй факт"])

    def test_the_context_is_built_from_a_validated_answer(self):
        """Тот же путь, что в бою: сырой ответ модели -> схема -> контекст."""
        slide = validate_slide(
            '{"heading": "Кто имеет право", '
            '"bullets": ["Первый факт", "Второй факт"], '
            '"citations": [{"source_id": 1, "chunk_id": 10}]}',
            allowed_citations={"10": 1},
        )

        built = context(slides=[slide], sources=make_sources(1))

        self.assertEqual(built["slides"][0]["heading"], "Кто имеет право")
        self.assertEqual(built["slides"][0]["citations"], ["[1]"])


class CitationLabelTests(unittest.TestCase):
    def test_labels_follow_the_order_of_the_sources_list(self):
        built = context(
            slides=make_slides(1, source_ids=(2, 1)), sources=make_sources(3)
        )

        self.assertEqual([source["label"] for source in built["sources"]],
                         ["[1]", "[2]", "[3]"])
        # Порядок меток слайда — порядок первого упоминания, а не сортировка:
        # он повторяет порядок цитат, в котором их поставила модель.
        self.assertEqual(built["slides"][0]["citations"], ["[2]", "[1]"])

    def test_one_document_cited_twice_gives_one_label(self):
        """На слайде метка — это ДОКУМЕНТ, а не фрагмент.

        Схема схлопывает цитаты по паре (source_id, chunk_id), поэтому две
        ссылки на разные фрагменты одного документа доезжают сюда обе. Напечатать
        «[1] [1]» значило бы показать пользователю бессмыслицу.
        """
        slide = PresentationSlide.model_validate(
            {
                "heading": "Заголовок",
                "bullets": ["Факт", "Ещё факт"],
                "citations": [
                    {"source_id": 1, "chunk_id": 10},
                    {"source_id": 1, "chunk_id": 11},
                ],
            }
        )

        built = context(slides=[slide], sources=make_sources(1))

        self.assertEqual(built["slides"][0]["citations"], ["[1]"])


class HtmlLangTests(unittest.TestCase):
    def test_tajik_becomes_tg_for_the_lang_attribute(self):
        """Внутренний код "tj" в HTML не попадает: в BCP-47 таджикский — "tg"."""
        built = context(language="tj")

        self.assertEqual(built["html_lang"], "tg")
        # Внутренний код при этом остаётся собой: по нему шаблон рисует
        # машинные подписи, и переименовывать его ради атрибута нельзя.
        self.assertEqual(built["language"], "tj")

    def test_russian_stays_ru(self):
        built = context(language="ru")

        self.assertEqual(built["html_lang"], "ru")
        self.assertEqual(built["language"], "ru")

    def test_captions_follow_the_language(self):
        built = context(language="tj")

        self.assertEqual(built["strings"]["sources_heading"], "Манбаъҳо")
        self.assertEqual(built["strings"]["sources_more"], SOURCES_MORE["tj"])


class SourceFitTests(unittest.TestCase):
    def test_a_short_list_is_not_truncated_and_stays_quiet(self):
        """Обратная сторона громкой обрезки.

        Без неё проверка «жалуется, когда не влезло» была бы совместима с
        рендерером, который жалуется всегда, — а такой WARNING перестают читать
        на второй неделе.
        """
        with self.assertNoLogs(RENDERER_LOGGER, level=logging.WARNING):
            built = context(sources=make_sources(3))

        self.assertEqual(built["sources_truncated"], 0)
        self.assertEqual(len(built["sources"]), 3)

    def test_an_overflowing_list_is_cut_counted_and_logged(self):
        capacity = SOURCE_FIT_BUDGET // source_cost(60)
        extra = 5

        with self.assertLogs(RENDERER_LOGGER, level=logging.WARNING) as captured:
            built = context(sources=make_sources(capacity + extra, name_length=60))

        self.assertEqual(len(built["sources"]), capacity)
        self.assertEqual(built["sources_truncated"], extra)
        complaint = "\n".join(captured.output)
        self.assertIn(str(capacity), complaint)
        self.assertIn(str(capacity + extra), complaint)

    def test_the_kept_labels_do_not_shift(self):
        """Недобор приходится на ХВОСТ, метки оставшихся не съезжают.

        Перенумеровать после обрезки значило бы поменять смысл ссылок на уже
        написанных слайдах: «[3]» указывал бы на другой документ, а слайды
        печатаются из того же контекста.
        """
        capacity = SOURCE_FIT_BUDGET // source_cost(60)
        built = context(sources=make_sources(capacity + 3, name_length=60))

        self.assertEqual(
            [source["label"] for source in built["sources"]],
            [f"[{index}]" for index in range(1, capacity + 1)],
        )

    def test_a_slide_may_cite_a_document_that_did_not_fit(self):
        """Ссылка на невлезшую строку честнее, чем ссылка на чужую.

        Метки раздаются по ПОЛНОМУ списку, поэтому слайд сохраняет номер
        документа, которого в перечне не видно, а хвост «не показано ещё N»
        объясняет, почему его там нет.
        """
        capacity = SOURCE_FIT_BUDGET // source_cost(60)
        built = context(
            slides=make_slides(1, source_ids=(capacity + 2,)),
            sources=make_sources(capacity + 3, name_length=60),
        )

        self.assertEqual(built["slides"][0]["citations"], [f"[{capacity + 2}]"])
        self.assertGreater(built["sources_truncated"], 0)

    def test_a_name_longer_than_measured_is_cut_visibly(self):
        """Граница знания, а не защитная обрезка.

        Ёмкость слайда измерена до SOURCE_NAME_MAX_CHARS знаков; для имени
        вдвое длиннее ответа нет. Из двух вариантов — подрезать имя или гадать о
        ёмкости — выбран видимый пользователю.
        """
        built = context(
            sources=[RenderedSource(source_id=1, name="и" * 500, pages=[1])]
        )

        name = built["sources"][0]["name"]
        self.assertEqual(len(name), SOURCE_NAME_MAX_CHARS)
        self.assertTrue(name.endswith("…"))

    def test_a_nameless_document_falls_back_to_its_id(self):
        built = context(sources=[RenderedSource(source_id=42, name="", pages=[])])

        self.assertEqual(built["sources"][0]["name"], "#42")

    def test_a_multiline_name_is_flattened(self):
        # Перевод строки в имени документа сломал бы и вёрстку, и подсчёт длины,
        # по которому считается ёмкость слайда.
        built = context(
            sources=[RenderedSource(source_id=1, name=" Кодекс\n  РТ ", pages=[])]
        )

        self.assertEqual(built["sources"][0]["name"], "Кодекс РТ")


class FitModelTests(unittest.TestCase):
    """Порог обрезки ВЫВЕДЕН из замеров, а не назначен рядом с ними."""

    def test_the_cost_model_reproduces_every_measurement(self):
        for name_length, capacity in SOURCE_FIT_MEASURED:
            with self.subTest(name_length=name_length):
                self.assertEqual(SOURCE_FIT_BUDGET // source_cost(name_length), capacity)

    def test_the_budget_divides_every_measured_capacity_exactly(self):
        """Иначе цена записи округляется, и замер перестаёт воспроизводиться.

        Бюджет — НОК замеренных ёмкостей ровно для этого. Правка таблицы
        замеров без правки бюджета обязана уронить тест здесь, а не проявиться
        одной пропавшей строкой на слайде.
        """
        for _, capacity in SOURCE_FIT_MEASURED:
            with self.subTest(capacity=capacity):
                self.assertEqual(SOURCE_FIT_BUDGET % capacity, 0)

    def test_cost_grows_with_the_name_length(self):
        lengths = [10, 60, 61, 75, 90, 120, SOURCE_NAME_MAX_CHARS]
        costs = [source_cost(length) for length in lengths]

        self.assertEqual(costs, sorted(costs))
        # И ни одна запись не дороже целого слайда: иначе список из одного
        # длинного источника оказался бы пустым.
        self.assertLess(max(costs), SOURCE_FIT_BUDGET)

    def test_a_mixed_list_is_measured_by_its_own_names(self):
        """Ёмкость считается по КАЖДОМУ имени, а не по самому длинному.

        Один длинный документ среди коротких не должен выбрасывать со слайда
        девять коротких: они там помещаются, и выбросить их значило бы потерять
        данные, которые влезали.
        """
        mixed = [
            RenderedSource(source_id=1, name="и" * SOURCE_NAME_MAX_CHARS, pages=[1]),
            *make_sources(8, name_length=30)[1:],
        ]

        kept, truncated = fit_sources(mixed, "ru")

        self.assertEqual(truncated, 0)
        self.assertEqual(len(kept), len(mixed))


class PagesTextTests(unittest.TestCase):
    def test_consecutive_pages_collapse_into_a_range(self):
        self.assertEqual(format_pages([12, 13, 14, 41], "ru"), "стр. 12–14, 41")

    def test_pages_are_deduplicated_and_sorted(self):
        self.assertEqual(format_pages([5, 1, 5, 3], "ru"), "стр. 1, 3, 5")

    def test_a_pair_is_a_range_too(self):
        self.assertEqual(format_pages([7, 8], "ru"), "стр. 7–8")

    def test_tajik_label(self):
        self.assertEqual(format_pages([2], "tj"), "саҳ. 2")

    def test_no_pages_gives_an_empty_string(self):
        # Шаблон печатает это поле безусловно, и пустая строка — рабочий ответ:
        # отдельной ветки «страниц нет» ему не нужно.
        self.assertEqual(format_pages([], "ru"), "")


class GeneratedOnTests(unittest.TestCase):
    def test_russian_date_is_written_in_words(self):
        self.assertEqual(format_date(CREATED_AT, "ru"), "4 августа 2026 г.")

    def test_tajik_date_is_written_in_words(self):
        self.assertEqual(format_date(CREATED_AT, "tj"), "4 августи соли 2026")

    def test_the_date_does_not_depend_on_the_system_locale(self):
        """Таблица месяцев, а не strftime("%B").

        Системная локаль в контейнере — C, и «%B» вернул бы «August» под русским
        заголовком; ставить же зависимость от установленных локалей значит менять
        оформление колоды от состава пакетов на машине.
        """
        built = context(language="ru", created_at=datetime(2026, 1, 31, tzinfo=timezone.utc))

        self.assertEqual(built["generated_on"], "31 января 2026 г.")


if __name__ == "__main__":
    unittest.main()
