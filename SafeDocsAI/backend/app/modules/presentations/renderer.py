"""Печать колоды: провалидированная структура -> HTML шаблона -> PDF из Chrome.

Модуль делает ровно две вещи, и разделение между ними принципиально:

  * СОБИРАЕТ КОНТЕКСТ (build_render_context) — чистая функция из плана, слайдов
    и источников в словарь, который ждут шаблоны. Ни файлов, ни процессов, ни
    базы: её можно проверить без браузера и без PostgreSQL, и почти всё, что в
    рендере способно соврать, живёт именно здесь;
  * ПЕЧАТАЕТ (render_presentation) — раскладывает страницу рядом с ресурсами
    шаблона и отдаёт её headless Chrome.

Почему HTML, а не pptx. Прежний рендерер собирал .pptx через python-pptx:
позиционирование фигур кодом, шрифты из темы, никакой типографики сложнее
абзаца. Вёрстка колоды — задача вёрстки, и решается она CSS'ом; браузер при
этом остаётся ВНЕШНИМ инструментом, а не библиотекой в процессе, и это меняет
устройство отказов. Библиотека бросает исключение, а внешний процесс умеет
зависнуть, наплодить детей и уйти в своп — поэтому здесь есть стадийный таймаут
и убийство ГРУППЫ процессов, которых у прежнего рендерера быть не могло.

ГРАНИЦЫ ДЛИНЫ СЮДА НЕ ВОЗВРАЩАЮТСЯ. У прежнего рендерера был свой набор
пределов (заголовок 120, буллет 300, строка источника 200) — защитная обрезка
внутри сборки файла. Настоящая граница всего, что пишет модель, — схема ответа
(llm_schemas): через неё проходит каждое поле, и шаблоны свёрстаны именно под
её числа. Второй набор пределов означал бы второй источник истины, который
разойдётся с первым на первой же правке схемы, — и разойдётся молча, потому что
обрезанный текст выглядит как текст. С появлением раскладок правило только
окрепло: полей, которые пишет модель, стало вчетверо больше, и у каждого свой
предел, посчитанный от места на слайде, — свой набор чисел здесь разошёлся бы со
схемой не в одном месте, а в четырнадцати.

Исключение ровно одно, и оно вынужденное: ИМЕНА ДОКУМЕНТОВ. Схема их не видит
(они приходят из метаданных чанков, а не от модели), слайд «Источники» не
прокручивается, а `overflow: hidden` в HTML режет молча — в отличие от pptx, где
текст сам уменьшался. Поэтому список источников подрезает сам рендерер, считает
непоместившиеся и ГРОМКО об этом сообщает: см. fit_sources ниже.

РАСКЛАДОК ПЯТЬ, и инвариант раздела от этого не меняется: модель пишет
структуру, код рисует. Из ответа модели приезжает ИМЯ раскладки из закрытого
списка и поля этой раскладки — ни HTML, ни CSS, ни выбора вёрстки. Разложить
структуру в поля контекста — работа build_render_context; что с ними делать
дальше, решает шаблон. См. таблицу раскладок ниже.

ОДИН БИТЫЙ СЛАЙД НЕ СТОИТ ЦЕЛОЙ КОЛОДЫ. Шаблоны ветвятся по layout, а окружение
у них строгое: слайд, в контексте которого нет полей своей ветки, роняет не
слайд, а всю страницу — и пользователь вместо колоды получает отказ после минут
работы модели. Поэтому неизвестная раскладка и недостающее поле здесь не
исключение, а деградация в список с ERROR в журнале (см. _layout_fields): текст
модели сохраняется, форма теряется, а узнаём об этом мы, а не пользователь.
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from jinja2 import TemplateError

from app.modules.presentations.chromium import (
    KILL_DRAIN_SECONDS,
    STDERR_NOISE,
    describe_failure,
    ensure_chromium_available,
    kill_process_group,
    pdf_command,
)
from app.modules.presentations.constants import (
    DATE_TEMPLATE,
    HTML_LANG,
    MONTH_NAMES,
    PAGES_LABEL,
    RENDER_PRINT_TIMEOUT,
    RENDER_STDERR_TAIL,
    SOURCES_HEADING,
    SOURCES_MORE,
    normalize_language,
)
from app.modules.presentations.llm_schemas import (
    LAYOUT_BULLETS,
    LAYOUT_COMPARE,
    LAYOUT_METRIC,
    LAYOUT_QUOTE,
    LAYOUT_STEPS,
    PresentationSlide,
)
from app.modules.presentations.templates import (
    TEMPLATE_FILENAME,
    build_environment,
    stage_page,
    template_registry,
)
from app.shared.models import as_utc

logger = logging.getLogger(__name__)


class RenderError(RuntimeError):
    """Колоду собрать не удалось, и причина известна.

    Отдельный тип, а не голый RuntimeError: сервис переводит его в
    generation_failed с текстом БЕЗ дополнительной обёртки, потому что текст уже
    написан для человека — «печать не уложилась в 270 с» плюс хвост stderr
    Chrome. Всё прочее, что может прилететь из рендера (OSError на записи,
    падение Jinja), сервис оборачивает сам и подписывает типом исключения.

    От RendererUnavailable (chromium.py) отличается смыслом: там «инструмента
    нет вовсе» — состояние машины, чинит администратор; здесь «инструмент
    отработал плохо» на конкретном заказе.
    """


# --- Источники: сколько их влезает на слайд -------------------------------
#
# Слайд «Источники» не прокручивается, а `overflow: hidden` режет МОЛЧА. Это
# тихая потеря данных: пользователь получает список, который выглядит полным.
# Значит, решать, сколько записей поместится, обязан код — до печати и вслух.
#
# ЗАМЕРЫ, а не назначенное число. Потолки сняты по ГОТОВОМУ PDF (не по DOM:
# Chrome при печати расставляет многоколоночник иначе, чем на экране, и запись,
# видимая в браузере, в файле может отсутствовать), с критерием «запись
# уместилась, только если видны и её метка, и её последняя строка»:
#
#     длина имени | draft | aurora | editorial | blueprint
#     60 знаков   |   18  |   16   |    20     |    12
#     90 знаков   |   12  |   12   |    14     |    10
#     148 знаков  |   10  |   10   |    10     |     8
#
# ОБЩИЙ порог по худшему дизайну, а не своё число на шаблон. Порог на шаблон
# пришлось бы держать в реестре — то есть превратить манифест оформления в
# код, который надо пересчитывать после каждой правки CSS, и который однажды
# разойдётся с этим самым CSS. Худший столбец (здесь blueprint во всех трёх
# строках) даёт число, верное для всех четырёх дизайнов сразу; остальные просто
# показывают меньше, чем могли бы, и это честная цена за один источник истины.
#
# ЗАМЕР ОБЯЗАН ВКЛЮЧАТЬ ОДНОСЛОВНЫЕ ИМЕНА, иначе он завышен. Таблица выше снята
# на именах ИЗ СЛОВ — самом добром для вёрстки случае: браузеру есть где
# перенести строку. Из базы приходит и другое: скан «skan_2026_final_v3…»,
# URL-подобное имя файла, слитная кириллица без единого пробела. Точек переноса
# там нет вовсе, и пока в .source-name не стоял overflow-wrap: anywhere,
# blueprint при тех же 60/90/148 знаках вмещал 6, 5 и 0 записей вместо 12, 10 и
# 8 — то есть замер на словах завышал ёмкость вдвое, а порог, снятый с него,
# молча срезал половину списка. Потому потолок берётся как МИНИМУМ и по
# дизайнам, и по стилям имени; сторожит это tests/test_presentation_print.py
# ::test_every_source_the_renderer_kept_is_actually_visible — он печатает все
# четыре шаблона на четырёх стилях имени и требует, чтобы каждая оставленная
# порогом запись целиком лежала внутри листа.
#
# Строку под хвост «не показано ещё N» вычитать НЕ НАДО: в состоянии
# sources_truncated > 0 шаблоны уплотняют список ровно на высоту плашки, и
# ёмкость от появления хвоста не падает (проверено обеими сторонами).
SOURCE_FIT_MEASURED = ((60, 12), (90, 10), (148, 8))

# Бюджет слайда в условных единицах. 120 — НОК замеренных ёмкостей (12, 10, 8),
# поэтому цена записи получается целой при каждом замере, а не приблизительной:
# арифметика с плавающей точкой на границе (120 / 12.0 -> 9.999...) отняла бы
# ровно одну запись, и понять, почему, было бы невозможно.
SOURCE_FIT_BUDGET = 120

# Цена одной записи выводится из замера, а не выписывается рядом с ним: два
# набора чисел про одно и то же — это то, чему предстоит разойтись. Округление
# ВВЕРХ, потому что ошибаться можно только в сторону «показали меньше»: обратная
# ошибка — молча срезанная браузером запись.
_SOURCE_COSTS = tuple(
    (length, -(-SOURCE_FIT_BUDGET // capacity))
    for length, capacity in SOURCE_FIT_MEASURED
)

# Длиннее самого длинного ЗАМЕРЕННОГО имени не бывает: имя подрезается.
#
# Это не «защитная обрезка на всякий случай», а граница знания. Ёмкость слайда
# измерена до 148 знаков; для имени в 255 (столько допускает Document.name)
# ответа нет, а экстраполяция прямой уводит ёмкость в отрицательные числа. Из
# двух честных вариантов — подрезать имя или гадать о ёмкости — выбран первый:
# он виден пользователю (многоточие в конце имени), а второй проявился бы
# исчезнувшими без следа источниками.
SOURCE_NAME_MAX_CHARS = SOURCE_FIT_MEASURED[-1][0]


@dataclass
class RenderedSource:
    """Документ в списке источников: имя и страницы, на которые ссылались."""

    source_id: int
    name: str
    pages: list[int] = field(default_factory=list)


def source_cost(name_length: int) -> int:
    """Во сколько единиц бюджета обходится запись с именем такой длины.

    Между замерами — линейная интерполяция с округлением вверх. Прямая здесь
    не модель физики, а способ не выдумывать: между 60 и 90 знаками ёмкость
    никто не мерил, и любая кривая была бы такой же догадкой, только менее
    понятной. Округление вверх делает догадку консервативной.
    """
    previous_length, previous_cost = _SOURCE_COSTS[0]
    if name_length <= previous_length:
        return previous_cost
    for length, cost in _SOURCE_COSTS[1:]:
        if name_length <= length:
            grown = (cost - previous_cost) * (name_length - previous_length)
            return previous_cost + -(-grown // (length - previous_length))
        previous_length, previous_cost = length, cost
    # Сюда доезжает только имя длиннее замеренного, а такого не бывает: имена
    # подрезаны до SOURCE_NAME_MAX_CHARS. Отдаём цену самой дорогой записи.
    return previous_cost


def fit_name(name: str) -> str:
    """Имя документа одной строкой и не длиннее замеренной границы."""
    text = " ".join((name or "").split())
    if len(text) <= SOURCE_NAME_MAX_CHARS:
        return text
    return text[: SOURCE_NAME_MAX_CHARS - 1].rstrip() + "…"


def format_pages(pages: list[int], language: str) -> str:
    """«стр. 12–14, 41»: подряд идущие страницы схлопываются в диапазон.

    Схлопывание не украшение, а ёмкость: документ, процитированный на десяти
    страницах подряд, иначе съедает строку целиком и вытесняет соседей со
    слайда. Пустой список даёт пустую строку — шаблон печатает её как есть, и
    отдельной ветки «страниц нет» ему не нужно.
    """
    ordered = sorted(
        {page for page in pages if isinstance(page, int) and not isinstance(page, bool)}
    )
    if not ordered:
        return ""

    spans: list[str] = []
    start = previous = ordered[0]
    for page in ordered[1:]:
        if page == previous + 1:
            previous = page
            continue
        spans.append(str(start) if start == previous else f"{start}–{previous}")
        start = previous = page
    spans.append(str(start) if start == previous else f"{start}–{previous}")
    return f"{PAGES_LABEL[language]} {', '.join(spans)}"


def format_date(moment: datetime, language: str) -> str:
    """Дата титульного слайда словами на языке колоды.

    Через таблицу месяцев, а не strftime («%d.%m.%Y» или «%B»): цифровая дата
    на титуле выглядит машинной, а «%B» зависит от локали, установленной в
    образе, — то есть оформление колоды менялось бы от состава пакетов.
    """
    local = as_utc(moment)
    return DATE_TEMPLATE[language].format(
        day=local.day, month=MONTH_NAMES[language][local.month - 1], year=local.year
    )


def fit_sources(
    sources: list[RenderedSource], language: str
) -> tuple[list[dict[str, Any]], int]:
    """Подрезанный список источников и число НЕпоместившихся.

    Записи набираются по порядку, пока хватает бюджета слайда, и в этом порядке
    же им розданы метки — поэтому недобор всегда приходится на ХВОСТ списка, а
    метки оставшихся записей не съезжают. Иначе перенумерация меняла бы смысл
    ссылок на уже написанных слайдах: «[3]» указывал бы на другой документ.

    Отбрасывание — WARNING в журнал, и это не перестраховка. Пользователь видит
    на слайде честное «не показано ещё N», а вот ПОЧЕМУ их столько (длинные
    имена файлов? слишком много документов в блокноте?) видно только по логу с
    самой длинной строкой. Тихая обрезка тут была бы потерей данных, о которой
    никто не узнал бы.
    """
    prepared: list[RenderedSource] = [
        RenderedSource(
            source_id=source.source_id,
            name=fit_name(source.name) or f"#{source.source_id}",
            pages=list(source.pages),
        )
        for source in sources
    ]

    spent = 0
    kept: list[dict[str, Any]] = []
    for position, source in enumerate(prepared, start=1):
        cost = source_cost(len(source.name))
        if spent + cost > SOURCE_FIT_BUDGET:
            break
        spent += cost
        kept.append(
            {
                "label": f"[{position}]",
                "name": source.name,
                "pages": sorted({page for page in source.pages if isinstance(page, int)}),
                "pages_text": format_pages(source.pages, language),
            }
        )

    truncated = len(prepared) - len(kept)
    if truncated:
        logger.warning(
            "Presentation render: the sources slide fits %d of %d documents, "
            "%d are hidden (budget %d units spent %d, longest name %d chars). "
            "The deck says so on the slide, but the list IS incomplete.",
            len(kept),
            len(prepared),
            truncated,
            SOURCE_FIT_BUDGET,
            spent,
            max(len(source.name) for source in prepared),
        )
    return kept, truncated


def source_labels(sources: list[RenderedSource]) -> dict[int, str]:
    """source_id -> метка «[N]» по порядку списка источников.

    Метки раздаются по ПОЛНОМУ списку, до подрезки: слайд, сославшийся на
    документ, который не поместился в перечень, обязан сохранить его номер.
    Ссылка на строку, которой не видно, — это честное «список неполон»;
    перенумерованная ссылка на ЧУЖОЙ документ — это ложь.
    """
    return {source.source_id: f"[{index}]" for index, source in enumerate(sources, 1)}


# --- Раскладки: структура слайда -> поля контекста -------------------------
#
# Раскладка — это ответ на вопрос «что модель нашла», а не «как это нарисовать».
# Сравнение двух режимов налогообложения, одно число с подписью, порядок
# действий, цитата из документа и обычный список фактов — пять РАЗНЫХ находок, и
# до раскладок все они превращались в один и тот же список буллетов. Выбирает из
# пяти модель, но выбирает она ИМЯ из закрытого списка; вёрстку по этому имени
# рисует шаблон, а между ними стоит таблица ниже — какие поля контекста получает
# шаблон для каждой раскладки.
#
# СПИСОК ЗАКРЫТ СХЕМОЙ, А НЕ ЗДЕСЬ, и имена раскладок приезжают ОТТУДА же —
# отсюда импорт LAYOUT_* из llm_schemas, а не пять своих строковых констант.
# Одним импортом, впрочем, не обойтись: разъехаться со схемой таблица может не
# написанием, а СОСТАВОМ — раскладку, добавленную в схему и забытую здесь,
# валидация пропустит, а нарисовать её будет нечем. Такой слайд деградирует в
# список (см. _layout_fields), то есть теряет форму; чтобы этого не случалось,
# полноту таблицы сторожит tests/test_presentation_layout_render.py, сверяя её
# ключи со SLIDE_LAYOUTS.


class _LayoutMismatch(Exception):
    """Структура слайда разошлась с таблицей раскладок.

    Внутренняя и наружу не выходит: снаружи у неё есть ровно один исход —
    деградация слайда в список (см. _layout_fields). Отдельный тип нужен, чтобы
    отличить «поля раскладки не хватает» от любой другой беды внутри сборки,
    которую глушить нельзя.
    """


def _need(source: Any, name: str) -> Any:
    """Обязательное поле раскладки — или несовпадение с именем поля.

    Обязательность держит схема, и пустое поле доезжает сюда только при
    расхождении её с таблицей выше. Молча отдать шаблону None значит напечатать
    «None» посреди слайда: в строгом окружении определённая пустота ошибкой не
    считается.
    """
    value = getattr(source, name, None)
    if value is None:
        raise _LayoutMismatch(f"в структуре слайда нет поля {name!r}")
    return value


def _bullets_fields(slide: Any) -> dict[str, Any]:
    return {"bullets": list(_need(slide, "bullets"))}


def _compare_fields(slide: Any) -> dict[str, Any]:
    """Две колонки со своими заголовками и своими списками.

    Метки цитат по колонкам НЕ разносятся и остаются у слайда целиком. В схеме
    цитата относится к слайду, а не к столбцу; разложить её надвое можно было бы
    только угадав, какое из утверждений на неё опиралось, — а «[1]» под левой
    колонкой утверждает, что источник подтверждает именно её.
    """
    return {
        "left": _compare_column(_need(slide, "left")),
        "right": _compare_column(_need(slide, "right")),
    }


def _compare_column(column: Any) -> dict[str, Any]:
    return {
        "heading": _need(column, "heading"),
        "bullets": list(_need(column, "bullets")),
    }


def _metric_fields(slide: Any) -> dict[str, Any]:
    """Одно число, подпись под ним и необязательная сноска.

    note — единственное необязательное поле всех пяти раскладок, и пусто здесь
    значит пусто на слайде. Отдаётся именно None, а не пустая строка: шаблон
    читает поле как «есть сноска или нет», и два способа сказать «нет» рано или
    поздно разъедутся. Подставить же вместо него что-нибудь разумное («по данным
    документа») значит написать за модель то, чего в документах не нашлось.
    """
    return {
        "value": _need(slide, "value"),
        "caption": _need(slide, "caption"),
        # Не через _need: у необязательного поля пустота — законный ответ, а не
        # расхождение со схемой.
        "note": getattr(slide, "note", None) or None,
    }


def _steps_fields(slide: Any) -> dict[str, Any]:
    """Шаги в том порядке, в котором их написала модель.

    Номеров здесь нет: их рисует шаблон по позиции в списке. Номер, приехавший
    полем, был бы вторым источником истины о порядке — и однажды разошёлся бы с
    самим списком, поставив «шаг 2» третьей карточкой.
    """
    return {
        "steps": [
            {"title": _need(step, "title"), "text": _need(step, "text")}
            for step in _need(slide, "steps")
        ]
    }


def _quote_fields(slide: Any) -> dict[str, Any]:
    return {"text": _need(slide, "text"), "attribution": _need(slide, "attribution")}


_LAYOUT_FIELDS = {
    LAYOUT_BULLETS: _bullets_fields,
    LAYOUT_COMPARE: _compare_fields,
    LAYOUT_METRIC: _metric_fields,
    LAYOUT_STEPS: _steps_fields,
    LAYOUT_QUOTE: _quote_fields,
}

# Служебные поля слайда: в спасённый текст они не входят. heading шаблон печатает
# сам и отдельно, citations — идентификаторы фрагментов, layout — имя раскладки,
# а не текст.
_SALVAGE_SKIP = frozenset({"heading", "citations", "layout"})


def _salvage_texts(slide: Any) -> list[str]:
    """Текст, который модель ВСЁ ЖЕ написала, — плоским списком строк.

    Ничего не сочиняет и ничего не переставляет: обходит структуру слайда и
    берёт непустые строки в том порядке, в котором они в ней лежат. Обход общий,
    а не по раскладкам, ровно потому, что зовут его тогда, когда раскладка
    ОКАЗАЛАСЬ НЕ ТОЙ: разбирать по раскладке то, что раскладке не соответствует,
    — способ потерять ещё и остаток.
    """
    dump = getattr(slide, "model_dump", None)
    try:
        structure = dump() if callable(dump) else getattr(slide, "__dict__", {})
    except Exception:  # noqa: BLE001 - спасение не имеет права само уронить печать
        structure = getattr(slide, "__dict__", {})
    return _plain_texts(structure)


def _plain_texts(value: Any) -> list[str]:
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, dict):
        return [
            text
            for key, nested in value.items()
            if key not in _SALVAGE_SKIP
            for text in _plain_texts(nested)
        ]
    if isinstance(value, (list, tuple)):
        return [text for nested in value for text in _plain_texts(nested)]
    return []


def _layout_fields(slide: Any, index: int) -> dict[str, Any]:
    """Раскладка слайда и поля ИМЕННО ЭТОЙ раскладки — с гарантией.

    Гарантия здесь двойная и она принадлежит РЕНДЕРУ, а не схеме: на выходе
    layout всегда одно из пяти известных значений, и поля этой раскладки всегда
    на месте. Схема сегодня обещает то же самое, но обещание схемы — про ответ
    модели, а шаблон читает контекст; между ними стоит эта функция, и держать
    контракт шаблона обязана она.

    ЧУЖАЯ РАСКЛАДКА ДЕГРАДИРУЕТ В СПИСОК, А НЕ РОНЯЕТ КОЛОДУ. Шаблоны ветвятся
    по layout и всё незнакомое отправляют в ветку bullets — а там читается поле
    `bullets`, которого у чужой раскладки нет. В строгом окружении это не
    испорченный слайд, а UndefinedError на всей странице: пользователь ждал
    минуты работы модели и не получает НИЧЕГО. Цена другого исхода несравнимо
    ниже: один слайд теряет свою форму, но сохраняет заголовок, ссылки и весь
    написанный моделью текст — списком.

    И ГРОМКО. Деградация — это дефект СБОРКИ (схема и рендерер приехали из
    разных релизов), а не плохой ответ модели, и увидеть его должны мы, а не
    пользователь: в колоде он выглядит как обычный слайд-список, поэтому
    единственное место, где о нём можно узнать, — ERROR в журнале.
    """
    layout = getattr(slide, "layout", None)
    build = _LAYOUT_FIELDS.get(layout)
    if build is not None:
        try:
            return {"layout": layout, **build(slide)}
        except _LayoutMismatch as exc:
            complaint = f"раскладка {layout!r}: {exc}"
        except Exception as exc:  # noqa: BLE001 - гарантия здесь абсолютная
            # Ловится ВСЁ, и это осознанно. Поля раскладки собираются из чужой
            # структуры, и «поля нет» — не единственный способ ей не совпасть:
            # список, оказавшийся строкой, или шаг, оказавшийся числом, дадут
            # TypeError на ровном месте. Для колоды разницы нет — слайд всё
            # равно нечем рисовать, — а исключение отсюда стоило бы всей
            # страницы. Тип ошибки не теряется: он уходит в ERROR ниже.
            complaint = f"раскладка {layout!r}: {type(exc).__name__}: {exc}"
    else:
        complaint = (
            f"раскладка {layout!r} рендереру неизвестна, известны только "
            f"{', '.join(_LAYOUT_FIELDS)}"
        )

    salvaged = _salvage_texts(slide)
    logger.error(
        "Presentation render: slide %d degraded to the %r layout (%s). The slide "
        "keeps its heading, its citations and %d line(s) of what the model wrote, "
        "but the shape it was written in is LOST. This is a build mismatch "
        "between the response schema and the renderer, not a bad answer from the "
        "model; the deck itself is printed to the end on purpose.",
        index,
        LAYOUT_BULLETS,
        complaint,
        len(salvaged),
    )
    return {"layout": LAYOUT_BULLETS, "bullets": salvaged}


def build_render_context(
    *,
    title: str,
    slides: list[PresentationSlide],
    sources: list[RenderedSource],
    language: str,
    notebook_name: str,
    created_at: datetime,
) -> dict[str, Any]:
    """Контекст шаблона из того же провалидированного JSON, что и раньше.

    Инвариант раздела не меняется: модель пишет структуру, код рисует. Сюда
    приходят PresentationPlan/PresentationSlide, уже прошедшие схему, и ни одно
    поле контекста не берётся из сырого ответа модели.

    Контракт полей зафиксирован и его читают все четыре шаблона. Два поля в нём
    существуют только ради шаблонов и заслуживают отдельного слова:

      * html_lang — код BCP-47, то есть "tg" там, где внутри проекта "tj";
      * sources_truncated — сколько источников не поместилось. Ноль здесь
        значит «список полон», и шаблон по нему решает, печатать ли хвост.
        Забыть это поле нельзя: окружение шаблонов строгое (StrictUndefined),
        и отсутствие ключа роняет рендер, а не рисует пустоту.

    У КАЖДОГО СЛАЙДА ЕСТЬ layout из закрытого списка и поля ИМЕННО ЭТОЙ
    раскладки — гарантированно, чем бы ни оказался слайд на входе (см.
    _layout_fields). Общего у всех четыре ключа — index, layout, heading,
    citations, — остальное зависит от раскладки.

    Титульный слайд и «Источники» в slides не входят вовсе: их рисует шаблон из
    title/notebook_name/generated_on и sources, раскладки у них нет и быть не
    может — их содержимое написала не модель.
    """
    language = normalize_language(language)
    labels = source_labels(sources)
    fitted, truncated = fit_sources(sources, language)

    return {
        "title": title,
        "notebook_name": notebook_name,
        "generated_on": format_date(created_at, language),
        "language": language,
        "html_lang": HTML_LANG[language],
        "slides": [
            {
                "index": index,
                "heading": slide.heading,
                "citations": _slide_labels(slide, labels),
                **_layout_fields(slide, index),
            }
            for index, slide in enumerate(slides, start=1)
        ],
        "sources": fitted,
        "sources_truncated": truncated,
        "strings": {
            "sources_heading": SOURCES_HEADING[language],
            "sources_more": SOURCES_MORE[language],
        },
    }


def _slide_labels(slide: PresentationSlide, labels: dict[int, str]) -> list[str]:
    """Готовые метки «[1]» слайда: без повторов, в порядке первого упоминания.

    Схема уже схлопнула цитаты по паре (source_id, chunk_id), но на слайде
    метка — это ДОКУМЕНТ: две цитаты на разные фрагменты одного документа дают
    один «[1]», и печатать его дважды значит показать пользователю бессмыслицу.
    """
    seen: set[str] = set()
    ordered: list[str] = []
    for citation in slide.citations:
        label = labels.get(citation.source_id)
        if label is None or label in seen:
            continue
        seen.add(label)
        ordered.append(label)
    return ordered


# --- Печать ----------------------------------------------------------------


def render_presentation(
    *,
    title: str,
    slides: list[PresentationSlide],
    sources: list[RenderedSource],
    language: str,
    template_key: str,
    notebook_name: str,
    created_at: datetime,
    output_path: str,
) -> None:
    """Собрать колоду и оставить готовый PDF по output_path.

    Функция СИНХРОННАЯ и блокирующая: она ждёт внешний процесс. Звать её можно
    только через run_in_threadpool (см. service.py) — тот же приём, что и для
    прежнего python-pptx, и по той же причине.

    Состав файла фиксирован контрактом slide_count: титульная страница, по
    странице на секцию плана и финальные «Источники». Разбиение на страницы
    делает CSS шаблона (`@page` плюс `break-after: page` на слайде), а не этот
    код: геометрия — часть дизайна, и навязывать её отсюда значило бы завести
    вторую, спорящую с первой.

    Chrome печатает СРАЗУ в output_path (у вызывающего это временный файл рядом
    с итоговым, который он потом переставит через os.replace). Промежуточный
    PDF во временном каталоге означал бы лишнее копирование между файловыми
    системами — /tmp на стенде бывает отдельным томом — и второй способ
    получить полуфайл.
    """
    template = template_registry.get(template_key)
    if template is None:
        # Не откат на «тему по умолчанию»: у HTML-рендера темы по умолчанию нет
        # и быть не может — без шаблона нет ни вёрстки, ни шрифтов. Реестр
        # выбрасывает битые шаблоны при старте с ERROR, а HTTP-слой не даёт
        # заказать несуществующий ключ; сюда это доезжает, только если шаблон
        # сломали между заказом и генерацией.
        raise RenderError(
            f"Шаблон {template_key!r} недоступен: он не прошёл проверку при "
            f"старте или был удалён"
        )

    binary = ensure_chromium_available()
    context = build_render_context(
        title=title,
        slides=slides,
        sources=sources,
        language=language,
        notebook_name=notebook_name,
        created_at=created_at,
    )
    html = render_html(template, context)

    with tempfile.TemporaryDirectory(prefix="deck-") as workspace:
        root = Path(workspace)
        try:
            page = stage_page(template.directory, html, root / "page")
        except OSError as exc:
            raise RenderError(f"Не удалось разложить страницу колоды: {exc}") from exc
        print_pdf(binary, page, Path(output_path), root / "profile")


def render_html(template: Any, context: dict[str, Any]) -> str:
    """Шаблон плюс контекст -> строка HTML.

    Окружение берётся у реестра (templates.build_environment), а не строится
    здесь: autoescape=True и StrictUndefined — это ЕГО решения, и смоук-рендер
    при старте проверяет шаблоны именно в нём. Своё окружение означало бы, что
    при старте проверяется одно, а пользователю печатается другое.

    Отдельного экранирования тут нет и не должно быть: весь пользовательский
    текст подставляется как данные, экранирует их Jinja. Фильтр |safe в
    шаблонах запрещён (см. комментарий к --no-sandbox в chromium.py: именно на
    этом держится решение печатать без песочницы).
    """
    environment = build_environment(template.directory)
    try:
        return environment.get_template(TEMPLATE_FILENAME).render(**context)
    except TemplateError as exc:
        # Прежде всего UndefinedError: контекст рендера разошёлся с шаблоном.
        # Такое ловит смоук-рендер при старте, и до заказа пользователя оно
        # доезжает только если шаблоны и код приехали из разных релизов.
        raise RenderError(
            f"Шаблон {template.key!r} не собрался с этим контекстом "
            f"({type(exc).__name__}): {exc}"
        ) from exc


# Сколько ждём смерти уже убитой группы. Это не бюджет работы, а время на
# доставку SIGKILL и закрытие труб: секунды здесь означали бы, что ядро не
# доставило сигнал, чего не бывает.


def print_pdf(binary: str, page: Path, pdf_file: Path, profile_dir: Path) -> None:
    """Запустить Chrome и дождаться PDF; зависший — убить группой процессов.

    ТАЙМАУТ СТОИТ ЗДЕСЬ, на самом процессе, а не только снаружи, и это главное
    отличие от прежнего рендера. asyncio.wait_for вокруг стадии (service.py)
    умеет отменить ОЖИДАНИЕ, но не работу: браузер, который он «снял», продолжил
    бы жить, держать память и писать в файл, который вызывающий уже считает
    брошенным. Поэтому первым обязан сработать subprocess-таймаут — он не ждёт,
    а убивает.

    УБИВАЕТСЯ ГРУППА, А НЕ ПРОЦЕСС. Chrome — это дерево: zygote, рендерер,
    gpu-процесс, утилиты. Убийство одного родителя оставляет детей живыми,
    осиротевшими и по-прежнему занимающими память и /dev/shm; на стенде это
    накапливается до тех пор, пока следующий рендер не упрётся в OOM. Отсюда
    start_new_session=True при запуске (у дерева появляется своя группа, чей id
    равен pid родителя) и killpg по ней.

    SIGKILL, а не SIGTERM: процесс уже перебрал отведённое время, его вывод
    непригоден, и вежливое завершение — это ещё одно ожидание с той же
    неопределённостью. Мягкий сигнал имел бы смысл, если бы нам был нужен
    результат его работы; нам он не нужен.
    """
    command = pdf_command(binary, page, pdf_file, profile_dir)
    try:
        process = subprocess.Popen(  # noqa: S603 - argv наш, шелла нет
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            errors="replace",
            # Своя сессия => своя группа процессов: только по ней можно убить
            # всё дерево браузера разом.
            start_new_session=True,
        )
    except OSError as exc:
        raise RenderError(f"Не удалось запустить {Path(binary).name}: {exc}") from exc

    try:
        _, stderr = process.communicate(timeout=RENDER_PRINT_TIMEOUT)
    except subprocess.TimeoutExpired:
        stderr = kill_process_group(process)
        raise RenderError(
            f"Печать колоды не уложилась в {int(RENDER_PRINT_TIMEOUT)} с и была "
            f"прервана{_stderr_suffix(stderr)}"
        )
    except BaseException:
        # Любой другой выход отсюда (в том числе отмена потока) не имеет права
        # оставить браузер работать: файл, в который он пишет, вызывающий уже
        # считает своим и вот-вот удалит.
        kill_process_group(process)
        raise

    if process.returncode != 0:
        raise RenderError(
            f"Печать колоды не удалась: "
            f"{describe_failure(command, process.returncode, stderr or '')}"
        )
    if not pdf_file.is_file() or pdf_file.stat().st_size == 0:
        # Chrome умеет выйти с нулём, не написав файла (страница не загрузилась,
        # каталог недоступен). Верить коду возврата на слово нельзя — иначе
        # заказ станет 'ready' с пустым файлом.
        raise RenderError(
            f"{Path(binary).name} отчитался об успехе, но PDF не появился"
            f"{_stderr_suffix(stderr)}"
        )


def _stderr_suffix(stderr: str) -> str:
    """Хвост осмысленного stderr для текста отказа, или пусто.

    Шум среды (см. STDERR_NOISE в chromium.py) выбрасывается ДО обрезки: Chrome
    пишет «Failed to adjust OOM score» на каждом запуске в контейнере, и без
    фильтра хвост состоял бы из него одного, вытеснив настоящую причину. Именно
    поэтому обрезаем хвост, а не начало: Chrome сообщает беду последней строкой.
    """
    lines = [
        line.strip()
        for line in (stderr or "").strip().splitlines()
        if line.strip() and not any(noise in line for noise in STDERR_NOISE)
    ]
    tail = " | ".join(lines)[-RENDER_STDERR_TAIL:]
    return f". Chrome сказал: {tail}" if tail else ""
