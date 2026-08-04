"""Общая оснастка тестов рендера: поддельный браузер, реестр без превью, PDF.

Не модуль с тестами (имя не начинается на test_), поэтому unittest discover его
не подбирает — это именно библиотека для трёх файлов, которым нужна одна и та же
подготовка: печать колоды проверяется и без базы (tests/
test_presentation_print.py), и на полном пайплайне (tests/
test_presentation_pipeline_db.py), и подготовка у них совпадает дословно.

Три вещи, ради которых заведён файл:

  * ПОДДЕЛЬНЫЙ БРАУЗЕР. Настоящий Chrome не умеет зависать по заказу, а именно
    зависание и надо проверить: стадийный таймаут, убийство группы процессов,
    хвост stderr в отказе. Подделка — shell-скрипт, который отвечает на
    `--version` (иначе его отбракует проверка chromium.probe_chromium) и дальше
    делает то, что попросил тест;
  * РЕЕСТР БЕЗ ПРЕВЬЮ. Боевой реестр при первом обращении рисует превью всех
    шаблонов, то есть запускает Chrome четыре раза. Тестам печати это лишние
    секунды на каждом классе и лишняя зависимость от браузера там, где его
    проверять не собирались;
  * ЧТЕНИЕ PDF. Проверять надо ГОТОВЫЙ PDF, а не DOM: Chrome при печати
    расставляет содержимое иначе, чем на экране, и запись, видимая в браузере, в
    файле может отсутствовать. Тест, смотрящий в HTML, подтвердил бы вёрстку,
    которой в файле нет.
"""

import os
import stat
import sys
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.modules.presentations import chromium as chromium_module  # noqa: E402
from app.modules.presentations import renderer as renderer_module  # noqa: E402
from app.modules.presentations.chromium import (  # noqa: E402
    CHROMIUM_BINARY_ENV,
    chromium_status,
)
from app.modules.presentations.llm_schemas import PresentationSlide  # noqa: E402
from app.modules.presentations.renderer import RenderedSource  # noqa: E402
from app.modules.presentations.templates import TemplateRegistry  # noqa: E402

# Ключи всех шаблонов комплекта. Тесты вёрстки обязаны идти по КАЖДОМУ: порог
# обрезки один на все дизайны, и проверка на одном из них означала бы, что про
# остальные три мы просто ничего не знаем.
TEMPLATE_KEYS = ("draft", "aurora", "editorial", "blueprint")

# Ответ поддельного браузера на --version. Проверка chromium.probe_chromium
# ищет в нём подстроку "chrom" — без неё бинарник считается заглушкой и до
# печати дело не доходит.
FAKE_VERSION = "Chromium 999.0.0.0 (fake)"


def offline_registry() -> TemplateRegistry:
    """Настоящие шаблоны с диска, но без генерации превью.

    Каталог берётся по умолчанию — то есть проверяются ТЕ ЖЕ четыре шаблона,
    которые поедут в прод, а не сочинённая для теста заглушка. Расходятся они с
    боевым реестром ровно одним: превью не рисуются, потому что галерея к печати
    заказа отношения не имеет.
    """
    return TemplateRegistry(generate_previews=False)


def use_offline_registry(test) -> TemplateRegistry:
    """Подменить реестр рендерера на время теста."""
    registry = offline_registry()
    patcher = patch.object(renderer_module, "template_registry", registry)
    patcher.start()
    test.addCleanup(patcher.stop)
    return registry


def write_fake_chromium(path: Path, body: str) -> Path:
    """Скрипт, который отвечает на --version и делает body вместо печати."""
    path.write_text(
        "#!/bin/sh\n"
        'if [ "$1" = "--version" ]; then\n'
        f"  echo '{FAKE_VERSION}'\n"
        "  exit 0\n"
        "fi\n"
        f"{body}\n",
        encoding="utf-8",
    )
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


@contextmanager
def fake_chromium(path: Path, body: str):
    """То же, что use_fake_chromium, но на ограниченный участок теста.

    Нужен там, где подделка обязана уйти ДО конца теста: например, когда первый
    заказ должен упасть на зависшем браузере, а следующий — доехать до конца на
    настоящем. С addCleanup такой сценарий не выразить.
    """
    binary = write_fake_chromium(path, body)
    previous = chromium_module._status
    with patch.dict(os.environ, {CHROMIUM_BINARY_ENV: str(binary)}):
        chromium_module._status = None
        status = chromium_status(force=True)
        assert status.available, f"поддельный браузер не прошёл проверку: {status.error}"
        try:
            yield binary
        finally:
            chromium_module._status = previous


def use_fake_chromium(test, path: Path, body: str) -> Path:
    """Подсунуть рендеру поддельный браузер на время теста.

    Кэш статуса сбрасывается ДО и ПОСЛЕ: он живёт на уровне модуля, и без сброса
    после теста следующий получил бы в ответ на «есть ли браузер» подделку,
    которой к тому времени уже нет на диске.
    """
    binary = write_fake_chromium(path, body)
    patcher = patch.dict(os.environ, {CHROMIUM_BINARY_ENV: str(binary)})
    patcher.start()
    test.addCleanup(patcher.stop)

    chromium_module._status = None
    test.addCleanup(setattr, chromium_module, "_status", None)
    # Проверка выполняется сразу: если подделка не отвечает на --version, пусть
    # это выяснится здесь, а не в середине проверяемого сценария.
    status = chromium_status(force=True)
    assert status.available, f"поддельный браузер не прошёл проверку: {status.error}"
    return binary


def real_chromium_available() -> bool:
    """Есть ли на машине настоящий браузер (для тестов, которым нужен именно он)."""
    return chromium_status(force=True).available


def make_slides(count: int, *, source_ids=(1,)) -> list[PresentationSlide]:
    """Провалидированные слайды — через схему, а не голыми объектами.

    Контекст рендера обязан собираться из ТОГО ЖЕ провалидированного JSON, что и
    в бою; фикстура, собранная в обход схемы, могла бы содержать то, чего в
    проде не бывает, и тест проверял бы несуществующий случай.
    """
    return [
        PresentationSlide.model_validate(
            {
                "heading": f"Заголовок {index}",
                "bullets": [f"Первый факт слайда {index}", "Второй факт"],
                "citations": [
                    {"source_id": source_id, "chunk_id": index * 10 + source_id}
                    for source_id in source_ids
                ],
            }
        )
        for index in range(1, count + 1)
    ]


# Стили имён документов. Порог обрезки один на все имена, а переносятся они
# по-разному: в имени из слов у браузера есть где рвать строку, а у скана или
# выгрузки («skan_2026_final_v3», «Сканированныйдокумент…») точек переноса нет
# вовсе, и строка либо ломается посреди слова, либо уезжает за лист целиком.
# Мерить ёмкость только на словах значит мерить самый добрый случай и выдавать
# его за общий: именно на однословных именах порог врал сильнее всего.
SOURCE_NAME_STYLES = ("words", "long_words", "underscored", "solid")

# Заготовки под каждый стиль и знак, которым заготовка склеивается сама с собой,
# когда имя заказали длиннее её самой.
_SOURCE_NAME_BODIES = {
    # Обычное человеческое имя: короткие и средние слова, много пробелов.
    "words": (
        "Постановление Правительства Республики Таджикистан о порядке "
        "налогообложения и электронной отчётности для юридических лиц",
        " ",
    ),
    # Слова есть, но каждое почти в полстроки: переносить можно, а получается
    # рвано, и последняя строка имени оказывается длиннее, чем при коротких.
    "long_words": (
        "Постановление Правительства налогообложения документооборота "
        "Таджикистан государственной регистрационный",
        " ",
    ),
    # Имя файла со сканера: латиница, ни одного пробела. Подчёркивание точкой
    # переноса не считается (UAX#14), так что для вёрстки это одно слово.
    "underscored": (
        "skanirovannyy_dokument_ministerstva_finansov_respubliki_tadzhikistan_"
        "o_poryadke_nalogooblozheniya_i_elektronnoy_otchetnosti_2026_final_v3",
        "_",
    ),
    # Худший случай: кириллица сплошняком, ни пробела, ни подчёркивания. Рвать
    # строку негде совсем, и без overflow-wrap: anywhere имя уезжает за лист.
    "solid": (
        "Сканированныйдокументминистерствафинансовреспубликитаджикистанопорядке"
        "налогообложенияиэлектроннойотчётностидвадцатьдвадцатьшестьподписанная",
        "",
    ),
}

# Уникальный ХВОСТ имени — три знака, и без пробела перед ними. Проверка «запись
# видна» ищет на листе конец имени, и одинаковые окончания у соседних записей
# делали бы её неопровержимой: хвост уехавшей за лист записи находился бы у
# соседа, оставшегося на листе. Пробел не годится: он дал бы браузеру точку
# переноса ровно там, где её у настоящего однословного имени нет.
_SOURCE_NAME_SUFFIXES = {
    "words": "№{:02d}",
    "long_words": "№{:02d}",
    "underscored": "_{:02d}",
    "solid": "н{:02d}",
}


def _source_name(index: int, length: int, style: str = "words") -> str:
    """Имя документа РОВНО в length знаков, с неповторимым хвостом."""
    body, joiner = _SOURCE_NAME_BODIES[style]
    suffix = _SOURCE_NAME_SUFFIXES[style].format(index)
    if style == "words":
        body = f"Документ {index:02d} " + body
    while len(body) < length:
        body = body + joiner + body

    cut = length - len(suffix)
    stem = body[:cut]
    if stem.endswith(" "):
        # Обрыв ровно на пробеле дал бы двойной пробел перед хвостом, а
        # renderer.fit_name схлопывает пробелы — имя вышло бы на знак короче
        # заказанного, то есть фикстура врала бы о длине, по которой считается
        # цена записи. Берём вместо пробела следующий знак заготовки.
        stem = stem[:-1] + body[cut]
    return stem + suffix


def make_sources(
    count: int, *, name_length: int = 40, style: str = "words"
) -> list[RenderedSource]:
    """Источники с именами заданной длины и заданной механики переноса.

    Длина имени — единственный вход порога обрезки, поэтому она задаётся точно.
    Стиль — второй вход, о котором порог не знает, но от которого зависит
    ёмкость слайда: см. SOURCE_NAME_STYLES.
    """
    return [
        RenderedSource(
            source_id=index, name=_source_name(index, name_length, style), pages=[index]
        )
        for index in range(1, count + 1)
    ]


def pdf_pages(path: str) -> int:
    import fitz

    with fitz.open(path) as document:
        return document.page_count


def pdf_text(path: str, page: int | None = None) -> str:
    """Текст PDF: всей колоды или одной страницы.

    Пробелы схлопываются намеренно: перенос строки внутри имени документа —
    решение вёрстки, а проверяем мы наличие текста, а не место переноса.
    """
    import fitz

    with fitz.open(path) as document:
        pages = range(document.page_count) if page is None else [page]
        chunks = [document[number].get_text() for number in pages]
    return " ".join(" ".join(chunk.split()) for chunk in chunks)


# Допуск на границе листа, в пунктах. Bbox глифа считается по метрикам шрифта и
# у строки, прижатой к самому краю, может выступить за него на доли пункта, не
# пропав при этом с бумаги. Речь же идёт о потере ЦЕЛЫХ строк, уехавших за лист
# на десятки пунктов, поэтому доли пункта прощаются.
MEDIABOX_TOLERANCE = 0.6


def pdf_visible_text(path: str, page: int) -> str:
    """Знаки страницы, чей bbox ЦЕЛИКОМ внутри MediaBox, без пробелов.

    get_text() отдаёт и то, что уехало за край листа: Chrome при печати не
    выбрасывает переполнившее содержимое из потока текста, а просто рисует его
    там, где оно оказалось, — за границей страницы. Поиск по get_text() поэтому
    доказывает, что строка ЕСТЬ В ФАЙЛЕ, а не что она видна на листе, то есть
    молчит ровно про ту потерю, ради которой заведён порог обрезки: замерено
    живьём — aurora, имена по 148 знаков, одиннадцать записей, все одиннадцать
    «находятся», и при этом 29 знаков лежат вне MediaBox.

    Пробелы выкидываются по той же причине, что и в pdf_text: перенос строки
    внутри имени — решение вёрстки, а спрашиваем мы про видимость знаков.
    """
    import fitz

    with fitz.open(path) as document:
        sheet = document[page]
        media = sheet.mediabox
        visible = []
        for block in sheet.get_text("rawdict")["blocks"]:
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    for char in span.get("chars", []):
                        if char["c"].isspace():
                            continue
                        x0, y0, x1, y1 = char["bbox"]
                        if (
                            x0 >= media.x0 - MEDIABOX_TOLERANCE
                            and y0 >= media.y0 - MEDIABOX_TOLERANCE
                            and x1 <= media.x1 + MEDIABOX_TOLERANCE
                            and y1 <= media.y1 + MEDIABOX_TOLERANCE
                        ):
                            visible.append(char["c"])
    return "".join(visible)


def pdf_is_a_pdf(path: str) -> bool:
    with open(path, "rb") as handle:
        return handle.read(5) == b"%PDF-"
