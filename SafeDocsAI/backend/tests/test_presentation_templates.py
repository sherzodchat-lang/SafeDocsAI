"""Реестр шаблонов презентаций v2: каталог HTML/CSS/шрифтов, отбраковка, превью.

Что закрепляем.

  * **Битый шаблон выбрасывается, галерея живёт.** Пропавший файл, неразбираемая
    Jinja, отсутствующий шрифт, внешняя ссылка — любая из этих бед стоит ОДНОЙ
    записи и ERROR в журнал; остальные дизайны остаются выбираемыми. Это принцип
    «интерфейс не врёт и не падает целиком»: показывать шаблон, на котором заказ
    потом упадёт, нельзя, но и гасить весь выбор из-за одного дизайна нельзя
    тоже. Каждая ветка проверяется отдельно и вместе с уцелевшим соседом.
  * **Проверки бывают только при старте.** Смоук-рендер эталонной фикстуры ловит
    то, чего не видит парсер: обращение к несуществующему полю, цикл по строке,
    фильтр с неверными аргументами. Без него первой жертвой опечатки был бы
    заказ пользователя — уже после того, как модель отработала.
  * **autoescape=True — свойство окружения, а не привычка автора шаблона.**
    Контекст рендера целиком состоит из пользовательских данных и ответов
    модели, а страницу мы сами открываем в браузере с --no-sandbox. `<script>`
    в буллите обязан приехать в вывод текстом.
  * **StrictUndefined падает громко.** Пропущенная переменная — это дефект,
    который пользователь заметит в готовой колоде, а мы нет.
  * **Превью перерисовывается по СОДЕРЖИМОМУ, а не по времени файла.** Тест
    проверяет обе стороны: правка шаблона перерисовывает, а `touch` без правки —
    нет. Иначе «работает» неотличимо от «делает всегда».

Настоящих backend/templates/presentations тесты не требуют: каждый случай
собирает свой каталог во временной папке — с template.html, styles.css,
шрифтом-пустышкой и собственным манифестом. Chrome тоже подменяется скриптом,
который считает, сколько раз его позвали: проверяется НАША логика отпечатков, а
не версия браузера на машине. Отдельная проверка на поставляемый комплект есть,
но она пропускается, пока комплекта нет в репозитории.
"""

import json
import os
import shutil
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from jinja2 import UndefinedError

# Каталог с тестами в sys.path: unittest discover кладёт его туда сам, но при
# запуске `python -m unittest tests.test_presentation_templates` этого не происходит.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.modules.presentations import templates as templates_module  # noqa: E402
from app.modules.presentations.chromium import ChromiumStatus  # noqa: E402
from app.modules.presentations.templates import (  # noqa: E402
    FIXTURE_FILENAME,
    MANIFEST_FILENAME,
    NAME_LANGUAGES,
    STYLES_FILENAME,
    TAJIK_GLYPHS,
    TEMPLATE_FILENAME,
    TemplateRegistry,
    build_environment,
    default_preview_dir,
    default_templates_dir,
    template_registry,
)

LOGGER_NAME = "app.modules.presentations.templates"

# Шаблон-пустышка, повторяющий контракт контекста рендера: title, notebook_name,
# generated_on, language, slides, sources, strings. Он намеренно трогает КАЖДОЕ
# поле — смоук-рендер проверяет ровно то, что шаблон умеет прочитать всё, что
# ему обещали дать.
TEMPLATE_HTML = """<!doctype html>
<html lang="{{ language }}">
<head><meta charset="utf-8"><title>{{ title }}</title>
<link rel="stylesheet" href="styles.css"></head>
<body>
<h1>{{ title }}</h1>
<p>{{ notebook_name }} — {{ generated_on }} — {{ strings.sources }}</p>
{% for slide in slides %}
<section id="s{{ slide.index }}"><h2>{{ slide.heading }}</h2>
<ul>{% for bullet in slide.bullets %}<li>{{ bullet }}</li>{% endfor %}</ul>
<div>{% for citation in slide.citations %}<span>{{ citation }}</span>{% endfor %}</div>
</section>
{% endfor %}
<footer><ul>{% for source in sources %}
<li>{{ source.label }} {{ source.name }}
{{ source.pages_text }} ({{ source.pages|length }})</li>
{% endfor %}</ul></footer>
</body></html>
"""

STYLES_CSS = """@font-face {
  font-family: "Deck";
  src: url("fonts/deck.woff2") format("woff2");
}
body { font-family: "Deck", sans-serif; }
"""

# Эталонная фикстура. Таджикские глифы стоят и в заголовке, и в буллите — так
# каждый смоук-рендер и каждое превью проверяют покрытие шрифта бесплатно.
FIXTURE = {
    "title": f"Отчёт {TAJIK_GLYPHS}",
    "notebook_name": "Блокнот",
    "generated_on": "04.08.2026",
    "language": "tj",
    "slides": [
        {
            "index": 1,
            "heading": "Хулоса",
            "bullets": [f"Матни санҷишӣ {TAJIK_GLYPHS}", "Второй пункт"],
            "citations": ["[1, с. 3]"],
        }
    ],
    "sources": [
        {"label": "[1]", "name": "Кодекс.pdf", "pages": [3, 4], "pages_text": "с. 3–4"}
    ],
    "strings": {"sources": "Манбаъҳо"},
}


class RegistryTestCase(unittest.TestCase):
    """Каталог шаблонов во временной папке плюс поддельный Chrome."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="templates-test-")
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.templates_dir = self.root / "templates"
        self.templates_dir.mkdir()
        self.preview_dir = self.root / "previews"
        self.calls_file = self.root / "chrome-calls.log"
        self.chrome = self._fake_chrome()
        self.write_fixture(FIXTURE)

    # --- сборка каталога ---------------------------------------------------

    def _fake_chrome(self) -> Path:
        """Скрипт, который пишет PNG туда, куда просили, и считает вызовы.

        Счётчик — единственный способ отличить «превью не перерисовалось,
        потому что логика отпечатков работает» от «перерисовалось молча».
        """
        path = self.root / "fake-chrome"
        path.write_text(
            "#!/bin/sh\n"
            f'echo call >> "{self.calls_file}"\n'
            'out=""\n'
            'for arg in "$@"; do\n'
            '  case "$arg" in\n'
            "    --version) echo 'Chromium 1.0'; exit 0;;\n"
            '    --screenshot=*) out="${arg#--screenshot=}";;\n'
            "  esac\n"
            "done\n"
            '[ -n "$out" ] || exit 3\n'
            "printf '\\211PNG\\r\\n\\032\\nfake-preview' > \"$out\"\n",
            encoding="utf-8",
        )
        path.chmod(path.stat().st_mode | stat.S_IXUSR)
        return path

    def chrome_calls(self) -> int:
        if not self.calls_file.is_file():
            return 0
        return len(self.calls_file.read_text(encoding="utf-8").splitlines())

    def make_template(
        self,
        key: str,
        html: str | None = TEMPLATE_HTML,
        css: str | None = STYLES_CSS,
        font: bool = True,
    ) -> Path:
        directory = self.templates_dir / key
        directory.mkdir(parents=True, exist_ok=True)
        if html is not None:
            (directory / TEMPLATE_FILENAME).write_text(html, encoding="utf-8")
        if css is not None:
            (directory / STYLES_FILENAME).write_text(css, encoding="utf-8")
        if font:
            fonts = directory / "fonts"
            fonts.mkdir(exist_ok=True)
            (fonts / "deck.woff2").write_bytes(b"wOF2 not really a font")
        return directory

    def write_manifest(self, *keys: str, entries: list | None = None) -> None:
        if entries is None:
            entries = [
                {"key": key, "name": {"ru": f"Дизайн {key}", "tj": f"Тарҳи {key}"}, "dir": key}
                for key in keys
            ]
        (self.templates_dir / MANIFEST_FILENAME).write_text(
            json.dumps({"templates": entries}, ensure_ascii=False), encoding="utf-8"
        )

    def write_fixture(self, data) -> None:
        (self.templates_dir / FIXTURE_FILENAME).write_text(
            json.dumps(data, ensure_ascii=False), encoding="utf-8"
        )

    def drop_fixture(self) -> None:
        (self.templates_dir / FIXTURE_FILENAME).unlink()

    # --- реестр ------------------------------------------------------------

    def registry(self, previews: bool = False) -> TemplateRegistry:
        """Реестр над временным каталогом.

        previews=False по умолчанию: подавляющему большинству проверок Chrome не
        нужен, и запускать его на каждую — это секунды на пустом месте.
        """
        return TemplateRegistry(
            self.templates_dir, self.preview_dir, generate_previews=previews
        )

    def with_fake_chrome(self, available: bool = True):
        status = ChromiumStatus(
            available=available,
            binary=str(self.chrome) if available else None,
            version="Chromium 1.0" if available else None,
            error=None if available else "no browser on this machine",
        )
        return patch.object(templates_module, "chromium_status", return_value=status)

    def assertRejected(self, registry: TemplateRegistry, key: str, needle: str) -> list[str]:
        """Шаблон исключён, ERROR записан, остальные живы."""
        with self.assertLogs(LOGGER_NAME, level="ERROR") as logs:
            listed = registry.list()

        keys = [info.key for info in listed]
        self.assertNotIn(key, keys)
        self.assertTrue(
            any(key in line and needle in line for line in logs.output),
            f"в журнале нет ERROR про {key!r} с {needle!r}: {logs.output}",
        )
        return keys


class HappyPathTests(RegistryTestCase):
    def test_usable_templates_are_listed_in_manifest_order(self) -> None:
        self.make_template("draft")
        self.make_template("bold")
        self.write_manifest("draft", "bold")

        listed = self.registry().list()

        self.assertEqual([info.key for info in listed], ["draft", "bold"])

    def test_paths_are_absolute_and_already_exist(self) -> None:
        self.make_template("draft")
        self.write_manifest("draft")

        info = self.registry().get("draft")

        self.assertTrue(info.html_file.is_absolute())
        self.assertTrue(info.html_file.is_file())
        self.assertTrue(info.css_file.is_file())
        self.assertTrue(info.fonts_dir.is_dir())
        self.assertEqual(info.directory.name, "draft")

    def test_names_arrive_on_both_languages(self) -> None:
        self.make_template("draft")
        self.write_manifest("draft")

        info = self.registry().get("draft")

        self.assertEqual(set(info.name), set(NAME_LANGUAGES))

    def test_an_unknown_key_is_none_not_an_exception(self) -> None:
        self.make_template("draft")
        self.write_manifest("draft")

        self.assertIsNone(self.registry().get("нет-такого"))

    def test_a_template_without_fonts_is_fine_when_it_asks_for_none(self) -> None:
        # Шаблон на системных шрифтах — законный дизайн; каталога fonts/ у него
        # просто нет, и требовать его было бы придиркой.
        self.make_template("plain", css="body { font-family: sans-serif; }", font=False)
        self.write_manifest("plain")

        info = self.registry().get("plain")

        self.assertIsNotNone(info)
        self.assertIsNone(info.fonts_dir)

    def test_inline_references_are_not_looked_for_on_disk(self) -> None:
        # url(#gradient) — ссылка на элемент этой же страницы, url(data:…) —
        # вшитый ресурс. Искать их на диске значило бы отбраковывать законную
        # вёрстку, и линт превратился бы в помеху вместо сторожа.
        self.make_template(
            "fancy",
            css=(
                "body { fill: url(#gradient); }\n"
                '@font-face { src: url("data:font/woff2;base64,d09GMg=="); }\n'
            ),
            font=False,
        )
        self.write_manifest("fancy")

        self.assertIsNotNone(self.registry().get("fancy"))

    def test_the_registry_is_read_once(self) -> None:
        # Сознательное отличие от runtime_settings.json: шаблоны меняются с
        # релизом, а не из админ-панели. Перечитывание означало бы парс Jinja и
        # смоук-рендер на каждый запрос списка — и ни одного изменённого ответа.
        self.make_template("draft")
        self.write_manifest("draft")
        registry = self.registry()

        first = registry.list()
        (self.templates_dir / MANIFEST_FILENAME).write_text("не json", encoding="utf-8")

        self.assertEqual([i.key for i in registry.list()], [i.key for i in first])

    def test_warm_up_never_raises_even_on_an_empty_machine(self) -> None:
        # Прогрев зовётся из lifespan приложения. Ни одна беда с шаблонами не
        # имеет права уронить старт: без галереи бэкенд работает, без бэкенда
        # админ не починит галерею.
        registry = TemplateRegistry(
            self.root / "нет-такого-каталога", self.preview_dir, generate_previews=False
        )

        with self.assertLogs(LOGGER_NAME, level="ERROR"):
            self.assertEqual(registry.warm_up(), [])

    def test_reload_picks_up_a_fixed_manifest(self) -> None:
        self.make_template("draft")
        self.write_manifest("draft")
        registry = self.registry()
        registry.list()

        self.make_template("bold")
        self.write_manifest("draft", "bold")

        self.assertEqual([i.key for i in registry.reload()], ["draft", "bold"])


class RejectionTests(RegistryTestCase):
    """Каждая ветка отбраковки: запись исключена, ERROR записан, сосед жив."""

    def setUp(self) -> None:
        super().setUp()
        # Здоровый сосед в каждом тесте: без него «реестр жив» неотличимо от
        # «реестр пуст».
        self.make_template("good")

    def test_a_missing_directory_is_rejected(self) -> None:
        self.write_manifest("good", "ghost")

        keys = self.assertRejected(self.registry(), "ghost", "does not exist")

        self.assertEqual(keys, ["good"])

    def test_a_missing_html_file_is_rejected(self) -> None:
        self.make_template("broken", html=None)
        self.write_manifest("good", "broken")

        keys = self.assertRejected(self.registry(), "broken", TEMPLATE_FILENAME)

        self.assertEqual(keys, ["good"])

    def test_a_missing_css_file_is_rejected(self) -> None:
        self.make_template("broken", css=None)
        self.write_manifest("good", "broken")

        keys = self.assertRejected(self.registry(), "broken", STYLES_FILENAME)

        self.assertEqual(keys, ["good"])

    def test_unparseable_jinja_is_rejected(self) -> None:
        self.make_template(
            "broken", html="<p>{% for slide in slides %}{{ slide.heading }}</p>"
        )
        self.write_manifest("good", "broken")

        keys = self.assertRejected(self.registry(), "broken", "Jinja2")

        self.assertEqual(keys, ["good"])

    def test_a_missing_font_is_rejected(self) -> None:
        # Самая тихая из поломок: Chrome не ругается, а молча берёт системный
        # шрифт, в котором таджикских ӣ ӯ қ ҳ ҷ ғ может не быть. Пользователь
        # получит колоду с квадратиками, и узнаем мы об этом от него.
        self.make_template("broken", font=False)
        self.write_manifest("good", "broken")

        keys = self.assertRejected(self.registry(), "broken", "missing asset")

        self.assertEqual(keys, ["good"])

    def test_an_external_url_in_css_is_rejected(self) -> None:
        self.make_template(
            "broken",
            css="@import url('https://fonts.googleapis.com/css2?family=Inter');",
        )
        self.write_manifest("good", "broken")

        keys = self.assertRejected(self.registry(), "broken", "external URL")

        self.assertEqual(keys, ["good"])

    def test_an_external_url_in_html_is_rejected(self) -> None:
        # Внешних ресурсов нет ПО ПОСТРОЕНИЮ, а не по договорённости: шаблон с
        # ссылкой на CDN рендерится по-разному в зависимости от того, была ли у
        # сервера сеть, и сообщает чужому хосту о каждой сборке.
        self.make_template(
            "broken",
            html=(
                '<html><head>'
                '<script src="http://cdn.example.com/x.js"></script>'
                '</head></html>'
            ),
        )
        self.write_manifest("good", "broken")

        keys = self.assertRejected(self.registry(), "broken", "external URL")

        self.assertEqual(keys, ["good"])

    def test_a_template_that_fails_the_smoke_render_is_rejected(self) -> None:
        # Опечатка в имени поля контекста: парсер её не видит, StrictUndefined
        # видит. Без смоук-рендера это всплыло бы в фоновой задаче пользователя.
        self.make_template("broken", html="<h1>{{ titel }}</h1>")
        self.write_manifest("good", "broken")

        keys = self.assertRejected(self.registry(), "broken", "fixture")

        self.assertEqual(keys, ["good"])

    def test_an_asset_outside_the_template_directory_is_rejected(self) -> None:
        self.make_template("broken", css='@font-face { src: url("../../../etc/passwd"); }')
        self.write_manifest("good", "broken")

        keys = self.assertRejected(self.registry(), "broken", "outside")

        self.assertEqual(keys, ["good"])

    def test_a_directory_outside_the_templates_root_is_rejected(self) -> None:
        self.write_manifest(
            entries=[
                {"key": "good", "name": {"ru": "Хороший", "tj": "Хуб"}, "dir": "good"},
                {"key": "escape", "name": {"ru": "Побег", "tj": "Гурез"}, "dir": "../.."},
            ]
        )

        keys = self.assertRejected(self.registry(), "escape", "outside")

        self.assertEqual(keys, ["good"])

    def test_a_name_missing_tajik_is_rejected(self) -> None:
        # ISO-код таджикского — tg, но во всём проекте язык обозначен как tj.
        # Запись с "tg" негодна: иначе в интерфейсе появилась бы пустая подпись.
        self.make_template("broken")
        self.write_manifest(
            entries=[
                {"key": "good", "name": {"ru": "Хороший", "tj": "Хуб"}, "dir": "good"},
                {"key": "broken", "name": {"ru": "Плохой", "tg": "Бад"}, "dir": "broken"},
            ]
        )

        keys = self.assertRejected(self.registry(), "broken", "'tj'")

        self.assertEqual(keys, ["good"])

    def test_a_duplicate_key_keeps_the_first_entry(self) -> None:
        # Не «последний побеждает»: выбор шаблона пришёл бы к одному каталогу, а
        # превью в списке показывалось бы от другого.
        self.make_template("twin")
        second = self.make_template("twin-two")
        self.write_manifest(
            entries=[
                {"key": "twin", "name": {"ru": "Раз", "tj": "Як"}, "dir": "twin"},
                {"key": "twin", "name": {"ru": "Два", "tj": "Ду"}, "dir": "twin-two"},
            ]
        )

        with self.assertLogs(LOGGER_NAME, level="ERROR"):
            listed = self.registry().list()

        self.assertEqual([i.key for i in listed], ["twin"])
        self.assertNotEqual(listed[0].directory, second)

    def test_a_key_that_could_escape_the_preview_directory_is_rejected(self) -> None:
        # Из ключа строится имя файла превью. Ключ "../x" вывел бы запись за
        # пределы каталога превью ещё до того, как кто-то попробовал бы это
        # снаружи, — и делал бы это наш собственный код, а не запрос.
        self.write_manifest(
            entries=[
                {"key": "good", "name": {"ru": "Хороший", "tj": "Хуб"}, "dir": "good"},
                {"key": "../беглец", "name": {"ru": "Побег", "tj": "Гурез"}, "dir": "good"},
            ]
        )

        with self.assertLogs(LOGGER_NAME, level="ERROR") as logs:
            listed = self.registry().list()

        self.assertEqual([i.key for i in listed], ["good"])
        self.assertTrue(any("a-z0-9_-" in line for line in logs.output))

    def test_an_entry_that_is_not_an_object_is_rejected(self) -> None:
        self.write_manifest(
            entries=[
                "просто строка",
                {"key": "good", "name": {"ru": "Хороший", "tj": "Хуб"}, "dir": "good"},
            ]
        )

        with self.assertLogs(LOGGER_NAME, level="ERROR"):
            listed = self.registry().list()

        self.assertEqual([i.key for i in listed], ["good"])


class ManifestTests(RegistryTestCase):
    def test_a_missing_manifest_is_an_empty_registry_with_an_error(self) -> None:
        with self.assertLogs(LOGGER_NAME, level="ERROR") as logs:
            listed = self.registry().list()

        self.assertEqual(listed, [])
        self.assertTrue(any("manifest" in line for line in logs.output))

    def test_broken_json_is_an_empty_registry_with_an_error(self) -> None:
        (self.templates_dir / MANIFEST_FILENAME).write_text("{нет", encoding="utf-8")

        with self.assertLogs(LOGGER_NAME, level="ERROR"):
            self.assertEqual(self.registry().list(), [])

    def test_a_manifest_without_the_templates_list_is_empty(self) -> None:
        (self.templates_dir / MANIFEST_FILENAME).write_text('{"шаблоны": []}', encoding="utf-8")

        with self.assertLogs(LOGGER_NAME, level="ERROR") as logs:
            self.assertEqual(self.registry().list(), [])

        self.assertTrue(any("templates" in line for line in logs.output))

    def test_a_bare_list_is_no_longer_accepted(self) -> None:
        # Формат v1 (голый список) больше не читается; принять его молча значило
        # бы держать два формата одновременно и не знать, какой перед нами.
        (self.templates_dir / MANIFEST_FILENAME).write_text("[]", encoding="utf-8")

        with self.assertLogs(LOGGER_NAME, level="ERROR"):
            self.assertEqual(self.registry().list(), [])


class FixtureTests(RegistryTestCase):
    def test_a_missing_fixture_is_loud_but_does_not_empty_the_gallery(self) -> None:
        # Один забытый общий файл не должен объявлять битыми все дизайны разом —
        # это ровно то, чего реестр обязан не допускать. Структурные проверки
        # при этом остаются в силе.
        self.make_template("draft")
        self.write_manifest("draft")
        self.drop_fixture()

        with self.assertLogs(LOGGER_NAME, level="ERROR") as logs:
            listed = self.registry().list()

        self.assertEqual([i.key for i in listed], ["draft"])
        self.assertTrue(any(FIXTURE_FILENAME in line for line in logs.output))

    def test_a_fixture_without_the_tajik_probe_is_an_error(self) -> None:
        # Фикстура без глифов рабочая, но перестала быть сторожем: превью больше
        # не доказывают покрытие шрифта. Тихая потеря проверки — самая дорогая.
        self.make_template("draft")
        self.write_manifest("draft")
        stripped = json.loads(json.dumps(FIXTURE))
        stripped["title"] = "Отчёт"
        stripped["slides"][0]["bullets"] = ["Обычный текст"]
        self.write_fixture(stripped)

        with self.assertLogs(LOGGER_NAME, level="ERROR") as logs:
            listed = self.registry().list()

        self.assertEqual([i.key for i in listed], ["draft"])
        self.assertTrue(any("glyph" in line for line in logs.output))

    def test_the_fixture_carries_the_glyphs_into_the_rendered_html(self) -> None:
        self.make_template("draft")
        self.write_manifest("draft")

        environment = build_environment(self.templates_dir / "draft")
        html = environment.get_template(TEMPLATE_FILENAME).render(**FIXTURE)

        self.assertIn(TAJIK_GLYPHS, html)


class JinjaEnvironmentTests(unittest.TestCase):
    """Окружение Jinja — наше, значит и его свойства проверяем мы.

    Контекст рендера собирает другой модуль, но экранирование и строгость
    неопределённых переменных задаются здесь, при создании Environment, и
    сломать их можно одной строкой в этом файле.
    """

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="jinja-test-")
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)

    def render(self, source: str, **context) -> str:
        (self.dir / TEMPLATE_FILENAME).write_text(source, encoding="utf-8")
        environment = build_environment(self.dir)
        return environment.get_template(TEMPLATE_FILENAME).render(**context)

    def test_a_script_tag_in_a_bullet_arrives_as_text(self) -> None:
        # Буллиты приходят из документов пользователя и из ответа модели, а
        # страницу мы сами открываем в браузере с --no-sandbox. Без
        # экранирования это исполняемый код внутри нашего же процесса печати.
        html = self.render(
            "<ul>{% for bullet in bullets %}<li>{{ bullet }}</li>{% endfor %}</ul>",
            bullets=["<script>alert(1)</script>"],
        )

        self.assertNotIn("<script>", html)
        self.assertIn("&lt;script&gt;", html)

    def test_quotes_are_escaped_so_attributes_cannot_be_broken_out_of(self) -> None:
        html = self.render(
            '<div title="{{ heading }}"></div>',
            heading='" onmouseover="alert(1)',
        )

        self.assertNotIn('onmouseover="alert(1)"', html)
        self.assertIn("&#34;", html)

    def test_closing_tags_in_the_notebook_name_cannot_escape_their_element(self) -> None:
        html = self.render("<h1>{{ notebook_name }}</h1>", notebook_name="</h1><img src=x>")

        self.assertNotIn("<img", html)

    def test_autoescape_is_on_for_every_template_not_just_html_named_ones(self) -> None:
        # select_autoescape по расширению здесь был бы ловушкой: достаточно
        # завести partial с другим суффиксом, и экранирование тихо отключится.
        environment = build_environment(self.dir)

        self.assertTrue(environment.autoescape)

    def test_a_missing_variable_fails_loudly(self) -> None:
        # Молчаливый Undefined превратил бы опечатку в пустое место на слайде
        # готовой колоды — дефект, который заметит пользователь, а не мы.
        with self.assertRaises(UndefinedError):
            self.render("<h1>{{ title }}</h1>")

    def test_a_missing_attribute_fails_loudly_too(self) -> None:
        with self.assertRaises(UndefinedError):
            self.render("<h2>{{ slide.headnig }}</h2>", slide={"heading": "Хулоса"})


class PreviewTests(RegistryTestCase):
    """Превью рисует сам Chrome, и перерисовывает по содержимому шаблона."""

    def setUp(self) -> None:
        super().setUp()
        self.make_template("draft")
        self.write_manifest("draft")

    def test_a_preview_is_generated_and_logged(self) -> None:
        with self.with_fake_chrome():
            with self.assertLogs(LOGGER_NAME, level="INFO") as logs:
                info = self.registry(previews=True).get("draft")

        self.assertTrue(info.preview_file.is_file())
        self.assertEqual(info.preview_file.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
        # Лог о том, что перегенерировали: без него молчаливая перерисовка на
        # каждом старте неотличима от работающего кэша.
        self.assertTrue(any("regenerated" in line for line in logs.output))
        self.assertEqual(self.chrome_calls(), 1)

    def test_the_preview_path_comes_from_the_registry_and_is_named_by_key(self) -> None:
        # Путь к превью берётся ТОЛЬКО из реестра, никогда из запроса: иначе
        # выдача картинки превращается в чтение произвольного файла с диска.
        # Имя файла строится из ключа, который реестр уже проверил.
        with self.with_fake_chrome():
            info = self.registry(previews=True).get("draft")

        self.assertEqual(info.preview_file.parent, self.preview_dir)
        self.assertEqual(info.preview_file.name, "draft.png")

    def test_nothing_is_regenerated_when_nothing_changed(self) -> None:
        # Без этой половины теста «работает» неотличимо от «делает всегда».
        with self.with_fake_chrome():
            self.registry(previews=True).list()
            self.assertEqual(self.chrome_calls(), 1)

            self.registry(previews=True).list()

        self.assertEqual(self.chrome_calls(), 1)

    def test_touching_a_file_without_changing_it_regenerates_nothing(self) -> None:
        # Именно поэтому отпечаток считается по СОДЕРЖИМОМУ, а не по mtime:
        # git checkout, docker COPY и rsync ставят свежее время, не меняя ни
        # байта, и на mtime мы перерисовывали бы всю галерею каждый деплой.
        html = self.templates_dir / "draft" / TEMPLATE_FILENAME
        with self.with_fake_chrome():
            self.registry(previews=True).list()
            os.utime(html, (1_800_000_000, 1_800_000_000))

            self.registry(previews=True).list()

        self.assertEqual(self.chrome_calls(), 1)

    def test_editing_the_template_regenerates_the_preview(self) -> None:
        html = self.templates_dir / "draft" / TEMPLATE_FILENAME
        with self.with_fake_chrome():
            self.registry(previews=True).list()

            html.write_text(TEMPLATE_HTML.replace("<h1>", "<h1 class='x'>"), encoding="utf-8")
            with self.assertLogs(LOGGER_NAME, level="INFO") as logs:
                self.registry(previews=True).list()

        self.assertEqual(self.chrome_calls(), 2)
        self.assertTrue(any("regenerated" in line for line in logs.output))

    def test_editing_the_stylesheet_regenerates_the_preview(self) -> None:
        # Отпечаток берётся со всего каталога: картинку меняет и правка в CSS.
        css = self.templates_dir / "draft" / STYLES_FILENAME
        with self.with_fake_chrome():
            self.registry(previews=True).list()

            css.write_text(STYLES_CSS + "\nh1 { color: red; }", encoding="utf-8")
            self.registry(previews=True).list()

        self.assertEqual(self.chrome_calls(), 2)

    def test_editing_the_fixture_regenerates_the_preview(self) -> None:
        # Превью в галерее показывают один и тот же эталонный текст; сменился
        # он — сменились все картинки.
        with self.with_fake_chrome():
            self.registry(previews=True).list()

            changed = json.loads(json.dumps(FIXTURE))
            changed["title"] = f"Другой заголовок {TAJIK_GLYPHS}"
            self.write_fixture(changed)
            self.registry(previews=True).list()

        self.assertEqual(self.chrome_calls(), 2)

    def test_without_chromium_the_template_stays_listed_without_a_preview(self) -> None:
        # Дизайн рабочий, рисовать его нечем. Галерея без картинок хуже, чем с
        # картинками, но несравнимо лучше, чем пустая галерея.
        with self.with_fake_chrome(available=False):
            with self.assertLogs(LOGGER_NAME, level="ERROR") as logs:
                info = self.registry(previews=True).get("draft")

        self.assertIsNotNone(info)
        self.assertIsNone(info.preview_file)
        self.assertTrue(any("no preview" in line for line in logs.output))

    def test_an_up_to_date_preview_needs_no_browser_at_all(self) -> None:
        # Шаблон не менялся — картинка верна, и запускать Chrome незачем даже
        # если он есть. Это и есть смысл отпечатка: старт не платит за то, что
        # уже нарисовано.
        with self.with_fake_chrome():
            first = self.registry(previews=True).get("draft").preview_file

        with self.with_fake_chrome(available=False):
            second = self.registry(previews=True).get("draft").preview_file

        self.assertEqual(first, second)
        self.assertTrue(second.is_file())

    def test_an_old_preview_survives_a_browser_outage(self) -> None:
        # Chrome пропал, а шаблон при этом отредактировали: картинка устарела,
        # но показать старую всё равно лучше, чем не показать ничего.
        with self.with_fake_chrome():
            first = self.registry(previews=True).get("draft").preview_file

        html = self.templates_dir / "draft" / TEMPLATE_FILENAME
        html.write_text(TEMPLATE_HTML.replace("<h1>", "<h1 id='t'>"), encoding="utf-8")

        with self.with_fake_chrome(available=False):
            with self.assertLogs(LOGGER_NAME, level="ERROR") as logs:
                second = self.registry(previews=True).get("draft").preview_file

        self.assertEqual(first, second)
        self.assertTrue(second.is_file())
        # Журнал обязан сказать именно «устаревшее», а не «нет превью». Раньше
        # здесь стояло второе, и это была ложь оператору: картинка есть, её
        # показывают, она просто снята с прошлой версии шаблона. Оператор шёл
        # искать пустую карточку в галерее, не находил и не узнавал главного —
        # что пользователи видят не тот дизайн, который лежит на диске.
        self.assertTrue(
            any("STALE preview" in line for line in logs.output),
            f"журнал не назвал превью устаревшим: {logs.output}",
        )
        self.assertFalse(
            any("has no preview" in line for line in logs.output),
            "журнал говорит «превью нет», хотя оно есть и отдаётся",
        )

    def test_a_browser_that_writes_nothing_is_not_believed(self) -> None:
        # Chrome умеет выйти нулём, не написав файла (например, когда страница
        # не загрузилась). Верить коду возврата на слово нельзя.
        self.chrome.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        self.chrome.chmod(self.chrome.stat().st_mode | stat.S_IXUSR)

        with self.with_fake_chrome():
            with self.assertLogs(LOGGER_NAME, level="ERROR") as logs:
                info = self.registry(previews=True).get("draft")

        self.assertIsNotNone(info)
        self.assertIsNone(info.preview_file)
        self.assertTrue(any("preview" in line for line in logs.output))

    def test_a_failing_browser_does_not_drop_the_template(self) -> None:
        self.chrome.write_text("#!/bin/sh\necho 'tab crashed' >&2\nexit 21\n", encoding="utf-8")
        self.chrome.chmod(self.chrome.stat().st_mode | stat.S_IXUSR)

        with self.with_fake_chrome():
            with self.assertLogs(LOGGER_NAME, level="ERROR") as logs:
                listed = self.registry(previews=True).list()

        self.assertEqual([i.key for i in listed], ["draft"])
        self.assertTrue(any("21" in line for line in logs.output))

    def test_staging_puts_the_page_next_to_its_assets(self) -> None:
        # Chrome разрешает href="styles.css" относительно самой страницы: HTML
        # обязан лежать рядом с ресурсами шаблона, иначе снимок выйдет без
        # оформления и без шрифтов, а мы этого даже не заметим — PNG-то будет.
        staged = templates_module.stage_page(
            self.templates_dir / "draft", "<html>тест</html>", self.root / "staged"
        )

        self.assertEqual(staged.name, "index.html")
        self.assertTrue((staged.parent / STYLES_FILENAME).is_file())
        self.assertTrue((staged.parent / "fonts" / "deck.woff2").is_file())

    def test_the_template_directory_is_never_written_to(self) -> None:
        # Страница печатается из КОПИИ каталога во временной папке: templates/ —
        # это исходники в репозитории, а в проде каталог бывает read-only.
        before = sorted(p.name for p in (self.templates_dir / "draft").rglob("*"))

        with self.with_fake_chrome():
            self.registry(previews=True).list()

        after = sorted(p.name for p in (self.templates_dir / "draft").rglob("*"))
        self.assertEqual(before, after)


@unittest.skipIf(shutil.which("google-chrome-stable") is None, "нет настоящего Chrome")
class RealPreviewTests(RegistryTestCase):
    """Одна проверка сквозь настоящий браузер: превью действительно PNG."""

    def test_chrome_draws_the_fixture_into_a_png(self) -> None:
        self.make_template("draft")
        self.write_manifest("draft")

        info = TemplateRegistry(self.templates_dir, self.preview_dir).get("draft")

        self.assertIsNotNone(info.preview_file, "превью не нарисовалось")
        self.assertEqual(info.preview_file.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
        # Пустая картинка тоже PNG; 16:9 при 1600px весит заметно больше.
        self.assertGreater(info.preview_file.stat().st_size, 1000)


class ShippedTemplatesTests(unittest.TestCase):
    """Комплект, который лежит в репозитории и уезжает в образ.

    Пропускается, пока каталогов v2 в репозитории нет: комплект кладёт
    параллельная работа, и тесты не должны зависеть от того, успела ли она.
    """

    def test_default_directory_is_next_to_backend_data(self) -> None:
        directory = default_templates_dir()

        self.assertEqual(directory.name, "presentations")
        self.assertEqual(directory.parent.name, "templates")
        # Каталог шаблонов — сосед data/, а не его содержимое: data монтируется
        # томом и переживает выкладку, шаблоны обязаны приезжать с кодом.
        self.assertTrue((directory.parent.parent / "data").exists())

    def test_previews_live_in_data_not_in_the_repository(self) -> None:
        # Превью — машинный результат и кэш (он зависит ещё и от версии Chrome
        # на машине), поэтому ему место в data/, которую не жалко потерять, а не
        # в templates/, где лежат исходники.
        preview_dir = default_preview_dir()

        self.assertEqual(preview_dir.parent.name, "data")
        self.assertFalse(
            default_templates_dir() in preview_dir.parents,
            "превью не должны писаться внутрь каталога шаблонов",
        )

    def _require_shipped_bundle(self) -> None:
        manifest = default_templates_dir() / MANIFEST_FILENAME
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            raise unittest.SkipTest(f"{manifest} ещё не в формате v2")
        if not isinstance(data, dict) or not data.get("templates"):
            raise unittest.SkipTest(f"{manifest} ещё не в формате v2")

    def test_shipped_manifest_and_directories_are_consistent(self) -> None:
        # Единственная проверка на настоящие файлы: она ловит момент, когда
        # манифест и каталоги шаблонов в репозитории разъехались.
        self._require_shipped_bundle()

        infos = template_registry.list()

        self.assertGreaterEqual(len(infos), 1)
        for info in infos:
            with self.subTest(template=info.key):
                self.assertEqual(set(info.name), set(NAME_LANGUAGES))
                self.assertTrue(info.html_file.is_file())
                self.assertTrue(info.css_file.is_file())
                self.assertIs(template_registry.get(info.key), info)

    def test_no_shipped_entry_is_rejected(self) -> None:
        self._require_shipped_bundle()
        manifest = default_templates_dir() / MANIFEST_FILENAME
        entries = json.loads(manifest.read_text(encoding="utf-8"))["templates"]

        self.assertEqual(
            sorted(entry["key"] for entry in entries),
            sorted(info.key for info in template_registry.list()),
        )


if __name__ == "__main__":
    unittest.main()
