"""Печать колоды: настоящий PDF, ёмкость слайда источников и зависший браузер.

Два класса задач, и они требуют разных инструментов.

НАСТОЯЩИЙ CHROME. Всё, что касается вёрстки, проверяется по ГОТОВОМУ PDF, а не
по HTML: при печати Chrome раскладывает содержимое иначе, чем на экране, и
запись, видимая в браузере, в файле может отсутствовать бесследно. Тест,
смотрящий в разметку, подтвердил бы вёрстку, которой в файле нет, — а
пользователь скачивает именно файл. Отсюда PyMuPDF и проверки «сколько страниц»
и «какой текст на последней».

ПОДДЕЛЬНЫЙ CHROME. Настоящий браузер не умеет зависать по заказу, не плодит
детей на команду и не падает с нужным кодом возврата. Всё это — поведение,
ради которого в рендерере есть стадийный таймаут и убийство ГРУППЫ процессов, и
проверить его можно только скриптом, который делает ровно то, что попросили.

Отдельно закрепляется убийство группы: Chrome — это дерево процессов, и
убийство одного родителя оставляет детей живыми и занимающими память. Тест
заводит подделку, которая плодит ребёнка, и требует, чтобы после таймаута умер
и он.
"""

import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.modules.presentations import renderer as renderer_module  # noqa: E402
from app.modules.presentations.constants import SOURCES_MORE  # noqa: E402
from app.modules.presentations.llm_schemas import (  # noqa: E402
    RENDERER_ADDED_SLIDES,
    SLIDE_LAYOUTS,
)
from app.modules.presentations.renderer import (  # noqa: E402
    SOURCE_FIT_BUDGET,
    SOURCE_FIT_MEASURED,
    RenderError,
    RenderedSource,
    render_presentation,
    source_cost,
)
from render_fixtures import (  # noqa: E402
    SOURCE_NAME_STYLES,
    TEMPLATE_KEYS,
    make_slide,
    make_slides,
    make_sources,
    maxed_slide_payload,
    pdf_is_a_pdf,
    pdf_pages,
    pdf_text,
    pdf_visible_text,
    real_chromium_available,
    structure_texts,
    use_fake_chromium,
    use_offline_registry,
)

CREATED_AT = datetime(2026, 8, 4, 9, 15, tzinfo=timezone.utc)

# Сколько последних знаков имени считать «последней строкой». Ровно строку взять
# неоткуда — где браузер перенесёт, зависит от шрифта и ширины колонки, — но
# хвост в 14 знаков заведомо в неё не помещается целиком только у самых узких
# колонок, а там он захватит ещё и предыдущую строку: обе видны или обеих нет.
# Ошибка тут возможна только в сторону строгости, и это правильная сторона.
SOURCE_NAME_TAIL = 14

# Двенадцать таджикских букв, которых нет в большинстве кириллических шрифтов.
# Если шаблон растерял свои .woff2, Chrome молча подставит системный шрифт, и в
# PDF на их месте окажутся пустые прямоугольники — то есть текста не будет
# вовсе. Извлечение текста из PDF это ловит: подставленный глиф в поток текста
# не попадает.
TAJIK_GLYPHS = "ӣ ӯ қ ҳ ҷ ғ Ӣ Ӯ Қ Ҳ Ҷ Ғ"

# Известные расхождения предела схемы с вёрсткой: пара (шаблон, раскладка) и
# причина, по которой проверка «предельный слайд целиком на листе» её пока не
# требует. Список именно ЗДЕСЬ, а не в виде удалённой проверки: пропуск с
# причиной виден в каждом прогоне, а вычеркнутая строка не видна нигде.
#
# ЗАМЕРЕНО (04.08.2026, слайд из SLIDE_BULLETS_MAX буллетов по
# SLIDE_BULLET_MAX_CHARS знаков плюс заголовок в SLIDE_HEADING_MAX_CHARS):
# blueprint выпускает за нижний край листа 64 знака, самый дальний — на 11.9 pt.
# Остальные девятнадцать пар «дизайн + раскладка» чисты полностью.
#
# Беда НЕ НОВАЯ: на коммите 58604d1 (до раскладок) тот же слайд на том же
# blueprint выпускал за лист 1 знак на 4.2 pt, то есть предел списка и раньше
# стоял на самой границе, а новая вёрстка эту границу перешла. Чинится это не
# здесь и не в рендере: либо кеглем и интерлиньяжем blueprint, либо пределом
# SLIDE_BULLET_MAX_CHARS в схеме. До тех пор строка ниже — единственное место,
# где расхождение записано числом.
KNOWN_OVERFLOW = {
    ("blueprint", "bullets"): (
        "blueprint не вмещает предельный список (замер: 64 знака за листом, до "
        "11.9 pt); расхождение старше раскладок, чинится в CSS или в пределе схемы"
    ),
}


def process_facts(pid: int) -> tuple[str, str] | None:
    """Состояние процесса и время его старта, или None, если процесса нет.

    Время старта берётся не из любопытства: pid'ы переиспользуются, и в длинном
    прогоне тестов это происходит регулярно. Проверка «жив ли pid» без него
    отвечает на вопрос «есть ли СЕЙЧАС процесс с таким номером», а спрашивали мы
    про другой — про тот, который убивали. Пара (pid, время старта) отличает их
    надёжно.
    """
    try:
        fields = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").rsplit(")", 1)[1]
    except (OSError, IndexError):
        return None
    parts = fields.split()
    # После отрезанного «pid (comm)» нумерация полей proc(5) начинается с 3-го:
    # состояние — первое, время старта — 22-е, то есть двадцатое отсюда.
    return parts[0], parts[19]


def process_is_gone(pid: int, started: str | None = None) -> bool:
    """Того самого процесса больше нет: он исчез, стал зомби или pid переиспользован."""
    facts = process_facts(pid)
    if facts is None:
        return True
    state, current_start = facts
    if started is not None and current_start != started:
        return True
    # Убитый, но ещё не подобранный родителем процесс остаётся в таблице и на
    # сигнал 0 отвечает как живой. Работать он при этом уже перестал.
    return state == "Z"


def wait_until_gone(pid: int, started: str | None = None, seconds: float = 5.0) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if process_is_gone(pid, started):
            return True
        time.sleep(0.05)
    return process_is_gone(pid, started)


class PrintTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="print-test-")
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.output = str(self.tmp / "deck.pdf")
        use_offline_registry(self)

    def render(self, **overrides) -> str:
        params = {
            "title": "Налоговые льготы",
            "slides": make_slides(3),
            "sources": make_sources(2),
            "language": "ru",
            "template_key": "draft",
            "notebook_name": "Налоги",
            "created_at": CREATED_AT,
            "output_path": self.output,
        }
        params.update(overrides)
        render_presentation(**params)
        return params["output_path"]


@unittest.skipUnless(
    real_chromium_available(), "на машине нет headless Chrome — печатать нечем"
)
class RealPdfTests(PrintTestCase):
    """Печать настоящим браузером: файл, число страниц, язык, шрифты."""

    def test_every_template_prints_a_real_pdf_of_the_ordered_length(self):
        """Состав файла — контракт slide_count: титул + секции + «Источники».

        Проверяется по PDF, а не по разметке: страницы расставляет Chrome, и
        именно их считает пользователь, открывший файл.
        """
        slides = make_slides(4)
        for key in TEMPLATE_KEYS:
            with self.subTest(template=key):
                path = self.render(template_key=key, slides=slides)

                self.assertTrue(pdf_is_a_pdf(path), "на выходе не PDF")
                self.assertGreater(os.path.getsize(path), 0)
                self.assertEqual(
                    pdf_pages(path), len(slides) + RENDERER_ADDED_SLIDES
                )

    def test_the_deck_carries_the_text_it_was_given(self):
        path = self.render(slides=make_slides(2))
        text = pdf_text(path)

        self.assertIn("Налоговые льготы", text)
        self.assertIn("Налоги", text)
        self.assertIn("4 августа 2026", text)
        self.assertIn("Заголовок 1", text)
        self.assertIn("Первый факт слайда 1", text)
        self.assertIn("Источники", text)

    def test_a_tajik_deck_keeps_its_glyphs_in_every_template(self):
        """Пропавший .woff2 — самая тихая поломка: Chrome подставит системный
        шрифт, а таджикские ӣ ӯ қ ҳ ҷ ғ в нём могут отсутствовать. В PDF это
        видно сразу: подставленный пустой глиф в поток текста не попадает.
        """
        slides = make_slides(1)
        slides[0].bullets = [f"Ҳамаи ҳарфҳо: {TAJIK_GLYPHS}", "Банди дуюм"]
        for key in TEMPLATE_KEYS:
            with self.subTest(template=key):
                path = self.render(
                    template_key=key,
                    language="tj",
                    slides=slides,
                    sources=[
                        RenderedSource(
                            source_id=1, name="Дастури амалӣ ӯқҳҷғ", pages=[21]
                        )
                    ],
                )
                text = pdf_text(path)

                for glyph in TAJIK_GLYPHS.split():
                    self.assertIn(glyph, text, f"{key}: глиф {glyph} не напечатан")
                self.assertIn("Манбаъҳо", text)
                self.assertIn("саҳ. 21", text)

    def test_every_source_the_renderer_kept_is_actually_visible(self):
        """Порог обрезки проверяется ПО ФАЙЛУ: все дизайны, все стили имён.

        Это главный тест ёмкости: рендерер обещает, что оставленные им записи
        видны, — и обещание проверяется там же, где его нарушение проявилось бы
        у пользователя, то есть в PDF. Порог общий для четырёх дизайнов, поэтому
        и проверка идёт по всем четырём: верный для одного, он ничего не говорит
        об остальных.

        Сколько записей проверять, спрашиваем у САМОГО порога, а не берём из
        замера рядом с ним. Иначе тест подтверждал бы замер сам собой и молчал
        бы про то, ради чего он тут стоит: про случай, когда порог отдаёт
        больше, чем влезает.

        Стилей имён четыре, потому что порог о них не знает, а вёрстка — знает.
        Имя из слов браузеру есть где перенести; у скана или выгрузки точек
        переноса нет вовсе, и именно на них порог врал сильнее всего. Проверка
        на одних словах прошла бы на шаблоне, теряющем половину списка.

        Критерий «видна» строгий вдвойне: видны должны быть и метка, и КОНЕЦ
        имени, и оба — ВНУТРИ листа (pdf_visible_text, а не pdf_text). Запись,
        у которой видно только начало, уехала под нижний край; запись, чей текст
        «находится» в файле, но лежит за MediaBox, не видна вовсе. Считать любую
        из них поместившейся значит закрепить ровно ту потерю, ради которой
        порог и заведён.
        """
        for name_length, _ in SOURCE_FIT_MEASURED:
            kept = SOURCE_FIT_BUDGET // source_cost(name_length)
            for style in SOURCE_NAME_STYLES:
                sources = make_sources(kept, name_length=name_length, style=style)
                # Сторож сторожа. Всё, что проверяется ниже, держится на том,
                # что хвост принадлежит ОДНОЙ записи: с одинаковыми окончаниями
                # хвост уехавшей за лист записи находился бы у соседа, и тест
                # проходил бы на сломанной вёрстке. Так уже было.
                self.assertEqual(
                    len({source.name[-SOURCE_NAME_TAIL:] for source in sources}),
                    kept,
                    f"{style}/{name_length}: хвосты имён в оснастке повторяются",
                )
                for key in TEMPLATE_KEYS:
                    with self.subTest(
                        template=key, name_length=name_length, style=style
                    ):
                        path = self.render(
                            template_key=key, sources=sources, slides=make_slides(1)
                        )
                        seen = pdf_visible_text(path, page=pdf_pages(path) - 1)

                        for index, source in enumerate(sources, start=1):
                            tail = source.name[-SOURCE_NAME_TAIL:].replace(" ", "")
                            self.assertIn(
                                f"[{index}]",
                                seen,
                                f"{key}/{style}/{name_length}: порог оставил "
                                f"{kept} записей, а метки [{index}] на листе нет",
                            )
                            self.assertIn(
                                tail,
                                seen,
                                f"{key}/{style}/{name_length}: порог оставил "
                                f"{kept} записей, а конец имени № {index} "
                                f"({tail!r}) за пределами листа",
                            )

    def test_the_tail_appears_only_when_something_did_not_fit(self):
        """Обе стороны хвоста «не показано ещё N» — на всех четырёх шаблонах.

        Без второй половины тест не отличил бы работающий шаблон от того, что
        печатает хвост всегда: колода с полным списком уверяла бы пользователя,
        что список неполон.
        """
        capacity = SOURCE_FIT_BUDGET // source_cost(60)
        overflowing = make_sources(capacity + 4, name_length=60)
        fitting = make_sources(capacity, name_length=60)
        marker = SOURCES_MORE["ru"].split("{count}")[0].strip()

        for key in TEMPLATE_KEYS:
            with self.subTest(template=key, truncated=True):
                path = self.render(template_key=key, sources=overflowing)
                text = pdf_text(path, page=pdf_pages(path) - 1)

                self.assertIn(marker, text)
                self.assertIn(SOURCES_MORE["ru"].format(count=4), text)

            with self.subTest(template=key, truncated=False):
                path = self.render(template_key=key, sources=fitting)
                text = pdf_text(path, page=pdf_pages(path) - 1)

                self.assertNotIn(marker, text)

    def test_user_text_cannot_become_markup(self):
        """Разметка из ответа модели остаётся текстом.

        Экранирует Jinja (autoescape=True в окружении реестра), и своего
        экранирования у рендерера нет и не должно быть. Но именно на этом
        свойстве держится решение печатать без песочницы (--no-sandbox), поэтому
        оно проверяется на настоящем файле: строка попала на слайд как текст, а
        не была исполнена и не исчезла из вывода.
        """
        slides = make_slides(1)
        slides[0].heading = "R&D <script>alert(1)</script>"
        slides[0].bullets = ['Кавычки "ёлочки" & <b>жирный</b>', "Второй факт"]

        path = self.render(slides=slides)
        text = pdf_text(path)

        self.assertIn("<script>alert(1)</script>", text)
        self.assertIn("<b>жирный</b>", text)

    def test_a_slide_filled_to_the_schema_limits_stays_on_the_sheet(self):
        """Пределы схемы сходятся с вёрсткой: предельный слайд ВЕСЬ на листе.

        Рендерер не режет ничего из написанного моделью — за длину отвечает
        схема, и её числа выведены из места на слайде (см. llm_schemas). Это
        обещание двух сторон друг другу, и до сих пор его никто не проверял:
        слайд, набитый до пределов схемы, собирается и печатается одинаково
        успешно и когда он помещается на лист, и когда его нижняя половина ушла
        под край. `overflow: hidden` режет молча — ровно та беда, ради которой у
        слайда «Источники» заведён порог обрезки, только здесь резать нельзя, и
        сходиться обязаны ЧИСЛА.

        Проверяются все четыре дизайна: пределы одни на всех, и верный для
        одного ничего не говорит про остальные три. Критерий тот же, что у
        источников, и такой же строгий: знаки обязаны лежать ВНУТРИ MediaBox
        (pdf_visible_text, а не pdf_text — Chrome при печати не выбрасывает
        переполнившее содержимое, а рисует его за границей страницы), а у
        каждого поля свой неповторимый хвост из двух знаков: строка из
        одинаковых букв «находилась» бы на листе, даже уехав с него целиком.

        Регистр не учитывается: дизайны набирают подписи капителью
        (text-transform: uppercase), и в поток текста PDF они попадают уже
        прописными. Это решение вёрстки, а спрашиваем мы про видимость.
        """
        slides = [
            make_slide(maxed_slide_payload(layout)) for layout in SLIDE_LAYOUTS
        ]
        for key in TEMPLATE_KEYS:
            path = self.render(template_key=key, slides=slides, sources=make_sources(1))
            for number, layout in enumerate(SLIDE_LAYOUTS, start=1):
                seen = pdf_visible_text(path, page=number).casefold()
                for text in structure_texts(maxed_slide_payload(layout)):
                    with self.subTest(template=key, layout=layout, tail=text[-2:]):
                        if (key, layout) in KNOWN_OVERFLOW:
                            self.skipTest(KNOWN_OVERFLOW[(key, layout)])
                        self.assertIn(
                            text.replace(" ", "").casefold(),
                            seen,
                            f"{key}/{layout}: поле с хвостом {text[-2:]!r} на "
                            f"листе {number} видно не целиком — предел схемы и "
                            f"вёрстка разошлись",
                        )


class UnknownTemplateTests(PrintTestCase):
    def test_a_missing_template_is_a_render_error_not_a_default_theme(self):
        """Отката на «тему по умолчанию» у HTML-рендера нет и быть не может.

        У прежнего pptx-рендерера он был: python-pptx умеет собрать файл без
        шаблона. Здесь без каталога шаблона нет ни вёрстки, ни шрифтов, и
        «колода на чём-нибудь» означала бы пустую страницу вместо честного
        отказа.
        """
        with self.assertRaises(RenderError) as caught:
            self.render(template_key="no-such-template")

        self.assertIn("no-such-template", str(caught.exception))
        self.assertFalse(os.path.exists(self.output))


class HangingBrowserTests(PrintTestCase):
    """Зависший Chrome: свой таймаут, убитая группа, диагностика в отказе."""

    def setUp(self) -> None:
        super().setUp()
        self.child_pid_file = self.tmp / "child.pid"
        self.child_stat_file = self.tmp / "child.stat"

    def use_chromium(self, body: str) -> None:
        use_fake_chromium(self, self.tmp / "fake-chrome", body)

    def test_a_hanging_browser_is_stopped_by_the_stage_budget(self):
        """Печать снимает СВОЙ таймаут, а не потолок джобы.

        Бюджет теста — доли секунды, подделка висит минуту: если бы зависание
        ждал кто-то снаружи, тест не уложился бы и близко.
        """
        self.use_chromium("sleep 60")

        started = time.perf_counter()
        with patch.object(renderer_module, "RENDER_PRINT_TIMEOUT", 0.5):
            with self.assertRaises(RenderError) as caught:
                self.render()
        elapsed = time.perf_counter() - started

        self.assertLess(elapsed, 20.0, "зависание ждал не бюджет стадии")
        self.assertIn("не уложилась", str(caught.exception))
        self.assertFalse(os.path.exists(self.output))

    def test_the_whole_process_group_dies_not_just_the_parent(self):
        """Chrome — дерево процессов, и убивать надо дерево.

        Убийство одного родителя оставляет детей живыми, осиротевшими и
        по-прежнему держащими память и /dev/shm; на стенде это копится до
        первого OOM. Подделка здесь плодит ребёнка ровно так же, как настоящий
        браузер плодит рендерер и zygote.
        """
        # Подделка сама записывает и pid ребёнка, и его /proc/PID/stat — то есть
        # снимок ребёнка, снятый пока он ЖИВ. Без времени старта из этого снимка
        # проверка «процесса больше нет» отвечала бы на вопрос «есть ли сейчас
        # процесс с таким номером», а pid'ы в длинном прогоне переиспользуются.
        self.use_chromium(
            f"sleep 60 &\n"
            f"echo $! > {self.child_pid_file}\n"
            f"cat /proc/$!/stat > {self.child_stat_file}\n"
            f"sleep 60"
        )

        with patch.object(renderer_module, "RENDER_PRINT_TIMEOUT", 1.0):
            with self.assertRaises(RenderError):
                self.render()

        child_pid = int(self.child_pid_file.read_text().strip())
        child_started = self.child_stat_file.read_text().rsplit(")", 1)[1].split()[19]
        self.assertTrue(
            wait_until_gone(child_pid, child_started),
            f"ребёнок браузера (pid {child_pid}) пережил убийство группы",
        )

    def test_the_failure_carries_the_tail_of_the_browser_output(self):
        """Хвост stderr — единственная диагностика, объясняющая отказ печати.

        Он берётся именно ХВОСТОМ: Chrome многословен, и настоящую причину
        («не нашёл страницу», «упала вкладка») он пишет последней строкой.
        """
        self.use_chromium(
            "echo 'Fatal: the tab crashed for real' >&2\n"
            "sleep 60"
        )

        with patch.object(renderer_module, "RENDER_PRINT_TIMEOUT", 0.7):
            with self.assertRaises(RenderError) as caught:
                self.render()

        self.assertIn("the tab crashed for real", str(caught.exception))

    def test_environment_noise_does_not_crowd_out_the_real_cause(self):
        """«Failed to adjust OOM score» Chrome пишет и при полном успехе.

        Это шум контейнера без CAP_SYS_RESOURCE, а не диагностика. Пропущенный в
        отказ, он вытеснил бы настоящую причину из предела error_text — то есть
        пользователь и администратор увидели бы ровно то, что ничего не значит.
        """
        noise = "\n".join(
            f"echo 'Failed to adjust OOM score of renderer with pid {pid}' >&2"
            for pid in range(100, 140)
        )
        self.use_chromium(f"{noise}\necho 'Fatal: no such file' >&2\nsleep 60")

        with patch.object(renderer_module, "RENDER_PRINT_TIMEOUT", 0.7):
            with self.assertRaises(RenderError) as caught:
                self.render()

        message = str(caught.exception)
        self.assertIn("Fatal: no such file", message)
        self.assertNotIn("Failed to adjust OOM score", message)


class BrokenBrowserTests(PrintTestCase):
    """Браузер отработал, но колоды нет: код возврата и пустой выход."""

    def use_chromium(self, body: str) -> None:
        use_fake_chromium(self, self.tmp / "fake-chrome", body)

    def test_a_nonzero_exit_code_is_a_render_error(self):
        self.use_chromium("echo 'Fatal: cannot create user data dir' >&2\nexit 21")

        with self.assertRaises(RenderError) as caught:
            self.render()

        self.assertIn("21", str(caught.exception))
        self.assertIn("cannot create user data dir", str(caught.exception))

    def test_success_without_a_file_is_still_a_failure(self):
        """Chrome умеет выйти с нулём, не написав PDF.

        Так бывает, когда страница не загрузилась. Поверить коду возврата на
        слово значит записать заказ в 'ready' с пустым файлом — то есть отказ на
        скачивании уже после «готово».
        """
        self.use_chromium("exit 0")

        with self.assertRaises(RenderError) as caught:
            self.render()

        self.assertIn("PDF не появился", str(caught.exception))

    def test_an_empty_pdf_is_not_a_deck_either(self):
        self.use_chromium("exit 0")
        Path(self.output).write_bytes(b"")

        with self.assertRaises(RenderError):
            self.render()


class KillGroupUnitTests(unittest.TestCase):
    """Убийство группы отдельно от рендера: сигнал и сбор вывода."""

    def test_the_group_is_killed_and_stderr_is_collected_after_that(self):
        """Диагностика забирается ПОСЛЕ убийства, и это не мелочь.

        Пока труба открыта хоть одним процессом группы, читать её до конца
        нельзя — communicate() ждал бы. SIGKILL по группе закрывает её у всех
        сразу, поэтому чтение после убийства не блокируется.
        """
        with tempfile.TemporaryDirectory(prefix="killgroup-") as workspace:
            marker = Path(workspace) / "spoke"
            process = subprocess.Popen(
                # Отметка на диске, а не sleep перед убийством: тест обязан
                # убивать процесс ПОСЛЕ того, как тот успел сказать своё, иначе
                # он проверяет скорость запуска sh, а не сбор диагностики.
                ["sh", "-c", f"echo 'last words' >&2; : > {marker}; sleep 60 & sleep 60"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
            )
            self.addCleanup(process.kill)
            deadline = time.monotonic() + 10.0
            while not marker.exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertTrue(marker.exists(), "подделка не успела запуститься")
        started = process_facts(process.pid)[1]

        stderr = renderer_module.kill_process_group(process)

        self.assertIn("last words", stderr)
        self.assertEqual(process.poll(), -signal.SIGKILL)
        self.assertTrue(wait_until_gone(process.pid, started))
        # Что умерла ВСЯ группа, а не только родитель, проверяет соседний тест
        # на дереве процессов: там ребёнок опознан по времени старта. Спросить
        # об этом группу (killpg с сигналом 0) нельзя — номер группы равен pid
        # её лидера, а pid'ы переиспользуются, и в длинном прогоне такой вопрос
        # однажды получает ответ про чужой процесс.


if __name__ == "__main__":
    unittest.main()
