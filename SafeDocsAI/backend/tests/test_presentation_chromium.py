"""Headless Chrome как внешний инструмент: поиск, проверка, аргументы.

Что закрепляем.

  * **Проверяется ответ, а не наличие файла.** Бинарник может лежать на месте и
    при этом не запускаться: потерянный бит +x, обрубленная установка, каталог
    вместо файла, заглушка-обёртка, отвечающая пустотой. Все эти случаи —
    существующий путь, которым ничего не напечатать, и все они обязаны
    отбиваться до того, как пользователь закажет колоду.
  * **Отказ громкий.** ERROR в журнал плюс исключение своего типа. Тихий None
    означал бы, что вызывающий волен продолжить, а продолжать нечем.
  * **Старт не падает.** log_chromium_state отвечает статусом при любой беде:
    отсутствие браузера ломает один сценарий, а отказ на старте отобрал бы и
    админ-панель, и логи — то, чем этот стенд чинят.
  * **--no-sandbox на месте и объяснён.** Это уступка в безопасности, принятая
    осознанно (контейнер не даёт user namespaces), и тест следит, чтобы она не
    исчезла случайно вместе с рефакторингом списка флагов.
  * **Никаких походов в сеть.** Флаги, отключающие фоновые соединения, — часть
    того же обещания, что и стартовый линт на http:// в шаблонах.

Настоящий Chrome нужен ровно одному тесту, и тот пропускается, если браузера на
машине нет. Остальные работают с бинарником-подделкой: shell-скрипт, который
отвечает на --version что попросили. Так тест проверяет НАШУ логику разбора
ответа, а не версию Chrome на конкретной машине.
"""

import os
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

# Каталог с тестами в sys.path: unittest discover кладёт его туда сам, но при
# запуске `python -m unittest tests.test_presentation_chromium` этого не происходит.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.modules.presentations import chromium as chromium_module  # noqa: E402
from app.modules.presentations.chromium import (  # noqa: E402
    CHROMIUM_BINARY_ENV,
    CHROMIUM_CANDIDATES,
    ChromiumStatus,
    RendererUnavailable,
    chromium_status,
    describe_failure,
    ensure_chromium_available,
    find_chromium,
    log_chromium_state,
    pdf_command,
    probe_chromium,
    screenshot_command,
)

LOGGER_NAME = "app.modules.presentations.chromium"


def write_fake_binary(directory: Path, name: str, body: str, executable: bool = True) -> Path:
    """Скрипт, притворяющийся браузером ровно настолько, насколько нужно тесту."""
    path = directory / name
    path.write_text(f"#!/bin/sh\n{body}\n", encoding="utf-8")
    if executable:
        path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


class TempDirTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="chromium-test-")
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        # Кэш статуса живёт на уровне модуля: без сброса второй тест получил бы
        # ответ первого и проверял бы не то, что настроил.
        self.addCleanup(setattr, chromium_module, "_status", None)
        chromium_module._status = None


class FindingTheBinaryTests(TempDirTestCase):
    def test_environment_variable_wins_over_path(self) -> None:
        named = write_fake_binary(self.tmp, "my-chrome", "echo 'Chromium 1.0'")

        with patch.dict(os.environ, {CHROMIUM_BINARY_ENV: str(named)}):
            self.assertEqual(find_chromium(), str(named))

    def test_named_but_missing_binary_is_not_replaced_by_a_search(self) -> None:
        # Админ назвал бинарник, а его там нет — это ошибка конфигурации.
        # Подставить вместо него другой браузер значит скрыть её и печатать не
        # тем, чем просили: вывод молча поедет на другом движке.
        missing = self.tmp / "no-such-chrome"

        with patch.dict(os.environ, {CHROMIUM_BINARY_ENV: str(missing)}):
            self.assertIsNone(find_chromium())

    def test_path_is_searched_when_the_variable_is_empty(self) -> None:
        write_fake_binary(self.tmp, CHROMIUM_CANDIDATES[0], "echo 'Chromium 1.0'")

        with patch.dict(os.environ, {CHROMIUM_BINARY_ENV: "  ", "PATH": str(self.tmp)}):
            found = find_chromium()

        self.assertEqual(Path(found).name, CHROMIUM_CANDIDATES[0])

    def test_candidates_are_a_list_not_a_hardcoded_name(self) -> None:
        # Пакет зовётся google-chrome-stable на стенде и chromium в debian-slim.
        # Одно захардкоженное имя означало бы «не запускается в проде».
        self.assertIn("google-chrome-stable", CHROMIUM_CANDIDATES)
        self.assertIn("chromium", CHROMIUM_CANDIDATES)
        self.assertEqual(CHROMIUM_CANDIDATES[0], "google-chrome-stable")


class ProbeTests(TempDirTestCase):
    def _probe_with(self, binary: Path | str) -> ChromiumStatus:
        with patch.dict(os.environ, {CHROMIUM_BINARY_ENV: str(binary)}):
            return probe_chromium()

    def test_a_working_binary_is_available_with_its_version(self) -> None:
        binary = write_fake_binary(self.tmp, "chrome", "echo 'Chromium 128.0.0.0'")

        status = self._probe_with(binary)

        self.assertTrue(status.available)
        self.assertEqual(status.version, "Chromium 128.0.0.0")
        self.assertIsNone(status.error)

    def test_a_missing_binary_is_unavailable_and_names_the_variable(self) -> None:
        status = self._probe_with(self.tmp / "nope")

        self.assertFalse(status.available)
        self.assertIn(CHROMIUM_BINARY_ENV, status.error)

    def test_a_file_without_the_execute_bit_is_unavailable(self) -> None:
        # Ровно тот случай, ради которого проверяется ОТВЕТ, а не существование:
        # файл на месте, is_file() истинно, запустить нельзя.
        binary = write_fake_binary(self.tmp, "chrome", "echo 'Chromium 1.0'", executable=False)

        status = self._probe_with(binary)

        self.assertFalse(status.available)
        self.assertIn("cannot be executed", status.error)

    def test_a_directory_named_like_a_binary_is_unavailable(self) -> None:
        directory = self.tmp / "chrome"
        directory.mkdir()

        status = self._probe_with(directory)

        self.assertFalse(status.available)

    def test_a_binary_that_fails_is_unavailable_and_quotes_the_exit_code(self) -> None:
        binary = write_fake_binary(
            self.tmp, "chrome", "echo 'error while loading shared libraries' >&2\nexit 127"
        )

        status = self._probe_with(binary)

        self.assertFalse(status.available)
        self.assertIn("127", status.error)
        self.assertIn("shared libraries", status.error)

    def test_a_binary_answering_nonsense_is_unavailable(self) -> None:
        # Обёртка-заглушка (snap-stub, чужой скрипт с тем же именем) умеет выйти
        # нулём и написать что угодно. «Ноль» не означает «это браузер».
        binary = write_fake_binary(self.tmp, "chrome", "echo 'hello there'")

        status = self._probe_with(binary)

        self.assertFalse(status.available)
        self.assertIn("does not look like", status.error)

    def test_a_silent_binary_is_unavailable(self) -> None:
        binary = write_fake_binary(self.tmp, "chrome", "exit 0")

        status = self._probe_with(binary)

        self.assertFalse(status.available)

    def test_version_may_arrive_on_stderr(self) -> None:
        # Некоторые сборки пишут версию в stderr; это всё ещё рабочий браузер.
        binary = write_fake_binary(self.tmp, "chrome", "echo 'Chromium 120.0' >&2")

        status = self._probe_with(binary)

        self.assertTrue(status.available)


class EnsureTests(TempDirTestCase):
    def test_a_working_binary_is_returned(self) -> None:
        binary = write_fake_binary(self.tmp, "chrome", "echo 'Chromium 1.0'")

        with patch.dict(os.environ, {CHROMIUM_BINARY_ENV: str(binary)}):
            self.assertEqual(ensure_chromium_available(), str(binary))

    def test_a_missing_binary_raises_and_shouts_into_the_log(self) -> None:
        with patch.dict(os.environ, {CHROMIUM_BINARY_ENV: str(self.tmp / "nope")}):
            with self.assertLogs(LOGGER_NAME, level="ERROR") as logs:
                with self.assertRaises(RendererUnavailable):
                    ensure_chromium_available()

        self.assertTrue(any("unavailable" in line for line in logs.output))

    def test_an_unexecutable_binary_raises(self) -> None:
        binary = write_fake_binary(self.tmp, "chrome", "echo hi", executable=False)

        with patch.dict(os.environ, {CHROMIUM_BINARY_ENV: str(binary)}):
            with self.assertLogs(LOGGER_NAME, level="ERROR"):
                with self.assertRaises(RendererUnavailable):
                    ensure_chromium_available()

    def test_the_exception_carries_the_reason(self) -> None:
        with patch.dict(os.environ, {CHROMIUM_BINARY_ENV: str(self.tmp / "nope")}):
            with self.assertLogs(LOGGER_NAME, level="ERROR"):
                with self.assertRaises(RendererUnavailable) as caught:
                    ensure_chromium_available()

        # HTTP-слой навесит на это исключение свой код; текст нужен админу в
        # журнале, поэтому пустым он быть не может.
        self.assertTrue(str(caught.exception))

    def test_the_answer_is_cached_until_asked_to_recheck(self) -> None:
        binary = write_fake_binary(self.tmp, "chrome", "echo 'Chromium 1.0'")

        with patch.dict(os.environ, {CHROMIUM_BINARY_ENV: str(binary)}):
            first = chromium_status()
            binary.unlink()
            self.assertTrue(chromium_status().available)
            # Список шаблонов зовут часто, и каждый вызов не должен
            # превращаться в fork; перепроверку заказывают явно.
            self.assertIs(chromium_status(), first)
            self.assertFalse(chromium_status(force=True).available)


class StartupCheckTests(TempDirTestCase):
    def test_a_working_binary_is_reported_once(self) -> None:
        binary = write_fake_binary(self.tmp, "chrome", "echo 'Chromium 1.0'")

        with patch.dict(os.environ, {CHROMIUM_BINARY_ENV: str(binary)}):
            with self.assertLogs(LOGGER_NAME, level="INFO") as logs:
                status = log_chromium_state()

        self.assertTrue(status.available)
        self.assertTrue(any("Chromium 1.0" in line for line in logs.output))

    def test_a_missing_binary_is_an_error_but_not_an_exception(self) -> None:
        with patch.dict(os.environ, {CHROMIUM_BINARY_ENV: str(self.tmp / "nope")}):
            with self.assertLogs(LOGGER_NAME, level="ERROR") as logs:
                status = log_chromium_state()

        self.assertFalse(status.available)
        self.assertTrue(any("unavailable" in line for line in logs.output))

    def test_even_a_broken_probe_returns_a_status(self) -> None:
        # Старт не должен падать ни от чего, включая беду в самой проверке.
        with patch.object(chromium_module, "probe_chromium", side_effect=OSError("boom")):
            with self.assertLogs(LOGGER_NAME, level="ERROR"):
                status = log_chromium_state()

        self.assertFalse(status.available)

    def test_the_status_can_be_turned_into_a_refusal(self) -> None:
        # Формулировка результата, которой воспользуется эндпоинт: сам решает,
        # отвечать 503 или показывать список, а причина уже в руках.
        status = ChromiumStatus(available=False, binary=None, version=None, error="нет")

        with self.assertRaises(RendererUnavailable):
            status.require()

        ok = ChromiumStatus(available=True, binary="/x", version="Chromium", error=None)
        self.assertEqual(ok.require(), "/x")


class CommandTests(unittest.TestCase):
    def test_pdf_command_prints_to_the_asked_file(self) -> None:
        command = pdf_command(
            "/usr/bin/chrome", Path("/tmp/deck.html"), Path("/tmp/deck.pdf"), Path("/tmp/p")
        )

        self.assertEqual(command[0], "/usr/bin/chrome")
        self.assertIn("--print-to-pdf=/tmp/deck.pdf", command)
        self.assertIn("--no-pdf-header-footer", command)
        self.assertEqual(command[-1], "file:///tmp/deck.html")

    def test_screenshot_command_sets_an_explicit_window(self) -> None:
        # Без размера окна Chrome берёт 800x600, и превью шаблона, свёрстанного
        # под 16:9, приезжает обрезанным.
        command = screenshot_command(
            "/usr/bin/chrome",
            Path("/tmp/p.html"),
            Path("/tmp/p.png"),
            Path("/tmp/p"),
            (1600, 900),
        )

        self.assertIn("--window-size=1600,900", command)
        self.assertIn("--screenshot=/tmp/p.png", command)

    def test_paths_with_spaces_and_cyrillic_become_a_valid_url(self) -> None:
        # Во временный каталог рендера попадает имя блокнота; «file://» + str
        # отдал бы Chrome строку с пробелами, и тот открыл бы не тот файл.
        command = pdf_command(
            "/usr/bin/chrome",
            Path("/tmp/мой блокнот/deck.html"),
            Path("/tmp/out.pdf"),
            Path("/tmp/p"),
        )

        url = command[-1]
        self.assertTrue(url.startswith("file:///"))
        self.assertNotIn(" ", url)
        self.assertIn("%20", url)

    def test_the_sandbox_concession_is_present_and_deliberate(self) -> None:
        # Контейнер не отдаёт user namespaces, и без флага Chrome умирает на
        # старте («Failed to move to new namespace»). Компенсация: в браузер
        # попадает только наш HTML, собранный Jinja с autoescape, без JS и без
        # внешних ресурсов. Тест стоит здесь, чтобы флаг не исчез случайно при
        # уборке списка аргументов — убирать его можно, только когда изменится
        # среда запуска.
        command = pdf_command("/x", Path("/a.html"), Path("/b.pdf"), Path("/p"))

        self.assertIn("--no-sandbox", command)
        self.assertTrue(any(arg.startswith("--headless") for arg in command))

    def test_each_run_gets_its_own_profile(self) -> None:
        # Без личного --user-data-dir второй Chrome передаёт задание первому и
        # выходит нулём, не написав файла.
        command = pdf_command("/x", Path("/a.html"), Path("/b.pdf"), Path("/profiles/one"))

        self.assertIn("--user-data-dir=/profiles/one", command)

    def test_background_networking_is_off(self) -> None:
        # Шаблоны без внешних ресурсов по построению; браузер тоже не должен
        # никуда ходить, иначе рендер зависит от наличия сети.
        command = pdf_command("/x", Path("/a.html"), Path("/b.pdf"), Path("/p"))

        self.assertIn("--disable-background-networking", command)
        self.assertIn("--disable-component-update", command)


class FailureDescriptionTests(unittest.TestCase):
    def test_environment_noise_does_not_crowd_out_the_real_cause(self) -> None:
        # Chrome сыплет «Failed to adjust OOM score» даже на успешной печати;
        # если пустить хвост как есть, настоящая причина в лог не попадёт.
        stderr = (
            "[1:1] ERROR:zygote_host_impl_linux.cc:279] Failed to adjust OOM score of "
            "renderer with pid 42: Permission denied (13)\n"
            "ERROR:Failed to connect to the bus: Could not parse server address\n"
            "[1:1] ERROR: Cannot open file /tmp/deck.html\n"
            "[1:1] ERROR:zygote_host_impl_linux.cc:279] Failed to adjust OOM score of "
            "renderer with pid 43: Permission denied (13)\n"
        )

        described = describe_failure(["/usr/bin/google-chrome-stable"], 21, stderr)

        self.assertIn("Cannot open file", described)
        self.assertNotIn("OOM score", described)
        self.assertIn("21", described)

    def test_silence_is_reported_as_silence(self) -> None:
        described = describe_failure(["/usr/bin/chrome"], 1, "")

        self.assertIn("no diagnostics", described)


@unittest.skipIf(
    find_chromium() is None, "на машине нет ни одного бинарника Chrome/Chromium"
)
class RealChromiumTests(unittest.TestCase):
    """Единственный тест на настоящий браузер: наши аргументы ему понятны."""

    def setUp(self) -> None:
        chromium_module._status = None
        self.addCleanup(setattr, chromium_module, "_status", None)

    def test_the_installed_browser_answers_the_probe(self) -> None:
        status = probe_chromium()

        self.assertTrue(status.available, status.error)
        self.assertIn("chrom", status.version.lower())

    def test_it_actually_writes_a_png_with_our_arguments(self) -> None:
        binary = ensure_chromium_available()
        with tempfile.TemporaryDirectory(prefix="chromium-real-") as workspace:
            root = Path(workspace)
            page = root / "page.html"
            # Таджикские глифы в самой странице: снимок, на котором они не
            # отрисовались бы, всё равно был бы PNG, но пусть тот же текст,
            # что и в эталонной фикстуре, проходит через настоящий браузер.
            page.write_text(
                "<html><body style='font-size:40px'>ӣ ӯ қ ҳ ҷ ғ Ӣ Ӯ Қ Ҳ Ҷ Ғ</body></html>",
                encoding="utf-8",
            )
            image = root / "shot.png"
            command = screenshot_command(binary, page, image, root / "profile", (400, 200))

            completed = subprocess.run(
                command, capture_output=True, text=True, timeout=120, check=False
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertTrue(image.is_file())
            # Сигнатура PNG: код возврата ноль Chrome умеет отдать и не написав
            # ничего, поэтому верим файлу, а не коду.
            self.assertEqual(image.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")


if __name__ == "__main__":
    unittest.main()
