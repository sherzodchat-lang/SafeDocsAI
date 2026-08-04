"""Headless Chrome как внешний инструмент печати: поиск, проверка, аргументы.

Рендерер презентаций собирает HTML и отдаёт его Chrome, чтобы получить PDF.
Chrome здесь — не библиотека, а сторонний бинарник, который может отсутствовать,
оказаться неисполняемым или приехать битым из наполовину прошедшего apt. Всё
это — состояние машины, а не ошибка запроса, и узнавать о нём в момент, когда
пользователь уже полторы минуты ждёт колоду, поздно. Поэтому проверка вынесена
в отдельный модуль и зовётся на старте.

Модуль сознательно тонкий. Он умеет ровно три вещи: найти бинарник, убедиться,
что тот отвечает, и собрать командную строку. Он НЕ управляет процессом —
стадийные таймауты, убийство группы процессов и разбор кодов возврата живут в
рендерере, потому что только там известно, сколько времени этому конкретному
заказу ещё позволено идти. Обёртка, которая «на всякий случай» ставила бы свой
таймаут, добавила бы второй, невидимый снаружи предел, и заказ умирал бы по
чужому будильнику.
"""

from __future__ import annotations

import logging
import os
import shutil
import signal
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Переменная окружения, которой можно назвать бинарник явно.
#
# Имя с префиксом модуля, а не общий CHROME_BIN: CHROME_BIN и CHROMIUM_PATH —
# имена, которые уже могут стоять в окружении ради puppeteer, playwright или
# selenium, и подхватить чужое значение молча хуже, чем не подхватить ничего.
# Значение читается через os.environ, а не через Settings: app/shared/settings
# правит другой агент, и лишний ключ в общей модели — гарантированный конфликт.
CHROMIUM_BINARY_ENV = "PRESENTATIONS_CHROMIUM_BINARY"

# Кого искать в PATH, когда переменная не задана. Порядок не случайный:
# google-chrome-stable стоит первым, потому что это то, что реально установлено
# на стенде. Пакеты chromium/chromium-browser идут последними намеренно — в
# Ubuntu 22.04 это заглушки, перенаправляющие на snap, а snapd на стенде нет:
# найденный «бинарник» ответил бы отказом или предложением поставить snap.
# Проверка через --version это поймает, но лучше до неё не доводить.
#
# Ищем список, а не одно имя: в debian-slim и alpine пакет зовётся chromium, и
# хардкод под google-chrome-stable означал бы «не запускается в проде» с
# починкой через переменную окружения, о которой никто не знает.
CHROMIUM_CANDIDATES = (
    "google-chrome-stable",
    "google-chrome",
    "chromium-browser",
    "chromium",
)

# Абсолютные пути на случай, когда PATH процесса урезан (systemd-юнит без
# окружения логина — обычное дело). shutil.which их не найдёт, а файл лежит.
CHROMIUM_FALLBACK_PATHS = (
    "/usr/bin/google-chrome-stable",
    "/opt/google/chrome/chrome",
    "/usr/bin/chromium-browser",
    "/usr/bin/chromium",
    "/snap/bin/chromium",
)

# Сколько ждём ответа на `--version`. Это не рендер, а вопрос «ты живой»:
# нормальный ответ приходит за десятки миллисекунд, а секунды означают, что
# бинарник висит на чём-то своём (например, лезет в недоступный X-сервер).
# Ждать дольше нечего — старт приложения не должен стоять на этом.
VERSION_PROBE_TIMEOUT_SECONDS = 20.0

# Строка, по которой узнаём, что ответил именно браузер. Битая установка умеет
# отвечать нулевым кодом возврата и пустым выводом — тогда «бинарник работает»
# было бы неправдой.
VERSION_MARKER = "chrom"


class RendererUnavailable(RuntimeError):
    """Печатать нечем: браузера нет, он не запускается или отвечает мусором.

    Собственный тип, а не ApiError и не запись в core/exceptions: HTTP-код на
    него навесит слой API (у него для этого свой словарь), а модулю рендера
    нужно уметь отличать «инструмента нет» от «инструмент отработал плохо» без
    зависимости от FastAPI. Наследуемся от RuntimeError, чтобы код, который
    ловит только его, всё-таки не пропустил эту беду мимо себя.
    """


@dataclass(frozen=True)
class ChromiumStatus:
    """Результат проверки — в виде, пригодном и для лога, и для эндпоинта.

    Не bool: эндпоинту, отвечающему 503, нужно объяснить пользователю (точнее,
    админу в логах) не «нет», а «нет вот почему», и путь, по которому искали.
    Не исключение: проверка на старте обязана вернуться, а не выстрелить.
    """

    available: bool
    binary: str | None
    version: str | None
    error: str | None

    def require(self) -> str:
        """Путь к бинарнику или RendererUnavailable с человеческой причиной."""
        if not self.available or self.binary is None:
            raise RendererUnavailable(self.error or "Chromium is not available")
        return self.binary


# Результат последней проверки. Хранится на уровне модуля, потому что вопрос
# «есть ли браузер» задаёт и старт приложения, и каждый заказ, и эндпоинт
# списка: запускать `--version` на каждый вопрос — это лишний процесс на пустом
# месте. Состояние машины между рестартами не меняется само, а если админ
# доставил пакет — перепроверку заказывают явно (force=True).
_status: ChromiumStatus | None = None
_status_lock = threading.Lock()


def configured_binary() -> str | None:
    """Значение переменной окружения, если оно задано и непустое."""
    raw = os.environ.get(CHROMIUM_BINARY_ENV, "")
    value = raw.strip()
    return value or None


def find_chromium() -> str | None:
    """Путь к бинарнику браузера или None. Существование, но не работу.

    Явно заданная переменная окружения не «дополняется» поиском по PATH: если
    админ назвал бинарник, а его там нет, это ошибка конфигурации, и молча
    подставить другой браузер — значит скрыть её и печатать не тем, чем просили.
    """
    configured = configured_binary()
    if configured is not None:
        # Имя без слэша (`chromium`) ищем в PATH, путь — проверяем как путь.
        located = shutil.which(configured)
        if located is not None:
            return located
        return configured if Path(configured).exists() else None

    for name in CHROMIUM_CANDIDATES:
        located = shutil.which(name)
        if located is not None:
            return located

    for path in CHROMIUM_FALLBACK_PATHS:
        if Path(path).exists():
            return path
    return None


def probe_chromium() -> ChromiumStatus:
    """Проверить браузер прямо сейчас, ничего не кэшируя и не бросая.

    Проверяется не наличие файла, а ответ на `--version`: неисполняемый файл
    (потерянный +x, каталог вместо бинарника), обрубленная установка без
    библиотек и обёртка-заглушка, отвечающая пустотой, — всё это существующие
    пути, которыми ничего не напечатать. `--version` — самый дешёвый вопрос,
    на который умеет ответить только настоящий браузер.
    """
    binary = find_chromium()
    if binary is None:
        configured = configured_binary()
        if configured is not None:
            return ChromiumStatus(
                available=False,
                binary=None,
                version=None,
                error=(
                    f"{CHROMIUM_BINARY_ENV}={configured!r} points at a file that "
                    f"does not exist"
                ),
            )
        return ChromiumStatus(
            available=False,
            binary=None,
            version=None,
            error=(
                "no Chromium binary found; tried "
                + ", ".join(CHROMIUM_CANDIDATES)
                + f" in PATH (set {CHROMIUM_BINARY_ENV} to point at one)"
            ),
        )

    try:
        completed = subprocess.run(  # noqa: S603 - путь наш, не пользовательский
            [binary, "--version"],
            capture_output=True,
            text=True,
            timeout=VERSION_PROBE_TIMEOUT_SECONDS,
            check=False,
        )
    except OSError as exc:
        # PermissionError (файл без +x), IsADirectoryError, ENOEXEC у обрывка
        # загрузки — разные errno с одним смыслом: запустить это нельзя.
        return ChromiumStatus(
            available=False,
            binary=binary,
            version=None,
            error=f"{binary} cannot be executed: {exc}",
        )
    except subprocess.TimeoutExpired:
        return ChromiumStatus(
            available=False,
            binary=binary,
            version=None,
            error=(
                f"{binary} did not answer --version in "
                f"{VERSION_PROBE_TIMEOUT_SECONDS:.0f}s"
            ),
        )

    version = (completed.stdout or "").strip() or (completed.stderr or "").strip()
    if completed.returncode != 0:
        return ChromiumStatus(
            available=False,
            binary=binary,
            version=None,
            error=(
                f"{binary} --version exited with {completed.returncode}: "
                f"{version or 'no output'}"
            ),
        )
    if VERSION_MARKER not in version.lower():
        return ChromiumStatus(
            available=False,
            binary=binary,
            version=None,
            error=(
                f"{binary} --version answered {version!r}, which does not look "
                f"like a Chrome/Chromium build"
            ),
        )
    return ChromiumStatus(available=True, binary=binary, version=version, error=None)


def chromium_status(force: bool = False) -> ChromiumStatus:
    """Кэшированный результат проверки. Старт зовёт с force=True.

    Отдельная функция от probe_chromium, чтобы эндпоинт мог спросить «ну как
    там» без права запустить процесс: на горячем пути список шаблонов зовут
    часто, и каждый вызов не должен превращаться в fork.
    """
    global _status
    with _status_lock:
        if force or _status is None:
            _status = probe_chromium()
        return _status


def ensure_chromium_available(force: bool = False) -> str:
    """Путь к рабочему браузеру или громкий отказ.

    Громкий — это ERROR в журнал и исключение: тихо вернуть None означало бы,
    что вызывающий волен «продолжить как-нибудь», а продолжать нечем. ERROR
    пишется здесь, а не в вызывающем коде, потому что причина (какой путь,
    что именно ответил бинарник) известна только тут, а вызывающих несколько.
    """
    status = chromium_status(force=force)
    if status.available and status.binary is not None:
        return status.binary
    logger.error(
        "Headless Chrome is unavailable, presentations cannot be rendered: %s",
        status.error,
    )
    raise RendererUnavailable(status.error or "Chromium is not available")


def log_chromium_state(force: bool = True) -> ChromiumStatus:
    """Одна строка о браузере при старте. Никогда не бросает.

    Отсутствие браузера не роняет приложение: сломан один сценарий из
    полутора десятков, и отказ на старте отнял бы у админа и логи, и
    админ-панель, через которую он этот стенд чинит. Заказы отклонит слой API,
    спросив chromium_status().
    """
    try:
        status = chromium_status(force=force)
    except Exception as exc:  # noqa: BLE001 - старт не роняем ничем
        logger.error("Chromium probe itself failed: %s", exc, exc_info=True)
        return ChromiumStatus(
            available=False, binary=None, version=None, error=str(exc)
        )

    if status.available:
        logger.info("Headless Chrome ready: %s (%s)", status.version, status.binary)
    else:
        logger.error(
            "Headless Chrome is unavailable: %s. Presentation rendering will be "
            "refused until this is fixed.",
            status.error,
        )
    return status


# --- Командные строки -----------------------------------------------------
#
# Аргументы собираются здесь, а запускает их вызывающий: ему держать таймаут и
# ему убивать зависший процесс. Функции возвращают список argv, а не строку —
# строка потребовала бы шелла, а шелл в пути, куда попадают имена файлов, это
# лишняя пара рук на подстановке.

# --no-sandbox — УСТУПКА В БЕЗОПАСНОСТИ, а не деталь запуска. Стоит первым и
# описан отдельно от остальных флагов именно поэтому.
#
# Почему он здесь. Песочница Chrome строится на user namespaces, а контейнер
# стенда их создавать не даёт. Без флага браузер не «работает похуже», а
# умирает на старте:
#   Failed to move to new namespace: PID namespaces supported, Network namespace
#   supported, but failed: errno = Operation not permitted
#   FATAL ... Check failed: . : Operation not permitted (1)
# PDF при этом не создаётся вовсе.
#
# Чем компенсируется. Песочница защищает от враждебного содержимого страницы, а
# враждебного содержимого здесь нет по построению — и каждое звено этого «по
# построению» проверяется машиной, а не памятью автора:
#   * страница — локальный file://, собранный нашим же Jinja-шаблоном; чужой
#     HTML в браузер не попадает никогда;
#   * пользовательский текст (имя блокнота, заголовки, буллиты, ответы модели)
#     подставляется как ДАННЫЕ: окружение Jinja создаётся с autoescape=True
#     безусловно (templates.build_environment), и это закреплено тестом;
#   * JavaScript в шаблонах не используется;
#   * внешних ресурсов нет: стартовый линт отбраковывает шаблон, в котором
#     встретился http:// или https:// (templates.EXTERNAL_URL_PATTERN), а
#     сетевые походы самого браузера отключены флагами ниже.
# То есть песочница защищала бы нас от нашего же статического HTML.
#
# Когда флаг можно убрать. Когда изменится СРЕДА ЗАПУСКА — контейнер начнёт
# отдавать user namespaces (или рендер переедет на хост, где они есть). Проверка
# ровно одна: запустить Chrome без флага и убедиться, что он не падает с текстом
# выше. Убирать флаг «на всякий случай», не поменяв среду, бессмысленно: браузер
# просто перестанет стартовать и презентации отвалятся целиком.
#
# Условным флаг сознательно НЕ сделан. Схема «попробовать без него, откатиться с
# предупреждением» удваивает число запусков Chrome на каждом рендере и создаёт
# два разных режима работы, из которых в проде живёт всегда один и тот же.
# Молчаливая же деградация была бы хуже всего: уступка в безопасности,
# случающаяся сама и незаметно, — это уже не решение, а случайность.
#
# Остальные флаги — про то, чтобы браузер не врал и никуда не ходил:
# --disable-gpu, --disable-dev-shm-usage: /dev/shm в контейнере 64 МБ, и
#   Chrome на большом документе тихо падает с «tab crashed».
# --no-first-run, --no-default-browser-check: иначе первый запуск тратит время
#   на диалоги, которых в headless никто не увидит.
# --disable-extensions, --disable-background-networking, --disable-sync,
#   --disable-component-update: убираем всё, что ходит в сеть. Шаблоны по
#   построению без внешних ресурсов, и браузер тоже не должен никуда ходить.
# --hide-scrollbars: иначе полоса прокрутки попадает в скриншот превью.
BASE_ARGS = (
    "--headless=new",
    "--no-sandbox",
    "--disable-gpu",
    "--disable-dev-shm-usage",
    "--no-first-run",
    "--no-default-browser-check",
    "--disable-extensions",
    "--disable-background-networking",
    "--disable-sync",
    "--disable-component-update",
    "--hide-scrollbars",
)


def _file_url(path: Path) -> str:
    """file:// URL для локального HTML.

    Через as_uri(), а не «file://» + str: пробелы и кириллица в пути
    (а имя блокнота попадает во временный каталог) требуют процентного
    кодирования, иначе Chrome откроет не тот файл или не откроет ничего.
    """
    return path.resolve().as_uri()


def base_args(user_data_dir: Path) -> list[str]:
    """Общие флаги плюс личный профиль под этот запуск.

    Отдельный --user-data-dir обязателен: без него параллельные запуски дерутся
    за один профиль в $HOME, а второй Chrome вместо печати просто передаёт
    задание первому и выходит с нулевым кодом — файл при этом не появляется.
    """
    return [
        *BASE_ARGS,
        f"--user-data-dir={user_data_dir}",
        f"--crash-dumps-dir={user_data_dir}",
    ]


def pdf_command(
    binary: str,
    html_file: Path,
    pdf_file: Path,
    user_data_dir: Path,
) -> list[str]:
    """argv для печати HTML в PDF.

    --no-pdf-header-footer убирает колонтитулы с URL и датой, которые Chrome
    печатает по умолчанию: в презентации это мусор поверх дизайна.
    """
    return [
        binary,
        *base_args(user_data_dir),
        "--no-pdf-header-footer",
        f"--print-to-pdf={pdf_file}",
        _file_url(html_file),
    ]


def screenshot_command(
    binary: str,
    html_file: Path,
    png_file: Path,
    user_data_dir: Path,
    window_size: tuple[int, int],
) -> list[str]:
    """argv для снимка первой страницы HTML в PNG.

    Размер окна задаётся явно: без него Chrome берёт 800x600, и превью
    шаблона, свёрстанного под 16:9, приезжает обрезанным по ширине.
    """
    width, height = window_size
    return [
        binary,
        *base_args(user_data_dir),
        f"--window-size={width},{height}",
        f"--screenshot={png_file}",
        _file_url(html_file),
    ]


# Строки, которые Chrome пишет в stderr и при полностью успешной печати. Это
# шум среды, а не диагностика: «Failed to adjust OOM score» появляется на каждом
# запуске в контейнере без CAP_SYS_RESOURCE, D-Bus в headless нет по замыслу.
# Отфильтровываются они только при СОСТАВЛЕНИИ ОБЪЯСНЕНИЯ неудачи, чтобы
# настоящая причина не оказалась вытеснена из хвоста; признаком провала не
# является ни одна из них — провал определяется кодом возврата и отсутствием
# файла на выходе.
STDERR_NOISE = (
    "Failed to adjust OOM score",
    "Failed to connect to the bus",
    "dbus",
    "GLib",
    "Fontconfig",
    "libva error",
    "DEPRECATED_ENDPOINT",
)


def describe_failure(command: list[str], returncode: int, stderr: str) -> str:
    """Однострочное объяснение неудачного запуска для журнала.

    Вывод Chrome многословен и почти весь состоит из шума среды; в лог идёт
    хвост осмысленных строк, потому что настоящая причина (не нашёл файл, упала
    вкладка) пишется последней.
    """
    lines = [
        line.strip()
        for line in stderr.strip().splitlines()
        if line.strip() and not any(noise in line for noise in STDERR_NOISE)
    ]
    tail = " | ".join(lines[-3:])
    return (
        f"{Path(command[0]).name} exited with {returncode}"
        + (f": {tail}" if tail else " with no diagnostics")
    )


# Сколько ждать, пока убитая группа отпустит трубу. Секунды, а не минуты:
# SIGKILL не игнорируют, и если через десять секунд труба всё ещё держится —
# держит её не наш процесс, и ждать дальше бессмысленно.
KILL_DRAIN_SECONDS = 10.0


def kill_process_group(process: subprocess.Popen) -> str:
    """Убить группу процессов браузера и вернуть то, что она сказала в stderr.

    Живёт здесь, а не рядом с печатью, потому что нужна каждому, кто запускает
    Chrome: и печати колоды, и съёмке превью. Раньше её знала только печать, а
    превью убивало прямого потомка через subprocess.run(timeout=...) — и дерево
    браузера переживало таймаут, оставляя процессы с ppid=1 копиться от
    перезапуска к перезапуску (поймано на приёмке 2026-08-04).

    УБИВАЕТСЯ ГРУППА, А НЕ ПРОЦЕСС. Chrome — дерево: zygote, рендерер,
    gpu-процесс, утилиты. Убийство одного родителя оставляет детей живыми и
    по-прежнему занимающими память и /dev/shm. Отсюда start_new_session=True
    при запуске (у дерева своя группа, чей id равен pid родителя) и killpg.

    SIGKILL, а не SIGTERM: процесс уже перебрал отведённое время, его вывод
    непригоден, и вежливое завершение — ещё одно ожидание с той же
    неопределённостью.

    Диагностику забираем ПОСЛЕ убийства: у зависшего браузера в трубе обычно
    лежит самое интересное (последняя строка перед зависанием), а пока труба не
    закрыта, communicate() ждать её не может. Убийство группы закрывает трубу у
    всех, кто её держал, — поэтому чтение после SIGKILL не блокируется.
    """
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError) as exc:
        # Процесс успел умереть сам между таймаутом и сигналом, либо группа нам
        # не принадлежит (так бывает под чужим init'ом в контейнере). Родителя
        # убиваем в любом случае: он держит трубу.
        logger.warning("Cannot kill the Chrome process group: %s", exc)
        process.kill()

    try:
        _, stderr = process.communicate(timeout=KILL_DRAIN_SECONDS)
    except subprocess.TimeoutExpired:  # pragma: no cover - SIGKILL не игнорируют
        logger.error("Chrome survived SIGKILL for %.0fs", KILL_DRAIN_SECONDS)
        return ""
    return stderr or ""
