"""Реестр шаблонов презентаций v2: каталог с HTML/CSS/шрифтами -> запись реестра.

Шаблон больше не pptx-файл, а КАТАЛОГ: template.html (Jinja2), styles.css и
fonts/*.woff2. Печатает его headless Chrome, и это меняет цену ошибки. У pptx
беда была видна сразу — файл либо открывается, либо нет. У HTML почти всё
ломается тихо: опечатка в Jinja всплывает на рендере конкретного заказа,
пропавший .woff2 даёт не ошибку, а подстановку системного шрифта (и таджикские
ӣ ӯ қ ҳ ҷ ғ уезжают в квадратики), ссылка на внешний CDN превращает сборку
презентации в поход в интернет с чужого сервера. Поэтому проверок стало больше
и все они выполняются ОДИН РАЗ ПРИ СТАРТЕ, а не на заказе пользователя.

Проверяется по каждому шаблону:
  * каталог и оба обязательных файла на месте;
  * ни в template.html, ни в styles.css нет http:// и https:// — внешних
    ресурсов у нас нет по построению, а не по договорённости;
  * все упомянутые шрифты существуют на диске;
  * template.html разбирается Jinja2;
  * смоук-рендер эталонной фикстуры проходит целиком.

Битый шаблон выбрасывается из реестра с ERROR, остальные остаются. Это принцип
«интерфейс не врёт и не падает целиком»: один сломанный дизайн не должен гасить
галерею выбора, но и предлагать выбрать шаблон, на котором генерация потом
упадёт, нельзя. Оба отказа — и «показать битое», и «не показать ничего» — хуже
третьего: показать то, что точно работает, и написать в журнал про остальное.

Реестр читается один раз и живёт в памяти. Это сознательное отличие от
runtime_settings.json, который перечитывается на каждое обращение: там значение
правит админ во время работы, здесь файлы приезжают из образа вместе с кодом.
Перечитывание означало бы разбор JSON, парс Jinja и смоук-рендер на каждый
запрос списка шаблонов — и не изменило бы ни одного ответа.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
import subprocess
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined, TemplateError

from app.modules.presentations.chromium import (
    ChromiumStatus,
    chromium_status,
    describe_failure,
    kill_process_group,
    screenshot_command,
)

logger = logging.getLogger(__name__)

MANIFEST_FILENAME = "manifest.json"
FIXTURE_FILENAME = "fixture.json"
TEMPLATE_FILENAME = "template.html"
STYLES_FILENAME = "styles.css"
FONTS_DIRNAME = "fonts"

# Языки, на которых обязано быть название шаблона: это подписи в интерфейсе
# выбора, и отсутствие одного из них означает пустую строку в списке.
#
# ISO был бы tg; используем tj для согласованности с i18n проекта — таджикский
# обозначен как "tj" везде: определение языка документа, доменные профили,
# инструкции модели, словари фронтенда. Третий код языка в одной системе
# опаснее расхождения со стандартом: сравнение language == "tj" где-нибудь в
# рендерере молча даст False, и презентация уедет на русском.
NAME_LANGUAGES = ("ru", "tj")

# Внешние ссылки в шаблоне. Ловим сам префикс схемы, а не тег: <link href>,
# @import, url() в CSS и src у картинки — четыре разных синтаксиса с одним
# последствием. Схема одна на всех, и её достаточно.
#
# Зачем вообще: шаблон с внешним шрифтом рендерится по-разному в зависимости от
# того, была ли у сервера сеть, — то есть иногда без таджикских глифов и всегда
# непредсказуемо по времени. Плюс это утечка: каждый рендер сообщал бы чужому
# CDN, что клиент собрал презентацию. Локальные ресурсы обязательны, и
# проверяется это здесь, а не глазами на ревью.
EXTERNAL_URL_PATTERN = re.compile(r"https?://", re.IGNORECASE)

# Допустимый ключ шаблона. Из ключа строится имя файла превью и он попадает в
# URL эндпоинта, поэтому набор символов узкий: латиница в нижнем регистре,
# цифры, дефис и подчёркивание. До 64 символов — столько же, сколько принимает
# путь /presentations/templates/{template_key}/preview.
KEY_PATTERN = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}")

# Ссылки на файлы шрифтов. url(...) в @font-face — основной способ, но упомянуть
# .woff2 можно и в preload-теге внутри HTML, поэтому ищем оба.
CSS_URL_PATTERN = re.compile(r"""url\(\s*['"]?([^'")]+)['"]?\s*\)""", re.IGNORECASE)
FONT_FILE_PATTERN = re.compile(r"""['"(]([^'")\s]+\.woff2?)['")]""", re.IGNORECASE)

# Строка, которая обязана быть в эталонной фикстуре. Смысл фикстуры не в том,
# чтобы «что-нибудь отрисовать», а в том, чтобы каждый смоук-рендер и каждое
# превью бесплатно проверяли таджикские глифы: они есть далеко не в каждом
# шрифте, и подстановка происходит молча. Если её нет — фикстура перестала быть
# сторожем, и об этом надо знать до того, как в галерее появятся квадратики.
TAJIK_GLYPHS = "ӣ ӯ қ ҳ ҷ ғ Ӣ Ӯ Қ Ҳ Ҷ Ғ"

# Размер снимка превью — ровно слайдовая коробка комплекта: 1280x720, 16:9.
# Число не «покрасивее», а подобранное: --screenshot снимает первый экран как
# есть, и окно шире слайда даёт поля, а выше — полосу следующего слайда снизу.
# При совпадении с коробкой в кадр попадает ровно титульный слайд.
#
# Шаблон с другой коробкой всё равно получит осмысленное превью — свой первый
# экран в том же соотношении. Это по-прежнему НАСТОЯЩИЙ вывод, а не рисунок,
# так что галерея не соврёт; она лишь покажет кадр менее удачно, и чинится это
# в CSS шаблона, а не здесь.
PREVIEW_SIZE = (1280, 720)

# Формат превью — PNG. Chrome умеет --screenshot и пишет именно PNG, поэтому
# любой другой формат означал бы второй инструмент в цепочке (Pillow или
# ImageMagick) ради конвертации из того, что уже готово. PNG к тому же без
# потерь: превью — это текст и тонкие линии на плоской заливке, а JPEG именно
# на такой картинке даёт кольца вокруг букв. Расширение .png ещё и уже описано
# в PREVIEW_MEDIA_TYPES эндпоинта — менять его пришлось бы в чужом периметре.
PREVIEW_SUFFIX = ".png"

# Версия «рецепта» превью. Входит в отпечаток вместе с содержимым шаблона:
# когда меняется размер снимка или набор флагов Chrome, картинки обязаны
# перерисоваться, хотя ни один файл шаблона не тронут.
PREVIEW_RECIPE_VERSION = "1"

# Сколько ждём Chrome на одном превью. Это не рендер заказа (там свой,
# стадийный таймаут в рендерере), а старт приложения: превью — три-четыре
# снимка статичной страницы, и если один занял минуту, значит браузер завис,
# и ждать его дольше — задерживать подъём всего бэкенда.
PREVIEW_TIMEOUT_SECONDS = 60.0


@dataclass(frozen=True)
class TemplateInfo:
    """Проверенная запись реестра.

    Пути абсолютные и уже существующие на момент создания записи: рендереру не
    нужно ни склеивать их с каталогом, ни проверять существование заново.

    preview_file — Path | None, и это честно: превью рисует Chrome, а Chrome
    может отсутствовать. Шаблон без картинки остаётся выбираемым (дизайн-то
    рабочий), поэтому отсутствие превью не повод выкидывать запись; но и
    делать вид, что файл есть, нельзя — вызывающий обязан увидеть None и
    решить, что показать.
    """

    key: str
    name: dict[str, str]
    directory: Path
    html_file: Path
    css_file: Path
    fonts_dir: Path | None
    preview_file: Path | None


def default_templates_dir() -> Path:
    """Каталог шаблонов: backend/templates/presentations.

    Путь считается от файла модуля, а не от текущего каталога процесса: бэкенд
    запускают и из backend/ (run.py), и из корня репозитория (start.sh), и из
    /app в контейнере — относительный путь в одном из этих случаев обязательно
    промахнулся бы.

    Рядом лежит backend/data — туда шаблоны НЕ относятся: data монтируется
    томом и переживает релиз, а шаблоны обязаны приезжать вместе с кодом,
    который на них рассчитывает.
    """
    return _backend_dir() / "templates" / "presentations"


def default_preview_dir() -> Path:
    """Куда складываются сгенерированные превью: backend/data/.

    Именно в data, а не рядом с шаблонами. Три довода, и каждый достаточен:
    templates/ — это исходники в репозитории, и складывать туда машинный
    результат значит либо коммитить его, либо вечно держать в git status;
    каталог образа в проде бывает read-only; и наконец превью зависят не только
    от шаблона, но и от версии Chrome на машине, то есть это кэш, а не артефакт
    сборки. data/ для кэша и предназначена — она переживает рестарт, но её не
    жалко потерять.
    """
    return _backend_dir() / "data" / "presentation_previews"


def _backend_dir() -> Path:
    return Path(__file__).resolve().parents[3]


def build_environment(directory: Path) -> Environment:
    """Окружение Jinja2 для одного каталога шаблона.

    autoescape=True БЕЗУСЛОВНО, а не select_autoescape по расширению. В
    шаблонах презентаций нет ни одного файла, который не является HTML, зато
    есть контекст, целиком собранный из пользовательских данных: имя блокнота,
    заголовки слайдов и буллиты приходят из документов и от модели. Строка
    вида `</div><script>` в буллите без экранирования — это уже не «сломанная
    вёрстка», а исполняемый код внутри страницы, которую мы сами открываем в
    браузере с --no-sandbox. Отключать экранирование точечно (|safe) можно
    только там, где строку построил код шаблона, и каждое такое место обязано
    быть видно в диффе.

    StrictUndefined, а не молчаливый Undefined: пропущенная переменная должна
    падать на смоук-рендере при старте, а не превращаться в пустое место в
    готовой колоде. «Слайд без заголовка» — это дефект, который пользователь
    заметит, а мы нет.
    """
    return Environment(
        loader=FileSystemLoader(str(directory)),
        autoescape=True,
        undefined=StrictUndefined,
        keep_trailing_newline=True,
    )


def stage_page(template_dir: Path, html: str, destination: Path) -> Path:
    """Разложить готовый HTML рядом с копией ресурсов шаблона; вернуть путь.

    Chrome открывает страницу как file:// и разрешает href="styles.css" и
    url("fonts/…") относительно НЕЁ. Значит, готовый HTML обязан лежать рядом с
    этими файлами — а писать его внутрь templates/ нельзя: это исходники в
    репозитории, в проде каталог бывает смонтирован только на чтение, и два
    параллельных рендера затёрли бы файл друг друга. Отсюда копия во временном
    каталоге, который заводит и убирает вызывающий.

    Функция общая для превью и для печати заказа: снимок и PDF обязаны
    получаться из одинаково разложенной страницы, иначе превью в галерее
    перестанет соответствовать тому, что скачает пользователь.
    """
    shutil.copytree(template_dir, destination)
    page = destination / "index.html"
    page.write_text(html, encoding="utf-8")
    return page


class TemplateRegistry:
    """Реестр шаблонов, прочитанный и проверенный с диска один раз.

    Чтение ленивое, а не на импорте модуля: импорт тянул бы за собой парс Jinja,
    смоук-рендеры и запуск Chrome, и тогда любой тест, случайно импортировавший
    модуль, платил бы за это временем, а сбой файловой системы превращался бы в
    ImportError — ошибку, из которой не видно, что дело в шаблонах.
    """

    def __init__(
        self,
        directory: Path | None = None,
        preview_dir: Path | None = None,
        *,
        generate_previews: bool = True,
    ) -> None:
        # None означает «каталог по умолчанию, вычисленный на момент чтения».
        # Тесты передают временный каталог и работают с собственным манифестом.
        self._directory = Path(directory) if directory is not None else None
        self._preview_dir = Path(preview_dir) if preview_dir is not None else None
        self._generate_previews = generate_previews
        self._templates: dict[str, TemplateInfo] | None = None
        # Первое обращение может прийти из нескольких потоков сразу (FastAPI
        # уводит синхронные обработчики в пул): без блокировки манифест
        # разбирался бы дважды, Chrome запускался бы дважды на одно превью, и в
        # журнале двоились бы строки об ошибках.
        self._lock = threading.Lock()

    @property
    def directory(self) -> Path:
        if self._directory is not None:
            return self._directory
        return default_templates_dir()

    @property
    def preview_dir(self) -> Path:
        if self._preview_dir is not None:
            return self._preview_dir
        return default_preview_dir()

    def get(self, key: str) -> TemplateInfo | None:
        """Шаблон по ключу или None, если такого нет (или он битый)."""
        return self._loaded().get(key)

    def list(self) -> list[TemplateInfo]:
        """Пригодные шаблоны в порядке манифеста."""
        return list(self._loaded().values())

    def warm_up(self) -> list[TemplateInfo]:
        """Прочитать и проверить всё прямо сейчас. Зовётся при старте.

        Отдельное имя, а не list(): на старте нас интересует побочный эффект —
        разбор манифеста, смоук-рендеры и генерация превью, — и вызов, который
        выглядит как «получить список», прочитать как «выполнить проверки»
        нельзя. Первый же запрос пользователя иначе оплачивал бы запуск Chrome.
        """
        return self.list()

    def reload(self) -> list[TemplateInfo]:
        """Сбросить кэш и перечитать. Нужно тестам и ручной починке стенда."""
        with self._lock:
            self._templates = None
        return self.list()

    def _loaded(self) -> dict[str, TemplateInfo]:
        templates = self._templates
        if templates is not None:
            return templates
        with self._lock:
            if self._templates is None:
                self._templates = self._read()
            return self._templates

    # --- Чтение и проверка ------------------------------------------------

    def _read(self) -> dict[str, TemplateInfo]:
        directory = self.directory
        manifest_path = directory / MANIFEST_FILENAME
        entries = self._read_manifest(manifest_path)
        fixture = self._read_fixture(directory / FIXTURE_FILENAME)
        status = self._chromium_status()

        templates: dict[str, TemplateInfo] = {}
        for position, entry in enumerate(entries):
            info = self._build(entry, position, directory, fixture, status)
            if info is None:
                continue
            if info.key in templates:
                # Дубликат ключа — не «последний побеждает»: выбор шаблона
                # пришёл бы к одному каталогу, а превью в списке показывалось
                # бы от другого. Оставляем первый и говорим об этом.
                logger.error(
                    "Presentation template %r duplicated in %s; keeping the first entry",
                    info.key,
                    manifest_path,
                )
                continue
            templates[info.key] = info

        logger.info(
            "Presentation templates loaded from %s: %d of %d entries usable (%s)",
            directory,
            len(templates),
            len(entries),
            ", ".join(templates) or "none",
        )
        return templates

    def _chromium_status(self) -> ChromiumStatus | None:
        if not self._generate_previews:
            return None
        return chromium_status()

    def _read_manifest(self, manifest_path: Path) -> list[Any]:
        """Список записей манифеста; при любой беде — пустой список и ERROR.

        Отсутствие манифеста здесь не «норма чистой установки», как у
        runtime_settings.json: файл кладёт сборка, и без него презентации
        собрать не из чего. Но и падать нельзя — молчать тоже, поэтому ERROR.
        """
        data = self._read_json(manifest_path, "manifest")
        if data is None:
            return []

        # Объект с ключом "templates", а не голый список: у манифеста есть
        # шанс обрасти общими полями (версия формата, дефолтный ключ), и
        # заворачивать список задним числом дороже, чем сразу.
        if not isinstance(data, dict):
            logger.error(
                "Presentation templates manifest %s must contain an object, got %s",
                manifest_path,
                type(data).__name__,
            )
            return []
        entries = data.get("templates")
        if not isinstance(entries, list):
            logger.error(
                "Presentation templates manifest %s has no 'templates' list",
                manifest_path,
            )
            return []
        return entries

    def _read_fixture(self, fixture_path: Path) -> dict[str, Any] | None:
        """Эталонный контекст для смоук-рендера и превью.

        Фикстура одна на все шаблоны — это её смысл: превью в галерее должны
        показывать один и тот же текст, иначе пользователь сравнивает не
        дизайны, а содержание. Отсутствие фикстуры — дефект сборки, но НЕ повод
        объявить битыми все шаблоны разом: тогда один забытый файл гасил бы
        галерею целиком, ровно то, чего этот модуль обязан не допускать.
        Пишем ERROR, пропускаем смоук-рендер и превью, структурные проверки
        оставляем в силе.
        """
        data = self._read_json(fixture_path, "fixture")
        if data is None:
            return None
        if not isinstance(data, dict):
            logger.error(
                "Presentation template fixture %s must contain an object, got %s",
                fixture_path,
                type(data).__name__,
            )
            return None
        if TAJIK_GLYPHS not in json.dumps(data, ensure_ascii=False):
            # Не отказ: фикстура рабочая, но перестала быть сторожем глифов, а
            # это тихая потеря проверки — самая дорогая из возможных.
            logger.error(
                "Presentation template fixture %s no longer contains the Tajik "
                "glyph probe %r; previews stop proving font coverage",
                fixture_path,
                TAJIK_GLYPHS,
            )
        return data

    @staticmethod
    def _read_json(path: Path, what: str) -> Any | None:
        try:
            raw = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            logger.error("Presentation templates %s %s is missing", what, path)
            return None
        except OSError as exc:
            logger.error(
                "Presentation templates %s %s is unreadable: %s", what, path, exc
            )
            return None
        try:
            # ValueError накрывает и JSONDecodeError, и UnicodeDecodeError.
            return json.loads(raw)
        except ValueError as exc:
            logger.error(
                "Presentation templates %s %s is not valid JSON: %s", what, path, exc
            )
            return None

    def _build(
        self,
        entry: Any,
        position: int,
        directory: Path,
        fixture: dict[str, Any] | None,
        status: ChromiumStatus | None,
    ) -> TemplateInfo | None:
        """Одна запись манифеста -> TemplateInfo или None с ERROR в журнале."""
        if not isinstance(entry, dict):
            logger.error(
                "Presentation template entry #%d is not an object (%s); skipped",
                position,
                type(entry).__name__,
            )
            return None

        key = entry.get("key")
        if not isinstance(key, str) or not key.strip():
            logger.error(
                "Presentation template entry #%d has no usable 'key'; skipped", position
            )
            return None
        key = key.strip()
        if KEY_PATTERN.fullmatch(key) is None:
            # Ключ — не просто подпись: из него строится имя файла превью и он
            # приезжает в URL. Ключ вида "../secrets" вывел бы запись превью за
            # пределы своего каталога ещё до того, как кто-то попробовал бы это
            # снаружи. Ограничение узкое намеренно: манифест наш, и придумать
            # ключ по этим правилам не стоит ничего.
            logger.error(
                "Presentation template entry #%d has key %r outside [a-z0-9_-]; "
                "skipped",
                position,
                key,
            )
            return None

        name = self._build_name(key, entry.get("name"))
        if name is None:
            return None

        template_dir = self._resolve_dir(key, entry.get("dir"), directory)
        if template_dir is None:
            return None

        html_file = template_dir / TEMPLATE_FILENAME
        css_file = template_dir / STYLES_FILENAME
        for path in (html_file, css_file):
            if not path.is_file():
                logger.error(
                    "Presentation template %r is missing %s; skipped", key, path
                )
                return None

        sources = self._read_sources(key, html_file, css_file)
        if sources is None:
            return None
        html_text, css_text = sources

        if not self._external_links_absent(key, html_file, html_text):
            return None
        if not self._external_links_absent(key, css_file, css_text):
            return None
        if not self._fonts_present(key, template_dir, html_text, css_text):
            return None

        rendered = self._smoke_render(key, template_dir, fixture)
        if rendered is False:
            return None

        fonts_dir = template_dir / FONTS_DIRNAME
        preview_file = self._preview(
            key=key,
            template_dir=template_dir,
            html=rendered if isinstance(rendered, str) else None,
            status=status,
        )

        return TemplateInfo(
            key=key,
            name=name,
            directory=template_dir,
            html_file=html_file,
            css_file=css_file,
            fonts_dir=fonts_dir if fonts_dir.is_dir() else None,
            preview_file=preview_file,
        )

    @staticmethod
    def _build_name(key: str, raw: Any) -> dict[str, str] | None:
        if not isinstance(raw, dict):
            logger.error("Presentation template %r has no 'name' object; skipped", key)
            return None
        name: dict[str, str] = {}
        for language in NAME_LANGUAGES:
            value = raw.get(language)
            if not isinstance(value, str) or not value.strip():
                logger.error(
                    "Presentation template %r has no name for language %r; skipped",
                    key,
                    language,
                )
                return None
            name[language] = value.strip()
        return name

    @staticmethod
    def _resolve_dir(key: str, raw: Any, directory: Path) -> Path | None:
        if not isinstance(raw, str) or not raw.strip():
            logger.error("Presentation template %r has no 'dir'; skipped", key)
            return None

        path = (directory / raw.strip()).resolve()
        # Манифест — свой же артефакт, но путь из него всё равно не выпускается
        # за пределы каталога шаблонов: запись вида "../../data/uploads" сделала
        # бы шаблоном произвольный каталог на диске, а его файлы — доступными
        # через выдачу превью.
        if not path.is_relative_to(directory.resolve()):
            logger.error(
                "Presentation template %r has dir %r outside %s; skipped",
                key,
                raw,
                directory,
            )
            return None
        if not path.is_dir():
            logger.error(
                "Presentation template %r has dir %s that does not exist; skipped",
                key,
                path,
            )
            return None
        return path

    @staticmethod
    def _read_sources(
        key: str, html_file: Path, css_file: Path
    ) -> tuple[str, str] | None:
        try:
            return (
                html_file.read_text(encoding="utf-8"),
                css_file.read_text(encoding="utf-8"),
            )
        except (OSError, ValueError) as exc:
            # ValueError здесь — UnicodeDecodeError: шаблон не в UTF-8 означает,
            # что таджикские буквы в нём уже испорчены.
            logger.error(
                "Presentation template %r cannot be read: %s; skipped", key, exc
            )
            return None

    @staticmethod
    def _external_links_absent(key: str, path: Path, text: str) -> bool:
        match = EXTERNAL_URL_PATTERN.search(text)
        if match is None:
            return True
        line = text.count("\n", 0, match.start()) + 1
        logger.error(
            "Presentation template %r references an external URL in %s:%d; "
            "templates must be fully self-contained; skipped",
            key,
            path.name,
            line,
        )
        return False

    @staticmethod
    def _fonts_present(key: str, template_dir: Path, html: str, css: str) -> bool:
        """Все упомянутые шрифты лежат на диске.

        Пропавший .woff2 — самая тихая из поломок: Chrome не ругается, а молча
        берёт системный шрифт, в котором таджикских ӣ ӯ қ ҳ ҷ ғ может не быть.
        Пользователь получит колоду с квадратиками вместо букв, и узнаем мы об
        этом от него, а не из журнала.
        """
        references: set[str] = set()
        for value in CSS_URL_PATTERN.findall(css):
            references.add(value.strip())
        for text in (html, css):
            for value in FONT_FILE_PATTERN.findall(text):
                references.add(value.strip())

        root = template_dir.resolve()
        for reference in sorted(references):
            if reference.startswith("data:"):
                # Шрифт или картинка, вшитые в CSS base64-строкой, на диске и не
                # должны лежать: они уже внутри файла.
                continue
            if reference.startswith("#"):
                # url(#gradient) — ссылка на элемент этой же страницы (градиент,
                # маска, фильтр в inline-SVG), а не на файл. Искать её на диске
                # значило бы отбраковывать совершенно законную вёрстку.
                continue
            path = (template_dir / reference).resolve()
            if not path.is_relative_to(root):
                logger.error(
                    "Presentation template %r references %r outside its own "
                    "directory; skipped",
                    key,
                    reference,
                )
                return False
            if not path.is_file():
                logger.error(
                    "Presentation template %r references a missing asset %r "
                    "(expected at %s); skipped",
                    key,
                    reference,
                    path,
                )
                return False
        return True

    @staticmethod
    def _smoke_render(
        key: str, template_dir: Path, fixture: dict[str, Any] | None
    ) -> str | bool:
        """Разбор шаблона и рендер эталонной фикстуры.

        Возвращает готовый HTML, True (разбор прошёл, фикстуры нет) или False
        (шаблон негоден). Смоук-рендер ловит то, чего не видит парсер: обращение
        к несуществующему полю контекста, фильтр с неверным числом аргументов,
        цикл по строке вместо списка. Без него первой жертвой опечатки был бы
        заказ пользователя — уже после того, как модель отработала.
        """
        environment = build_environment(template_dir)
        try:
            template = environment.get_template(TEMPLATE_FILENAME)
        except TemplateError as exc:
            logger.error(
                "Presentation template %r does not parse as Jinja2: %s; skipped",
                key,
                exc,
            )
            return False
        except OSError as exc:
            logger.error(
                "Presentation template %r cannot be loaded: %s; skipped", key, exc
            )
            return False

        if fixture is None:
            return True

        try:
            return template.render(**fixture)
        except Exception as exc:  # noqa: BLE001
            # Перечислить исключения рендера нельзя: StrictUndefined даёт
            # UndefinedError, чужой фильтр — что угодно своё, а выражение вида
            # slides[0] на пустом списке — IndexError. Смысл один: на этом
            # шаблоне заказ пользователя упадёт.
            logger.error(
                "Presentation template %r fails to render the reference "
                "fixture (%s): %s; skipped",
                key,
                type(exc).__name__,
                exc,
            )
            return False

    # --- Превью -----------------------------------------------------------

    def _preview(
        self,
        key: str,
        template_dir: Path,
        html: str | None,
        status: ChromiumStatus | None,
    ) -> Path | None:
        """Путь к превью шаблона; при нужде — перерисовывает его Chrome'ом.

        Превью рисует сам браузер из того же шаблона и той же фикстуры, что
        уйдут в печать. Нарисованная дизайнером картинка рано или поздно
        разъезжается с реальным выводом, и галерея начинает врать: пользователь
        выбирает одно, получает другое. Здесь такое невозможно по построению —
        превью и есть вывод.
        """
        target = self.preview_dir / f"{key}{PREVIEW_SUFFIX}"
        stamp_file = target.with_suffix(f"{PREVIEW_SUFFIX}.stamp")

        if html is None:
            # Без фикстуры рисовать нечего. Старое превью, если оно осталось от
            # прошлого запуска, лучше пустого места: дизайн не менялся.
            return target if target.is_file() else None

        # Отпечаток — HASH СОДЕРЖИМОГО, а не mtime. Mtime здесь врёт в обе
        # стороны: git checkout, docker COPY и rsync ставят файлам свежее время,
        # не меняя ни байта (и мы перерисовывали бы всё на каждом деплое), а
        # восстановление файла из архива возвращает СТАРОЕ время при новом
        # содержимом (и мы бы не перерисовали ничего). Хэш отвечает на вопрос,
        # который нас на самом деле интересует: изменился ли шаблон.
        stamp = self._fingerprint(template_dir, html)
        if target.is_file() and self._read_stamp(stamp_file) == stamp:
            return target

        if status is None:
            # Генерация выключена явно (так реестр создают тесты, которым
            # Chrome не нужен). Молчим: это не беда, а заказанное поведение.
            return target if target.is_file() else None

        if not status.available or status.binary is None:
            # Не отказ шаблона: дизайн рабочий, рисовать его нечем. Галерея
            # покажет запись без картинки — это хуже, чем с картинкой, но
            # несравнимо лучше, чем пустая галерея.
            #
            # Два разных сообщения, а не одно: сюда попадают два несравнимых
            # исхода. Если старая картинка на диске осталась, галерея её и
            # покажет — то есть пользователь увидит превью, просто снятое с
            # прошлой версии шаблона. Сказать про это «has no preview» значит
            # соврать оператору в журнале: он пойдёт искать пустую карточку,
            # которой нет, и не узнает главного — что показывается устаревшее.
            stale = target.is_file()
            if stale:
                logger.error(
                    "Presentation template %r keeps a STALE preview (template "
                    "changed, cannot redraw): %s",
                    key,
                    status.error,
                )
            else:
                logger.error(
                    "Presentation template %r has no preview: %s", key, status.error
                )
            return target if stale else None

        rendered = self._shoot(key, template_dir, html, target, status.binary)
        if not rendered:
            return target if target.is_file() else None

        self._write_stamp(stamp_file, stamp)
        logger.info(
            "Presentation template %r preview regenerated at %s (fingerprint %s)",
            key,
            target,
            stamp[:12],
        )
        return target

    @staticmethod
    def _fingerprint(template_dir: Path, html: str) -> str:
        """Отпечаток всего, от чего зависит картинка превью.

        Считается по КАЖДОМУ файлу каталога, а не только по template.html:
        превью меняет и правка в styles.css, и подменённый шрифт. Готовый HTML
        входит сюда же — вместе с ним в отпечаток попадает содержимое фикстуры,
        то есть смена эталонного текста тоже перерисовывает галерею.
        """
        digest = hashlib.sha256()
        digest.update(PREVIEW_RECIPE_VERSION.encode("utf-8"))
        digest.update(repr(PREVIEW_SIZE).encode("utf-8"))
        digest.update(html.encode("utf-8"))
        for path in sorted(p for p in template_dir.rglob("*") if p.is_file()):
            digest.update(str(path.relative_to(template_dir)).encode("utf-8"))
            try:
                digest.update(path.read_bytes())
            except OSError:
                # Нечитаемый файл — тоже изменение состояния; пусть отпечаток
                # отличается от того, что был при удачном чтении.
                digest.update(b"<unreadable>")
        return digest.hexdigest()

    @staticmethod
    def _read_stamp(stamp_file: Path) -> str | None:
        try:
            return stamp_file.read_text(encoding="utf-8").strip()
        except (OSError, ValueError):
            return None

    @staticmethod
    def _write_stamp(stamp_file: Path, stamp: str) -> None:
        try:
            stamp_file.write_text(stamp, encoding="utf-8")
        except OSError as exc:
            # Без отпечатка превью просто перерисуется в следующий раз — это
            # трата времени, а не поломка, поэтому WARNING и работаем дальше.
            logger.warning("Cannot write preview stamp %s: %s", stamp_file, exc)

    def _shoot(
        self, key: str, template_dir: Path, html: str, target: Path, binary: str
    ) -> bool:
        """Снимок готового HTML браузером. True, если файл появился.

        Страница печатается из КОПИИ каталога шаблона во временной папке, а не
        из самого каталога: относительные ссылки на styles.css и fonts/ должны
        разрешаться, а писать index.html внутрь templates/ нельзя — это
        исходники в репозитории, и в проде каталог бывает смонтирован только на
        чтение.
        """
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.error("Cannot create preview directory %s: %s", target.parent, exc)
            return False

        with tempfile.TemporaryDirectory(prefix="preview-") as workspace:
            root = Path(workspace)
            try:
                page = stage_page(template_dir, html, root / "page")
            except OSError as exc:
                logger.error(
                    "Presentation template %r preview: cannot stage the page: %s",
                    key,
                    exc,
                )
                return False

            command = screenshot_command(
                binary, page, target, root / "profile", PREVIEW_SIZE
            )
            # Своя группа процессов и убийство ГРУППЫ, а не subprocess.run с
            # timeout=. Тот по таймауту убивает только прямого потомка, а Chrome
            # — дерево: zygote и рендереры переживают смерть родителя и остаются
            # висеть. Поймано на приёмке: после снятого по таймауту снимка в
            # системе остался осиротевший процесс с ppid=1, и такие копятся от
            # перезапуска к перезапуску. Путь печати колоды это уже закрыл тем
            # же приёмом; здесь была та же дыра.
            try:
                process = subprocess.Popen(  # noqa: S603 - пути наши
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    start_new_session=True,
                )
            except OSError as exc:
                logger.error(
                    "Presentation template %r preview: cannot start %s: %s",
                    key,
                    binary,
                    exc,
                )
                return False

            try:
                stdout, stderr = process.communicate(
                    timeout=PREVIEW_TIMEOUT_SECONDS
                )
            except subprocess.TimeoutExpired:
                # kill_process_group сам вычитывает stderr после SIGKILL:
                # пока труба открыта хоть одним потомком, communicate не
                # вернётся, поэтому порядок «убить, потом читать» обязателен.
                kill_process_group(process)
                logger.error(
                    "Presentation template %r preview: %s did not finish in %.0fs",
                    key,
                    binary,
                    PREVIEW_TIMEOUT_SECONDS,
                )
                return False

            completed = subprocess.CompletedProcess(
                command, process.returncode, stdout, stderr
            )

        if completed.returncode != 0:
            logger.error(
                "Presentation template %r preview failed: %s",
                key,
                describe_failure(command, completed.returncode, completed.stderr or ""),
            )
            return False
        if not target.is_file() or target.stat().st_size == 0:
            # Chrome умеет выйти с нулём, не написав файла (например, когда
            # страница не загрузилась). Верить коду возврата на слово нельзя.
            logger.error(
                "Presentation template %r preview: %s reported success but wrote "
                "no image to %s",
                key,
                binary,
                target,
            )
            return False
        return True


# Единственный экземпляр на процесс: кэш имеет смысл только общий, а каталог
# шаблонов один. Тесты создают собственные экземпляры с временным каталогом.
template_registry = TemplateRegistry()
