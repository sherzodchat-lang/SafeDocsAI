"""Реестр шаблонов презентаций: манифест на диске -> проверенные записи.

Шаблон — это ДЕПЛОЙ-АРТЕФАКТ: pptx с оформлением, картинка-превью и запись в
templates/presentations/manifest.json. Меняется он вместе с релизом, а не из
админ-панели, поэтому реестр читается один раз и живёт в памяти. Это
сознательное отличие от runtime_settings.json, который перечитывается на каждое
обращение: там значение правит админ прямо во время работы, здесь — никто,
файлы приезжают из образа. Перечитывание дало бы разбор JSON и открытие трёх
pptx на каждый запрос списка шаблонов и не изменило бы ни одного ответа.

Битая запись не роняет старт. Отсутствующий файл, нечитаемый pptx или индекс
layout'а за пределами файла — это ошибка сборки одного шаблона, а не отказ
модуля презентаций: такая запись выбрасывается с ERROR в журнал, остальные
работают. Обратное поведение (падать на старте) означало бы, что забытый в
.dockerignore png гасит приложение целиком, включая экраны, шаблонов не
касающиеся.

Пустой реестр — тоже допустимое состояние; отвечать за «шаблонов нет вообще»
будет вызывающий код, который выбирает шаблон под запрос пользователя.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pptx import Presentation

logger = logging.getLogger(__name__)

MANIFEST_FILENAME = "manifest.json"

# Языки, на которых обязано быть название шаблона: это подписи в интерфейсе
# выбора, и отсутствие одного из них означает пустую строку в списке.
#
# ISO был бы tg; используем tj для согласованности с i18n проекта — таджикский
# обозначен как "tj" везде: определение языка документа, доменные профили,
# инструкции модели, словари фронтенда. Третий код языка в одной системе
# опаснее расхождения со стандартом: сравнение language == "tj" где-нибудь в
# рендерере молча даст False, и презентация уедет на русском.
NAME_LANGUAGES = ("ru", "tj")

# Роли layout'ов, которые обязан предоставить шаблон. Их ровно четыре, потому
# что столько видов слайдов умеет собирать рендерер: титул, разделитель,
# слайд с буллитами и финальные «Источники». Ключи здесь — контракт с
# рендерером, а числа за ними живут в манифесте каждого шаблона: у разных
# дизайнов порядок layout'ов внутри pptx разный.
LAYOUT_ROLES = ("title", "section", "bullets", "sources")


@dataclass(frozen=True)
class TemplateInfo:
    """Проверенная запись реестра.

    Пути — абсолютные и уже существующие на момент создания записи: рендереру
    не нужно ни склеивать их с каталогом, ни проверять существование заново.
    """

    key: str
    name: dict[str, str]
    template_file: Path
    preview_file: Path
    layouts: dict[str, int]


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
    backend_dir = Path(__file__).resolve().parents[3]
    return backend_dir / "templates" / "presentations"


class TemplateRegistry:
    """Реестр шаблонов, прочитанный с диска один раз.

    Чтение ленивое, а не на импорте модуля: импорт тянет за собой открытие всех
    pptx, и тогда любой тест, случайно импортировавший модуль, платил бы за
    разбор трёх файлов, а сбой файловой системы превращался бы в ImportError —
    ошибку, из которой не видно, что дело в шаблонах.
    """

    def __init__(self, directory: Path | None = None) -> None:
        # None означает «каталог по умолчанию, вычисленный на момент чтения».
        # Тесты передают временный каталог и работают с собственным манифестом.
        self._directory = Path(directory) if directory is not None else None
        self._templates: dict[str, TemplateInfo] | None = None
        # Первое обращение может прийти из нескольких потоков сразу (FastAPI
        # уводит синхронные обработчики в пул): без блокировки манифест
        # разбирался бы дважды, и в журнале двоились бы строки об ошибках.
        self._lock = threading.Lock()

    @property
    def directory(self) -> Path:
        if self._directory is not None:
            return self._directory
        return default_templates_dir()

    def get(self, key: str) -> TemplateInfo | None:
        """Шаблон по ключу или None, если такого нет (или он битый)."""
        return self._loaded().get(key)

    def list(self) -> list[TemplateInfo]:
        """Пригодные шаблоны в порядке манифеста."""
        return list(self._loaded().values())

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
        templates: dict[str, TemplateInfo] = {}
        for position, entry in enumerate(entries):
            info = self._build(entry, position, directory)
            if info is None:
                continue
            if info.key in templates:
                # Дубликат ключа — не «последний побеждает»: выбор шаблона
                # пришёл бы к одному файлу, а превью в списке показывалось бы
                # от другого. Оставляем первый и говорим об этом.
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

    def _read_manifest(self, manifest_path: Path) -> list[Any]:
        """Список записей манифеста; при любой беде — пустой список и ERROR.

        Отсутствие манифеста здесь не «норма чистой установки», как у
        runtime_settings.json: файл кладёт сборка, и без него презентации
        собрать не из чего. Но и падать нельзя — молчать тоже, поэтому ERROR.
        """
        try:
            raw = manifest_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            logger.error("Presentation templates manifest %s is missing", manifest_path)
            return []
        except OSError as exc:
            logger.error(
                "Presentation templates manifest %s is unreadable: %s",
                manifest_path,
                exc,
            )
            return []

        try:
            # ValueError накрывает и JSONDecodeError, и UnicodeDecodeError.
            data = json.loads(raw)
        except ValueError as exc:
            logger.error(
                "Presentation templates manifest %s is not valid JSON: %s",
                manifest_path,
                exc,
            )
            return []

        if not isinstance(data, list):
            logger.error(
                "Presentation templates manifest %s must contain a list, got %s",
                manifest_path,
                type(data).__name__,
            )
            return []
        return data

    def _build(
        self, entry: Any, position: int, directory: Path
    ) -> TemplateInfo | None:
        """Одна запись манифеста -> TemplateInfo или None с ERROR в журнале.

        Проверяется всё, на что рендерер обопрётся молча: форма записи, наличие
        обоих файлов, открываемость pptx и присутствие в нём всех четырёх
        layout'ов по индексам из манифеста. Последнее — не педантизм: индекс за
        границей списка обнаружится только в момент сборки презентации
        пользователя, то есть в фоновой задаче, а не при выкладке.
        """
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

        name = self._build_name(key, entry.get("name"))
        if name is None:
            return None

        layouts = self._build_layouts(key, entry.get("layouts"))
        if layouts is None:
            return None

        template_file = self._resolve_file(
            key, entry.get("template_file"), directory, "template_file"
        )
        if template_file is None:
            return None

        preview_file = self._resolve_file(
            key, entry.get("preview_file"), directory, "preview_file"
        )
        if preview_file is None:
            return None

        if not self._layouts_present(key, template_file, layouts):
            return None

        return TemplateInfo(
            key=key,
            name=name,
            template_file=template_file,
            preview_file=preview_file,
            layouts=layouts,
        )

    @staticmethod
    def _build_name(key: str, raw: Any) -> dict[str, str] | None:
        if not isinstance(raw, dict):
            logger.error(
                "Presentation template %r has no 'name' object; skipped", key
            )
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
    def _build_layouts(key: str, raw: Any) -> dict[str, int] | None:
        if not isinstance(raw, dict):
            logger.error(
                "Presentation template %r has no 'layouts' object; skipped", key
            )
            return None
        layouts: dict[str, int] = {}
        for role in LAYOUT_ROLES:
            value = raw.get(role)
            # isinstance(True, int) истинно, а layout под номером True — это
            # индекс 1 по случайности, а не по замыслу манифеста.
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                logger.error(
                    "Presentation template %r has invalid layout index "
                    "for role %r: %r; skipped",
                    key,
                    role,
                    value,
                )
                return None
            layouts[role] = value
        return layouts

    @staticmethod
    def _resolve_file(
        key: str, raw: Any, directory: Path, field: str
    ) -> Path | None:
        if not isinstance(raw, str) or not raw.strip():
            logger.error(
                "Presentation template %r has no %r; skipped", key, field
            )
            return None

        path = (directory / raw.strip()).resolve()
        # Манифест — свой же артефакт, но путь из него всё равно не выпускается
        # за пределы каталога шаблонов: запись вида "../../data/uploads/…"
        # превратила бы выдачу превью в чтение произвольного файла на диске.
        if not path.is_relative_to(directory.resolve()):
            logger.error(
                "Presentation template %r has %s %r outside %s; skipped",
                key,
                field,
                raw,
                directory,
            )
            return None

        if not path.is_file():
            logger.error(
                "Presentation template %r has %s %s that does not exist; skipped",
                key,
                field,
                path,
            )
            return None
        return path

    @staticmethod
    def _layouts_present(key: str, template_file: Path, layouts: dict[str, int]) -> bool:
        try:
            presentation = Presentation(str(template_file))
        except Exception as exc:  # noqa: BLE001
            # Перечислить исключения python-pptx нельзя: битый zip даёт
            # PackageNotFoundError, zip не от pptx — KeyError на отсутствующей
            # части, обрезанный XML — ошибку lxml. Здесь любое из них означает
            # одно и то же: файл к сборке презентации непригоден.
            logger.error(
                "Presentation template %r cannot be opened (%s): %s",
                key,
                template_file,
                exc,
            )
            return False

        available = len(presentation.slide_layouts)
        for role, index in layouts.items():
            if index >= available:
                logger.error(
                    "Presentation template %r points role %r at layout %d, "
                    "but %s has only %d layouts; skipped",
                    key,
                    role,
                    index,
                    template_file.name,
                    available,
                )
                return False
        return True


# Единственный экземпляр на процесс: кэш имеет смысл только общий, а каталог
# шаблонов один. Тесты создают собственные экземпляры с временным каталогом.
template_registry = TemplateRegistry()
