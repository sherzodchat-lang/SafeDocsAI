"""Реестр шаблонов презентаций: манифест, отбраковка и кэш.

Что закрепляем.

  * **Битая запись не роняет старт.** Шаблон — деплой-артефакт, и ошибиться в
    нём легко: забыть png, переименовать pptx, оставить в манифесте индекс
    layout'а от прежнего дизайна. Любая из этих бед выбрасывает ОДНУ запись с
    ERROR в журнал; остальные шаблоны остаются доступны, а модуль презентаций
    продолжает работать. Обратное поведение означало бы, что забытая картинка
    гасит приложение целиком.
  * **Проверяется то, на что рендерер обопрётся молча.** Существование обоих
    файлов, открываемость pptx и наличие всех четырёх layout'ов по индексам из
    манифеста. Индекс за границей списка иначе всплыл бы в фоновой задаче
    пользователя, а не при выкладке.
  * **Реестр читается один раз.** Это сознательное отличие от
    runtime_settings.json (тот перечитывается на каждое обращение): шаблоны
    меняются с релизом, а не из админ-панели. Здесь это проверяется дважды —
    манифест после первого чтения можно испортить, и ответы не изменятся, а
    python-pptx после первого чтения больше не зовут.
  * **Язык названия — tj.** ISO-код таджикского — tg, но во всём проекте язык
    обозначен как tj, и реестр обязан требовать именно его: запись с "tg"
    негодна, иначе в интерфейсе появилась бы пустая подпись.

Реальные backend/templates/presentations тесты почти не трогают: каждый случай
собирает свой каталог во временной папке — с pptx, сделанным python-pptx, и
собственным манифестом. Отдельная проверка на поставляемый комплект всё же
есть: она ловит момент, когда манифест и файлы в репозитории разъехались.
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pptx import Presentation

# Каталог с тестами в sys.path: unittest discover кладёт его туда сам, но при
# запуске `python -m unittest tests.test_presentation_templates` этого не происходит.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.modules.presentations import templates as templates_module  # noqa: E402
from app.modules.presentations.templates import (  # noqa: E402
    LAYOUT_ROLES,
    NAME_LANGUAGES,
    TemplateRegistry,
    default_templates_dir,
    template_registry,
)

LOGGER_NAME = "app.modules.presentations.templates"

# Индексы layout'ов стандартного шаблона python-pptx: 0 Title Slide,
# 1 Title and Content, 2 Section Header, 3 Two Content. Фикстурам достаточно
# того, что такие индексы в файле есть.
FIXTURE_LAYOUTS = {"title": 0, "section": 2, "bullets": 1, "sources": 3}


def make_entry(key: str, **overrides) -> dict:
    entry = {
        "key": key,
        "name": {"ru": f"Шаблон {key}", "tj": f"Қолиби {key}"},
        "template_file": f"{key}.pptx",
        "preview_file": f"{key}.png",
        "layouts": dict(FIXTURE_LAYOUTS),
    }
    entry.update(overrides)
    return entry


class TemplateRegistryTestCase(unittest.TestCase):
    """База: временный каталог шаблонов и помощники для его наполнения."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.directory = Path(self._tmp.name)

    def write_pptx(self, key: str) -> Path:
        path = self.directory / f"{key}.pptx"
        Presentation().save(str(path))
        return path

    def write_preview(self, key: str) -> Path:
        path = self.directory / f"{key}.png"
        # Содержимое картинки реестр не разбирает — ему важно, что файл есть.
        path.write_bytes(b"\x89PNG\r\n\x1a\n")
        return path

    def write_files(self, key: str) -> None:
        self.write_pptx(key)
        self.write_preview(key)

    def write_manifest(self, entries) -> Path:
        path = self.directory / "manifest.json"
        path.write_text(json.dumps(entries, ensure_ascii=False), encoding="utf-8")
        return path

    def registry(self) -> TemplateRegistry:
        return TemplateRegistry(self.directory)


class ManifestReadingTests(TemplateRegistryTestCase):
    """Чтение манифеста и форма записей реестра."""

    def test_reads_entries_in_manifest_order(self) -> None:
        for key in ("alpha", "beta"):
            self.write_files(key)
        self.write_manifest([make_entry("alpha"), make_entry("beta")])

        registry = self.registry()

        self.assertEqual([info.key for info in registry.list()], ["alpha", "beta"])

    def test_entry_fields_are_ready_for_the_renderer(self) -> None:
        self.write_files("alpha")
        self.write_manifest([make_entry("alpha")])

        info = self.registry().get("alpha")

        self.assertIsNotNone(info)
        assert info is not None
        self.assertEqual(info.key, "alpha")
        self.assertEqual(set(info.name), set(NAME_LANGUAGES))
        self.assertEqual(info.name["ru"], "Шаблон alpha")
        # Пути абсолютные и существующие: склеивать их с каталогом и проверять
        # заново вызывающему не нужно.
        self.assertTrue(info.template_file.is_absolute())
        self.assertTrue(info.template_file.is_file())
        self.assertTrue(info.preview_file.is_absolute())
        self.assertTrue(info.preview_file.is_file())
        self.assertEqual(set(info.layouts), set(LAYOUT_ROLES))
        self.assertEqual(info.layouts, FIXTURE_LAYOUTS)

    def test_unknown_key_returns_none(self) -> None:
        self.write_files("alpha")
        self.write_manifest([make_entry("alpha")])

        self.assertIsNone(self.registry().get("no-such-template"))

    def test_missing_manifest_gives_empty_registry_and_error(self) -> None:
        registry = self.registry()

        with self.assertLogs(LOGGER_NAME, level="ERROR") as captured:
            self.assertEqual(registry.list(), [])
        self.assertTrue(
            any("manifest" in message for message in captured.output), captured.output
        )

    def test_broken_manifest_json_gives_empty_registry_and_error(self) -> None:
        (self.directory / "manifest.json").write_text("{not json", encoding="utf-8")
        registry = self.registry()

        with self.assertLogs(LOGGER_NAME, level="ERROR"):
            self.assertEqual(registry.list(), [])

    def test_manifest_object_instead_of_list_is_rejected(self) -> None:
        self.write_manifest({"alpha": make_entry("alpha")})
        registry = self.registry()

        with self.assertLogs(LOGGER_NAME, level="ERROR"):
            self.assertEqual(registry.list(), [])

    def test_duplicate_key_keeps_the_first_entry(self) -> None:
        self.write_files("alpha")
        first = make_entry("alpha")
        second = make_entry("alpha", name={"ru": "Второй", "tj": "Дуюм"})
        self.write_manifest([first, second])
        registry = self.registry()

        with self.assertLogs(LOGGER_NAME, level="ERROR"):
            infos = registry.list()

        self.assertEqual(len(infos), 1)
        self.assertEqual(infos[0].name["ru"], "Шаблон alpha")


class BrokenEntryTests(TemplateRegistryTestCase):
    """Битая запись выбрасывается, соседние живут, в журнале ERROR."""

    def load_with_broken(self, broken_entry: dict) -> tuple[list[str], list[str]]:
        """Реестр из «хорошей» записи и переданной битой; вернуть ключи и лог."""
        self.write_files("good")
        self.write_manifest([broken_entry, make_entry("good")])
        registry = self.registry()

        with self.assertLogs(LOGGER_NAME, level="ERROR") as captured:
            keys = [info.key for info in registry.list()]
        return keys, captured.output

    def assert_only_good_survived(self, keys: list[str], output: list[str]) -> None:
        self.assertEqual(keys, ["good"])
        self.assertTrue(any("broken" in message for message in output), output)

    def test_missing_template_file(self) -> None:
        self.write_preview("broken")

        keys, output = self.load_with_broken(make_entry("broken"))

        self.assert_only_good_survived(keys, output)
        self.assertTrue(any("template_file" in message for message in output), output)

    def test_missing_preview_file(self) -> None:
        self.write_pptx("broken")

        keys, output = self.load_with_broken(make_entry("broken"))

        self.assert_only_good_survived(keys, output)
        self.assertTrue(any("preview_file" in message for message in output), output)

    def test_pptx_that_cannot_be_opened(self) -> None:
        # Не zip вовсе: python-pptx на таком файле бросает, и реестр обязан
        # поймать это здесь, а не в фоновой задаче сборки презентации.
        (self.directory / "broken.pptx").write_bytes(b"not a pptx at all")
        self.write_preview("broken")

        keys, output = self.load_with_broken(make_entry("broken"))

        self.assert_only_good_survived(keys, output)
        self.assertTrue(any("opened" in message for message in output), output)

    def test_layout_index_beyond_the_file(self) -> None:
        self.write_files("broken")
        layouts = dict(FIXTURE_LAYOUTS)
        layouts["sources"] = 99

        keys, output = self.load_with_broken(make_entry("broken", layouts=layouts))

        self.assert_only_good_survived(keys, output)
        self.assertTrue(any("sources" in message for message in output), output)

    def test_missing_layout_role(self) -> None:
        self.write_files("broken")
        layouts = dict(FIXTURE_LAYOUTS)
        del layouts["section"]

        keys, output = self.load_with_broken(make_entry("broken", layouts=layouts))

        self.assert_only_good_survived(keys, output)

    def test_layout_index_that_is_not_an_index(self) -> None:
        self.write_files("broken")
        layouts = dict(FIXTURE_LAYOUTS)
        layouts["title"] = "0"

        keys, output = self.load_with_broken(make_entry("broken", layouts=layouts))

        self.assert_only_good_survived(keys, output)

    def test_name_without_tajik_is_rejected(self) -> None:
        # ISO-код таджикского — tg, но канон проекта — tj: запись с "tg"
        # негодна, иначе подпись шаблона в интерфейсе оказалась бы пустой.
        self.write_files("broken")
        entry = make_entry("broken", name={"ru": "Шаблон", "tg": "Қолиб"})

        keys, output = self.load_with_broken(entry)

        self.assert_only_good_survived(keys, output)
        self.assertTrue(any("'tj'" in message for message in output), output)

    def test_entry_without_key_is_rejected(self) -> None:
        self.write_files("good")
        nameless = {"name": {"ru": "Без ключа", "tj": "Бе калид"}}
        self.write_manifest([nameless, make_entry("good")])
        registry = self.registry()

        with self.assertLogs(LOGGER_NAME, level="ERROR"):
            self.assertEqual([info.key for info in registry.list()], ["good"])

    def test_file_outside_the_templates_directory_is_rejected(self) -> None:
        # Путь из манифеста не выпускается за пределы каталога шаблонов: иначе
        # выдача превью превращалась бы в чтение произвольного файла.
        outside = self.directory.parent / "outside.png"
        outside.write_bytes(b"\x89PNG\r\n\x1a\n")
        self.addCleanup(outside.unlink)
        self.write_pptx("broken")

        keys, output = self.load_with_broken(
            make_entry("broken", preview_file="../outside.png")
        )

        self.assert_only_good_survived(keys, output)
        self.assertTrue(any("outside" in message for message in output), output)

    def test_all_entries_broken_leaves_empty_registry_without_raising(self) -> None:
        self.write_manifest([make_entry("one"), make_entry("two")])
        registry = self.registry()

        with self.assertLogs(LOGGER_NAME, level="ERROR"):
            self.assertEqual(registry.list(), [])
        self.assertIsNone(registry.get("one"))


class CacheTests(TemplateRegistryTestCase):
    """Манифест читается один раз за жизнь процесса."""

    def test_manifest_is_not_reread_after_the_first_access(self) -> None:
        self.write_files("alpha")
        manifest = self.write_manifest([make_entry("alpha")])
        registry = self.registry()
        self.assertEqual([info.key for info in registry.list()], ["alpha"])

        # Файл после первого чтения испорчен вдребезги. Реестр обязан отвечать
        # тем же, что прочитал при старте: шаблоны меняются с релизом, и
        # заглядывать на диск повторно ему незачем.
        manifest.write_text("{ garbage", encoding="utf-8")

        self.assertEqual([info.key for info in registry.list()], ["alpha"])
        self.assertIsNotNone(registry.get("alpha"))

    def test_pptx_is_not_reopened_after_the_first_access(self) -> None:
        self.write_files("alpha")
        self.write_manifest([make_entry("alpha")])
        registry = self.registry()
        registry.list()

        # Открытие pptx — самая дорогая часть чтения; после прогрева его не
        # должно быть вовсе. Подменяем python-pptx взрывающейся заглушкой.
        def explode(*args, **kwargs):
            raise AssertionError("pptx re-opened on a cached registry")

        with patch.object(templates_module, "Presentation", explode):
            self.assertEqual([info.key for info in registry.list()], ["alpha"])
            self.assertIsNotNone(registry.get("alpha"))

    def test_empty_result_is_cached_too(self) -> None:
        # Пустой реестр — тоже прочитанное состояние, а не «ещё не читали»:
        # иначе отсутствующий манифест давал бы попытку чтения (и строку ERROR)
        # на каждое обращение к списку шаблонов.
        registry = self.registry()
        with self.assertLogs(LOGGER_NAME, level="ERROR"):
            self.assertEqual(registry.list(), [])

        self.write_files("alpha")
        self.write_manifest([make_entry("alpha")])

        with patch.object(templates_module, "Presentation", None):
            self.assertEqual(registry.list(), [])


class ShippedTemplatesTests(unittest.TestCase):
    """Комплект, который лежит в репозитории и уезжает в образ."""

    def test_default_directory_is_next_to_backend_data(self) -> None:
        directory = default_templates_dir()

        self.assertEqual(directory.name, "presentations")
        self.assertEqual(directory.parent.name, "templates")
        # Каталог шаблонов — сосед data/, а не его содержимое: data монтируется
        # томом и переживает выкладку, шаблоны обязаны приезжать с кодом.
        self.assertTrue((directory.parent.parent / "data").exists())

    def test_shipped_manifest_and_files_are_consistent(self) -> None:
        # Единственная проверка на настоящие файлы: она ловит момент, когда
        # манифест и комплект pptx/png в репозитории разъехались.
        infos = template_registry.list()

        self.assertGreaterEqual(len(infos), 2)
        for info in infos:
            with self.subTest(template=info.key):
                self.assertEqual(set(info.name), set(NAME_LANGUAGES))
                self.assertEqual(set(info.layouts), set(LAYOUT_ROLES))
                self.assertTrue(info.template_file.is_file())
                self.assertTrue(info.preview_file.is_file())
                self.assertIs(template_registry.get(info.key), info)

    def test_shipped_manifest_has_no_rejected_entries(self) -> None:
        manifest_path = default_templates_dir() / "manifest.json"
        entries = json.loads(manifest_path.read_text(encoding="utf-8"))

        self.assertEqual(
            sorted(entry["key"] for entry in entries),
            sorted(info.key for info in template_registry.list()),
        )


if __name__ == "__main__":
    unittest.main()
