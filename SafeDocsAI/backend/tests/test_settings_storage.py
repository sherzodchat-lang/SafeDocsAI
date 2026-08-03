"""Хранение настроек: запись, чтение и сброс runtime_settings.json.

Что закрепляем.

  * **Запись атомарна.** Было `path.write_text(...)`: он открывает файл на
    запись, то есть СНАЧАЛА обрезает его в ноль и только потом наполняет.
    Настройки читают все — чат, поиск, индексация, — и читатель, попавший в
    это окно, получал обрезанный JSON. Дальше срабатывал молчаливый
    `except Exception: data = {}`, то есть откат на умолчания — включая ДРУГУЮ
    embedding-модель, а имя коллекции ChromaDB выводится из неё: поиск уезжал
    в коллекцию, которую никто не заполнял, и отвечал так, будто документов
    нет. Теперь содержимое пишется во временный файл в том же каталоге и
    переставляется через os.replace.
  * **Порча файла не проходит молча.** Отсутствие файла — норма (чистая
    установка), нечитаемый или битый файл — авария, и о ней должна остаться
    строка в журнале с уровнем ERROR.
  * **Read-modify-write идёт под блокировкой.** update_settings читает
    настройки целиком и целиком же переписывает; два сохранения в одном окне
    затирали правки друг друга.
  * **Сброс к умолчаниям существует.** Вернуться назад было нельзя: сброса в
    разделе нет, а любой удачный PUT оставляет постоянный файл. Сброс
    возвращает и embedding_model, поэтому подтверждается так же, как её ручная
    смена.

Базы данных здесь не нужно: RuntimeSettingsService в неё не ходит. Каталог
моделей подменён — иначе валидация ходила бы в Ollama. Файл настроек живёт во
временном каталоге: рабочий backend/data/runtime_settings.json тесты не
трогают.
"""

import asyncio
import json
import os
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

# Каталог с тестами в sys.path: unittest discover кладёт его туда сам, но при
# запуске `python -m unittest tests.test_settings_storage` этого не происходит.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.core.exceptions import SettingsError, SettingsErrors  # noqa: E402
from app.shared.settings import runtime_settings as runtime_settings_module  # noqa: E402
from app.shared.settings.runtime_settings import RuntimeSettingsService  # noqa: E402


LOGGER_NAME = "app.shared.settings.runtime_settings"

CHAT_MODEL = "gemma4:26b"
# Умолчание должно оставаться выбираемым: иначе к нему нельзя вернуться ни
# сбросом, ни руками.
DEFAULT_EMBEDDING = RuntimeSettingsService.DEFAULTS["embedding_model"]
OTHER_EMBEDDING = "bge-m3"

FAKE_CATALOG = {
    "available_models": [CHAT_MODEL, DEFAULT_EMBEDDING, OTHER_EMBEDDING],
    "available_chat_models": [CHAT_MODEL],
    "available_embedding_models": [DEFAULT_EMBEDDING, OTHER_EMBEDDING],
    "ollama_available": True,
    "ollama_error": None,
}


class SettingsFileMixin:
    """Файл настроек во временном каталоге и каталог моделей без Ollama."""

    def set_up_settings_file(self) -> None:
        self._dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._dir.cleanup)
        self.settings_path = Path(self._dir.name) / "runtime_settings.json"

        path_patcher = patch.object(
            RuntimeSettingsService, "_settings_path", return_value=self.settings_path
        )
        path_patcher.start()
        self.addCleanup(path_patcher.stop)

        catalog_patcher = patch.object(
            RuntimeSettingsService, "model_catalog", return_value=FAKE_CATALOG
        )
        catalog_patcher.start()
        self.addCleanup(catalog_patcher.stop)

    def saved(self) -> dict:
        return json.loads(self.settings_path.read_text(encoding="utf-8"))

    def leftovers(self) -> list[str]:
        """Всё, что осталось в каталоге настроек кроме самого файла."""
        return sorted(
            entry.name
            for entry in Path(self._dir.name).iterdir()
            if entry.name != self.settings_path.name
        )


# --- Атомарность записи -------------------------------------------------


class AtomicWriteTests(SettingsFileMixin, unittest.TestCase):
    def setUp(self) -> None:
        self.set_up_settings_file()

    def test_the_file_readers_look_at_is_never_seen_truncated(self):
        """Главная проверка: в момент подмены на месте лежит ПРЕЖНИЙ файл целиком.

        os.replace перехвачен, потому что окно между обрезанием и наполнением
        у write_text измеряется микросекундами и ловится только случайно.
        Здесь же вопрос задан детерминированно: что видит читатель в последний
        момент перед подменой. С write_text ответ был бы «пустой файл».
        """
        RuntimeSettingsService.update_settings({"top_k": 7})
        before = self.settings_path.read_text(encoding="utf-8")

        seen: list[str] = []
        real_replace = os.replace

        def spying_replace(src, dst):
            # Файл на месте назначения обязан быть либо целым старым, либо
            # отсутствовать — но не обрезанным.
            seen.append(Path(dst).read_text(encoding="utf-8"))
            return real_replace(src, dst)

        with patch.object(runtime_settings_module.os, "replace", spying_replace):
            RuntimeSettingsService.update_settings({"top_k": 9})

        self.assertEqual(len(seen), 1, "запись должна идти ровно одной подменой")
        self.assertEqual(seen[0], before, "старый файл был испорчен до подмены")
        self.assertEqual(json.loads(seen[0])["top_k"], 7)
        self.assertEqual(self.saved()["top_k"], 9)

    def test_a_parallel_reader_never_sees_invalid_json(self):
        """То же самое, но без подмен: читатель в отдельном потоке.

        Полезная нагрузка намеренно раздута — с прежней записью через
        write_text окно обрезания растягивается до заметного, и битые чтения
        пошли бы пачками.
        """
        RuntimeSettingsService._write_settings({"top_k": 5, "pad": "A" * 2_000_000})

        broken: list[str] = []
        reads = 0
        stop = threading.Event()

        def reader():
            nonlocal reads
            while not stop.is_set():
                try:
                    json.loads(self.settings_path.read_text(encoding="utf-8"))
                    reads += 1
                except Exception as exc:  # noqa: BLE001 - ровно это и ловим
                    broken.append(f"{type(exc).__name__}: {exc}")

        thread = threading.Thread(target=reader)
        thread.start()
        try:
            for index in range(30):
                RuntimeSettingsService._write_settings(
                    {"top_k": index, "pad": "A" * 2_000_000}
                )
        finally:
            stop.set()
            thread.join(timeout=10)

        self.assertEqual(broken, [], "читатель увидел файл в промежуточном виде")
        self.assertGreater(reads, 0, "читатель не успел ни разу прочитать файл")

    def test_the_temporary_file_lives_in_the_same_directory(self):
        """os.replace атомарен только в пределах одной файловой системы.

        Временный файл из /tmp переехал бы через копирование, то есть ровно с
        тем окном, ради устранения которого всё и затевалось.
        """
        seen_dirs: list[str] = []
        real_mkstemp = tempfile.mkstemp

        def spying_mkstemp(*args, **kwargs):
            seen_dirs.append(kwargs.get("dir"))
            return real_mkstemp(*args, **kwargs)

        with patch.object(
            runtime_settings_module.tempfile, "mkstemp", spying_mkstemp
        ):
            RuntimeSettingsService.update_settings({"top_k": 6})

        self.assertEqual(seen_dirs, [str(self.settings_path.parent)])

    def test_a_successful_write_leaves_no_temporary_files(self):
        RuntimeSettingsService.update_settings({"top_k": 7})

        self.assertEqual(self.leftovers(), [])

    def test_a_failed_write_keeps_the_old_file_and_cleans_up_after_itself(self):
        """Сорванная запись не должна ни портить настройки, ни сорить в data/."""
        RuntimeSettingsService.update_settings({"top_k": 7})
        before = self.settings_path.read_text(encoding="utf-8")

        def failing_replace(src, dst):
            raise OSError("диск переполнен")

        with patch.object(runtime_settings_module.os, "replace", failing_replace):
            with self.assertRaises(OSError):
                RuntimeSettingsService.update_settings({"top_k": 9})

        self.assertEqual(self.settings_path.read_text(encoding="utf-8"), before)
        self.assertEqual(self.leftovers(), [])


# --- Чтение: журнал вместо тишины ---------------------------------------


class SettingsReadLoggingTests(SettingsFileMixin, unittest.TestCase):
    def setUp(self) -> None:
        self.set_up_settings_file()

    def test_a_broken_file_is_reported_as_an_incident(self):
        self.settings_path.write_text("{ это не json", encoding="utf-8")

        with self.assertLogs(LOGGER_NAME, level="ERROR") as captured:
            values = RuntimeSettingsService.get_settings()

        message = "\n".join(captured.output)
        self.assertIn(str(self.settings_path), message)
        self.assertIn("JSONDecodeError", message)
        # Тихий откат и был опасен тем, что менял embedding-модель, поэтому она
        # названа в сообщении прямо.
        self.assertIn(RuntimeSettingsService.DEFAULTS["embedding_model"], message)
        self.assertEqual(
            values["embedding_model"],
            RuntimeSettingsService.DEFAULTS["embedding_model"],
        )

    def test_a_file_that_is_not_an_object_is_reported_too(self):
        """JSON разобрался, но настройками не является — тот же откат."""
        self.settings_path.write_text('["top_k", 7]', encoding="utf-8")

        with self.assertLogs(LOGGER_NAME, level="ERROR") as captured:
            values = RuntimeSettingsService.get_settings()

        self.assertIn("list", "\n".join(captured.output))
        self.assertEqual(values["top_k"], RuntimeSettingsService.DEFAULTS["top_k"])

    def test_an_unreadable_file_is_reported_as_an_incident(self):
        self.settings_path.write_text("{}", encoding="utf-8")

        with patch.object(
            runtime_settings_module.Path,
            "read_text",
            side_effect=PermissionError("отказано в доступе"),
        ):
            with self.assertLogs(LOGGER_NAME, level="ERROR") as captured:
                RuntimeSettingsService.get_settings()

        self.assertIn("PermissionError", "\n".join(captured.output))

    def test_a_missing_file_is_normal_and_stays_silent(self):
        """Чистая установка — не авария: журнал засорять нечем."""
        self.assertFalse(self.settings_path.exists())

        with self.assertNoLogs(LOGGER_NAME, level="WARNING"):
            values = RuntimeSettingsService.get_settings()

        self.assertEqual(values["top_k"], RuntimeSettingsService.DEFAULTS["top_k"])

    def test_a_file_vanishing_mid_read_is_not_an_incident_either(self):
        """Между exists() и чтением прошёл сброс настроек — это не поломка."""
        self.settings_path.write_text("{}", encoding="utf-8")

        with patch.object(
            runtime_settings_module.Path,
            "read_text",
            side_effect=FileNotFoundError("нет такого файла"),
        ):
            with self.assertLogs(LOGGER_NAME, level="INFO") as captured:
                RuntimeSettingsService.get_settings()

        output = "\n".join(captured.output)
        self.assertIn("disappeared", output)
        self.assertNotIn("ERROR", output)


# --- Блокировка ---------------------------------------------------------


class SettingsWriteLockTests(SettingsFileMixin, unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.set_up_settings_file()

    async def test_a_second_write_waits_for_the_first(self):
        lock = RuntimeSettingsService._lock()
        await lock.acquire()
        try:
            task = asyncio.create_task(
                RuntimeSettingsService.update_settings_locked({"top_k": 7})
            )
            done, _ = await asyncio.wait({task}, timeout=0.2)

            self.assertEqual(done, set(), "запись прошла мимо блокировки")
        finally:
            lock.release()

        await asyncio.wait_for(task, timeout=10)
        self.assertEqual(self.saved()["top_k"], 7)

    async def test_parallel_updates_do_not_lose_each_other(self):
        """Правка первого не должна пропадать под записью второго."""
        await asyncio.gather(
            RuntimeSettingsService.update_settings_locked({"top_k": 7}),
            RuntimeSettingsService.update_settings_locked({"retrieval_top_k": 33}),
        )

        saved = self.saved()
        self.assertEqual(saved["top_k"], 7)
        self.assertEqual(saved["retrieval_top_k"], 33)


# --- Сброс к умолчаниям -------------------------------------------------


class ResetSettingsTests(SettingsFileMixin, unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.set_up_settings_file()

    async def test_reset_returns_every_setting_to_its_default(self):
        await RuntimeSettingsService.update_settings_locked(
            {"top_k": 7, "retrieval_top_k": 33, "chat_model": CHAT_MODEL}
        )

        restored = await RuntimeSettingsService.reset_settings()

        defaults = RuntimeSettingsService.DEFAULTS
        self.assertEqual(restored["top_k"], defaults["top_k"])
        self.assertEqual(restored["retrieval_top_k"], defaults["retrieval_top_k"])
        self.assertEqual(restored["chat_model"], defaults["chat_model"])
        # И на диске тоже: иначе следующий запуск поднялся бы с прежними.
        self.assertEqual(self.saved()["top_k"], defaults["top_k"])

    async def test_reset_that_changes_the_embedding_model_needs_confirmation(self):
        """У сброса те же последствия, что у ручной смены модели."""
        await RuntimeSettingsService.update_settings_locked(
            {"embedding_model": OTHER_EMBEDDING, "confirm_reindex": True}
        )
        before = self.saved()

        with self.assertRaises(SettingsError) as raised:
            await RuntimeSettingsService.reset_settings()

        self.assertEqual(
            raised.exception.error_code,
            SettingsErrors.REINDEX_CONFIRMATION_REQUIRED,
        )
        self.assertEqual(self.saved(), before, "отклонённый сброс ничего не менял")

    async def test_confirmed_reset_switches_the_model_back_and_flags_a_reindex(self):
        await RuntimeSettingsService.update_settings_locked(
            {"embedding_model": OTHER_EMBEDDING, "confirm_reindex": True}
        )

        restored = await RuntimeSettingsService.reset_settings(confirm_reindex=True)

        self.assertEqual(
            restored["embedding_model"],
            RuntimeSettingsService.DEFAULTS["embedding_model"],
        )
        self.assertTrue(restored["reindex_required"])

    async def test_reset_without_a_model_change_does_not_ask_for_confirmation(self):
        """Обычный случай: embedding-модель никто не трогал.

        Требовать подтверждение и здесь значило бы приучить клиента слать
        confirm_reindex=true всегда — и предохранитель перестал бы работать.
        """
        await RuntimeSettingsService.update_settings_locked({"top_k": 7})

        restored = await RuntimeSettingsService.reset_settings()

        self.assertEqual(
            restored["top_k"], RuntimeSettingsService.DEFAULTS["top_k"]
        )
        self.assertFalse(restored["reindex_required"])

    async def test_reset_does_not_forget_a_reindex_that_was_already_due(self):
        """Флаг описывает состояние ChromaDB, а не настройку.

        Долг за прежней сменой модели остаётся долгом: сброс настроек векторы
        не пересчитывает.
        """
        await RuntimeSettingsService.update_settings_locked(
            {"embedding_model": OTHER_EMBEDDING, "confirm_reindex": True}
        )
        # Возвращаемся к умолчанию руками — теперь сброс модель не меняет, но
        # переиндексация всё ещё не выполнена.
        await RuntimeSettingsService.update_settings_locked(
            {
                "embedding_model": RuntimeSettingsService.DEFAULTS["embedding_model"],
                "confirm_reindex": True,
            }
        )

        restored = await RuntimeSettingsService.reset_settings()

        self.assertTrue(restored["reindex_required"])


# --- Флаг переиндексации ------------------------------------------------


class ReindexFlagTests(SettingsFileMixin, unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.set_up_settings_file()

    async def test_the_flag_is_raised_by_a_confirmed_model_change(self):
        updated = await RuntimeSettingsService.update_settings_locked(
            {"embedding_model": OTHER_EMBEDDING, "confirm_reindex": True}
        )

        self.assertTrue(updated["reindex_required"])
        self.assertTrue(self.saved()["reindex_required"])
        self.assertTrue(RuntimeSettingsService.get_settings()["reindex_required"])

    async def test_the_confirmation_flag_itself_is_not_a_setting(self):
        """confirm_reindex — признак операции: в файле ему не место."""
        await RuntimeSettingsService.update_settings_locked(
            {"top_k": 7, "confirm_reindex": True}
        )

        self.assertNotIn("confirm_reindex", self.saved())

    async def test_clearing_the_flag_keeps_the_rest_of_the_settings(self):
        await RuntimeSettingsService.update_settings_locked(
            {"top_k": 7, "embedding_model": OTHER_EMBEDDING, "confirm_reindex": True}
        )

        self.assertFalse(await RuntimeSettingsService.clear_reindex_required())

        saved = self.saved()
        self.assertFalse(saved["reindex_required"])
        self.assertEqual(saved["top_k"], 7)
        self.assertEqual(saved["embedding_model"], OTHER_EMBEDDING)

    async def test_clearing_an_unset_flag_does_not_create_a_settings_file(self):
        """Иначе переиндексация на чистой установке замораживала бы умолчания
        в постоянном файле — ровно то, что мешает вернуться назад."""
        self.assertFalse(self.settings_path.exists())

        self.assertFalse(await RuntimeSettingsService.clear_reindex_required())

        self.assertFalse(self.settings_path.exists())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
