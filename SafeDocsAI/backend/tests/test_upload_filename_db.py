"""Имя файла при загрузке источника: длина и разделители каталогов.

Что закрепляем.

  * Загрузка с ЛЕГИТИМНЫМ длинным именем проходит. Раньше граница проходила
    по пределу ext4: имя в 222 байта вместе с uuid-префиксом в 33 давало ровно
    255 и записывалось, а 223 байта давали 256 — open() отвечал
    OSError [Errno 36], DocumentModuleService.upload_document ловил только
    ValueError, и наружу уходил пустой 500 без JSON и без error_code.
    Теперь имя на диске обрезается до предела файловой системы с сохранением
    расширения, а полное имя остаётся в document.name.
  * Если файловая система откажет и после обрезки (свой NAME_MAX строже
    ext4 — так живёт eCryptfs со своими 143 байтами), клиент получает 400 с
    кодом source.filename_too_long, а не 500. Проверяется на настоящем
    open(), а не на подменённом: интересен именно errno ENAMETOOLONG.
  * Файла-сироты не остаётся ни в одном из отказов: документа в БД ещё нет,
    и убрать записанный хвост больше некому.
  * document.name не хранит разделители каталогов. На запись они и раньше не
    влияли (файл ложится под собственным uuid-именем), но это же значение
    уходит в GET /sources/ и в Content-Disposition у GET /{id}/preview.

Почему настоящий PostgreSQL, а не заглушка сессии: проверки смотрят на то,
что реально записано в document.name, а загрузка пишет document и job одной
транзакцией через настоящий внешний ключ job.source_id -> document.id.

Тяжёлые части не поднимаем: lifespan через ASGITransport не запускается,
поэтому фоновый воркер индексации не стартует, а обработчик upload только
кладёт задачу в очередь.
"""

import os
import sys
import unittest
from unittest.mock import patch

# Каталог с тестами в sys.path: unittest discover кладёт его туда сам, но при
# запуске `python -m unittest tests.test_upload_filename_db` этого не происходит.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dbfixtures import DatabaseBackedTestCase  # noqa: E402

import app.services.document_service as document_service_module  # noqa: E402
from app.api.endpoints.documents import upload_limiter  # noqa: E402
from app.core.exceptions import SourceErrors  # noqa: E402
from app.services.document_service import (  # noqa: E402
    MAX_NAME_BYTES,
    STORED_NAME_PREFIX_BYTES,
)
from app.shared.models import Document, Notebook  # noqa: E402


SOURCES = "/api/v1/sources"

# Сколько байт остаётся имени на диске после uuid-префикса. Ровно эта величина
# и была границей 500 на стенде.
NAME_BUDGET = MAX_NAME_BYTES - STORED_NAME_PREFIX_BYTES


def name_of_length(total_bytes: int, suffix: str = ".txt") -> str:
    """Имя ровно в total_bytes байт, оканчивающееся на suffix."""
    return "A" * (total_bytes - len(suffix.encode("utf-8"))) + suffix


class UploadFilenameTestCase(DatabaseBackedTestCase):
    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()

        self.user = await self.make_user("owner", "user")
        self.notebook = await self.seed(
            Notebook(
                name="Свой блокнот",
                description=None,
                domain_profile="general",
                owner_id=self.user.id,
            )
        )

        # Загруженные файлы — во временный каталог, а не в data/uploads.
        self.uploads = os.path.join(self._tmpdir.name, "uploads")
        os.makedirs(self.uploads, exist_ok=True)
        upload_patcher = patch.object(
            document_service_module, "UPLOAD_DIR", self.uploads
        )
        upload_patcher.start()
        self.addCleanup(upload_patcher.stop)

        # Счётчики лимитера живут в памяти процесса и общие на весь набор.
        upload_limiter.clients.clear()
        self.addCleanup(upload_limiter.clients.clear)

        self.as_user(self.user)

    async def upload(self, name: str):
        return await self.client.post(
            f"{SOURCES}/upload",
            data={"notebook_id": str(self.notebook.id)},
            files={"file": (name, b"tekst novogo istochnika", "text/plain")},
        )

    def stored_files(self) -> list[str]:
        return os.listdir(self.uploads)


class LongFilenameTests(UploadFilenameTestCase):
    async def test_name_at_the_filesystem_limit_is_accepted(self):
        """222 байта + префикс = 255: работало и раньше, не должно сломаться."""
        name = name_of_length(NAME_BUDGET)

        response = await self.upload(name)

        self.assertEqual(response.status_code, 200, response.text)
        stored = await self.get_row(Document, response.json()["id"])
        self.assertEqual(stored.name, name)
        self.assertTrue(os.path.exists(stored.path))
        self.assertEqual(
            len(os.path.basename(stored.path).encode("utf-8")), MAX_NAME_BYTES
        )

    async def test_name_over_the_filesystem_limit_no_longer_returns_500(self):
        """223 байта + префикс = 256. Здесь стенд отвечал пустым 500."""
        name = name_of_length(NAME_BUDGET + 1)

        response = await self.upload(name)

        self.assertEqual(response.status_code, 200, response.text)
        stored = await self.get_row(Document, response.json()["id"])
        self.assertTrue(
            os.path.exists(stored.path), "файл не записан на диск"
        )
        # Имя на диске обрезано до предела файловой системы...
        basename = os.path.basename(stored.path)
        self.assertLessEqual(len(basename.encode("utf-8")), MAX_NAME_BYTES)
        self.assertTrue(basename.endswith(".txt"), basename)
        # ...а пользователю показывается его собственное имя целиком.
        self.assertEqual(stored.name, name)

    async def test_absurdly_long_name_is_truncated_for_the_indexed_column(self):
        """document.name лежит под btree-индексом: строка в килобайты роняет
        INSERT по размеру индексной записи — снова 500, только глубже."""
        name = name_of_length(4000)

        response = await self.upload(name)

        self.assertEqual(response.status_code, 200, response.text)
        stored = await self.get_row(Document, response.json()["id"])
        self.assertLessEqual(len(stored.name.encode("utf-8")), MAX_NAME_BYTES)
        self.assertTrue(stored.name.endswith(".txt"), stored.name)
        self.assertTrue(os.path.exists(stored.path))

    async def test_multibyte_name_is_cut_on_a_character_boundary(self):
        """NAME_MAX считается в байтах, а кириллица занимает по два: срез по
        символам упёрся бы в тот же ENAMETOOLONG, срез по байтам вслепую —
        разрубил бы букву пополам."""
        name = "б" * 300 + ".txt"

        response = await self.upload(name)

        self.assertEqual(response.status_code, 200, response.text)
        stored = await self.get_row(Document, response.json()["id"])
        basename = os.path.basename(stored.path)
        self.assertLessEqual(len(basename.encode("utf-8")), MAX_NAME_BYTES)
        # Строка декодируется целиком — обрубка многобайтовой
        # последовательности превратилась бы в U+FFFD или UnicodeDecodeError.
        self.assertNotIn("�", basename)
        self.assertTrue(basename.endswith(".txt"), basename)

    async def test_filesystem_refusal_is_reported_with_a_machine_code(self):
        """Файловая система со своим, более строгим NAME_MAX.

        Подменяем не open(), а известный коду предел: тогда обрезка не спасает
        и настоящий open() отвечает настоящим ENAMETOOLONG — ровно та ветка,
        которую и надо закрепить.
        """
        with patch.object(document_service_module, "MAX_NAME_BYTES", 100_000):
            response = await self.upload(name_of_length(400))

        self.assertEqual(response.status_code, 400, response.text)
        body = response.json()
        self.assertEqual(body["error_code"], SourceErrors.FILENAME_TOO_LONG)
        # 413 здесь был бы неверен: он означает «файл великоват, разбей его»,
        # а лечится это переименованием.
        self.assertNotEqual(response.status_code, 413)

    async def test_filesystem_refusal_leaves_no_orphan_file(self):
        with patch.object(document_service_module, "MAX_NAME_BYTES", 100_000):
            response = await self.upload(name_of_length(400))

        self.assertEqual(response.status_code, 400, response.text)
        self.assertEqual(self.stored_files(), [], "остался файл-сирота")
        self.assertEqual(await self.all_rows(Document), [])


class FilenameSanitizationTests(UploadFilenameTestCase):
    """document.name не хранит путь: он уходит в API и в Content-Disposition."""

    async def test_relative_traversal_is_stripped_from_the_stored_name(self):
        response = await self.upload("../../../../etc/passwd.txt")

        self.assertEqual(response.status_code, 200, response.text)
        stored = await self.get_row(Document, response.json()["id"])
        self.assertEqual(stored.name, "passwd.txt")

    async def test_absolute_path_is_stripped_from_the_stored_name(self):
        response = await self.upload("/etc/passwd.txt")

        self.assertEqual(response.status_code, 200, response.text)
        stored = await self.get_row(Document, response.json()["id"])
        self.assertEqual(stored.name, "passwd.txt")

    async def test_windows_separators_are_stripped_from_the_stored_name(self):
        response = await self.upload(r"..\..\..\etc\passwd.txt")

        self.assertEqual(response.status_code, 200, response.text)
        stored = await self.get_row(Document, response.json()["id"])
        self.assertEqual(stored.name, "passwd.txt")

    async def test_file_still_lands_inside_the_uploads_directory(self):
        """Выхода за каталог загрузок не было и раньше — закрепляем, что
        очистка имени этого не изменила."""
        response = await self.upload("../../../../etc/passwd.txt")

        self.assertEqual(response.status_code, 200, response.text)
        stored = await self.get_row(Document, response.json()["id"])
        self.assertEqual(
            os.path.dirname(os.path.abspath(stored.path)),
            os.path.abspath(self.uploads),
        )

    async def test_ordinary_name_is_left_alone(self):
        response = await self.upload("обычный отчёт (2024).txt")

        self.assertEqual(response.status_code, 200, response.text)
        stored = await self.get_row(Document, response.json()["id"])
        self.assertEqual(stored.name, "обычный отчёт (2024).txt")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
