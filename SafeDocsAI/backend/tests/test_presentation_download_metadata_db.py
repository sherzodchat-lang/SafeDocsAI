"""Метаданные скачивания, отказ без браузера и шаблон без превью.

Три ветки HTTP-слоя, которые появились вместе с переездом рендерера на
HTML -> PDF. Все три роднит одно: правильный ответ здесь неотличим от
неправильного на глаз, и увидеть подмену можно только по заголовкам.

  * **Тип и имя файла выводятся из ХРАНИМОГО ПУТИ, а не из константы.** Новые
    колоды печатает Chrome, и это .pdf; но на дисках лежат колоды, собранные
    прежним рендерером в .pptx, и мигрировать их никто не будет — это
    пользовательские файлы. Константа «тип презентации» назвала бы половину из
    них чужим именем: браузер сохранил бы .pptx под именем .pdf, и открыть его
    не смог бы никто. Отсюда две ветки в тестах, а не одна: старое и новое
    поколение файлов проверяются по отдельности.
  * **Заказ без браузера отвергается на приёме, а не в очереди.** Иначе
    пользователь платит минутами ожидания за то, что было известно в момент
    клика, а администратор читает жалобу «презентации ломаются» вместо
    «браузер не установлен». Проверяется и порядок: владение и роль стоят ДО
    проверки браузера, иначе 503 стал бы оракулом существования чужих
    блокнотов и подсказкой пользователю, которому раздел вообще не положен.
  * **Шаблон без превью не роняет галерею.** Превью рисует Chrome, а Chrome
    может отсутствовать — реестр в этом случае оставляет шаблон выбираемым и
    отдаёт preview_file = None. Обращение к атрибуту такого None было бы
    пятисоткой на ровном месте: сервер цел, дизайн доступен, отсутствует
    только картинка.

Настоящая база нужна: скачивание — это стык строки в БД и файла на диске, а
заказ считает источники и очередь запросами. Ollama и Chrome не поднимаются:
HTTP-слой до них не доходит, а состояние браузера подменяется на уровне
кэша проверки (chromium._status), то есть проверяется наш разбор ответа, а не
комплектация конкретной машины.
"""

import os
import sys
import unittest
from dataclasses import replace
from unittest.mock import patch
from urllib.parse import quote, unquote

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dbfixtures import DatabaseBackedTestCase  # noqa: E402

from app.api.endpoints import presentations as presentations_endpoint  # noqa: E402
from app.core.exceptions import PresentationErrors, SourceErrors  # noqa: E402
from app.modules.presentations import chromium as chromium_module  # noqa: E402
from app.modules.presentations.chromium import ChromiumStatus  # noqa: E402
from app.modules.presentations.constants import (  # noqa: E402
    SLIDE_COUNT_DEFAULT,
    STATUS_READY,
)
from app.modules.presentations.templates import template_registry  # noqa: E402
from app.shared.models import Document, Notebook, Presentation, User  # noqa: E402

# Ответы проверки браузера. Подменяется именно КЭШ (chromium._status), а не
# функция проверки: так тест проходит через настоящие chromium_status и
# ensure_chromium_available — то есть проверяет, что слой API понимает их
# ответ, — но не запускает процессов и не зависит от того, стоит ли Chrome на
# машине, где идёт прогон.
CHROMIUM_READY = ChromiumStatus(
    available=True,
    binary="/usr/bin/google-chrome-stable",
    version="Google Chrome 141.0.0.0",
    error=None,
)
CHROMIUM_MISSING = ChromiumStatus(
    available=False,
    binary=None,
    version=None,
    error="no Chromium binary found; tried google-chrome-stable in PATH",
)


def chromium(status: ChromiumStatus):
    """Состояние браузера на время теста."""
    return patch.object(chromium_module, "_status", status)


class PresentationsMetadataTestCase(DatabaseBackedTestCase):
    """Общая обвязка: менеджер, блокнот с источником, заказ на диске."""

    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()

        self.manager: User = await self.make_user("manager", "content_manager")
        self.plain: User = await self.make_user("reader", "user")
        self.stranger: User = await self.make_user("neighbour", "content_manager")
        self.as_user(self.manager)

        self.notebook = await self.make_notebook("Налоги")
        # Ключ настоящего шаблона: тело заказа проверяется по реестру ДО
        # обработчика, и выдуманный ключ подменял бы проверяемый ответ на 422.
        self.template_key = self.a_template_key()

        # Лимитер общий на процесс: без сброса соседние наборы выбивали бы друг
        # другу 429 вместо проверяемого ответа.
        presentations_endpoint.order_limiter.clients.clear()
        self.addCleanup(presentations_endpoint.order_limiter.clients.clear)

    def a_template_key(self) -> str:
        templates = template_registry.list()
        if not templates:  # pragma: no cover - зависит от комплекта на диске
            self.skipTest("на диске нет ни одного пригодного шаблона")
        return templates[0].key

    async def make_notebook(
        self, name: str = "Блокнот", *, owner: User | None = None
    ) -> Notebook:
        owner = owner or self.manager
        notebook = await self.seed(
            Notebook(name=name, domain_profile="general", owner_id=owner.id)
        )
        # Имя файла на диске не выводится из имени блокнота: среди имён здесь
        # есть длиннее NAME_MAX, и фикстура падала бы раньше самой проверки.
        await self.seed(
            Document(
                name="источник.pdf",
                path=self.make_file(f"source_{notebook.id}.pdf"),
                size=10,
                status="indexed",
                notebook_id=notebook.id,
                owner_id=owner.id,
            )
        )
        return notebook

    async def make_ready_presentation(
        self,
        *,
        suffix: str,
        notebook: Notebook | None = None,
        owner: User | None = None,
        template_key: str | None = None,
    ) -> Presentation:
        notebook = notebook or self.notebook
        owner = owner or self.manager
        template_key = template_key or self.template_key
        path = self.make_file(f"presentation_{notebook.id}{suffix}", "%PDF-1.4")
        return await self.seed(
            Presentation(
                notebook_id=notebook.id,
                owner_id=owner.id,
                template_key=template_key,
                language="ru",
                slide_count=SLIDE_COUNT_DEFAULT,
                status=STATUS_READY,
                file_path=path,
                file_size=os.path.getsize(path),
            )
        )

    async def download(self, presentation_id: int):
        return await self.client.get(
            f"/api/v1/presentations/{presentation_id}/download"
        )

    async def order(self, notebook_id: int, template_key: str):
        return await self.client.post(
            f"/api/v1/notebooks/{notebook_id}/presentations",
            json={"template_key": template_key},
        )

    def assert_code(self, response, status_code: int, error_code: str) -> None:
        self.assertEqual(response.status_code, status_code, response.text)
        self.assertEqual(response.json().get("error_code"), error_code, response.text)

    def attachment_name(self, response) -> str:
        """Имя из Content-Disposition, как его прочитает браузер."""
        disposition = response.headers["content-disposition"]
        self.assertTrue(disposition.startswith("attachment"), disposition)
        if "filename*=utf-8''" in disposition:
            return unquote(disposition.split("filename*=utf-8''", 1)[1])
        return disposition.split('filename="', 1)[1].rsplit('"', 1)[0]


# --- Тип и имя скачиваемого файла ----------------------------------------


class DownloadMetadataTests(PresentationsMetadataTestCase):
    async def test_new_pdf_is_served_as_pdf(self):
        row = await self.make_ready_presentation(suffix=".pdf")

        response = await self.download(row.id)

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.headers["content-type"], "application/pdf")
        self.assertEqual(
            self.attachment_name(response), f"{self.notebook.name} — презентация.pdf"
        )

    async def test_legacy_pptx_is_served_exactly_as_before(self):
        """Главный тест набора: колоду прежнего поколения переезд не трогает.

        Файл лежит на диске .pptx, никакой миграции для него не задумано, и
        пользователь, нажавший «Скачать» через полгода после перехода на PDF,
        обязан получить работающий .pptx — с типом presentationml и с именем,
        которое откроется двойным щелчком. Тип, взятый из константы «сегодня мы
        печатаем PDF», сломал бы ровно это.
        """
        row = await self.make_ready_presentation(suffix=".pptx")

        response = await self.download(row.id)

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(
            response.headers["content-type"], presentations_endpoint.PPTX_MEDIA_TYPE
        )
        self.assertEqual(
            self.attachment_name(response), f"{self.notebook.name} — презентация.pptx"
        )

    async def test_name_and_type_always_agree(self):
        """Расширение имени и MIME читаются из одного места — из пути на диске.

        Разъезд этих двух означал бы файл, который система пользователя
        открывает не тем приложением; проверяется он поштучно, чтобы падение
        называло виноватую ветку.
        """
        for suffix, media_type in (
            (".pdf", "application/pdf"),
            (".pptx", presentations_endpoint.PPTX_MEDIA_TYPE),
        ):
            with self.subTest(suffix=suffix):
                notebook = await self.make_notebook(f"Блокнот {suffix}")
                row = await self.make_ready_presentation(
                    suffix=suffix, notebook=notebook
                )

                response = await self.download(row.id)

                self.assertEqual(response.headers["content-type"], media_type)
                self.assertTrue(
                    self.attachment_name(response).endswith(suffix),
                    self.attachment_name(response),
                )

    async def test_unknown_extension_is_not_guessed(self):
        """Расширение не из таблицы — octet-stream, а не «наверное, PDF».

        Путь пишет наш же рендерер, поэтому файл с посторонним расширением
        означает, что что-то пошло не по плану. Пусть браузер спросит
        пользователя, а не сделает вид, что понял.
        """
        row = await self.make_ready_presentation(suffix=".bin")

        response = await self.download(row.id)

        self.assertEqual(
            response.headers["content-type"], presentations_endpoint.FALLBACK_MEDIA_TYPE
        )

    async def test_pdf_name_is_sanitized_and_cut_to_the_byte_limit(self):
        """Имя блокнота пишет пользователь: и слэши, и 255 кириллических букв.

        Чистится и режется тем же средством, что имя источника
        (SourceService.sanitize_display_name): своей копии этого правила в
        разделе презентаций быть не должно. Расширение обязано пережить
        обрезку — иначе система пользователя не поймёт, чем открывать файл.
        """
        notebook = await self.make_notebook("../../etc/" + "я" * 255)
        row = await self.make_ready_presentation(suffix=".pdf", notebook=notebook)

        response = await self.download(row.id)
        name = self.attachment_name(response)

        self.assertNotIn("..", name)
        self.assertNotIn("/", name)
        self.assertLessEqual(len(name.encode("utf-8")), 255)
        self.assertTrue(name.endswith(".pdf"), name)

    async def test_non_ascii_name_is_encoded_by_starlette(self):
        """Кодирование заголовка — не наша забота, но проверить его стоит."""
        row = await self.make_ready_presentation(suffix=".pdf")

        disposition = (await self.download(row.id)).headers["content-disposition"]

        self.assertIn(quote(f"{self.notebook.name} — презентация.pdf"), disposition)


# --- Отказ, когда печатать нечем -----------------------------------------


class RendererUnavailableTests(PresentationsMetadataTestCase):
    async def test_order_without_a_browser_answers_503(self):
        # Логгер модуля проверки перехватывается, а не глушится: ERROR с
        # настоящей причиной (какой путь, что ответил бинарник) — это то, по
        # чему администратор поймёт, что чинить, и он обязан быть. Наружу
        # причина при этом не уходит: устройство сервера пользователю не нужно.
        with chromium(CHROMIUM_MISSING), patch.object(
            chromium_module.logger, "error"
        ) as complaint:
            response = await self.order(self.notebook.id, self.template_key)

        self.assert_code(response, 503, PresentationErrors.RENDERER_UNAVAILABLE)
        self.assertTrue(complaint.called, "отказ не оставил следа в журнале")
        self.assertNotIn(CHROMIUM_MISSING.error, response.text)

    async def test_refused_order_leaves_no_row_behind(self):
        """Отказ на приёме — это именно отказ, а не заказ, который не поедет.

        Строка в 'queued' пережила бы отказ и заняла бы блокнот: следующий
        заказ (уже с установленным браузером) получил бы 409 «уже
        генерируется» от заказа, которого никто не делал.
        """
        with chromium(CHROMIUM_MISSING), patch.object(chromium_module.logger, "error"):
            response = await self.order(self.notebook.id, self.template_key)

        self.assert_code(response, 503, PresentationErrors.RENDERER_UNAVAILABLE)
        self.assertEqual(await self.all_rows(Presentation), [])

    async def test_browser_on_the_machine_does_not_stand_in_the_way(self):
        """Обратная ветка: с браузером тот же запрос принимается.

        Без неё проверка «503, когда браузера нет» была бы совместима с
        эндпоинтом, который отказывает всегда.
        """
        with chromium(CHROMIUM_READY):
            response = await self.order(self.notebook.id, self.template_key)

        self.assertEqual(response.status_code, 202, response.text)

    async def test_foreign_notebook_answers_404_even_without_a_browser(self):
        """Владение проверяется ДО браузера.

        Иначе по разнице 404 и 503 перебором id подтверждается существование
        чужих блокнотов — ровно тот оракул, ради закрытия которого весь раздел
        отвечает 404 на чужое.
        """
        foreign = await self.make_notebook("Чужой", owner=self.stranger)

        with chromium(CHROMIUM_MISSING):
            response = await self.order(foreign.id, self.template_key)

        # Код здесь тот же, что у любого чужого блокнота (deps.get_owned_
        # notebook), — важен не он, а то, что ответ не 503.
        self.assert_code(response, 404, SourceErrors.NOTEBOOK_NOT_FOUND)

    async def test_role_is_checked_before_the_browser(self):
        """Пользователю без роли — 403, а не 503.

        Состояние сервера не его дело: раздел ему не положен вовсе, и «зайдите
        позже» подсказывало бы, что дело во временной беде.
        """
        notebook = await self.make_notebook("Свой", owner=self.plain)
        self.as_user(self.plain)

        with chromium(CHROMIUM_MISSING):
            response = await self.order(notebook.id, self.template_key)

        self.assert_code(response, 403, PresentationErrors.ROLE_NOT_ALLOWED)

    async def test_reading_and_downloading_survive_a_missing_browser(self):
        """Уже готовые колоды скачиваются и без браузера.

        Отсутствие Chrome ломает СБОРКУ новых, а не выдачу собранных: файл на
        диске лежит, и отказывать в нём значило бы отнимать у пользователя
        работу, которая уже сделана.
        """
        row = await self.make_ready_presentation(suffix=".pdf")

        with chromium(CHROMIUM_MISSING):
            listing = await self.client.get(
                f"/api/v1/notebooks/{self.notebook.id}/presentations"
            )
            download = await self.download(row.id)

        self.assertEqual(listing.status_code, 200, listing.text)
        self.assertEqual(download.status_code, 200, download.text)


# --- Шаблон, у которого нет картинки -------------------------------------


class TemplateWithoutPreviewTests(PresentationsMetadataTestCase):
    """Превью рисует Chrome; без него шаблон остаётся, а картинки нет.

    Реестр в этом случае отдаёт preview_file = None и НЕ выбрасывает запись:
    дизайн рабочий, заказывать по нему можно. Здесь проверяется, что слой API
    понимает это так же — а не падает на атрибуте None.
    """

    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()

        templates = template_registry.list()
        if not templates:  # pragma: no cover - зависит от комплекта на диске
            self.skipTest("на диске нет ни одного пригодного шаблона")

        # Копия настоящей записи без картинки: подменяется ответ реестра, а не
        # файловая система, — иначе тест удалял бы кэш превью, общий на прогон.
        self.blind = replace(templates[0], preview_file=None)
        patcher = patch.object(
            template_registry, "get", side_effect=lambda key: (
                self.blind if key == self.blind.key else None
            )
        )
        patcher.start()
        self.addCleanup(patcher.stop)

        listing = patch.object(template_registry, "list", return_value=[self.blind])
        listing.start()
        self.addCleanup(listing.stop)

    async def test_gallery_still_lists_the_template(self):
        response = await self.client.get("/api/v1/presentations/templates")

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(
            [item["key"] for item in response.json()], [self.blind.key]
        )

    async def test_preview_answers_404_instead_of_crashing(self):
        response = await self.client.get(
            f"/api/v1/presentations/templates/{self.blind.key}/preview"
        )

        # 404 file_missing, а не unsupported_template: шаблон существует и
        # выбираем, отсутствует ровно картинка.
        self.assert_code(response, 404, PresentationErrors.FILE_MISSING)

    async def test_the_template_can_still_be_ordered(self):
        """Картинки нет — заказ есть. Иначе отсутствие кэша превью выключало бы
        целый дизайн, хотя собрать по нему колоду ничто не мешает."""
        with chromium(CHROMIUM_READY):
            response = await self.order(self.notebook.id, self.blind.key)

        self.assertEqual(response.status_code, 202, response.text)
        self.assertEqual(response.json()["template_key"], self.blind.key)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
