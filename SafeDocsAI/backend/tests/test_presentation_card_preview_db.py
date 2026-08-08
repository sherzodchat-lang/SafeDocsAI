"""Превью карточки и заголовок колоды: две вещи, которые берутся из файла.

Раздел показывает сетку заказанных презентаций, и до этой правки карточка
подписывалась именем ШАБЛОНА, а картинкой брала превью того же шаблона. То есть
десять колод одного оформления выглядели десятью одинаковыми плитками: подпись
не различала, картинка не различала. Здесь проверяется то, что это чинит.

Четыре утверждения, каждое из которых на глаз неотличимо от своей поломки.

  * **Превью — это первая страница ЭТОЙ колоды, и она кэшируется.** Рисование
    ленивое: картинки нет, пока карточку не показали, а показанная колода
    рисуется один раз и дальше отдаётся с диска. Проверяется обоими концами —
    что файл кэша появился рядом с колодой и что второй запрос обходится без
    рендера вовсе (fitz на это время сломан намеренно).
  * **Превью закрыто ровно как скачивание.** Титульный слайд несёт название
    темы и имя блокнота, поэтому чужая картинка — это выдержка из чужой работы.
    Отдельная проверка на 404 у соседа: разойдись зависимость владения с той,
    что стоит у download, — дыра оказалась бы в месте, куда никто не смотрит.
  * **Заголовок хранится в БД и приходит в ответе.** Не выводится клиентом из
    описания: описание пишет пользователь ДО генерации как пожелание, заголовок
    формулирует модель ПОСЛЕ неё по найденному материалу.
  * **Колоды, собранные до появления колонки, заголовок тоже получают.**
    Восстанавливается он из /Title напечатанного PDF — туда его положил Chrome
    из того же plan.title. Проверяется и то, что проход идемпотентен, и то, что
    нечитаемый файл оставляет NULL, а не выдумывает подпись.

Настоящая база нужна: и превью, и заголовок — это стык строки в БД с файлом на
диске, а бэкфилл вообще написан запросом. Ollama и Chrome не поднимаются: до
них ни одна из проверяемых веток не доходит, а PDF собирается прямо здесь тем
же PyMuPDF, которым его потом читают.
"""

import os
import sys
import unittest
from unittest.mock import patch

import fitz

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dbfixtures import DatabaseBackedTestCase  # noqa: E402

from app.core.exceptions import PresentationErrors  # noqa: E402
from app.modules.presentations import preview as preview_module  # noqa: E402
from app.modules.presentations.constants import (  # noqa: E402
    SLIDE_COUNT_DEFAULT,
    STATUS_ERROR,
    STATUS_QUEUED,
    STATUS_READY,
)
from app.modules.presentations.llm_schemas import PLAN_TITLE_MAX_CHARS  # noqa: E402
from app.modules.presentations.preview import (  # noqa: E402
    CARD_PREVIEW_WIDTH,
    card_preview_path,
    ensure_card_preview,
    read_stored_title,
)
from app.modules.presentations.service import PresentationsService  # noqa: E402
from app.modules.presentations.templates import template_registry  # noqa: E402
from app.shared.models import Notebook, Presentation, User  # noqa: E402

# Размер печатной страницы колоды в точках. Значение не проверяется на
# совпадение с рендерером: важно только, что страница ШИРЕ картинки превью, —
# иначе масштабирование шло бы вверх и проверка ширины ничего бы не значила.
PAGE_WIDTH = 960.0
PAGE_HEIGHT = 540.0


def write_pdf(path: str, *, title: str | None, pages: int = 3) -> str:
    """Файл колоды, каким его оставляет печать: /Title и несколько страниц.

    Заголовок кладётся в метаданные ровно так же, как это делает Chrome из
    <title> напечатанной страницы, — иначе проверка восстановления читала бы
    не то, что бывает на диске.
    """
    document = fitz.open()
    for index in range(pages):
        page = document.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
        page.insert_text((72, 200), f"Страница {index + 1}", fontsize=48)
    if title is not None:
        document.set_metadata({"title": title})
    document.save(path)
    document.close()
    return path


class CardPreviewTestCase(DatabaseBackedTestCase):
    """Менеджер, его блокнот и готовая колода с настоящим PDF на диске."""

    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()

        self.manager: User = await self.make_user("manager", "content_manager")
        self.stranger: User = await self.make_user("neighbour", "content_manager")
        self.as_user(self.manager)

        self.notebook = await self.make_notebook("Налоги")
        self.template_key = self.a_template_key()

    def a_template_key(self) -> str:
        templates = template_registry.list()
        if not templates:  # pragma: no cover - зависит от комплекта на диске
            self.skipTest("на диске нет ни одного пригодного шаблона")
        return templates[0].key

    async def make_notebook(
        self, name: str = "Блокнот", *, owner: User | None = None
    ) -> Notebook:
        owner = owner or self.manager
        return await self.seed(
            Notebook(name=name, domain_profile="general", owner_id=owner.id)
        )

    async def make_presentation(
        self,
        *,
        status: str = STATUS_READY,
        file_name: str | None = "deck.pdf",
        pdf_title: str | None = "Налоговый кодекс РТ",
        title: str | None = None,
        notebook: Notebook | None = None,
        owner: User | None = None,
    ) -> Presentation:
        notebook = notebook or self.notebook
        owner = owner or self.manager
        path = None
        if file_name is not None:
            path = os.path.join(self._tmpdir.name, f"n{notebook.id}_{file_name}")
            if path.endswith(".pdf"):
                write_pdf(path, title=pdf_title)
            else:
                # Колода прежнего рендерера: файл настоящий, но страниц из него
                # не достать — ровно тот случай, ради которого превью обязано
                # отвечать отказом, а не пятисоткой.
                with open(path, "wb") as handle:
                    handle.write(b"PK\x03\x04not-a-pdf")
        return await self.seed(
            Presentation(
                notebook_id=notebook.id,
                owner_id=owner.id,
                template_key=self.template_key,
                language="ru",
                slide_count=SLIDE_COUNT_DEFAULT,
                status=status,
                title=title,
                file_path=path,
                file_size=os.path.getsize(path) if path else None,
            )
        )

    async def fetch_preview(self, presentation_id: int):
        return await self.client.get(
            f"/api/v1/presentations/{presentation_id}/preview"
        )

    def assert_code(self, response, status_code: int, error_code: str) -> None:
        self.assertEqual(response.status_code, status_code, response.text)
        self.assertEqual(response.json().get("error_code"), error_code, response.text)


# --- Картинка первой страницы --------------------------------------------


class CardPreviewImageTests(CardPreviewTestCase):
    async def test_preview_is_the_first_page_of_this_deck(self):
        row = await self.make_presentation()

        response = await self.fetch_preview(row.id)

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.headers["content-type"], "image/png")
        # Проверяем не «что-то пришло», а что пришла страница ИМЕННО этой
        # колоды нужного размера: PNG другой ширины означал бы, что картинка
        # взялась откуда-то ещё (например, осталась превью шаблона).
        picture = fitz.Pixmap(response.content)
        self.assertEqual(picture.width, CARD_PREVIEW_WIDTH)
        self.assertEqual(
            picture.height, round(CARD_PREVIEW_WIDTH * PAGE_HEIGHT / PAGE_WIDTH)
        )

    async def test_second_request_is_served_from_cache(self):
        """Рисуем один раз на колоду, а не на каждый показ карточки.

        Главная проверка ленивого рендера: без кэша сетка из двадцати колод
        заново декодировала бы двадцать PDF на каждое открытие раздела. fitz на
        время второго запроса сломан намеренно — если ответ всё равно 200,
        значит картинку взяли с диска, а не нарисовали снова.
        """
        row = await self.make_presentation()
        first = await self.fetch_preview(row.id)
        self.assertEqual(first.status_code, 200, first.text)
        self.assertTrue(os.path.exists(card_preview_path(row.file_path)))

        def explode(*_args, **_kwargs):  # pragma: no cover - вызов и есть провал
            raise AssertionError("превью нарисовано второй раз вместо чтения кэша")

        with patch.object(preview_module.fitz, "open", explode):
            second = await self.fetch_preview(row.id)

        self.assertEqual(second.status_code, 200, second.text)
        self.assertEqual(second.content, first.content)

    async def test_stale_cache_is_redrawn(self):
        """Картинка старше своей колоды считается протухшей.

        В норме этого не случается — файл колоды пишется один раз, — но правило
        дешёвое, а его отсутствие означало бы молча показанную чужую картинку.
        """
        row = await self.make_presentation()
        cached = card_preview_path(row.file_path)
        await self.fetch_preview(row.id)

        # Кэш «из прошлого»: файл колоды на минуту новее своей картинки. Время
        # ставится ПОСЛЕ записи — сама запись обновила бы его на текущее.
        with open(cached, "wb") as handle:
            handle.write("устаревшая картинка".encode("utf-8"))
        stale = os.path.getmtime(row.file_path) - 60
        os.utime(cached, (stale, stale))

        response = await self.fetch_preview(row.id)

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(fitz.Pixmap(response.content).width, CARD_PREVIEW_WIDTH)

    async def test_deck_in_progress_has_no_preview_yet(self):
        row = await self.make_presentation(status=STATUS_QUEUED, file_name=None)

        self.assert_code(
            await self.fetch_preview(row.id), 409, PresentationErrors.NOT_READY
        )

    async def test_ready_row_without_file_reports_file_missing(self):
        row = await self.make_presentation()
        os.remove(row.file_path)

        self.assert_code(
            await self.fetch_preview(row.id), 404, PresentationErrors.FILE_MISSING
        )

    async def test_legacy_pptx_deck_has_no_preview_but_stays_intact(self):
        """Колода прежнего рендерера не рисуется — и это не пятисотка.

        Файл на месте и по-прежнему скачивается; отсутствует только картинка, и
        сетка на этот отказ показывает ту же заглушку, что у шаблона без
        превью.
        """
        row = await self.make_presentation(file_name="deck.pptx")

        self.assert_code(
            await self.fetch_preview(row.id), 404, PresentationErrors.FILE_MISSING
        )
        self.assertTrue(os.path.exists(row.file_path))

    async def test_broken_pdf_does_not_break_the_grid(self):
        row = await self.make_presentation()
        with open(row.file_path, "wb") as handle:
            handle.write("%PDF-1.4 и дальше мусор".encode("utf-8"))

        self.assert_code(
            await self.fetch_preview(row.id), 404, PresentationErrors.FILE_MISSING
        )

    async def test_stranger_cannot_see_someone_elses_preview(self):
        """Чужая картинка — 404, тем же ответом, что и чужое скачивание.

        Титульный слайд несёт название темы и имя блокнота: отдать его соседу
        значит показать содержимое чужой работы. 404, а не 403, — иначе разница
        ответов делает эндпоинт оракулом для перебора id.
        """
        row = await self.make_presentation()
        self.as_user(self.stranger)

        self.assert_code(
            await self.fetch_preview(row.id), 404, PresentationErrors.NOT_FOUND
        )

    async def test_deleting_a_deck_removes_its_cached_preview(self):
        row = await self.make_presentation()
        await self.fetch_preview(row.id)
        cached = card_preview_path(row.file_path)
        self.assertTrue(os.path.exists(cached))

        response = await self.client.delete(f"/api/v1/presentations/{row.id}")

        self.assertEqual(response.status_code, 200, response.text)
        self.assertFalse(os.path.exists(row.file_path))
        self.assertFalse(os.path.exists(cached))


# --- Адрес картинки в ответе ---------------------------------------------


class PreviewUrlTests(CardPreviewTestCase):
    async def test_ready_deck_carries_its_preview_url(self):
        row = await self.make_presentation()

        response = await self.client.get(f"/api/v1/presentations/{row.id}")

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(
            response.json()["preview_url"],
            f"/api/v1/presentations/{row.id}/preview",
        )

    async def test_unfinished_deck_promises_no_picture(self):
        """У неготовой колоды адреса картинки нет вовсе.

        Пустой адрес клиент читает как «картинки не будет» и не делает запроса,
        который всё равно вернул бы 409.
        """
        for status in (STATUS_QUEUED, STATUS_ERROR):
            with self.subTest(status=status):
                row = await self.make_presentation(
                    status=status,
                    file_name=None,
                    notebook=await self.make_notebook(f"Блокнот {status}"),
                )

                response = await self.client.get(f"/api/v1/presentations/{row.id}")

                self.assertEqual(response.status_code, 200, response.text)
                self.assertIsNone(response.json()["preview_url"])


# --- Заголовок колоды ------------------------------------------------------


class DeckTitleTests(CardPreviewTestCase):
    async def test_title_is_stored_on_ready_and_returned_by_the_api(self):
        row = await self.make_presentation(status=STATUS_QUEUED, file_name=None)
        path = write_pdf(
            os.path.join(self._tmpdir.name, "ready.pdf"), title="что угодно"
        )

        async with self.session_factory() as session:
            await PresentationsService.mark_ready(
                session,
                row.id,
                file_path=path,
                file_size=os.path.getsize(path),
                title="  Налоговый кодекс РТ: что меняется  ",
            )

        stored = await self.get_row(Presentation, row.id)
        # Обрезка пробелов — на записи, а не на показе: колонка обязана хранить
        # то, что покажет карточка, иначе поиск и сортировка по ней однажды
        # разойдутся с глазами.
        self.assertEqual(stored.title, "Налоговый кодекс РТ: что меняется")

        response = await self.client.get(f"/api/v1/presentations/{row.id}")
        self.assertEqual(
            response.json()["title"], "Налоговый кодекс РТ: что меняется"
        )

    async def test_title_stays_within_the_column_limit(self):
        row = await self.make_presentation(status=STATUS_QUEUED, file_name=None)
        path = write_pdf(os.path.join(self._tmpdir.name, "long.pdf"), title="x")

        async with self.session_factory() as session:
            await PresentationsService.mark_ready(
                session,
                row.id,
                file_path=path,
                file_size=os.path.getsize(path),
                title="Я" * (PLAN_TITLE_MAX_CHARS + 40),
            )

        stored = await self.get_row(Presentation, row.id)
        self.assertEqual(len(stored.title), PLAN_TITLE_MAX_CHARS)

    async def test_title_is_separate_from_the_user_written_description(self):
        """Подпись карточки — не пересказ того, что просил пользователь.

        Описание и заголовок приезжают разными полями и не подменяют друг
        друга: описание — пожелание ДО генерации, заголовок — то, что модель
        сформулировала ПОСЛЕ неё по найденному материалу.
        """
        row = await self.make_presentation(title="Спорт в вузах Таджикистана")
        async with self.session_factory() as session:
            fresh = await session.get(Presentation, row.id)
            fresh.description = "сделай коротко и для студентов"
            session.add(fresh)
            await session.commit()

        body = (await self.client.get(f"/api/v1/presentations/{row.id}")).json()

        self.assertEqual(body["title"], "Спорт в вузах Таджикистана")
        self.assertEqual(body["description"], "сделай коротко и для студентов")


# --- Колоды, собранные до появления колонки -------------------------------


class TitleBackfillTests(CardPreviewTestCase):
    async def test_old_deck_recovers_its_title_from_the_pdf(self):
        """Заголовок старой колоды лежит в её файле, и это НЕ догадка.

        Chrome печатает в /Title то, что стояло в <title> страницы, а туда
        рендерер кладёт ровно plan.title. То есть восстанавливается тот же
        самый заголовок, просто доехавший через файл, — единственный источник
        для строк, созданных до появления колонки.
        """
        row = await self.make_presentation(
            title=None, pdf_title="Цели развития до 2030 года"
        )

        async with self.session_factory() as session:
            filled = await PresentationsService.backfill_titles(session)

        self.assertEqual(filled, 1)
        stored = await self.get_row(Presentation, row.id)
        self.assertEqual(stored.title, "Цели развития до 2030 года")

    async def test_backfill_repeats_without_touching_what_is_already_filled(self):
        """Проход идёт на каждом старте, но работает ровно один раз.

        Второй вызов не находит ничего и не перезаписывает заголовок, который
        уже стоит в колонке: строка, собранная заново, может нести заголовок
        точнее того, что лежит в старом файле.
        """
        row = await self.make_presentation(title=None, pdf_title="Первый заголовок")
        async with self.session_factory() as session:
            await PresentationsService.backfill_titles(session)

        write_pdf(row.file_path, title="Подменённый заголовок")
        async with self.session_factory() as session:
            again = await PresentationsService.backfill_titles(session)

        self.assertEqual(again, 0)
        stored = await self.get_row(Presentation, row.id)
        self.assertEqual(stored.title, "Первый заголовок")

    async def test_unreadable_file_leaves_the_title_empty(self):
        """Нечего прочитать — значит NULL, а не выдуманная подпись.

        Пустая колонка честна: клиент показывает по ней запасную подпись и
        объясняет, откуда она. Заголовок, собранный из имени файла или из
        описания, выглядел бы настоящим и врал бы про содержимое колоды.
        """
        missing = await self.make_presentation(title=None, pdf_title=None)
        legacy = await self.make_presentation(
            title=None,
            file_name="deck.pptx",
            notebook=await self.make_notebook("Второй"),
        )
        gone = await self.make_presentation(
            title=None, notebook=await self.make_notebook("Третий")
        )
        os.remove(gone.file_path)

        async with self.session_factory() as session:
            filled = await PresentationsService.backfill_titles(session)

        self.assertEqual(filled, 0)
        for row in (missing, legacy, gone):
            with self.subTest(presentation=row.id):
                self.assertIsNone((await self.get_row(Presentation, row.id)).title)

    async def test_unfinished_rows_are_not_touched(self):
        """Заказ в очереди заголовка не получает: его ещё не существует."""
        row = await self.make_presentation(
            status=STATUS_QUEUED, file_name=None, title=None
        )

        async with self.session_factory() as session:
            self.assertEqual(await PresentationsService.backfill_titles(session), 0)

        self.assertIsNone((await self.get_row(Presentation, row.id)).title)


# --- Модуль превью отдельно от HTTP ---------------------------------------


class PreviewModuleTests(unittest.TestCase):
    """Проверки, которым база не нужна: чтение файла и его отсутствие."""

    def setUp(self) -> None:
        import tempfile

        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)

    def path(self, name: str) -> str:
        return os.path.join(self._tmp.name, name)

    def test_title_is_trimmed_to_the_schema_limit(self):
        path = write_pdf(self.path("long.pdf"), title="Ю" * (PLAN_TITLE_MAX_CHARS + 5))

        self.assertEqual(len(read_stored_title(path)), PLAN_TITLE_MAX_CHARS)

    def test_pdf_without_title_reports_nothing(self):
        path = write_pdf(self.path("plain.pdf"), title=None)

        self.assertIsNone(read_stored_title(path))

    def test_missing_file_is_not_an_exception(self):
        """Пропавший файл — отсутствие превью, а не отказ сервера."""
        self.assertIsNone(ensure_card_preview(self.path("нет-такого.pdf")))
        self.assertIsNone(read_stored_title(self.path("нет-такого.pdf")))

    def test_failed_render_leaves_no_temporary_files(self):
        path = self.path("broken.pdf")
        with open(path, "wb") as handle:
            handle.write("%PDF-1.4 мусор".encode("utf-8"))

        self.assertIsNone(ensure_card_preview(path))
        self.assertEqual(
            [name for name in os.listdir(self._tmp.name) if ".tmp-" in name], []
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
