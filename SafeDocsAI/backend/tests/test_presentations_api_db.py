"""HTTP-контракт раздела презентаций — на настоящем PostgreSQL.

Что закрепляем.

  * **Порядок проверок на заказе.** Владение -> роль -> частота -> тело ->
    бизнес-условия. Это не украшение: чужой блокнот обязан отвечать 404 даже
    вызывающему, которому и по роли ничего не положено, — иначе по разнице
    403/404 чужие блокноты перебираются по id, и эндпоинт становится оракулом
    существования. Отказ по роли при этом не должен расходовать лимит частоты,
    а негодное тело — проезжать мимо лимита.
  * **422 приходят НАШИМИ кодами.** В проекте есть обработчик
    RequestValidationError, отдающий request.validation_failed без разбора по
    полям. Для «поле не того типа» это правильно, для «такого шаблона нет» —
    нет: пользователю нужно понять, что чинить. Поэтому у каждой ветки
    проверки значений свой код, и здесь проверяется именно он.
  * **Границы числа слайдов — одно число на весь проект.** Тесты нижней и
    верхней границы написаны ЧЕРЕЗ константы, а не через выписанные 5 и 15:
    тест на литералы краснел бы вместе с осознанной правкой границы, а разъезд
    между схемой, валидатором и константой — пропускал. Тот же приём, что в
    tests/test_settings_limits_single_source.py.
  * **queue_position вне очереди — null, а не 0.** Ноль читается как
    «следующая на очереди» и означал бы ровно обратное.
  * **Скачивание различает три беды.** Чужая строка — 404, не готовая — 409
    (продолжай опрос), готовая без файла — 404 file_missing (закажи заново).
  * **Удаление: строка, commit, потом файл.** И сбой os.remove не превращается
    в 500 — ровно как при удалении блокнота.
  * **Превью шаблона нельзя увести за каталог.** Путь берётся из реестра по
    ключу, поэтому «../../» здесь — просто неизвестный ключ.

Настоящая база нужна почти всем проверкам: 409 «уже генерируется» считается
запросом по таблице, пагинация — по строкам, а «строка удалена, файл стёрт» —
это стык БД и диска, ради которого тесты этого проекта и ходят в PostgreSQL.
Ollama и ChromaDB не поднимаются: HTTP-слой до них не доходит вовсе — он
только ставит строку в очередь.
"""

import os
import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch
from urllib.parse import quote

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dbfixtures import DatabaseBackedTestCase  # noqa: E402

from app.api.endpoints import presentations as presentations_endpoint  # noqa: E402
from app.api.endpoints.documents import (  # noqa: E402
    DEFAULT_PAGE_SIZE,
    MAX_PAGE_SIZE,
    TOTAL_COUNT_HEADER,
)
from app.core.exceptions import (  # noqa: E402
    PresentationErrors,
    RequestErrors,
    SourceErrors,
)
from app.main import app  # noqa: E402
from app.modules.presentations import service as presentation_service  # noqa: E402
from app.modules.presentations.constants import (  # noqa: E402
    DESCRIPTION_MAX,
    SLIDE_COUNT_DEFAULT,
    SLIDE_COUNT_MAX,
    SLIDE_COUNT_MIN,
    STATUS_ERROR,
    STATUS_GENERATING,
    STATUS_QUEUED,
    STATUS_READY,
    SUPPORTED_LANGUAGES,
)
from app.modules.presentations.llm_schemas import MIN_SLIDE_COUNT  # noqa: E402
from app.modules.presentations.templates import (  # noqa: E402
    default_preview_dir,
    default_templates_dir,
    template_registry,
)
from app.shared.models import Document, Notebook, Presentation, User  # noqa: E402

ENDPOINT_LOGGER = "app.api.endpoints.presentations"

TEMPLATES_URL = "/api/v1/presentations/templates"


def a_template_key() -> str:
    """Ключ любого пригодного шаблона из поставляемого комплекта."""
    templates = template_registry.list()
    if not templates:  # pragma: no cover - зависит от комплекта на диске
        raise unittest.SkipTest(
            f"В {default_templates_dir()} нет ни одного пригодного шаблона"
        )
    return templates[0].key


# --- Границы числа слайдов: один источник --------------------------------


class SlideCountSingleSourceTests(unittest.TestCase):
    """Три числа заказа объявлены один раз и читаются всеми, кто их проверяет.

    Ни базы, ни сети: проверяются сами объявления и сгенерированный контракт.
    """

    def test_product_floor_is_not_below_the_schema_floor(self):
        """Продуктовая нижняя граница не может уйти ниже собираемой колоды.

        MIN_SLIDE_COUNT (llm_schemas) — это титул, один контентный слайд и
        «Источники»: меньше рендерер не соберёт. SLIDE_COUNT_MIN — граница
        формы заказа, она обязана лежать не ниже.
        """
        self.assertGreaterEqual(SLIDE_COUNT_MIN, MIN_SLIDE_COUNT)

    def test_default_lies_inside_the_range(self):
        """Иначе форма предлагала бы значение, которое сама же не примет."""
        self.assertLess(SLIDE_COUNT_MIN, SLIDE_COUNT_MAX)
        self.assertLessEqual(SLIDE_COUNT_MIN, SLIDE_COUNT_DEFAULT)
        self.assertLessEqual(SLIDE_COUNT_DEFAULT, SLIDE_COUNT_MAX)

    def test_every_checker_reads_the_same_objects(self):
        """Схема запроса и защита пайплайна смотрят в ту же константу.

        Проверка на идентичность, а не на равенство: равные числа получаются и
        у двух независимо выписанных литералов — ровно того дефекта, который в
        проекте только что вычищали на настройках.
        """
        self.assertIs(presentations_endpoint.SLIDE_COUNT_MIN, SLIDE_COUNT_MIN)
        self.assertIs(presentations_endpoint.SLIDE_COUNT_MAX, SLIDE_COUNT_MAX)
        self.assertIs(presentations_endpoint.SLIDE_COUNT_DEFAULT, SLIDE_COUNT_DEFAULT)
        self.assertIs(presentation_service.SLIDE_COUNT_MIN, SLIDE_COUNT_MIN)
        self.assertIs(presentation_service.SLIDE_COUNT_MAX, SLIDE_COUNT_MAX)

    def test_openapi_documents_the_same_bounds(self):
        """Границы обязаны доезжать до /openapi.json.

        По ним генератор клиента и форма строят подсказку поля. При этом
        ограничением Pydantic они быть не должны — иначе отказ пришёл бы общим
        кодом request.validation_failed (это проверяет OrderValidationTests).
        """
        schema = app.openapi()["components"]["schemas"]["PresentationCreate"]
        slide_count = schema["properties"]["slide_count"]

        self.assertEqual(slide_count["minimum"], SLIDE_COUNT_MIN)
        self.assertEqual(slide_count["maximum"], SLIDE_COUNT_MAX)
        self.assertEqual(slide_count["default"], SLIDE_COUNT_DEFAULT)


# --- Общая обвязка -------------------------------------------------------


class PresentationsApiTestCase(DatabaseBackedTestCase):
    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()

        self.manager: User = await self.make_user("manager", "content_manager")
        self.admin: User = await self.make_user("root", "admin")
        self.plain: User = await self.make_user("reader", "user")
        self.stranger: User = await self.make_user("neighbour", "content_manager")
        self.as_user(self.manager)

        self.notebook = await self.make_notebook("Налоги")
        self.template_key = a_template_key()

        # Лимитер общий на процесс: без сброса тесты влияли бы друг на друга и
        # на соседние наборы. Ключ счётчика в тестах один на всех («unknown»:
        # у ASGI-транспорта нет адреса клиента).
        presentations_endpoint.order_limiter.clients.clear()
        self.addCleanup(presentations_endpoint.order_limiter.clients.clear)

    # --- данные ---

    async def make_notebook(
        self, name: str = "Блокнот", *, owner: User | None = None, indexed: bool = True
    ) -> Notebook:
        owner = owner or self.manager
        notebook = await self.seed(
            Notebook(name=name, domain_profile="general", owner_id=owner.id)
        )
        if indexed:
            # Имя файла на диске не выводится из имени блокнота: в тестах
            # встречаются имена длиннее NAME_MAX (см. проверку обрезки в
            # DownloadTests), и фикстура падала бы раньше самой проверки.
            await self.seed(
                Document(
                    name=f"{name}.pdf",
                    path=self.make_file(f"source_{notebook.id}.pdf"),
                    size=10,
                    status="indexed",
                    notebook_id=notebook.id,
                    owner_id=owner.id,
                )
            )
        return notebook

    async def make_presentation(
        self,
        notebook: Notebook | None = None,
        *,
        status: str = STATUS_READY,
        owner: User | None = None,
        with_file: bool = False,
        **overrides,
    ) -> Presentation:
        notebook = notebook or self.notebook
        owner = owner or self.manager
        fields = {
            "notebook_id": notebook.id,
            "owner_id": owner.id,
            "template_key": self.template_key,
            "language": "ru",
            "slide_count": SLIDE_COUNT_DEFAULT,
            "status": status,
        }
        if with_file:
            fields["file_path"] = self.make_file(
                f"presentation_{notebook.id}_{status}_{len(overrides)}.pptx", "PK\x03\x04"
            )
            fields["file_size"] = 4
        fields.update(overrides)
        return await self.seed(Presentation(**fields))

    # --- запросы ---

    def order_body(self, **overrides) -> dict:
        body = {"template_key": self.template_key}
        body.update(overrides)
        return body

    async def order(self, notebook_id: int | None = None, **overrides):
        notebook_id = notebook_id if notebook_id is not None else self.notebook.id
        return await self.client.post(
            f"/api/v1/notebooks/{notebook_id}/presentations",
            json=self.order_body(**overrides),
        )

    async def list_orders(self, notebook_id: int | None = None, **params):
        notebook_id = notebook_id if notebook_id is not None else self.notebook.id
        return await self.client.get(
            f"/api/v1/notebooks/{notebook_id}/presentations", params=params
        )

    async def get_order(self, presentation_id: int):
        return await self.client.get(f"/api/v1/presentations/{presentation_id}")

    async def download(self, presentation_id: int):
        return await self.client.get(
            f"/api/v1/presentations/{presentation_id}/download"
        )

    async def delete_order(self, presentation_id: int):
        return await self.client.delete(f"/api/v1/presentations/{presentation_id}")

    def assert_code(self, response, status_code: int, error_code: str) -> None:
        self.assertEqual(response.status_code, status_code, response.text)
        self.assertEqual(response.json().get("error_code"), error_code, response.text)


# --- Порядок проверок на заказе ------------------------------------------


class OrderCheckOrderTests(PresentationsApiTestCase):
    async def test_foreign_notebook_answers_404_even_to_a_wrong_role(self):
        """Главный тест набора: владение проверяется ПЕРВЫМ.

        Иначе по разнице 403 (роль не та) и 404 (блокнота нет) перебором id
        подтверждается существование чужих блокнотов.
        """
        foreign = await self.make_notebook("Чужой", owner=self.stranger)
        self.as_user(self.plain)

        response = await self.order(foreign.id)

        self.assert_code(response, 404, SourceErrors.NOTEBOOK_NOT_FOUND)

    async def test_foreign_notebook_answers_404_to_a_right_role_too(self):
        foreign = await self.make_notebook("Чужой", owner=self.stranger)

        response = await self.order(foreign.id)

        self.assert_code(response, 404, SourceErrors.NOTEBOOK_NOT_FOUND)

    async def test_missing_notebook_answers_the_same_404(self):
        response = await self.order(10_000_000)

        self.assert_code(response, 404, SourceErrors.NOTEBOOK_NOT_FOUND)

    async def test_own_notebook_wrong_role_answers_403(self):
        notebook = await self.make_notebook("Свой", owner=self.plain)
        self.as_user(self.plain)

        response = await self.order(notebook.id)

        self.assert_code(response, 403, PresentationErrors.ROLE_NOT_ALLOWED)
        self.assertEqual(await self.all_rows(Presentation), [])

    async def test_admin_may_order_in_a_foreign_notebook(self):
        """Админ видит всё — правило владения общее на весь API."""
        self.as_user(self.admin)

        response = await self.order()

        self.assertEqual(response.status_code, 202, response.text)

    async def test_role_refusal_does_not_spend_the_rate_limit(self):
        """403 дешевле лимита: он срабатывает раньше счётчика.

        Иначе обычный пользователь, тыкающий в недоступную кнопку, выбивал бы
        себе же час тишины, ничего при этом не заказав.
        """
        notebook = await self.make_notebook("Свой", owner=self.plain)
        self.as_user(self.plain)

        for _ in range(5):
            response = await self.order(notebook.id)
            self.assertEqual(response.status_code, 403, response.text)

        self.assertEqual(presentations_endpoint.order_limiter.clients, {})

    async def test_ownership_refusal_does_not_spend_the_rate_limit(self):
        foreign = await self.make_notebook("Чужой", owner=self.stranger)

        for _ in range(5):
            self.assertEqual((await self.order(foreign.id)).status_code, 404)

        self.assertEqual(presentations_endpoint.order_limiter.clients, {})

    async def test_rate_limit_is_checked_before_the_body(self):
        """Лимит стоит РАНЬШЕ разбора тела.

        Иначе вал заведомо негодных запросов ничего не стоил бы отправителю:
        каждый получал бы 422 в обход счётчика.
        """
        limit = presentations_endpoint.order_limiter.requests
        # Тратим лимит целиком. Первый заказ проходит, остальные упираются в
        # 409 «уже в очереди» — но счётчик расходуют все.
        for _ in range(limit):
            self.assertIn((await self.order()).status_code, (202, 409))

        with_bad_body = await self.order(template_key="нет-такого-шаблона")

        self.assertEqual(with_bad_body.status_code, 429, with_bad_body.text)
        self.assertIn("Retry-After", with_bad_body.headers)

    async def test_body_is_checked_before_the_business_rules(self):
        """Негодное тело не должно платить за запросы в базу.

        Блокнот без источников ответил бы 409 no_sources — но только на годном
        теле: сначала объясняем, что не так с запросом.
        """
        empty = await self.make_notebook("Пустой", indexed=False)

        response = await self.order(empty.id, language="en")

        self.assert_code(response, 422, PresentationErrors.UNSUPPORTED_LANGUAGE)


# --- Проверка тела -------------------------------------------------------


class OrderValidationTests(PresentationsApiTestCase):
    async def test_unknown_template_is_refused_and_not_substituted(self):
        response = await self.order(template_key="дизайн-которого-нет")

        self.assert_code(response, 422, PresentationErrors.UNSUPPORTED_TEMPLATE)
        # Никакой подмены на первый попавшийся: строки нет вовсе.
        self.assertEqual(await self.all_rows(Presentation), [])

    async def test_every_registered_template_is_accepted(self):
        for info in template_registry.list():
            with self.subTest(template=info.key):
                notebook = await self.make_notebook(f"Блокнот {info.key}")
                response = await self.order(notebook.id, template_key=info.key)

                self.assertEqual(response.status_code, 202, response.text)
                self.assertEqual(response.json()["template_key"], info.key)

    async def test_unsupported_language_is_refused(self):
        for language in ("en", "tg", "", "ru-RU"):
            with self.subTest(language=language):
                response = await self.order(language=language)
                self.assert_code(
                    response, 422, PresentationErrors.UNSUPPORTED_LANGUAGE
                )

    async def test_supported_languages_are_accepted_and_normalized(self):
        for language in SUPPORTED_LANGUAGES:
            with self.subTest(language=language):
                notebook = await self.make_notebook(f"Блокнот {language}")
                response = await self.order(notebook.id, language=f"  {language.upper()} ")

                self.assertEqual(response.status_code, 202, response.text)
                self.assertEqual(response.json()["language"], language)

    async def test_slide_count_outside_the_range_is_refused(self):
        """Числа берутся из констант: тест переживает осознанную правку границ."""
        for slide_count in (SLIDE_COUNT_MIN - 1, SLIDE_COUNT_MAX + 1, 0, -3, 1000):
            with self.subTest(slide_count=slide_count):
                response = await self.order(slide_count=slide_count)
                self.assert_code(
                    response, 422, PresentationErrors.VALUE_OUT_OF_RANGE
                )
                # Границы названы в тексте: клиент подставляет их в подсказку.
                self.assertIn(str(SLIDE_COUNT_MIN), response.json()["detail"])
                self.assertIn(str(SLIDE_COUNT_MAX), response.json()["detail"])

    async def test_both_ends_of_the_range_are_accepted(self):
        for slide_count in (SLIDE_COUNT_MIN, SLIDE_COUNT_MAX):
            with self.subTest(slide_count=slide_count):
                notebook = await self.make_notebook(f"Блокнот {slide_count}")
                response = await self.order(notebook.id, slide_count=slide_count)

                self.assertEqual(response.status_code, 202, response.text)
                self.assertEqual(response.json()["slide_count"], slide_count)

    async def test_slide_count_as_a_numeric_string_is_still_bounded(self):
        """«slide_count: "99"» не должен проезжать мимо проверки диапазона.

        Ровно та дыра, ради которой проверка стоит ПОСЛЕ приведения типа:
        валидатор «до» пропустил бы строку дальше, а Pydantic следом честно
        превратил бы её в число.
        """
        response = await self.order(slide_count=str(SLIDE_COUNT_MAX + 1))

        self.assert_code(response, 422, PresentationErrors.VALUE_OUT_OF_RANGE)

    async def test_missing_slide_count_falls_back_to_the_default(self):
        response = await self.order()

        self.assertEqual(response.status_code, 202, response.text)
        self.assertEqual(response.json()["slide_count"], SLIDE_COUNT_DEFAULT)

    async def test_description_over_the_limit_is_refused(self):
        response = await self.order(description="я" * (DESCRIPTION_MAX + 1))

        self.assert_code(response, 422, PresentationErrors.DESCRIPTION_TOO_LONG)

    async def test_description_at_the_limit_is_accepted(self):
        exact = "я" * DESCRIPTION_MAX

        response = await self.order(description=exact)

        self.assertEqual(response.status_code, 202, response.text)
        self.assertEqual(response.json()["description"], exact)

    async def test_padding_does_not_eat_into_the_description_limit(self):
        padded = "  " + "я" * DESCRIPTION_MAX + " \n\t"

        response = await self.order(description=padded)

        self.assertEqual(response.status_code, 202, response.text)
        self.assertEqual(response.json()["description"], "я" * DESCRIPTION_MAX)

    async def test_blank_description_is_allowed_and_stored_as_null(self):
        for description in (None, "", "   ", "\n\t "):
            with self.subTest(description=repr(description)):
                notebook = await self.make_notebook(f"Блокнот {description!r}")
                response = await self.order(notebook.id, description=description)

                self.assertEqual(response.status_code, 202, response.text)
                self.assertIsNone(response.json()["description"])
                row = await self.get_row(Presentation, response.json()["id"])
                self.assertIsNone(row.description)

    async def test_missing_template_key_keeps_the_pydantic_error(self):
        """Отсутствующее поле — отказ Pydantic: общий код и имя поля в detail.

        Свой код здесь был бы неправдой: шаблон не «неподдерживаемый», его
        просто не прислали, и разбирает такое клиент по массиву detail.
        """
        response = await self.client.post(
            f"/api/v1/notebooks/{self.notebook.id}/presentations", json={}
        )

        self.assertEqual(response.status_code, 422, response.text)
        body = response.json()
        self.assertEqual(body.get("error_code"), RequestErrors.VALIDATION_FAILED)
        self.assertIsInstance(body["detail"], list)
        self.assertEqual(body["detail"][0]["loc"][-1], "template_key")

    async def test_wrong_type_keeps_the_pydantic_error(self):
        for field, value in (
            ("slide_count", "много"),
            ("template_key", 5),
            ("language", {"code": "ru"}),
        ):
            with self.subTest(field=field):
                response = await self.order(**{field: value})

                self.assertEqual(response.status_code, 422, response.text)
                body = response.json()
                self.assertEqual(
                    body.get("error_code"), RequestErrors.VALIDATION_FAILED
                )
                self.assertEqual(body["detail"][0]["loc"][-1], field)

    async def test_notebook_id_outside_the_integer_range_is_refused(self):
        """id больше PostgreSQL integer роняет asyncpg — отсекаем на валидации."""
        response = await self.order(2_147_483_648)

        self.assertEqual(response.status_code, 422, response.text)
        self.assertEqual(
            response.json().get("error_code"), RequestErrors.VALIDATION_FAILED
        )


# --- Бизнес-условия ------------------------------------------------------


class OrderBusinessRulesTests(PresentationsApiTestCase):
    async def test_notebook_without_indexed_sources_is_refused(self):
        empty = await self.make_notebook("Пустой", indexed=False)

        response = await self.order(empty.id)

        self.assert_code(response, 409, PresentationErrors.NO_SOURCES)
        self.assertEqual(await self.all_rows(Presentation), [])

    async def test_sources_still_indexing_do_not_count(self):
        notebook = await self.make_notebook("Индексируется", indexed=False)
        await self.seed(
            Document(
                name="в работе.pdf",
                path=self.make_file("в работе.pdf"),
                size=10,
                status="pending",
                notebook_id=notebook.id,
                owner_id=self.manager.id,
            )
        )

        response = await self.order(notebook.id)

        self.assert_code(response, 409, PresentationErrors.NO_SOURCES)

    async def test_second_order_while_one_is_queued_is_refused(self):
        await self.make_presentation(status=STATUS_QUEUED)

        response = await self.order()

        self.assert_code(response, 409, PresentationErrors.GENERATION_IN_PROGRESS)
        self.assertEqual(len(await self.all_rows(Presentation)), 1)

    async def test_second_order_while_one_is_generating_is_refused(self):
        await self.make_presentation(status=STATUS_GENERATING)

        response = await self.order()

        self.assert_code(response, 409, PresentationErrors.GENERATION_IN_PROGRESS)

    async def test_finished_orders_do_not_block_a_new_one(self):
        for status in (STATUS_READY, STATUS_ERROR):
            with self.subTest(status=status):
                notebook = await self.make_notebook(f"Блокнот {status}")
                await self.make_presentation(notebook, status=status, with_file=True)

                response = await self.order(notebook.id)

                self.assertEqual(response.status_code, 202, response.text)

    async def test_an_active_order_of_another_notebook_does_not_block(self):
        other = await self.make_notebook("Соседний")
        await self.make_presentation(other, status=STATUS_GENERATING)

        response = await self.order()

        self.assertEqual(response.status_code, 202, response.text)

    async def test_no_sources_is_answered_before_generation_in_progress(self):
        """Порядок бизнес-проверок: сначала «нечего собирать»."""
        empty = await self.make_notebook("Пустой", indexed=False)
        await self.make_presentation(empty, status=STATUS_QUEUED)

        response = await self.order(empty.id)

        self.assert_code(response, 409, PresentationErrors.NO_SOURCES)


# --- Принятый заказ ------------------------------------------------------


class AcceptedOrderTests(PresentationsApiTestCase):
    async def test_accepted_order_returns_202_with_the_row_and_its_place(self):
        response = await self.order(
            slide_count=SLIDE_COUNT_MIN, language="tj", description="  Про льготы  "
        )

        self.assertEqual(response.status_code, 202, response.text)
        body = response.json()
        self.assertEqual(body["notebook_id"], self.notebook.id)
        self.assertEqual(body["template_key"], self.template_key)
        self.assertEqual(body["language"], "tj")
        self.assertEqual(body["slide_count"], SLIDE_COUNT_MIN)
        self.assertEqual(body["description"], "Про льготы")
        self.assertEqual(body["status"], STATUS_QUEUED)
        self.assertEqual(body["progress"], 0)
        self.assertEqual(body["queue_position"], 1)
        self.assertIsNone(body["error_code"])
        self.assertIsNone(body["file_size"])
        # Путь на сервере наружу не уходит никогда.
        self.assertNotIn("file_path", body)

    async def test_the_row_belongs_to_the_caller_and_to_the_notebook(self):
        response = await self.order()

        row = await self.get_row(Presentation, response.json()["id"])
        self.assertEqual(row.owner_id, self.manager.id)
        self.assertEqual(row.notebook_id, self.notebook.id)
        self.assertEqual(row.status, STATUS_QUEUED)

    async def test_queue_position_counts_the_whole_queue(self):
        """Очередь одна на систему: место считается по всем ждущим заказам."""
        first = await self.order()
        second_notebook = await self.make_notebook("Второй")
        second = await self.order(second_notebook.id)

        self.assertEqual(first.json()["queue_position"], 1)
        self.assertEqual(second.json()["queue_position"], 2)

    async def test_admin_order_belongs_to_the_admin(self):
        self.as_user(self.admin)

        response = await self.order()

        row = await self.get_row(Presentation, response.json()["id"])
        self.assertEqual(row.owner_id, self.admin.id)


# --- Чтение --------------------------------------------------------------


class ReadPresentationsTests(PresentationsApiTestCase):
    async def seed_page(self, count: int = 5) -> list[Presentation]:
        """Заказы с заведомо разным created_at — сверху самый свежий."""
        base = datetime(2026, 8, 1, 12, 0, 0)
        rows = []
        for index in range(count):
            rows.append(
                await self.make_presentation(
                    status=STATUS_READY,
                    created_at=base + timedelta(minutes=index),
                    updated_at=base + timedelta(minutes=index),
                )
            )
        return rows

    async def test_list_returns_the_freshest_first_with_a_total(self):
        rows = await self.seed_page()

        response = await self.list_orders()

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.headers[TOTAL_COUNT_HEADER], str(len(rows)))
        self.assertEqual(
            [item["id"] for item in response.json()],
            [row.id for row in reversed(rows)],
        )

    async def test_pagination_slices_the_same_order(self):
        rows = await self.seed_page()
        expected = [row.id for row in reversed(rows)]

        page = await self.list_orders(skip=1, limit=2)

        self.assertEqual([item["id"] for item in page.json()], expected[1:3])
        # Общее число — без учёта skip/limit.
        self.assertEqual(page.headers[TOTAL_COUNT_HEADER], str(len(rows)))

    async def test_equal_timestamps_do_not_shuffle_between_pages(self):
        """Тай-брейк по id: без него заказы одной миллисекунды прыгают."""
        same = datetime(2026, 8, 1, 12, 0, 0)
        rows = [
            await self.make_presentation(status=STATUS_READY, created_at=same)
            for _ in range(4)
        ]
        expected = [row.id for row in reversed(rows)]

        first = await self.list_orders(skip=0, limit=2)
        second = await self.list_orders(skip=2, limit=2)

        self.assertEqual(
            [item["id"] for item in first.json()] + [item["id"] for item in second.json()],
            expected,
        )

    async def test_pagination_bounds_are_the_same_as_for_sources(self):
        for params, ok in (
            ({"limit": MAX_PAGE_SIZE}, True),
            ({"limit": MAX_PAGE_SIZE + 1}, False),
            ({"limit": 0}, False),
            ({"skip": -1}, False),
        ):
            with self.subTest(params=params):
                response = await self.list_orders(**params)
                self.assertEqual(response.status_code, 200 if ok else 422)

    async def test_default_page_size_matches_the_sources_endpoint(self):
        self.assertIs(presentations_endpoint.DEFAULT_PAGE_SIZE, DEFAULT_PAGE_SIZE)
        self.assertIs(presentations_endpoint.MAX_PAGE_SIZE, MAX_PAGE_SIZE)

    async def test_list_of_a_foreign_notebook_is_404(self):
        foreign = await self.make_notebook("Чужой", owner=self.stranger)
        await self.make_presentation(foreign, owner=self.stranger)
        self.as_user(self.plain)

        response = await self.list_orders(foreign.id)

        self.assert_code(response, 404, SourceErrors.NOTEBOOK_NOT_FOUND)

    async def test_list_shows_only_this_notebook(self):
        mine = await self.make_presentation(status=STATUS_READY)
        other = await self.make_notebook("Соседний")
        await self.make_presentation(other, status=STATUS_READY)

        response = await self.list_orders()

        self.assertEqual([item["id"] for item in response.json()], [mine.id])

    async def test_queue_position_is_null_outside_the_queue(self):
        """Ноль читался бы как «следующая на очереди» — это ровно обратное."""
        for status in (STATUS_GENERATING, STATUS_READY, STATUS_ERROR):
            with self.subTest(status=status):
                row = await self.make_presentation(status=status)

                response = await self.get_order(row.id)

                self.assertEqual(response.status_code, 200, response.text)
                self.assertIsNone(response.json()["queue_position"])

    async def test_queue_position_is_reported_while_waiting(self):
        first = await self.make_presentation(
            status=STATUS_QUEUED, created_at=datetime(2026, 8, 1, 10, 0, 0)
        )
        other = await self.make_notebook("Второй")
        second = await self.make_presentation(
            other, status=STATUS_QUEUED, created_at=datetime(2026, 8, 1, 11, 0, 0)
        )

        self.assertEqual((await self.get_order(first.id)).json()["queue_position"], 1)
        self.assertEqual((await self.get_order(second.id)).json()["queue_position"], 2)

    async def test_polled_row_carries_progress_and_the_failure(self):
        row = await self.make_presentation(
            status=STATUS_ERROR,
            progress=40,
            error_code=PresentationErrors.GENERATION_TIMEOUT,
            error_text="Генерация не уложилась в отведённое время",
        )

        body = (await self.get_order(row.id)).json()

        self.assertEqual(body["status"], STATUS_ERROR)
        self.assertEqual(body["progress"], 40)
        self.assertEqual(body["error_code"], PresentationErrors.GENERATION_TIMEOUT)
        self.assertTrue(body["error_text"])

    async def test_ready_row_reports_its_size(self):
        row = await self.make_presentation(status=STATUS_READY, with_file=True)

        body = (await self.get_order(row.id)).json()

        self.assertEqual(body["file_size"], row.file_size)
        self.assertNotIn("file_path", body)

    async def test_dates_are_marked_as_utc(self):
        """Без явного смещения браузер в UTC+5 уводит дату на пять часов назад."""
        row = await self.make_presentation(
            status=STATUS_READY, created_at=datetime(2026, 8, 1, 9, 15, 0)
        )

        body = (await self.get_order(row.id)).json()

        self.assertTrue(body["created_at"].endswith("Z"), body["created_at"])
        self.assertTrue(body["updated_at"].endswith("Z"), body["updated_at"])
        self.assertEqual(
            datetime.fromisoformat(body["created_at"].replace("Z", "+00:00")),
            datetime(2026, 8, 1, 9, 15, tzinfo=timezone.utc),
        )

    async def test_foreign_presentation_is_404(self):
        foreign_notebook = await self.make_notebook("Чужой", owner=self.stranger)
        foreign = await self.make_presentation(
            foreign_notebook, owner=self.stranger, status=STATUS_READY
        )

        response = await self.get_order(foreign.id)

        self.assert_code(response, 404, PresentationErrors.NOT_FOUND)

    async def test_missing_presentation_is_the_same_404(self):
        response = await self.get_order(999_999)

        self.assert_code(response, 404, PresentationErrors.NOT_FOUND)

    async def test_admin_sees_a_foreign_presentation(self):
        foreign_notebook = await self.make_notebook("Чужой", owner=self.stranger)
        foreign = await self.make_presentation(
            foreign_notebook, owner=self.stranger, status=STATUS_READY
        )
        self.as_user(self.admin)

        self.assertEqual((await self.get_order(foreign.id)).status_code, 200)

    async def test_reading_does_not_require_the_role(self):
        """Роль нужна на заказе, а не на чтении: колоды живут на владении.

        Иначе пользователь, у которого забрали роль, терял бы доступ к уже
        заказанным им колодам — и они стали бы неудаляемым мусором на диске.
        """
        notebook = await self.make_notebook("Свой", owner=self.plain)
        row = await self.make_presentation(
            notebook, owner=self.plain, status=STATUS_READY
        )
        self.as_user(self.plain)

        self.assertEqual((await self.get_order(row.id)).status_code, 200)
        self.assertEqual((await self.list_orders(notebook.id)).status_code, 200)


# --- Скачивание ----------------------------------------------------------


class DownloadTests(PresentationsApiTestCase):
    async def test_ready_presentation_is_served(self):
        row = await self.make_presentation(status=STATUS_READY, with_file=True)

        response = await self.download(row.id)

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(
            response.headers["content-type"], presentations_endpoint.PPTX_MEDIA_TYPE
        )
        self.assertEqual(response.content, Path(row.file_path).read_bytes())

    async def test_filename_is_human_readable(self):
        row = await self.make_presentation(status=STATUS_READY, with_file=True)

        response = await self.download(row.id)

        disposition = response.headers["content-disposition"]
        expected = f"{self.notebook.name} — презентация.pptx"
        # Starlette кодирует не-ASCII имя сам (RFC 5987), поэтому сравниваем с
        # тем, что он должен был записать.
        self.assertIn(quote(expected), disposition)
        self.assertTrue(disposition.startswith("attachment"), disposition)

    async def test_filename_is_sanitized_and_cut_to_the_byte_limit(self):
        """Имя блокнота пишет пользователь: и слэши, и 255 кириллических букв.

        Чистится и режется теми же средствами, что имя источника, — иначе
        заголовок уносил бы в файловую систему скачивающего то, что там именем
        файла не является.
        """
        notebook = await self.make_notebook("../../etc/" + "я" * 255)
        row = await self.make_presentation(
            notebook, status=STATUS_READY, with_file=True
        )

        disposition = (await self.download(row.id)).headers["content-disposition"]

        self.assertNotIn("..", disposition)
        self.assertNotIn("/etc/", disposition)
        # Байтовая длина имени укладывается в NAME_MAX, а расширение уцелело.
        encoded = disposition.split("filename*=utf-8''", 1)[1]
        from urllib.parse import unquote

        name = unquote(encoded)
        self.assertLessEqual(len(name.encode("utf-8")), 255)
        self.assertTrue(name.endswith(".pptx"), name)

    async def test_unfinished_presentation_answers_409(self):
        for status in (STATUS_QUEUED, STATUS_GENERATING, STATUS_ERROR):
            with self.subTest(status=status):
                # Свой блокнот на каждый статус: 'queued' и 'generating' — оба
                # активные, а больше одного активного заказа на блокнот база не
                # держит (uq_presentation_active_notebook).
                notebook = await self.make_notebook(f"Блокнот {status}")
                row = await self.make_presentation(notebook, status=status)

                response = await self.download(row.id)

                self.assert_code(response, 409, PresentationErrors.NOT_READY)

    async def test_ready_row_without_a_file_answers_404(self):
        row = await self.make_presentation(status=STATUS_READY, with_file=True)
        os.remove(row.file_path)

        with self.assertLogs(ENDPOINT_LOGGER, level="WARNING"):
            response = await self.download(row.id)

        self.assert_code(response, 404, PresentationErrors.FILE_MISSING)

    async def test_ready_row_with_an_empty_path_answers_404(self):
        row = await self.make_presentation(status=STATUS_READY, with_file=False)

        with self.assertLogs(ENDPOINT_LOGGER, level="WARNING"):
            response = await self.download(row.id)

        self.assert_code(response, 404, PresentationErrors.FILE_MISSING)

    async def test_foreign_presentation_is_404_before_anything_else(self):
        foreign_notebook = await self.make_notebook("Чужой", owner=self.stranger)
        foreign = await self.make_presentation(
            foreign_notebook, owner=self.stranger, status=STATUS_QUEUED
        )

        response = await self.download(foreign.id)

        # Именно 404, а не 409: чужая строка не должна выдавать даже свой статус.
        self.assert_code(response, 404, PresentationErrors.NOT_FOUND)


# --- Удаление ------------------------------------------------------------


class DeleteTests(PresentationsApiTestCase):
    async def test_ready_presentation_and_its_file_are_removed(self):
        row = await self.make_presentation(status=STATUS_READY, with_file=True)
        path = row.file_path

        response = await self.delete_order(row.id)

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["id"], row.id)
        self.assertFalse(await self.exists(Presentation, row.id))
        self.assertFalse(os.path.exists(path), "файл готовой колоды остался")

    async def test_queued_and_failed_orders_are_removed_too(self):
        for status in (STATUS_QUEUED, STATUS_ERROR):
            with self.subTest(status=status):
                row = await self.make_presentation(
                    status=status, with_file=status == STATUS_ERROR
                )

                response = await self.delete_order(row.id)

                self.assertEqual(response.status_code, 200, response.text)
                self.assertFalse(await self.exists(Presentation, row.id))

    async def test_generating_presentation_is_refused(self):
        row = await self.make_presentation(status=STATUS_GENERATING)

        response = await self.delete_order(row.id)

        self.assert_code(response, 409, PresentationErrors.GENERATION_IN_PROGRESS)
        self.assertTrue(await self.exists(Presentation, row.id))

    async def test_row_without_a_file_is_removed_quietly(self):
        row = await self.make_presentation(status=STATUS_READY, with_file=True)
        os.remove(row.file_path)

        response = await self.delete_order(row.id)

        self.assertEqual(response.status_code, 200, response.text)
        self.assertFalse(await self.exists(Presentation, row.id))

    async def test_file_removal_failure_does_not_become_500(self):
        """Строки уже нет — врать клиенту про несделанную работу нельзя.

        Единственное требование: путь остаётся в журнале, чтобы файл можно
        было убрать руками.
        """
        row = await self.make_presentation(status=STATUS_READY, with_file=True)
        blocked = row.file_path

        def refuse(path, *args, **kwargs):
            raise PermissionError(13, "Permission denied")

        with patch(f"{ENDPOINT_LOGGER}.os.remove", side_effect=refuse):
            with self.assertLogs(ENDPOINT_LOGGER, level="WARNING") as logs:
                response = await self.delete_order(row.id)

        self.assertEqual(response.status_code, 200, response.text)
        self.assertFalse(await self.exists(Presentation, row.id))
        recorded = "\n".join(logs.output)
        self.assertIn(blocked, recorded)
        self.assertIn(str(row.id), recorded)
        self.assertTrue(os.path.exists(blocked), "файл всё-таки исчез")

    async def test_file_is_touched_only_after_the_row_is_committed(self):
        """Порядок «строка, commit, потом файл» — зеркало удаления блокнота.

        Проверяется с той стороны, с которой это вообще наблюдаемо: os.remove
        падает, а строки всё равно нет. Значит, commit случился ДО работы с
        диском, а не после неё. Обратный порядок при откате транзакции
        уничтожил бы файл у живой строки — 'ready' с путём в никуда.
        """
        row = await self.make_presentation(status=STATUS_READY, with_file=True)
        seen: list[bool] = []
        real_remove = os.remove

        def spy(path, *args, **kwargs):
            # В момент вызова файл ещё на месте: удаление строки его не
            # трогало, этим занята только эта строчка обработчика.
            seen.append(os.path.exists(path))
            real_remove(path)

        with patch(f"{ENDPOINT_LOGGER}.os.remove", side_effect=spy):
            response = await self.delete_order(row.id)

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(seen, [True])
        self.assertFalse(await self.exists(Presentation, row.id))

    async def test_refused_delete_touches_neither_row_nor_file(self):
        """409 на 'generating' не должен ничего снести по дороге."""
        row = await self.make_presentation(status=STATUS_GENERATING, with_file=True)

        response = await self.delete_order(row.id)

        self.assertEqual(response.status_code, 409, response.text)
        self.assertTrue(await self.exists(Presentation, row.id))
        self.assertTrue(os.path.exists(row.file_path))

    async def test_foreign_presentation_is_404(self):
        foreign_notebook = await self.make_notebook("Чужой", owner=self.stranger)
        foreign = await self.make_presentation(
            foreign_notebook, owner=self.stranger, status=STATUS_READY, with_file=True
        )

        response = await self.delete_order(foreign.id)

        self.assert_code(response, 404, PresentationErrors.NOT_FOUND)
        self.assertTrue(await self.exists(Presentation, foreign.id))
        self.assertTrue(os.path.exists(foreign.file_path))

    async def test_deleting_frees_the_notebook_for_a_new_order(self):
        """409 «уже в очереди» — не приговор: удалил заказ, заказал заново."""
        row = await self.make_presentation(status=STATUS_QUEUED)
        self.assert_code(
            await self.order(), 409, PresentationErrors.GENERATION_IN_PROGRESS
        )

        await self.delete_order(row.id)

        self.assertEqual((await self.order()).status_code, 202)


# --- Шаблоны -------------------------------------------------------------


class TemplatesTests(PresentationsApiTestCase):
    async def test_list_returns_every_usable_template(self):
        response = await self.client.get(TEMPLATES_URL)

        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(
            [item["key"] for item in body],
            [info.key for info in template_registry.list()],
        )
        for item in body:
            with self.subTest(template=item["key"]):
                # Название на обоих языках: подпись в форме выбора не должна
                # быть пустой ни при каком языке интерфейса.
                self.assertEqual(set(item["name"]), {"ru", "tj"})
                self.assertTrue(all(item["name"].values()))
                self.assertTrue(item["preview_url"].endswith("/preview"))

    async def test_templates_require_the_role(self):
        self.as_user(self.plain)

        response = await self.client.get(TEMPLATES_URL)

        self.assert_code(response, 403, PresentationErrors.ROLE_NOT_ALLOWED)

    async def test_admin_sees_the_templates(self):
        self.as_user(self.admin)

        self.assertEqual((await self.client.get(TEMPLATES_URL)).status_code, 200)

    async def test_preview_is_served_by_key(self):
        info = template_registry.list()[0]

        response = await self.client.get(f"{TEMPLATES_URL}/{info.key}/preview")

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.headers["content-type"], "image/png")
        self.assertEqual(response.content, info.preview_file.read_bytes())

    async def test_preview_url_from_the_list_actually_works(self):
        listed = (await self.client.get(TEMPLATES_URL)).json()

        for item in listed:
            with self.subTest(template=item["key"]):
                response = await self.client.get(item["preview_url"])
                self.assertEqual(response.status_code, 200, response.text)

    async def test_preview_requires_the_role(self):
        info = template_registry.list()[0]
        self.as_user(self.plain)

        response = await self.client.get(f"{TEMPLATES_URL}/{info.key}/preview")

        self.assert_code(response, 403, PresentationErrors.ROLE_NOT_ALLOWED)

    async def test_unknown_key_is_404(self):
        response = await self.client.get(f"{TEMPLATES_URL}/нет-такого/preview")

        self.assert_code(response, 404, PresentationErrors.UNSUPPORTED_TEMPLATE)

    async def test_traversal_attempts_read_nothing(self):
        """Путь берётся из реестра, поэтому «../» — просто неизвестный ключ.

        Проверяются три формы: закодированные слэши (маршрут такой путь не
        распознаёт вовсе), голый «..» (клиент нормализует его ещё до отправки,
        и запрос уходит вообще на другой маршрут) и ключ, похожий на имя файла
        в каталоге шаблонов. Утверждение общее для всех: ответ не 200 и в теле
        нет ни манифеста, ни pptx — то есть прочитать файл рядом с превью
        нельзя ни одним из способов.
        """
        for key in (
            "..%2F..%2Fmanifest.json",
            "%2e%2e%2f%2e%2e%2fmanifest.json",
            "..",
            "manifest.json",
            "classic.pptx",
        ):
            with self.subTest(key=key):
                response = await self.client.get(f"{TEMPLATES_URL}/{key}/preview")

                self.assertNotEqual(response.status_code, 200, response.text)
                self.assertNotIn(b"template_file", response.content)
                self.assertNotIn(b"PK\x03\x04", response.content)

    async def test_a_key_shaped_like_a_path_is_an_unknown_template(self):
        """Ключ, похожий на файл из каталога шаблонов, — обычный 404 с кодом."""
        for key in ("manifest.json", "classic.pptx"):
            with self.subTest(key=key):
                response = await self.client.get(f"{TEMPLATES_URL}/{key}/preview")

                self.assert_code(
                    response, 404, PresentationErrors.UNSUPPORTED_TEMPLATE
                )

    async def test_registry_paths_stay_inside_their_own_roots(self):
        """Страховка на реестр: наружу он путей не выпускает по построению.

        Корней теперь ДВА, и это не ослабление проверки, а следствие перехода
        на HTML. Исходники шаблона лежат в templates/ и приезжают с релизом;
        превью рисует Chrome при старте, и оно попадает в data/ — как всякий
        машинный результат, который зависит ещё и от версии браузера на машине.
        Утверждение осталось прежним: каждый путь заперт в СВОЁМ каталоге, и
        запись вида "../.." в манифесте или ключ вида "../x" не выводят реестр
        ни за один из них.
        """
        templates_root = default_templates_dir().resolve()
        preview_root = default_preview_dir().resolve()
        for info in template_registry.list():
            with self.subTest(template=info.key):
                self.assertTrue(
                    info.html_file.resolve().is_relative_to(templates_root)
                )
                self.assertTrue(info.css_file.resolve().is_relative_to(templates_root))
                if info.preview_file is not None:
                    self.assertTrue(
                        info.preview_file.resolve().is_relative_to(preview_root)
                    )


if __name__ == "__main__":
    unittest.main()
