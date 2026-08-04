"""Гонка двойного заказа: инвариант держит база, а не предпроверка.

Инвариант раздела — «не больше одной активной генерации на блокнот». На нём
стоит вся очередь: воркер берёт задачи по одной, позиция считается по всем
ждущим, а GPU — самый дефицитный ресурс системы. Нарушение стоит двойной работы
модели, двух файлов-дублей и спутанных позиций очереди.

Держать его предпроверкой в хендлере нельзя по устройству: SELECT активных
строк и INSERT — два разных запроса, и между ними помещается целиком второй
такой же запрос. Два клика в одну секунду проходят проверку ОБА. Лимит частоты
здесь не помощник — он считает десятки заказов в час, а не два в секунду.

Поэтому проверка живёт в базе: частичный уникальный индекс
`uq_presentation_active_notebook` — `UNIQUE (notebook_id) WHERE status IN
('queued','generating')`. Его нарушение хендлер переводит в тот же 409
`presentation.generation_in_progress`, что и предпроверка: клиент не должен
различать, кто именно отказал.

Что здесь проверяется и почему именно так:

  * запросы идут ПАРАЛЛЕЛЬНО (`asyncio.gather`). Последовательные два POST
    ловит предпроверка, и такой тест зеленел бы и без индекса — то есть не
    доказывал бы ничего;
  * оба запроса заведомо проходят предпроверку до первой вставки: створка в
    `PresentationsService.create` держит первого, пока не подойдёт второй.
    Без неё исход зависел бы от того, как лягут переключения корутин, и тест
    «мигал» бы — а мигающий тест на гонку хуже отсутствующего, потому что его
    зелёный цвет ничего не значит;
  * рядом стоит ОБРАТНЫЙ тест: с временно снятым индексом та же гонка даёт два
    202 и две активные строки. Он и доказывает, что основной тест краснеет без
    индекса, а не проходит благодаря предпроверке.
"""

import asyncio
import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dbfixtures import DatabaseBackedTestCase  # noqa: E402

from sqlalchemy import text  # noqa: E402
from sqlalchemy.exc import IntegrityError  # noqa: E402

import app.core.database as database_module  # noqa: E402
from app.api.endpoints import presentations as presentations_endpoint  # noqa: E402
from app.core.database import (  # noqa: E402
    PRESENTATION_ACTIVE_INDEX,
    PRESENTATION_ACTIVE_STATUSES,
)
from app.core.exceptions import PresentationErrors  # noqa: E402
from app.modules.presentations.constants import (  # noqa: E402
    STATUS_ERROR,
    STATUS_GENERATING,
    STATUS_QUEUED,
    STATUS_READY,
)
from app.modules.presentations.service import PresentationsService  # noqa: E402
from app.modules.presentations.templates import template_registry  # noqa: E402
from app.shared.models import Document, Notebook, Presentation, User  # noqa: E402


def a_template_key() -> str:
    templates = template_registry.list()
    if not templates:  # pragma: no cover - зависит от комплекта на диске
        raise unittest.SkipTest("В комплекте нет ни одного пригодного шаблона")
    return templates[0].key


class DoubleOrderTestCase(DatabaseBackedTestCase):
    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()

        self.manager: User = await self.make_user("manager", "content_manager")
        self.as_user(self.manager)
        self.notebook: Notebook = await self.seed(
            Notebook(name="Налоги", domain_profile="general", owner_id=self.manager.id)
        )
        await self.seed(
            Document(
                name="Кодекс.pdf",
                path=self.make_file("Кодекс.pdf"),
                size=10,
                status="indexed",
                notebook_id=self.notebook.id,
                owner_id=self.manager.id,
            )
        )
        self.template_key = a_template_key()

        presentations_endpoint.order_limiter.clients.clear()
        self.addCleanup(presentations_endpoint.order_limiter.clients.clear)

    async def order(self):
        return await self.client.post(
            f"/api/v1/notebooks/{self.notebook.id}/presentations",
            json={"template_key": self.template_key},
        )

    async def race(self):
        """Два POST, вставки которых заведомо разъезжаются с предпроверками.

        Створка отпускает обоих только после того, как оба дошли до создания
        строки, то есть уже прошли SELECT активных заказов. Это и есть гонка
        двойного заказа в её честном виде: без створки первый запрос успевал бы
        закоммитить строку раньше, чем второй доберётся до своей проверки, и
        тест проверял бы предпроверку, а не индекс.
        """
        gate = asyncio.Event()
        arrivals = 0
        original = PresentationsService.create

        async def gated_create(session, **kwargs):
            nonlocal arrivals
            arrivals += 1
            if arrivals >= 2:
                gate.set()
            await asyncio.wait_for(gate.wait(), timeout=10)
            return await original(session, **kwargs)

        with patch.object(
            PresentationsService, "create", staticmethod(gated_create)
        ):
            responses = await asyncio.gather(self.order(), self.order())

        self.assertEqual(arrivals, 2, "предпроверку прошёл не каждый запрос")
        return responses

    async def active_rows(self) -> list[Presentation]:
        rows = await self.rows_where(
            Presentation, Presentation.notebook_id == self.notebook.id
        )
        return [row for row in rows if row.status in PRESENTATION_ACTIVE_STATUSES]

    async def drop_index(self) -> None:
        """Снять индекс в тестовой схеме и вернуть его в cleanup.

        Возврат идёт вместе с уборкой строк, которые без индекса успели
        появиться: иначе CREATE UNIQUE INDEX упал бы на них же и унёс с собой
        весь остаток прогона — тестовая схема живёт один процесс на все файлы.
        """
        definition = await self.index_definition()
        self.assertIsNotNone(definition, "снимать нечего: индекса и так нет")
        async with self.engine.begin() as conn:
            await conn.execute(
                text(f'DROP INDEX IF EXISTS "{PRESENTATION_ACTIVE_INDEX}"')
            )
        self.addAsyncCleanup(self.restore_index, definition)

    async def restore_index(self, definition: str) -> None:
        # Снимаем и создаём заново, а не создаём поверх: тест мог позвать
        # init_db, и индекс уже вернулся бы на место сам (definition из
        # pg_indexes приходит без IF NOT EXISTS).
        async with self.engine.begin() as conn:
            await conn.execute(
                text(f'DROP INDEX IF EXISTS "{PRESENTATION_ACTIVE_INDEX}"')
            )
            await conn.execute(text("DELETE FROM presentation"))
            await conn.execute(text(definition))

    async def index_definition(self) -> str | None:
        async with self.engine.connect() as conn:
            result = await conn.execute(
                text(
                    """
                    SELECT indexdef FROM pg_indexes
                    WHERE schemaname = current_schema()
                      AND tablename = 'presentation'
                      AND indexname = :name
                    """
                ),
                {"name": PRESENTATION_ACTIVE_INDEX},
            )
            row = result.first()
        return None if row is None else row[0]


# --- Осмысленность окружения ---------------------------------------------


class SchemaSanityTests(DoubleOrderTestCase):
    """Без индекса в схеме остальные проверки файла ничего не доказывают."""

    async def test_partial_unique_index_exists(self):
        definition = await self.index_definition()

        self.assertIsNotNone(
            definition,
            f"init_db не создал {PRESENTATION_ACTIVE_INDEX} — воспроизводить "
            f"гонку не на чем",
        )
        self.assertIn("UNIQUE INDEX", definition)
        self.assertIn("(notebook_id)", definition)

    async def test_predicate_covers_exactly_the_active_statuses(self):
        """Предикат индекса и множество статусов хендлера — одно и то же.

        Разъедься они, одна из двух проверок начнёт пропускать то, что
        отвергает другая: заказ, отбитый предпроверкой, но разрешённый базой
        (или наоборот) — это либо дыра в инварианте, либо 500 на ровном месте.
        """
        definition = await self.index_definition()
        self.assertIsNotNone(definition)

        predicate = definition.split("WHERE", 1)[1]
        for status in PRESENTATION_ACTIVE_STATUSES:
            self.assertIn(f"'{status}'", predicate)
        for status in (STATUS_READY, STATUS_ERROR):
            self.assertNotIn(f"'{status}'", predicate)
        self.assertEqual(
            tuple(presentations_endpoint.ACTIVE_STATUSES), PRESENTATION_ACTIVE_STATUSES
        )

    async def test_finished_orders_are_outside_the_index(self):
        """Готовых и упавших колод у блокнота сколько угодно.

        Иначе индекс запрещал бы заказывать колоду повторно — то самое
        действие, ради которого пользователь и приходит после неудачи.
        """
        for status in (STATUS_READY, STATUS_READY, STATUS_ERROR):
            await self.seed(
                Presentation(
                    notebook_id=self.notebook.id,
                    owner_id=self.manager.id,
                    template_key=self.template_key,
                    language="ru",
                    slide_count=5,
                    status=status,
                )
            )

        self.assertEqual(len(await self.all_rows(Presentation)), 3)
        # И поверх них штатно встаёт новый заказ.
        self.assertEqual((await self.order()).status_code, 202)


# --- Сама гонка -----------------------------------------------------------


class ParallelOrderTests(DoubleOrderTestCase):
    async def test_two_parallel_orders_leave_exactly_one_active_row(self):
        first, second = await self.race()

        codes = sorted(response.status_code for response in (first, second))
        self.assertEqual(codes, [202, 409], f"{first.text} | {second.text}")

        refused = first if first.status_code == 409 else second
        self.assertEqual(
            refused.json().get("error_code"),
            PresentationErrors.GENERATION_IN_PROGRESS,
            refused.text,
        )
        # Главное: в базе ровно одна активная строка, а не две.
        active = await self.active_rows()
        self.assertEqual(len(active), 1, [row.status for row in active])
        self.assertEqual(active[0].status, STATUS_QUEUED)
        self.assertEqual(len(await self.all_rows(Presentation)), 1)

    async def test_refusal_is_indistinguishable_from_the_pre_check(self):
        """Отказ базы и отказ предпроверки — один ответ до последнего поля.

        Иначе клиенту пришлось бы знать про устройство сервера, чтобы понять,
        что делать: событие-то одно — «по этому блокноту уже идёт работа».
        """
        first, second = await self.race()
        by_race = first if first.status_code == 409 else second

        # Тот же отказ, пришедший обычным путём: строка уже стоит в очереди.
        by_pre_check = await self.order()

        self.assertEqual(by_pre_check.status_code, 409, by_pre_check.text)
        self.assertEqual(by_race.json(), by_pre_check.json())

    async def test_the_accepted_order_is_a_usable_one(self):
        """Победитель гонки — обычный заказ, а не покалеченная строка.

        Проигравшая транзакция откатывается, и откат не имеет права утащить с
        собой поля победителя или его место в очереди.
        """
        first, second = await self.race()
        accepted = first if first.status_code == 202 else second
        body = accepted.json()

        self.assertEqual(body["status"], STATUS_QUEUED)
        self.assertEqual(body["queue_position"], 1)
        self.assertEqual(body["notebook_id"], self.notebook.id)

        row = await self.get_row(Presentation, body["id"])
        self.assertIsNotNone(row)
        self.assertEqual(row.owner_id, self.manager.id)

        # И очередь его видит: воркер обязан забрать именно эту строку.
        async with self.session_factory() as session:
            self.assertEqual(await PresentationsService.claim_next(session), row.id)
        self.assertEqual(
            (await self.get_row(Presentation, row.id)).status, STATUS_GENERATING
        )

    async def test_a_foreign_integrity_error_is_not_disguised_as_409(self):
        """Чужое нарушение целостности остаётся ошибкой, а не «уже в работе».

        Перехват узкий намеренно: 409 на любой IntegrityError означал бы, что
        сломанный внешний ключ годами показывается пользователю как «дождитесь
        окончания генерации».
        """
        original = PresentationsService.create

        async def broken_create(session, **kwargs):
            await original(session, **kwargs)
            # Ссылка на несуществующий блокнот: presentation_notebook_id_fkey.
            await session.execute(
                text(
                    """
                    INSERT INTO presentation
                        (notebook_id, owner_id, template_key, language,
                         slide_count, status, progress, created_at, updated_at)
                    VALUES (:notebook_id, :owner_id, 'classic', 'ru', 5,
                            'ready', 0, NOW(), NOW())
                    """
                ),
                {"notebook_id": 10_000_000, "owner_id": self.manager.id},
            )
            await session.commit()

        with patch.object(
            PresentationsService, "create", staticmethod(broken_create)
        ):
            # raise_app_exceptions=True: непойманное исключение приложения
            # доходит до теста настоящим трейсбеком, а не превращается в 500.
            with self.assertRaises(IntegrityError) as caught:
                await self.order()

        self.assertIn("presentation_notebook_id_fkey", str(caught.exception))


# --- Обратная проверка: без индекса гонка проходит ------------------------


class WithoutTheIndexTests(DoubleOrderTestCase):
    """Контрольный опыт: что именно ловит гонку.

    Индекс снимается прямо в тестовой схеме и возвращается на место в cleanup
    (см. drop_index в общей обвязке).
    """

    async def test_without_the_index_both_orders_are_accepted(self):
        """Тот же тест, что и в ParallelOrderTests, но на схеме без индекса.

        Он и есть доказательство, что основной тест краснеет: предпроверка,
        рейт-лимит и порядок зависимостей остались нетронутыми — и пропустили
        оба заказа.
        """
        await self.drop_index()

        first, second = await self.race()

        self.assertEqual(
            [first.status_code, second.status_code],
            [202, 202],
            "без индекса гонку поймало что-то другое — проверьте створку теста",
        )
        active = await self.active_rows()
        self.assertEqual(
            len(active), 2, "двойной заказ не воспроизвёлся — тест ничего не доказывает"
        )

    async def test_sequential_orders_are_still_refused_without_the_index(self):
        """И почему одного последовательного теста было бы мало.

        Без индекса ДВА ПОДРЯД заказа по-прежнему отбиваются — предпроверкой.
        Значит, последовательный тест зеленеет на сломанном инварианте и
        доказывает только то, что предпроверка на месте.
        """
        await self.drop_index()

        self.assertEqual((await self.order()).status_code, 202)
        second = await self.order()

        self.assertEqual(second.status_code, 409, second.text)
        self.assertEqual(
            second.json().get("error_code"), PresentationErrors.GENERATION_IN_PROGRESS
        )


# --- Миграция на грязных данных ------------------------------------------


class DirtyDataStartupTests(DoubleOrderTestCase):
    """Старт приложения на базе, пережившей гонку ДО появления индекса.

    Там уже лежит блокнот с двумя активными заказами, и `CREATE UNIQUE INDEX`
    на таких данных падает. Падение внутри `init_db` означало бы, что
    приложение вообще не поднимается — и чинить данные пришлось бы вслепую из
    psql, без единого экрана. Поэтому шаг пропускается с ERROR в журнале, а
    инвариант до починки держится предпроверкой в обработчике.
    """

    async def make_active_pair(self) -> None:
        for _ in range(2):
            await self.seed(
                Presentation(
                    notebook_id=self.notebook.id,
                    owner_id=self.manager.id,
                    template_key=self.template_key,
                    language="ru",
                    slide_count=5,
                    status=STATUS_QUEUED,
                )
            )

    async def run_init_db(self):
        """init_db по тестовой схеме — тот же приём, что в dbfixtures."""
        with patch.object(database_module, "engine", self.engine):
            await database_module.init_db()

    async def test_startup_survives_a_notebook_with_two_active_orders(self):
        await self.drop_index()
        await self.make_active_pair()

        with self.assertLogs("app.core.database", level="ERROR") as logs:
            await self.run_init_db()  # не бросает — это и есть проверка

        recorded = "\n".join(logs.output)
        self.assertIn(PRESENTATION_ACTIVE_INDEX, recorded)
        # В журнале сказано, ЧТО чинить: номер блокнота и число заказов.
        self.assertIn(f"notebook {self.notebook.id}: 2", recorded)
        self.assertIsNone(
            await self.index_definition(), "индекс создан поверх грязных данных"
        )
        # Остальная схема доехала: старт состоялся целиком, а не наполовину.
        self.assertEqual((await self.order()).status_code, 409)

    async def test_the_index_arrives_on_the_first_start_after_the_cleanup(self):
        """Починка данных — это удалить лишние заказы и перезапустить."""
        await self.drop_index()
        await self.make_active_pair()
        await self.run_init_db()

        rows = sorted(await self.all_rows(Presentation), key=lambda row: row.id)
        async with self.engine.begin() as conn:
            await conn.execute(
                text("DELETE FROM presentation WHERE id = :id"), {"id": rows[-1].id}
            )

        await self.run_init_db()

        self.assertIsNotNone(await self.index_definition())


if __name__ == "__main__":
    unittest.main()
