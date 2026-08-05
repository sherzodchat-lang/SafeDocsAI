"""Бюджет времени презентации: потолок вызова и выведенный из него потолок джобы.

Ни базы, ни Ollama здесь нет — проверяется арифметика и её свойства. Смысл
файла в том, что потолок джобы ПЕРЕСТАЛ быть константой: раньше здесь стояло
PRESENTATION_JOB_TIMEOUT = 600, посчитанное для другой модели и другого
корпуса, и молчало ровно до того дня, когда модель сменили из админ-панели.
Теперь калибруется одна величина — LLM_CALL_TIMEOUT на ОДИН вызов, — а потолок
джобы считается из числа слайдов заказа.

Второе, что здесь закреплено, — ПРАВИЛО, по которому формула собрана: каждый
её член равен ВНЕШНЕЙ границе своей стадии, а не внутренней, которая «обычно
срабатывает первой». У вызова модели границ две (клиентский LLM_CALL_TIMEOUT и
страховка LLM_CALL_WATCHDOG_TIMEOUT), и формула, посчитанная по первой,
занижала потолок ровно на те случаи, ради которых заведена вторая.

Числа заказа в проверках не выписаны: они спрашиваются у констант. Тест на
конкретные секунды краснел бы при честной перекалибровке потолка вызова и
пропускал бы то единственное, ради чего заведён, — потерю связи между
потолком джобы и размером заказа.
"""

from __future__ import annotations

import unittest

from app.modules.presentations.constants import (
    LLM_CALL_ATTEMPTS,
    LLM_CALL_TIMEOUT,
    LLM_CALL_WATCHDOG_TIMEOUT,
    LLM_RETRY_PAUSE_AFTER_TIMEOUT,
    SLIDE_COUNT_MAX,
    SLIDE_COUNT_MIN,
    presentation_job_timeout,
)
from app.modules.presentations.llm_schemas import content_section_count
from app.modules.presentations.service import CallTimings


class JobCeilingIsDerivedTests(unittest.TestCase):
    """Потолок джобы выводится из slide_count, а не назначается."""

    def test_the_old_constant_is_gone(self) -> None:
        """Потолка джобы больше нет среди констант — ни под каким именем.

        Проверка не косметическая. Пока значение лежит константой, его можно
        импортировать и подставить в wait_for мимо формулы — и вернуться ровно
        к тому дефекту, из-за которого здоровые заказы падали в 'error'.
        """
        from app.modules.presentations import constants

        self.assertFalse(
            hasattr(constants, "PRESENTATION_JOB_TIMEOUT"),
            "потолок джобы снова записан числом",
        )

    def test_ceiling_grows_with_the_order(self) -> None:
        ceilings = [
            presentation_job_timeout(slide_count)
            for slide_count in range(SLIDE_COUNT_MIN, SLIDE_COUNT_MAX + 1)
        ]
        self.assertEqual(ceilings, sorted(ceilings))
        self.assertLess(ceilings[0], ceilings[-1])

    def test_ceiling_matches_the_declared_formula(self) -> None:
        """Вызовы по ВНЕШНЕЙ границе + паузы повторов + стадия рендера."""
        for slide_count in (SLIDE_COUNT_MIN, SLIDE_COUNT_MAX):
            with self.subTest(slide_count=slide_count):
                calls = 1 + content_section_count(slide_count)
                expected = (
                    calls * LLM_CALL_ATTEMPTS * LLM_CALL_WATCHDOG_TIMEOUT
                    + calls * (LLM_CALL_ATTEMPTS - 1) * LLM_RETRY_PAUSE_AFTER_TIMEOUT
                    + LLM_CALL_TIMEOUT
                )
                self.assertEqual(presentation_job_timeout(slide_count), expected)

    def test_every_term_is_the_outer_boundary_of_its_stage(self) -> None:
        """Правило формулы: внешняя граница, а не «обычно срабатывающая».

        У вызова модели границ две: клиент (LLM_CALL_TIMEOUT) и страховка
        wait_for (LLM_CALL_WATCHDOG_TIMEOUT). В норме первым сдаётся клиент —
        но именно «в норме»: страховка заведена ровно на тот случай, когда он
        не сработал (разбор уже полученного тела, незакрытый поток), и тогда
        стадия честно занимает watchdog. Потолок джобы, посчитанный по
        внутренней границе, короче правды на разницу, помноженную на число
        вызовов, — то есть ошибается в единственную запрещённую сторону.
        """
        self.assertGreater(
            LLM_CALL_WATCHDOG_TIMEOUT,
            LLM_CALL_TIMEOUT,
            "проверка выродилась: у вызова перестало быть двух границ",
        )
        for slide_count in (SLIDE_COUNT_MIN, SLIDE_COUNT_MAX):
            with self.subTest(slide_count=slide_count):
                calls = 1 + content_section_count(slide_count)
                self.assertGreaterEqual(
                    presentation_job_timeout(slide_count),
                    calls * LLM_CALL_ATTEMPTS * LLM_CALL_WATCHDOG_TIMEOUT,
                    "вызовы заложены по внутренней границе клиента",
                )

    def test_the_ceiling_covers_a_run_where_every_stage_hits_its_budget(self) -> None:
        """Заказ, у которого КАЖДАЯ стадия уложилась впритык, не снимается.

        Стадии перечисляются по одной, а не подставляются в формулу: так
        проверка ловит забытую стадию рендера, забытые паузы повторов и сдвиг
        на единицу в числе секций, то есть ровно те ошибки, из-за которых
        потолок джобы снимал бы заказ, где ни один вызов сам по себе не нарушил
        свой бюджет. Это и была бы ошибка в короткую сторону — та, ради
        устранения которой таймаут опустили на уровень вызова.

        Худший прогон здесь честный: каждая попытка досидела до ВНЕШНЕЙ границы
        вызова, и каждый повтор был повтором после таймаута, то есть с паузой.
        """
        for slide_count in range(SLIDE_COUNT_MIN, SLIDE_COUNT_MAX + 1):
            with self.subTest(slide_count=slide_count):
                stages = ["план"]
                stages += [
                    f"секция {index + 1}"
                    for index in range(content_section_count(slide_count))
                ]
                worst_run = 0
                for _ in stages:
                    worst_run += LLM_CALL_ATTEMPTS * LLM_CALL_WATCHDOG_TIMEOUT
                    worst_run += (
                        LLM_CALL_ATTEMPTS - 1
                    ) * LLM_RETRY_PAUSE_AFTER_TIMEOUT
                worst_run += LLM_CALL_TIMEOUT  # стадия рендера
                self.assertGreaterEqual(
                    presentation_job_timeout(slide_count), worst_run
                )

    def test_a_broken_slide_count_still_yields_a_positive_ceiling(self) -> None:
        """Мусорное число слайдов не должно превращаться в мгновенный таймаут.

        Такое число до пайплайна доезжать не должно (его отвергает HTTP-слой и
        сам пайплайн), но wait_for с отрицательным таймаутом снял бы джобу
        мгновенно и подменил понятный отказ «число слайдов вне границ» на
        невнятный generation_timeout.
        """
        for slide_count in (0, 1, 2, -5):
            with self.subTest(slide_count=slide_count):
                self.assertGreater(presentation_job_timeout(slide_count), 0)

    def test_the_ceiling_is_generous_against_the_measured_deck(self) -> None:
        """Замер приёмки: колода из 15 слайдов — 444 с целиком.

        Потолок джобы обязан лежать НАМНОГО выше: он страховка от
        бесконечного цикла, а не обещание времени. Ошибаться можно только в
        длинную сторону — короткий потолок убивает здоровые заказы, длинный
        стоит лишь «зависший заказ умрёт позже».
        """
        measured_full_deck_seconds = 444.0
        self.assertGreater(
            presentation_job_timeout(SLIDE_COUNT_MAX),
            measured_full_deck_seconds * 5,
        )


class CallTimeoutIsCalibratedTests(unittest.TestCase):
    """Потолок ОДНОГО вызова — против наблюдённых замеров приёмки."""

    # Замеры приёмки (gemma4:31b, корпус 714 чанков): худший наблюдённый
    # вызов — план за 69.9 с, худший слайд-вызов — 41.07 с.
    WORST_OBSERVED_CALL_SECONDS = 69.9

    def test_call_timeout_keeps_a_wide_margin(self) -> None:
        self.assertGreaterEqual(
            LLM_CALL_TIMEOUT, self.WORST_OBSERVED_CALL_SECONDS * 4
        )

    def test_the_retry_is_a_separate_call_in_the_arithmetic(self) -> None:
        """Повтор входит в потолок джобы множителем, а не слагаемым.

        У него свой бюджет: он заново генерирует весь ответ, получив исходный
        промпт, отвергнутый ответ и претензию валидатора. Заложи его слагаемым
        (одним лишним вызовом на джобу), и потолок оказался бы вдвое ниже
        честного худшего случая.
        """
        self.assertGreater(LLM_CALL_ATTEMPTS, 1, "повторной попытки нет вовсе")
        calls = 1 + content_section_count(SLIDE_COUNT_MAX)
        without_render = presentation_job_timeout(SLIDE_COUNT_MAX) - LLM_CALL_TIMEOUT
        without_pauses = without_render - calls * (
            LLM_CALL_ATTEMPTS - 1
        ) * LLM_RETRY_PAUSE_AFTER_TIMEOUT
        self.assertEqual(
            without_pauses, calls * LLM_CALL_ATTEMPTS * LLM_CALL_WATCHDOG_TIMEOUT
        )

    def test_the_retry_pause_is_a_term_of_the_formula(self) -> None:
        """Пауза перед повтором проходит под тем же потолком, что и вызовы.

        Забытая в формуле, она делает потолок короче правды ровно на себя,
        помноженную на число повторов, — и снимает заказ, у которого каждая
        стадия уложилась в свой бюджет. Проверка сравнивает потолок с тем же
        потолком без пауз, а не с константой: перекалибровка паузы её не
        покрасит, а вот исчезновение члена — покрасит.
        """
        self.assertGreater(
            LLM_RETRY_PAUSE_AFTER_TIMEOUT, 0, "паузы перед повтором нет вовсе"
        )
        for slide_count in (SLIDE_COUNT_MIN, SLIDE_COUNT_MAX):
            with self.subTest(slide_count=slide_count):
                calls = 1 + content_section_count(slide_count)
                pauses = (
                    calls * (LLM_CALL_ATTEMPTS - 1) * LLM_RETRY_PAUSE_AFTER_TIMEOUT
                )
                self.assertGreaterEqual(
                    presentation_job_timeout(slide_count)
                    - calls * LLM_CALL_ATTEMPTS * LLM_CALL_WATCHDOG_TIMEOUT
                    - LLM_CALL_TIMEOUT,
                    pauses,
                    "паузы повторов выпали из потолка джобы",
                )


class CallStatisticsTests(unittest.TestCase):
    """Строка p50/p90: то, чем заменена невозможная формула про скорость модели."""

    def test_percentiles_return_an_observed_call(self) -> None:
        # Ближайший ранг, а не интерполяция: на выборке из единиц значений
        # интерполяция рисовала бы точность, которой в замерах нет.
        values = [10.0, 20.0, 30.0, 40.0]
        self.assertEqual(CallTimings.percentile(values, 0.5), 20.0)
        self.assertEqual(CallTimings.percentile(values, 0.9), 40.0)
        for share in (0.5, 0.9):
            self.assertIn(CallTimings.percentile(values, share), values)

    def test_percentile_of_a_single_call(self) -> None:
        self.assertEqual(CallTimings.percentile([7.0], 0.5), 7.0)
        self.assertEqual(CallTimings.percentile([7.0], 0.9), 7.0)

    def test_percentile_of_nothing_does_not_explode(self) -> None:
        # Джоба может упасть до первого вызова модели (нет источников), и
        # строка статистики всё равно пишется.
        self.assertEqual(CallTimings.percentile([], 0.5), 0.0)

    def test_summary_reads_at_a_glance(self) -> None:
        timings = CallTimings()
        timings.record(stage="план презентации", attempt=1, seconds=69.9)
        for index in range(3):
            timings.record(
                stage=f"слайд {index + 1} из 3", attempt=1, seconds=32.3
            )
        timings.record(stage="слайд 3 из 3", attempt=2, seconds=41.1)

        summary = timings.summary()
        self.assertIn("вызовов модели 5", summary)
        self.assertIn("план 1", summary)
        self.assertIn("слайды 4", summary)
        self.assertIn("повторных 1", summary)
        self.assertIn("p50 32.3с", summary)
        self.assertIn("p90 69.9с", summary)
        self.assertIn("max 69.9с", summary)

    def test_summary_of_a_job_that_never_called_the_model(self) -> None:
        # Стадия рендера названа и здесь: у джобы, не дошедшей ни до модели, ни
        # до печати, обе половины строки обязаны сказать «ничего не было», а не
        # молча исчезнуть. Пустое место читалось бы как «прошло мгновенно».
        self.assertEqual(
            CallTimings().summary(), "вызовов модели 0; рендер не выполнялся"
        )

    def test_summary_names_the_render_stage_separately(self) -> None:
        """Длительность печати — своё поле, а не ещё одно значение в замерах.

        Природа у них разная: вызовы модели измеряют скорость чат-модели (её
        меняют из админ-панели), рендер — скорость внешнего браузера и диска.
        Смешав их в одном p50, мы получили бы число, не описывающее ни то, ни
        другое, и перестали бы замечать, что печать колоды поехала.
        """
        timings = CallTimings()
        timings.record(stage="план презентации", attempt=1, seconds=69.9)
        timings.render_seconds = 4.25

        summary = timings.summary()

        self.assertIn("рендер 4.2с", summary)
        # И не попала в статистику вызовов модели: их по-прежнему один.
        self.assertIn("вызовов модели 1", summary)
        self.assertIn("max 69.9с", summary)

    def test_unclassified_failures_show_up_next_to_the_retries(self) -> None:
        """Счётчик неразобранных причин — в той же строке, что и повторы.

        Отдельного места для него нет намеренно: смотрят в лог джобы один раз и
        одной строкой, а «повторных 0, неклассифицированных 3» читается сразу —
        медленно не было, просто классификатор не понял, что произошло, и
        повтора не случилось ни разу.
        """
        timings = CallTimings()
        timings.record(stage="план презентации", attempt=1, seconds=12.0)
        timings.record_unclassified()
        timings.record_unclassified()

        summary = timings.summary()
        self.assertIn("неклассифицированных 2", summary)
        # Рядом с повторами, а не в конце строки после замеров: обе величины
        # отвечают на вопрос «что было с вызовами», а не «сколько они шли».
        self.assertLess(summary.index("повторных"), summary.index("неклассиф"))
        self.assertLess(summary.index("неклассиф"), summary.index("p50"))

    def test_layout_mismatches_show_up_next_to_the_retries_too(self) -> None:
        """Слайдов, вернувшихся не в назначенной планом раскладке.

        Раскладку выбирает план — он один видит колоду целиком, — а пишется
        слайд по найденным под секцию чанкам, и второй стороны сравнения в них
        может не оказаться. Отдать тогда список честнее, и слайд за это не
        отвергается. Но без счётчика «материал не дал» неотличимо от «промпт не
        работает» — а это ровно тот вопрос, ради которого выбор и переехал в
        план.
        """
        timings = CallTimings()
        timings.record(stage="план презентации", attempt=1, seconds=12.0)
        timings.record(stage="слайд 1 из 2", attempt=1, seconds=30.0)
        timings.record(stage="слайд 2 из 2", attempt=1, seconds=30.0)
        timings.record_layout_mismatch()
        timings.record_layout_mismatch()

        summary = timings.summary()
        self.assertIn("раскладок не по плану 2", summary)
        self.assertLess(summary.index("повторных"), summary.index("не по плану"))
        self.assertLess(summary.index("не по плану"), summary.index("p50"))

    def test_a_zero_counter_does_not_litter_the_line(self) -> None:
        """Ноль не печатается — иначе счётчик перестаёт быть заметным.

        Величина заведена ради РЕДКОГО события: постоянный «0» в каждой строке
        журнала перестаёт читаться на второй неделе, и появление там единицы не
        заметит никто. Правило общее для обоих счётчиков: колода, целиком
        исполнившая план, — норма, и писать о ней в каждой строке незачем.
        """
        timings = CallTimings()
        timings.record(stage="план презентации", attempt=1, seconds=12.0)
        self.assertNotIn("неклассиф", timings.summary())
        self.assertNotIn("не по плану", timings.summary())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
