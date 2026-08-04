"""Бюджет времени презентации: потолок вызова и выведенный из него потолок джобы.

Ни базы, ни Ollama здесь нет — проверяется арифметика и её свойства. Смысл
файла в том, что потолок джобы ПЕРЕСТАЛ быть константой: раньше здесь стояло
PRESENTATION_JOB_TIMEOUT = 600, посчитанное для другой модели и другого
корпуса, и молчало ровно до того дня, когда модель сменили из админ-панели.
Теперь калибруется одна величина — LLM_CALL_TIMEOUT на ОДИН вызов, — а потолок
джобы считается из числа слайдов заказа.

Числа заказа в проверках не выписаны: они спрашиваются у констант. Тест на
конкретные 2700 и 8700 краснел бы при честной перекалибровке потолка вызова и
пропускал бы то единственное, ради чего заведён, — потерю связи между
потолком джобы и размером заказа.
"""

from __future__ import annotations

import unittest

from app.modules.presentations.constants import (
    LLM_CALL_ATTEMPTS,
    LLM_CALL_TIMEOUT,
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
        """(1 план + секции) × попытки × потолок вызова + буфер рендера."""
        for slide_count in (SLIDE_COUNT_MIN, SLIDE_COUNT_MAX):
            with self.subTest(slide_count=slide_count):
                calls = 1 + content_section_count(slide_count)
                expected = (
                    calls * LLM_CALL_ATTEMPTS + 1
                ) * LLM_CALL_TIMEOUT
                self.assertEqual(presentation_job_timeout(slide_count), expected)

    def test_the_ceiling_covers_a_run_where_every_stage_hits_its_budget(self) -> None:
        """Заказ, у которого КАЖДАЯ стадия уложилась впритык, не снимается.

        Стадии перечисляются по одной, а не подставляются в формулу: так
        проверка ловит забытый буфер рендера и сдвиг на единицу в числе секций,
        то есть ровно те ошибки, из-за которых потолок джобы снимал бы заказ,
        где ни один вызов сам по себе не нарушил свой бюджет. Это и была бы
        ошибка в короткую сторону — та, ради устранения которой таймаут
        опустили на уровень вызова.
        """
        for slide_count in range(SLIDE_COUNT_MIN, SLIDE_COUNT_MAX + 1):
            with self.subTest(slide_count=slide_count):
                stages = ["план"]
                stages += [
                    f"секция {index + 1}"
                    for index in range(content_section_count(slide_count))
                ]
                worst_run = sum(
                    LLM_CALL_ATTEMPTS * LLM_CALL_TIMEOUT for _ in stages
                )
                worst_run += LLM_CALL_TIMEOUT  # рендер
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
        self.assertEqual(
            without_render, calls * LLM_CALL_ATTEMPTS * LLM_CALL_TIMEOUT
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
        self.assertEqual(CallTimings().summary(), "вызовов модели 0")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
