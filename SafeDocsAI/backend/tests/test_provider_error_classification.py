"""Причина отказа провайдера разбирается по ТИПУ, а не по имени класса.

Ни Ollama, ни сети здесь нет: проверяется классификатор
(ModelManager._wrap_provider_error) и то, чем он отвечает на исключение, вида
которого раньше не видел.

Зачем файл. Классификация по именам классов была сознательным временным
решением, и его слабость записали сразу: новый вид исключения молча падает в
UNKNOWN, а UNKNOWN означает «не повторять». То есть цена ошибки классификатора
— не строчка в логе, а умерший с первой попытки заказ. httpx заводит новые
подклассы таймаутов и транспортных ошибок регулярно, и каждый такой подкласс
до перехода на isinstance был бы отказом без повтора.

Проверки написаны так, чтобы краснеть именно на возврате к именам: исключения
здесь наследуются от настоящих типов httpx, но названы так, как в наборах имён
не значится. По имени такой класс неотличим от экзотики, по типу — обычный
таймаут.

Отдельно закреплён фолбэк: имена НЕ удалены и обязаны продолжать работать для
исключений из библиотек, которых в этом модуле нет и быть не должно.
"""

from __future__ import annotations

import unittest
import urllib.error

import httpx
import ollama

from app.core.exceptions import ExternalServiceError, ExternalServiceErrorKind
from app.modules.rag.model_manager import ModelManager


def wrap(exc: Exception) -> ExternalServiceError:
    return ModelManager._wrap_provider_error("Ollama", exc)


class ПерегруженныйПул(httpx.PoolTimeout):
    """Наследник настоящего таймаута httpx с именем, которого нет в наборах."""


class СвязьНеВстала(httpx.ConnectError):
    """То же для транспортной ошибки."""


class ClassificationFollowsTheTypeTests(unittest.TestCase):
    def test_a_renamed_timeout_is_still_a_timeout(self) -> None:
        """Класс с незнакомым именем, но настоящего рода.

        До перехода на isinstance это был UNKNOWN: имени нет в наборе — значит,
        причина не разобрана, значит, повтора нет. Ровно так выглядел бы любой
        новый подкласс httpx.TimeoutException.
        """
        error = wrap(ПерегруженныйПул("pool is exhausted"))
        self.assertEqual(error.kind, ExternalServiceErrorKind.TIMEOUT)
        self.assertTrue(error.is_transient)

    def test_a_renamed_transport_error_is_still_unavailability(self) -> None:
        error = wrap(СвязьНеВстала("connection refused"))
        self.assertEqual(error.kind, ExternalServiceErrorKind.UNAVAILABLE)

    def test_the_whole_httpx_timeout_branch_is_covered_at_once(self) -> None:
        """Ветка иерархии, а не перечень имён.

        Смысл isinstance в том и есть: одна проверка закрывает и то, что httpx
        заведёт завтра.
        """
        for exc in (
            httpx.ReadTimeout("read"),
            httpx.WriteTimeout("write"),
            httpx.PoolTimeout("pool"),
        ):
            with self.subTest(exc=type(exc).__name__):
                self.assertEqual(
                    wrap(exc).kind, ExternalServiceErrorKind.TIMEOUT
                )

    def test_a_connect_timeout_stays_unavailability(self) -> None:
        """Порядок групп значим: ConnectTimeout — наследник TimeoutException.

        Соединение не установилось вовсе, отвечать было некому. Считать это
        «модель не успела» значило бы советовать пользователю колоду покороче
        при выключенном сервисе — и, с новой политикой повтора, ещё и ждать
        тридцать секунд перед второй попыткой достучаться в никуда.
        """
        self.assertEqual(
            wrap(httpx.ConnectTimeout("timed out")).kind,
            ExternalServiceErrorKind.UNAVAILABLE,
        )

    def test_the_urllib_fallback_path_is_classified_too(self) -> None:
        # Запасной путь embed() ходит через urllib, а не через httpx: его
        # отказы приходят сюда своим типом.
        self.assertEqual(
            wrap(urllib.error.URLError("connection refused")).kind,
            ExternalServiceErrorKind.UNAVAILABLE,
        )

    def test_a_builtin_connection_error_is_unavailability(self) -> None:
        self.assertEqual(
            wrap(ConnectionRefusedError("refused")).kind,
            ExternalServiceErrorKind.UNAVAILABLE,
        )

    def test_the_cause_is_examined_and_not_only_the_wrapper(self) -> None:
        """ollama оборачивает httpx: снаружи ResponseError, причина — в __cause__.

        Без разбора цепочки такой отказ уходил бы в REQUEST_REJECTED по статусу
        обёртки, то есть «повторять бессмысленно» — при том что повторить его
        как раз и стоило.
        """
        try:
            try:
                raise ПерегруженныйПул("pool is exhausted")
            except ПерегруженныйПул as cause:
                raise ollama.ResponseError("upstream error", 502) from cause
        except ollama.ResponseError as exc:
            error = wrap(exc)

        self.assertEqual(error.kind, ExternalServiceErrorKind.TIMEOUT)

    def test_the_status_code_branch_survived_the_move_to_types(self) -> None:
        """Сеть смотрится первой, статус — потом; оба ответа прежние."""
        self.assertEqual(
            wrap(ollama.ResponseError("boom", 500)).kind,
            ExternalServiceErrorKind.SERVER_ERROR,
        )
        self.assertEqual(
            wrap(ollama.ResponseError("model not found", 404)).kind,
            ExternalServiceErrorKind.REQUEST_REJECTED,
        )
        # -1 у ollama.ResponseError означает «сервер статуса не назвал»:
        # принимать его за код ответа нельзя.
        self.assertEqual(
            wrap(ollama.ResponseError("no status at all")).kind,
            ExternalServiceErrorKind.UNKNOWN,
        )


class NamesRemainAsTheLastResortTests(unittest.TestCase):
    """Фолбэк по именам не удалён — и это решение, а не забытый код.

    Он закрывает то, чего isinstance закрыть не может: исключение из
    библиотеки, которой в model_manager нет и быть не должно (urllib3,
    requests у соседнего клиента, будущая замена ollama). Импортировать её ради
    одной проверки — завести зависимость на пустом месте; не проверять вовсе —
    отдать её в UNKNOWN, где повтора не бывает.
    """

    def test_a_foreign_timeout_is_recognised_by_its_name(self) -> None:
        foreign = type("ReadTimeout", (Exception,), {})
        self.assertEqual(
            wrap(foreign("timed out")).kind, ExternalServiceErrorKind.TIMEOUT
        )

    def test_a_foreign_connection_error_is_recognised_by_its_name(self) -> None:
        foreign = type("ConnectError", (Exception,), {})
        self.assertEqual(
            wrap(foreign("refused")).kind, ExternalServiceErrorKind.UNAVAILABLE
        )


class UnknownIsLoudTests(unittest.TestCase):
    """Неразобранная причина обязана кричать, а не молчать.

    Она означает не «редкий отказ», а слепое пятно классификатора: повтора
    такому отказу не положено, то есть заказ умирает с первой попытки. WARNING
    здесь мало — такие строки в журнале живут рядом с рабочими предупреждениями
    ретривала и не читаются.
    """

    def test_an_unclassified_failure_is_logged_at_error_level(self) -> None:
        class СовсемНовыйОтказ(Exception):
            pass

        with self.assertLogs("app.modules.rag.model_manager", level="ERROR") as logs:
            error = wrap(СовсемНовыйОтказ("что-то новое"))

        self.assertEqual(error.kind, ExternalServiceErrorKind.UNKNOWN)
        self.assertFalse(error.is_transient)
        self.assertEqual(len(logs.records), 1)
        self.assertEqual(logs.records[0].levelname, "ERROR")

    def test_the_log_names_the_full_class_path(self) -> None:
        """Полный путь, а не имя класса.

        «TimeoutError» в журнале не отвечает на вопрос, чей он — встроенный,
        httpx'овый или чужого клиента, — а разбирают такую строку именно по
        имени модуля: по нему видно, какую библиотеку читать и какой тип
        добавлять в набор.
        """
        class СовсемНовыйОтказ(Exception):
            pass

        with self.assertLogs("app.modules.rag.model_manager", level="ERROR") as logs:
            wrap(СовсемНовыйОтказ("что-то новое"))

        line = logs.output[0]
        self.assertIn(f"{__name__}.", line)
        self.assertIn("СовсемНовыйОтказ", line)

    def test_the_cause_is_named_too(self) -> None:
        """Виновник чаще лежит в __cause__, а не в обёртке.

        Строка, называющая только внешний класс, отправляет читателя изучать
        обёртку, которая ни при чём.
        """
        class ЧужаяОбёртка(Exception):
            pass

        class НастоящийВиновник(Exception):
            pass

        try:
            try:
                raise НастоящийВиновник("вот он")
            except НастоящийВиновник as cause:
                raise ЧужаяОбёртка("обёртка") from cause
        except ЧужаяОбёртка as exc:
            with self.assertLogs(
                "app.modules.rag.model_manager", level="ERROR"
            ) as logs:
                wrap(exc)

        line = logs.output[0]
        self.assertIn("ЧужаяОбёртка", line)
        self.assertIn("НастоящийВиновник", line)

    def test_a_classified_failure_stays_quiet(self) -> None:
        """Разобранная причина ERROR'а не пишет — иначе он перестанет значить.

        Таймаут и недоступность — рабочие события с готовым ответом
        пользователю; поднимать по ним уровень значило бы утопить в них
        единственную строку, ради которой уровень и задран.
        """
        with self.assertNoLogs("app.modules.rag.model_manager", level="ERROR"):
            wrap(httpx.ReadTimeout("timed out"))
            wrap(httpx.ConnectError("refused"))
            wrap(ollama.ResponseError("model not found", 404))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
