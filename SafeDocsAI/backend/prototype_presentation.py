"""Синхронный прогон БОЕВОГО пайплайна презентаций с печатью таймингов.

Скрипт не содержит своей копии пайплайна и не должен её получить. Он заводит
одну строку заказа, зовёт generate_presentation — ту же функцию, которую в бою
зовёт воркер, — и печатает то, что она уже насчитала (CallTimings). Тем самым
всё, что влияет на замер, берётся из боевого кода: отбор чанков, промпты,
повторная попытка, wait_for вокруг каждого вызова модели (LLM_CALL_TIMEOUT) и
потолок джобы (presentation_job_timeout).

Прошлая версия была ФОРКОМ пайплайна: своим отбором чанков, своими промптами,
своими константами. Форк, показаниями которого принимают решения, хуже мёртвого
кода — по нему калибровали таймауты, а показывал он дефекты, которых в продукте
нет, и молчал бы о тех, что есть. Поэтому здесь нет и не должно появиться ни
одной строки, повторяющей пайплайн: всё, что понадобится измерить, добавляется
в service.py и вызывается отсюда, а не переписывается тут заново.

Отличий от боя ровно два, и оба намеренные:

  * ОЧЕРЕДИ НЕТ. Строка заводится сразу в 'generating' и ни секунды не бывает
    'queued', поэтому её не может подхватить воркер запущенного рядом сервера.
  * ВОРКЕРА НЕТ. Его пост-обработку скрипт повторяет только там, где без неё
    в базе останется неправда: отказ записывается в строку (иначе она навсегда
    'generating', а requeue_stuck при следующем старте сервера отправит её в
    настоящую очередь и владелец блокнота получит колоду, которую не
    заказывал). Строку в журнал блокнота скрипт не пишет: прогон
    измерительный, а журнал отвечает на вопрос «что здесь делал пользователь».

Скрипт ПИШЕТ В БАЗУ, и иначе быть не может: боевому пайплайну нужна строка
presentation, по которой он двигает прогресс и статус. По умолчанию созданная
строка и напечатанный PDF удаляются в конце прогона — заказ, которого
пользователь не делал, не должен появляться у него в интерфейсе. Флаг --keep
оставляет и строку, и файл: колода видна в списке блокнота и скачивается как
любая другая.

Запуск (переменные — те же, что у рабочего процесса; OLLAMA_MODEL_EMBEDDING
обязателен, иначе имя коллекции ChromaDB выведется другим и поиск уйдёт в
пустоту):

    POSTGRES_USER=andozai_user POSTGRES_PASSWORD=... POSTGRES_SERVER=localhost \
    POSTGRES_PORT=5432 POSTGRES_DB=andozai_db \
    OLLAMA_MODEL_EMBEDDING=qwen3-embedding:8b SECRET_KEY=... \
    ./venv/bin/python prototype_presentation.py --notebook-id 16 --language ru \
        --slide-count 10 --description "Обзор налоговых льгот"
"""

import argparse
import asyncio
import contextlib
import json
import logging
import os
import sys
from time import perf_counter

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from app.core.database import session_context
from app.modules.presentations.constants import (
    DESCRIPTION_MAX,
    LLM_CALL_TIMEOUT,
    SLIDE_COUNT_DEFAULT,
    STATUS_GENERATING,
    normalize_language,
    presentation_job_timeout,
)
from app.modules.presentations.service import (
    CallTimings,
    GenerationResult,
    PresentationsService,
    error_code_for,
    error_text_for,
    generate_presentation,
)
from app.modules.presentations.templates import template_registry
from app.services.runtime_settings_service import RuntimeSettingsService
from app.shared.models import Notebook, Presentation

# Как часто перечитывать прогресс заказа. Прогресс двигает сам пайплайн после
# каждой секции, и опрос строки — единственный способ видеть ход генерации, не
# расставляя по скрипту собственных отметок: свои отметки существовали бы
# только здесь и разошлись бы с тем, что видит пользователь.
PROGRESS_POLL_SECONDS = 5.0

# Прошлая версия скрипта принимала «tg» (код ISO) и переводила его внутри.
# В контракте проекта таджикский обозначен одним кодом — «tj», — но запуски с
# --language tg уже написаны в чужих шпаргалках, и ломать их незачем: перевод
# живёт ровно на границе CLI и дальше не проходит.
CLI_LANGUAGE_ALIASES = {"tg": "tj"}


def cli_language(value: str) -> str:
    try:
        return normalize_language(CLI_LANGUAGE_ALIASES.get(value.strip().lower(), value))
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


def cli_description(value: str) -> str:
    """Описание заказа, подрезанное по тому же пределу, что и приём по HTTP.

    Единственная проверка входа, которой у пайплайна нет своей: границы числа
    слайдов он проверяет сам, а длину описания — нет, её отсекает форма заказа.
    Описание длиннее бюджета не отказ бы дало, а тихо неверный замер: промпт
    такого размера в бою не собирается, потому что до пайплайна не доезжает.
    """
    trimmed = value.strip()
    if len(trimmed) > DESCRIPTION_MAX:
        raise SystemExit(
            f"--description длиннее {DESCRIPTION_MAX} знаков — такой заказ "
            "приём по HTTP отвергает (presentation.description_too_long)"
        )
    return trimmed


def resolve_template_key(value: str) -> str:
    """Ключ шаблона: из реестра, а не из своего списка.

    Умолчание — первый шаблон реестра, и неизвестный ключ отвергается так же,
    как его отвергает приём заказа по HTTP. Проверка стоит ЗДЕСЬ, до первого
    вызова модели: рендер неизвестный ключ не переживает (темы по умолчанию у
    HTML-печати нет — без каталога шаблона нет ни вёрстки, ни шрифтов), и без
    этой проверки замер умирал бы в самом конце, потратив все минуты генерации.
    Ровно так и вёл себя ключ "default" прошлой версии скрипта, которого в
    реестре нет.
    """
    known = template_registry.list()
    if not known:
        raise SystemExit(
            "реестр шаблонов пуст (backend/templates/presentations): "
            "рендерить нечем"
        )
    key = value.strip() or known[0].key
    if template_registry.get(key) is None:
        raise SystemExit(
            f"шаблон {key!r} не найден; доступны: "
            + ", ".join(info.key for info in known)
        )
    return key


def chat_model_name() -> str:
    """Имя чат-модели — только для шапки отчёта.

    Модель пайплайн выбирает сам, тем же выражением; общего доступа к «какой
    моделью пойдёт генерация» в коде нет — выбор зашит в generate_presentation.
    Разъехавшись, эта строка испортит подпись под замером, но не сам замер.
    Печатать замер без имени модели нельзя вовсе: LLM_CALL_TIMEOUT калибруется
    под конкретную модель, и время вызовов без её имени ни к чему не привязано.
    """
    settings = RuntimeSettingsService.get_settings()
    return str(settings.get("chat_model") or settings.get("model") or "<не задана>")


async def create_job(
    args: argparse.Namespace, *, language: str, template_key: str, description: str
) -> tuple[Presentation, str]:
    """Строка заказа СРАЗУ в 'generating' — мимо очереди.

    PresentationsService.create() тут не годится: она заводит строку в 'queued'
    и будит воркера, то есть отдаёт прогон ровно той очереди, мимо которой
    скрипт и затевался. Строка, никогда не бывавшая 'queued', не может быть
    захвачена ни этим процессом, ни сервером, работающим рядом, — гонки нет по
    построению, а не по везению.

    Владелец берётся у блокнота: скрипт ходит мимо HTTP, а пайплайн выводит из
    owner_id область поиска. С чужим владельцем замер считался бы по другому
    набору документов.
    """
    async with session_context() as session:
        notebook = await session.get(Notebook, args.notebook_id)
        if notebook is None:
            raise SystemExit(f"notebook id={args.notebook_id} not found")
        presentation = Presentation(
            notebook_id=notebook.id,
            owner_id=notebook.owner_id,
            template_key=template_key,
            language=language,
            slide_count=args.slide_count,
            description=description or None,
            status=STATUS_GENERATING,
            progress=0,
        )
        session.add(presentation)
        await session.commit()
        await session.refresh(presentation)
        return presentation, notebook.name or ""


async def load_job(presentation_id: int) -> Presentation | None:
    async with session_context() as session:
        return await session.get(Presentation, presentation_id)


async def watch_progress(presentation_id: int) -> None:
    """Печатать прогресс, который пайплайн публикует для интерфейса.

    Ровно то же, что делает клиент: опрос строки. Прогон идёт минутами, и без
    этого скрипт молчит от старта до итога — но собственных отметок по стадиям
    скрипт не расставляет, иначе они начнут расходиться с тем, что показывают
    пользователю.
    """
    started = perf_counter()
    last = -1
    while True:
        await asyncio.sleep(PROGRESS_POLL_SECONDS)
        try:
            presentation = await load_job(presentation_id)
        except Exception as exc:
            # Наблюдение не имеет права уронить прогон, ради которого всё
            # затевалось: пропущенный опрос — это пропущенная строка вывода.
            print(f"[прогресс] опрос не удался: {exc}")
            continue
        progress = -1 if presentation is None else presentation.progress
        if progress != last:
            last = progress
            print(f"[прогресс] {progress:3d}%  на {perf_counter() - started:7.1f}с")


def print_timings(
    *,
    timings: CallTimings,
    wall_seconds: float,
    ceiling: float,
) -> None:
    """Итог по времени.

    Первая строка — та же самая, что воркер пишет в лог каждой боевой джобы:
    считать перцентили здесь заново значило бы снова завести вторую арифметику
    рядом с боевой и разойтись с ней ровно в том месте, ради которого скрипт и
    переписывали.
    """
    print("=" * 72)
    print("ТАЙМИНГИ")
    print(f"  {timings.summary()}")
    if timings.durations:
        print("  вызовы по порядку : "
              + ", ".join(f"{seconds:.1f}с" for seconds in timings.durations))
    print(f"  потолок вызова    : {LLM_CALL_TIMEOUT}с")
    print(f"  потолок джобы     : {ceiling:.0f}с")
    print(f"  время стены       : {wall_seconds:.1f}с")
    # Разница между стеной и суммой вызовов — всё, что моделью не является:
    # ретривал, обращения к базе, рендер. Отдельного замера у них нет, а
    # заводить его здесь означало бы мерить то, чего боевая джоба не мерит.
    print(f"  вне вызовов модели: {wall_seconds - sum(timings.durations):.1f}с "
          f"(ретривал, база, рендер)")
    print("=" * 72)


def print_result(
    result: GenerationResult,
    presentation: Presentation | None,
    failure: tuple[str, str] | None,
) -> None:
    if failure is not None:
        print(f"ОТКАЗ: {failure[0]}: {failure[1]}")
        return
    print(f"слайдов собрано : {result.slides}")
    print(f"профиль         : {result.domain_profile}")
    for source in result.sources:
        pages = sorted(set(source.pages))
        print(f"  источник {source.source_id}: {source.name}"
              + (f", стр. {pages}" if pages else ""))
    if presentation is not None:
        print(f"файл            : {presentation.file_path} "
              f"({presentation.file_size} байт)")


async def drop_job(presentation: Presentation | None) -> None:
    """Убрать за собой строку и файл.

    Порядок тот же, что у боевого удаления заказа (delete_presentation в
    app/api/endpoints/presentations.py): сначала строка и commit, потом файл.
    Осиротевший файл невиден и безвреден, строка с путём в никуда — нет.
    """
    if presentation is None:
        return
    file_path = presentation.file_path
    async with session_context() as session:
        row = await session.get(Presentation, presentation.id)
        if row is not None:
            await session.delete(row)
            await session.commit()
    if file_path:
        try:
            os.remove(file_path)
        except FileNotFoundError:
            pass
        except OSError as exc:
            print(f"файл {file_path} остался на диске: {exc}")
    print(f"строка #{presentation.id} и файл удалены (--keep оставляет их)")


def write_metrics_json(
    path: str,
    *,
    presentation: Presentation | None,
    timings: CallTimings,
    wall_seconds: float,
    ceiling: float,
    result: GenerationResult,
    failure: tuple[str, str] | None,
) -> None:
    payload = {
        "presentation_id": presentation.id if presentation else None,
        "notebook_id": presentation.notebook_id if presentation else None,
        "language": presentation.language if presentation else None,
        "template_key": presentation.template_key if presentation else None,
        "slide_count": presentation.slide_count if presentation else None,
        "model": chat_model_name(),
        "llm_call_timeout": LLM_CALL_TIMEOUT,
        "job_timeout": ceiling,
        "wall_seconds": round(wall_seconds, 3),
        "summary": timings.summary(),
        "call_seconds": [round(seconds, 3) for seconds in timings.durations],
        "plan_calls": timings.plan_calls,
        "slide_calls": timings.slide_calls,
        "retries": timings.retries,
        "slides": result.slides,
        "sources": [
            {
                "doc_id": source.source_id,
                "doc_name": source.name,
                "pages": sorted(set(source.pages)),
            }
            for source in result.sources
        ],
        "error_code": failure[0] if failure else None,
        "error_text": failure[1] if failure else None,
    }
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
    print(f"замеры json: {path}")


async def run(args: argparse.Namespace) -> int:
    language = cli_language(args.language)
    template_key = resolve_template_key(args.template_key)

    presentation, notebook_name = await create_job(
        args,
        language=language,
        template_key=template_key,
        description=cli_description(args.description),
    )
    presentation_id = presentation.id
    # Потолок джобы берётся той же функцией, что у воркера, и накрывает вызов
    # так же: без него скрипт мерил бы пайплайн без верхней границы, то есть не
    # тот, что работает у пользователя. Потолки ОТДЕЛЬНЫХ вызовов ставит сам
    # пайплайн, здесь их дублировать нечем и незачем.
    ceiling = presentation_job_timeout(presentation.slide_count)
    print("=" * 72)
    print(f"заказ      : #{presentation_id} (создан этим прогоном, мимо очереди)")
    print(f"блокнот    : {presentation.notebook_id} ({notebook_name})")
    print(f"владелец   : {presentation.owner_id}")
    print(f"язык       : {presentation.language}")
    print(f"шаблон     : {presentation.template_key}")
    print(f"слайдов    : {presentation.slide_count}")
    print(f"модель     : {chat_model_name()}")
    print(f"embedding  : {RuntimeSettingsService.embedding_model() or '<не задана>'}")
    print(f"потолки    : вызов {LLM_CALL_TIMEOUT}с, джоба {ceiling:.0f}с")
    print("=" * 72)

    timings = CallTimings()
    result = GenerationResult()
    failure: tuple[str, str] | None = None
    watcher = asyncio.create_task(watch_progress(presentation_id))
    started = perf_counter()
    try:
        result = await asyncio.wait_for(
            generate_presentation(presentation_id, timings=timings), timeout=ceiling
        )
    except BaseException as exc:
        # BaseException, а не Exception: снятие прогона (Ctrl+C) приходит сюда
        # CancelledError. Строка, оставшаяся в 'generating', — не просто
        # неправда в интерфейсе. requeue_stuck при следующем старте сервера
        # вернёт её в настоящую очередь, и владелец блокнота получит колоду,
        # которую не заказывал. Поэтому исход фиксируется при ЛЮБОМ отказе, а
        # коды и тексты берутся у воркера — те же, что увидел бы пользователь.
        failure = (error_code_for(exc), error_text_for(exc))
    finally:
        wall_seconds = perf_counter() - started
        watcher.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await watcher

    if failure is not None:
        async with session_context() as session:
            await PresentationsService.mark_error(
                session, presentation_id, error_code=failure[0], error_text=failure[1]
            )

    finished = await load_job(presentation_id)
    print_timings(timings=timings, wall_seconds=wall_seconds, ceiling=ceiling)
    print_result(result, finished, failure)

    if args.metrics_json:
        write_metrics_json(
            args.metrics_json,
            presentation=finished or presentation,
            timings=timings,
            wall_seconds=wall_seconds,
            ceiling=ceiling,
            result=result,
            failure=failure,
        )

    if args.keep:
        print(f"строка #{presentation_id} осталась в базе и видна в интерфейсе блокнота")
    else:
        await drop_job(finished or presentation)
    return 0 if failure is None else 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--notebook-id", type=int, required=True)
    parser.add_argument(
        "--template-key",
        default="",
        help="ключ шаблона оформления; по умолчанию первый из реестра",
    )
    parser.add_argument("--language", default="ru")
    parser.add_argument("--slide-count", type=int, default=SLIDE_COUNT_DEFAULT)
    parser.add_argument("--description", default="")
    parser.add_argument(
        "--keep",
        action="store_true",
        help="оставить заказ и файл в базе (по умолчанию прогон убирает за собой)",
    )
    parser.add_argument(
        "--metrics-json",
        default="",
        help="куда сложить замеры прогона (необязательно)",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    # INFO, а не WARNING: о ходе стадий рассказывает лог самого пайплайна —
    # отвергнутые валидатором ответы, вызовы, снятые по таймауту, итоговая
    # строка про готовый файл. Своего параллельного рассказа у скрипта нет и
    # быть не должно: он повторял бы боевой лог и расходился бы с ним.
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    sys.exit(asyncio.run(run(parse_args())))
