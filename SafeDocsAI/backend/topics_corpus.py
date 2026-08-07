#!/usr/bin/env python3
"""Сборка таджикского корпуса с рубриками из собранных jsonl.

Зачем нужен ещё один корпус. Модель, которая сейчас подписывает документы,
обучена на task1_multilingual_dataset: 840 синтетических корпоративных текстов
(кадры, маркетинг, логистика) и 1438 выдержек Википедии. Боевые документы
системы — налоговый кодекс и паёмы президента. Тем «Законодательство»,
«Послания», «Энергетика» в модели попросту нет, поэтому каждый реальный
документ приписывается к ближайшему из двадцати чужих центров, а три паёма
разъезжаются по трём разным темам.

Собранный корпус — тот же жанр и тот же язык, что у боевых документов, и он уже
размечен людьми: у каждой статьи есть раздел сайта. Сведение разделов к списку
рубрик живёт в app/modules/topics/pipeline/rubrics.py и написано руками — это
решение о том, что считать одной темой, и оно должно быть видимым.

ЧТО ЗДЕСЬ ВАЖНО НЕ ПЕРЕПУТАТЬ.

*Размеченная часть и общий котёл — разные вещи.* Четыре сайта дают осмысленные
рубрики (khovar, jumhuriyat, ozodi, Википедия). У sputnik разделы — это
поисковые страницы вроде «Самые свежие новости России и мира сегодня онлайн с
видео и фото»; темой это не является. Поэтому sputnik попадает в корпус, но не
в измерение: его векторы нужны главным осям (там разметка не требуется вовсе), а
его 659 русских документов — единственный способ, чтобы первая главная
компонента действительно поймала ось языка и её можно было снять.

*Разбиения только для размеченных.* train/validation/test содержат лишь
документы с рубрикой. Всё остальное лежит в full со `split = "pool"` и в
метриках не участвует.

*Неизвестный раздел не выбрасывается молча.* Раздел, которого нет ни в таблице
рубрик, ни в списке сознательно проигнорированных, попадает в отчёт поимённо и
требует решения человека. Иначе таблица покрытия соврала бы.

Примеры:

    ./venv/bin/python topics_corpus.py                  # собрать и записать
    ./venv/bin/python topics_corpus.py --dry-run        # только отчёт
    ./venv/bin/python topics_corpus.py --out data/topics_tj
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

from app.modules.topics.pipeline.dataset import FULL_FILE, SPLIT_FILES  # noqa: E402
from app.modules.topics.pipeline.rubrics import (  # noqa: E402
    RUBRICS,
    UNLABELLED,
    normalize_section,
    rubric_names,
)

BACKEND_ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE_DIR = BACKEND_ROOT / "data" / "topics_dataset"
DEFAULT_OUT_DIR = BACKEND_ROOT / "data" / "topics_tj"

# Сайты с РЕДАКЦИОННЫМИ рубриками — теми, что поставил человек-редактор.
# Список закрытый и явный: источник, у которого рубрики окажутся мусорными,
# должен добавляться сюда осознанно, а не просто потому, что кто-то положил
# рядом файл.
#
# У остальных источников рубрику тоже ищем — часть их разделов осмысленна и
# сведена вручную (CURATED_TO_RUBRIC в rubrics.py). Разница между ними в другом:
# неизвестный раздел РЕДАКЦИОННОГО источника требует решения человека и
# называется в отчёте поимённо, а неизвестная страница-подборка у sputnik — это
# «Прогноз погоды» и «Курсы валют», и требовать по ним решения значит завалить
# отчёт шумом.
EDITORIAL_SOURCES = frozenset({"khovar.tj", "jumhuriyat.tj", "ozodi.org", "tg.wikipedia.org"})

# Ниже этого числа документов рубрика в измерении не участвует: ARI по классу
# из десятка документов — это шум, выданный за результат. Документы при этом
# остаются в корпусе.
#
# Двадцать пять, а не тридцать. Порог сдвинут не ради того, чтобы протащить
# слабые рубрики: двадцать пять — это то число, которое стояло в заданиях
# сборщикам («не меньше 25 статей на предмет»), и держать в сборке другое
# значило бы отбрасывать ровно то, что заказано и собрано. Расхождение уже
# стоило рубрики «Медицина» с её двадцатью семью документами.
MIN_RUBRIC_SIZE = 25

# Короче этого документ — анонс или обрывок навигации, а не текст. Эмбеддинг
# такого куска описывает шаблон сайта, а не тему.
MIN_TEXT_CHARS = 200

SPLIT_SHARES = (("train", 0.70), ("validation", 0.15), ("test", 0.15))
RANDOM_STATE = 42

# Язык записывается кодом ISO 639-1, как в прежнем корпусе и в отчёте: там
# таджикский — "tg". В собранных файлах он помечен "tj" (так его называет сам
# продукт). Расхождение известное; сводим к одному написанию здесь, чтобы
# сравнение двух корпусов в отчёте не спотыкалось о два имени одного языка.
LANGUAGE_ALIASES = {"tj": "tg"}


def document_id(url: str, text: str) -> str:
    """Устойчивый идентификатор: хэш от url, а при его отсутствии — от текста.

    Хэш, а не порядковый номер: кэш эмбеддингов ключуется по id, и добавление
    новых документов в середину файла не должно сдвигать идентификаторы уже
    посчитанных. Иначе каждый досбор стоил бы полного пересчёта векторов.
    """
    material = url.strip() or text[:2000]
    return hashlib.sha1(material.encode("utf-8")).hexdigest()[:20]


# Разметка формул из Википедии. Она приезжает в текст как «{\displaystyle
# r=c{\sqrt {n}}}» вперемешку с вертикальными столбиками пробелов, и её нашлось
# в 57 статьях из 983.
#
# Почему это не мелочь: модель собрала из них кластер, чьи характерные слова —
# «displaystyle, mathbf». То есть тема «математические формулы», куда попали
# законы о связи. Для читателя это шум, для эмбеддинга — сильный и очень
# однородный сигнал, ровно такой, какой кластеризация и ловит охотнее всего.
MATH_MARKUP = re.compile(r"\\[a-zA-Z]+")
# Фигурные скобки убираются целиком, а не аккуратно по парам: разобрать
# вложенность формулы регулярным выражением нельзя, а в обычной прозе скобок
# этих не бывает — остаются только обломки разметки.
BRACES = re.compile(r"[{}]")
BLANK_RUNS = re.compile(r"\n{3,}")
SPACE_RUNS = re.compile(r"[ \t]{3,}")


def clean_text(text: str) -> str:
    """Текст без формульной разметки и без её пробельных развалов."""
    without = MATH_MARKUP.sub(" ", text)
    without = BRACES.sub(" ", without)
    without = SPACE_RUNS.sub(" ", without)
    without = BLANK_RUNS.sub("\n\n", without)
    return "\n".join(line.rstrip() for line in without.splitlines()).strip()


def read_records(source_dir: Path) -> list[dict]:
    files = sorted(source_dir.glob("*.jsonl"))
    if not files:
        raise SystemExit(f"в {source_dir} нет ни одного .jsonl")
    records = []
    for path in files:
        with open(path, encoding="utf-8") as handle:
            for number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise SystemExit(f"{path}:{number}: не разбирается как JSON") from exc
    return records


def build(records: list[dict]) -> tuple[list[dict], dict]:
    """Записи корпуса плюс отчёт о том, что во что превратилось."""
    documents: list[dict] = []
    seen_ids: set[str] = set()
    seen_texts: set[str] = set()
    report = {
        "read": len(records),
        "skipped_short": 0,
        "skipped_duplicate": 0,
        "unknown_sections": Counter(),
        "by_rubric": Counter(),
        "by_source": Counter(),
        "by_language": Counter(),
    }

    for record in records:
        text = clean_text(str(record.get("text") or ""))
        if len(text) < MIN_TEXT_CHARS:
            report["skipped_short"] += 1
            continue

        # Дубликат ловится и по id, и по нормализованному тексту: одна и та же
        # новость нередко лежит под двумя адресами, и попади она в train и в
        # test одновременно, метрика на test оказалась бы завышена без единого
        # видимого признака.
        identifier = document_id(str(record.get("url") or ""), text)
        fingerprint = " ".join(text.split())[:400]
        if identifier in seen_ids or fingerprint in seen_texts:
            report["skipped_duplicate"] += 1
            continue
        seen_ids.add(identifier)
        seen_texts.add(fingerprint)

        source = str(record.get("source") or "")
        raw_section = str(record.get("section") or "")
        code = normalize_section(raw_section)
        if code is None:
            # Раздел есть, а решения по нему нет. Документ остаётся в корпусе
            # как неразмеченный. У редакционного источника раздел вдобавок
            # называется в отчёте: молча ссыпать его в корзину значило бы
            # потерять тему, которую стоило бы завести.
            if source in EDITORIAL_SOURCES:
                report["unknown_sections"][raw_section] += 1
            code = UNLABELLED

        language = str(record.get("lang") or "tg")
        language = LANGUAGE_ALIASES.get(language, language)
        tg_name, ru_name = rubric_names(code)
        documents.append(
            {
                "id": identifier,
                "text": text,
                "language": language,
                "topic_id": code,
                "topic": tg_name,
                "topic_ru": ru_name,
                # Исходный раздел сайта — более дробное деление той же рубрики
                # («Сиёсат» и «Сиёсати хориҷӣ» внутри политики). Хранится
                # ровно как на сайте: нормализованный вариант уже есть в
                # topic_id, и вторая нормализованная колонка была бы копией.
                "subtopic_id": raw_section or "—",
                "dataset_origin": "real",
                "source": source,
                "url": str(record.get("url") or ""),
                "title": str(record.get("title") or ""),
                "published_at": str(record.get("published_at") or ""),
                "word_count": len(text.split()),
                "split": "pool",
            }
        )
        report["by_rubric"][code] += 1
        report["by_source"][source] += 1
        report["by_language"][language] += 1

    return documents, report


def demote_small_rubrics(documents: list[dict], report: dict) -> list[str]:
    """Рубрики меньше MIN_RUBRIC_SIZE переводятся в неразмеченные.

    Возвращает список того, что понижено, — он идёт в отчёт. Понижение, а не
    удаление: документ нужен главным осям и кластеризации, ему просто нечем
    мерить попадание.
    """
    demoted = [
        code
        for code, count in report["by_rubric"].items()
        if code != UNLABELLED and count < MIN_RUBRIC_SIZE
    ]
    if not demoted:
        return []
    tg_name, ru_name = rubric_names(UNLABELLED)
    for document in documents:
        if document["topic_id"] in demoted:
            document["topic_id"] = UNLABELLED
            document["topic"] = tg_name
            document["topic_ru"] = ru_name
    report["by_rubric"] = Counter(document["topic_id"] for document in documents)
    return sorted(demoted)


def assign_splits(documents: list[dict]) -> None:
    """Стратифицированное 70/15/15 внутри каждой рубрики.

    Стратификация обязательна: рубрики разного размера, и случайная нарезка
    целиком могла бы оставить «Паём ва суханронӣ» без единого документа в test.
    Порядок внутри рубрики перемешивается с фиксированным seed — иначе
    разбиение повторяло бы порядок обхода файлов, то есть делило бы по сайтам.

    Неразмеченные остаются в `pool`: они не участвуют ни в обучении кластеров,
    ни в метриках, зато участвуют в подсчёте главных осей.
    """
    rng = np.random.default_rng(RANDOM_STATE)
    by_rubric: dict[str, list[dict]] = defaultdict(list)
    for document in documents:
        if document["topic_id"] != UNLABELLED:
            by_rubric[document["topic_id"]].append(document)

    for code in sorted(by_rubric):
        group = by_rubric[code]
        order = rng.permutation(len(group))
        n_train = int(round(len(group) * SPLIT_SHARES[0][1]))
        n_validation = int(round(len(group) * SPLIT_SHARES[1][1]))
        # Остаток отдаётся test, а не размазывается: сумма долей после
        # округления может не совпасть с размером группы, и «положим куда
        # придётся» означало бы разный размер test при разных k.
        for position, index in enumerate(order):
            if position < n_train:
                group[index]["split"] = "train"
            elif position < n_train + n_validation:
                group[index]["split"] = "validation"
            else:
                group[index]["split"] = "test"


def write_corpus(documents: list[dict], out_dir: Path) -> dict[str, int]:
    data_dir = out_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    def dump(path: Path, rows: list[dict]) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    dump(data_dir / FULL_FILE, documents)
    written = {"full": len(documents)}
    for name, filename in SPLIT_FILES.items():
        rows = [document for document in documents if document["split"] == name]
        dump(data_dir / filename, rows)
        written[name] = len(rows)
    written["pool"] = sum(1 for document in documents if document["split"] == "pool")
    return written


def print_report(documents: list[dict], report: dict, demoted: list[str], written: dict) -> None:
    say = print
    say(f"прочитано записей: {report['read']}")
    say(f"  пропущено коротких (< {MIN_TEXT_CHARS} символов): {report['skipped_short']}")
    say(f"  пропущено дубликатов: {report['skipped_duplicate']}")
    say(f"  документов в корпусе: {len(documents)}")
    say("")
    say("по источникам:")
    for source, count in report["by_source"].most_common():
        mark = "рубрики редакции" if source in EDITORIAL_SOURCES else "подборки, сведены вручную"
        say(f"  {count:5d}  {source:22s} {mark}")
    say("")
    say("по языкам: " + ", ".join(f"{k} {v}" for k, v in sorted(report["by_language"].items())))
    say("")
    say("рубрики:")
    for rubric in RUBRICS:
        count = report["by_rubric"].get(rubric.code, 0)
        if not count:
            continue
        say(f"  {count:5d}  {rubric.code}  {rubric.tg:26s} {rubric.ru}")
    unlabelled = report["by_rubric"].get(UNLABELLED, 0)
    say(f"  {unlabelled:5d}  {UNLABELLED}  без рубрики — в измерении не участвуют")
    if demoted:
        say("")
        say(
            f"понижено в «без рубрики» (меньше {MIN_RUBRIC_SIZE} документов): "
            + ", ".join(demoted)
        )
    if report["unknown_sections"]:
        say("")
        say("РАЗДЕЛЫ БЕЗ РЕШЕНИЯ — их нет ни в таблице рубрик, ни в списке игнорируемых:")
        for section, count in report["unknown_sections"].most_common():
            say(f"  {count:5d}  {section!r}")
        say("  добавьте их в rubrics.py или в IGNORED_SECTIONS — молча они не пропадут")
    say("")
    say("записано: " + ", ".join(f"{name} {count}" for name, count in written.items()))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", default=str(DEFAULT_SOURCE_DIR))
    parser.add_argument("--out", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--dry-run", action="store_true", help="только отчёт, ничего не писать")
    args = parser.parse_args()

    records = read_records(Path(args.source))
    documents, report = build(records)
    demoted = demote_small_rubrics(documents, report)
    assign_splits(documents)

    written = {"full": len(documents)}
    if not args.dry_run:
        written = write_corpus(documents, Path(args.out))
    else:
        written = {
            name: sum(1 for d in documents if d["split"] == name)
            for name in ("train", "validation", "test", "pool")
        }
        written["full"] = len(documents)

    print_report(documents, report, demoted, written)
    if args.dry_run:
        print("\n(--dry-run: файлы не записаны)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
