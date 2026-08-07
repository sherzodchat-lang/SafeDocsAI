#!/usr/bin/env python3
"""Какие метки принесли сборщики и что с ними делает таблица рубрик.

Между сбором и обучением стоит решение: как новые метки лечь на список рубрик.
Принимать его вслепую нельзя — класс из пяти документов кластером не станет, а
две метки об одном («здравоохранение» и «медицина и здравоохранение») дадут два
кластера, различающихся только тем, какой агент их собрал.

Скрипт отвечает на три вопроса разом:
  * какие метки вообще есть и сколько под каждой документов;
  * какие уже разбираются таблицей рубрик, а какие нет;
  * какие слишком малы, чтобы стать классом.

Ничего не меняет: только читает и печатает.

    ./venv/bin/python topics_sections.py
    ./venv/bin/python topics_sections.py --min 25
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.modules.topics.pipeline.rubrics import (  # noqa: E402
    RUBRIC_BY_CODE,
    UNLABELLED,
    normalize_section,
)

BACKEND_ROOT = Path(__file__).resolve().parent
DEFAULT_DIR = BACKEND_ROOT / "data" / "topics_dataset"

# Ниже этого числа документов метка классом не станет: k-means не выделит
# кластер из горстки точек, а если и выделит — метрика по нему будет шумом.
DEFAULT_MIN = 25


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dir", default=str(DEFAULT_DIR))
    parser.add_argument("--min", type=int, default=DEFAULT_MIN)
    args = parser.parse_args()

    directory = Path(args.dir)
    per_section: Counter = Counter()
    per_source: dict[str, Counter] = defaultdict(Counter)
    languages: dict[str, Counter] = defaultdict(Counter)
    total = 0

    for path in sorted(directory.glob("*.jsonl")):
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            section = " ".join(str(row.get("section") or "").split())
            per_section[section] += 1
            per_source[path.name][section] += 1
            languages[section][str(row.get("lang") or "?")] += 1
            total += 1

    print(f"документов всего: {total}, различных меток: {len(per_section)}\n")

    known: list[tuple[str, int, str]] = []
    ignored: list[tuple[str, int]] = []
    unknown: list[tuple[str, int]] = []
    for section, count in per_section.most_common():
        code = normalize_section(section)
        if code is None:
            unknown.append((section, count))
        elif code == UNLABELLED:
            ignored.append((section, count))
        else:
            known.append((section, count, code))

    print("РАЗБИРАЮТСЯ ТАБЛИЦЕЙ РУБРИК")
    by_code: Counter = Counter()
    for section, count, code in known:
        by_code[code] += count
    for code, count in by_code.most_common():
        rubric = RUBRIC_BY_CODE[code]
        mark = "" if count >= args.min else "  << МАЛО"
        print(f"  {count:5d}  {code}  {rubric.ru}{mark}")

    if unknown:
        print(f"\nМЕТКИ БЕЗ РЕШЕНИЯ — их {len(unknown)}, требуют строки в таблице:")
        for section, count in unknown:
            langs = ", ".join(f"{k} {v}" for k, v in languages[section].most_common())
            mark = "" if count >= args.min else "  << МАЛО, класса не выйдет"
            print(f"  {count:5d}  [{langs}]  {section!r}{mark}")

    if ignored:
        print(f"\nСОЗНАТЕЛЬНО НЕ СЧИТАЮТСЯ ТЕМОЙ: {sum(c for _, c in ignored)} документов")

    print("\nПО ФАЙЛАМ")
    for name, counter in sorted(per_source.items()):
        decided = sum(
            count
            for section, count in counter.items()
            if (normalize_section(section) or UNLABELLED) != UNLABELLED
        )
        print(f"  {sum(counter.values()):5d}  {name:26s} с рубрикой {decided}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
