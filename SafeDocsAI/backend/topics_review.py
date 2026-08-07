#!/usr/bin/env python3
"""Как модель разложила документы по темам — глазами человека, а не метрикой.

Метрика на отложенной выборке отвечает на вопрос «насколько разбиение похоже на
редакционные рубрики». Здесь вопрос другой и более важный: взял человек свои
документы, загрузил их, открыл папки — увидел он осмысленное или мусор.

Отвечает таблицей «папка — что в ней лежит» и, если у документов известен
предмет (манифест корпуса проверки), отдельной сверкой «предмет против папки».
Сверка нужна затем, что по одной только папке нельзя отличить «модель собрала
все законы о транспорте вместе» от «модель сложила туда всё подряд».

Ничего не меняет: только читает базу и печатает.

    ./venv/bin/python topics_review.py
    ./venv/bin/python topics_review.py --subjects   # со сверкой по предметам
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sqlalchemy import text  # noqa: E402

BACKEND_ROOT = Path(__file__).resolve().parent
CORPUS_ROOT = BACKEND_ROOT / "data" / "test_corpus"


def say(message: str = "") -> None:
    print(message, flush=True)


def subjects_by_file() -> dict[str, str]:
    """Предмет каждого файла из манифестов корпуса проверки.

    Ключ — имя файла, оно же имя документа в системе. Манифеста может не быть
    вовсе (корпус собирают отдельно), и тогда сверка просто не печатается: это
    не повод отказываться от основной таблицы.
    """
    result: dict[str, str] = {}
    if not CORPUS_ROOT.is_dir():
        return result
    for manifest in CORPUS_ROOT.glob("*/manifest.json"):
        try:
            raw = json.loads(manifest.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        rows = raw if isinstance(raw, list) else raw.get("files", [])
        for row in rows:
            if isinstance(row, dict) and row.get("file") and row.get("subject"):
                result[str(row["file"])] = str(row["subject"])
    return result


async def load_documents():
    from app.core.database import session_context

    async with session_context() as session:
        model = (
            await session.execute(
                text(
                    "SELECT version, k FROM topicmodelversion "
                    "WHERE is_active = true LIMIT 1"
                )
            )
        ).first()
        rows = (
            await session.execute(
                text(
                    """
                    SELECT name, language, topic_cluster_index, topic_label_ru,
                           topic_model_version, status
                    FROM document
                    WHERE status = 'indexed'
                    ORDER BY topic_cluster_index NULLS LAST, name
                    """
                )
            )
        ).all()
    return model, rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--subjects", action="store_true", help="сверка предмет против папки")
    args = parser.parse_args()

    model, rows = asyncio.run(load_documents())
    if not rows:
        say("проиндексированных документов нет")
        return 0
    if model is None:
        say("активной модели тем нет")
        return 1

    version, k = int(model[0]), int(model[1])
    say(f"модель версии {version}, кластеров {k}, документов {len(rows)}")
    say()

    by_folder: dict[str, list] = defaultdict(list)
    stale = 0
    unclear = 0
    untouched = 0
    for name, language, cluster, label, doc_version, _ in rows:
        if doc_version != version:
            stale += 1
            continue
        if cluster is None:
            # Версия проставлена, номера нет — модель посмотрела и не решилась.
            unclear += 1
            by_folder["— тема не определена —"].append((name, language))
            continue
        by_folder[label or f"Кластер {cluster}"].append((name, language))
    untouched = sum(1 for row in rows if row[4] is None)

    say("ПАПКИ И ИХ СОДЕРЖИМОЕ")
    for folder in sorted(by_folder, key=lambda key: (-len(by_folder[key]), key)):
        items = by_folder[folder]
        say(f"\n  {folder}  ({len(items)})")
        for name, language in items[:12]:
            say(f"      [{language}] {name}")
        if len(items) > 12:
            say(f"      … и ещё {len(items) - 12}")

    say()
    say(f"без темы (модель отказалась): {unclear}")
    if stale:
        say(f"размечены прошлой версией, ждут переразметки: {stale}")
    if untouched:
        say(f"не размечены вовсе: {untouched}")

    if not args.subjects:
        return 0

    known = subjects_by_file()
    if not known:
        say("\n(манифестов корпуса нет — сверку по предметам пропускаю)")
        return 0

    say()
    say("СВЕРКА: ПРЕДМЕТ ПРОТИВ ПАПКИ")
    say("Собрала ли модель документы одного предмета вместе — по одной только")
    say("папке этого не видно, а именно это и спрашивают.")
    per_subject: dict[str, Counter] = defaultdict(Counter)
    for folder, items in by_folder.items():
        for name, _ in items:
            subject = known.get(name)
            if subject:
                per_subject[subject][folder] += 1

    for subject in sorted(per_subject, key=lambda s: -sum(per_subject[s].values())):
        spread = per_subject[subject]
        total = sum(spread.values())
        top_folder, top_count = spread.most_common(1)[0]
        say(
            f"\n  {subject} ({total} док.): в самой частой папке {top_count} "
            f"из {total}, папок задействовано {len(spread)}"
        )
        for folder, count in spread.most_common(3):
            say(f"      {count:2d}  {folder}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
