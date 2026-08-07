#!/usr/bin/env python3
"""Что обученная модель скажет о БОЕВЫХ документах системы — до переразметки.

Зачем отдельный скрипт. Метрики на отложенной выборке отвечают на вопрос «как
модель ведёт себя на корпусе, из которого её учили». Вопрос, ради которого всё
затевалось, другой: что она скажет про налоговый кодекс и про паёмы президента.
Прежняя модель на этот вопрос отвечала так: пять посланий одного автора одного
жанра разъехались по двум темам («Экономика» и «Политика»), и ни одна из них не
называлась «Послания», потому что такой темы в ней не было вовсе.

Скрипт НИЧЕГО НЕ МЕНЯЕТ: он читает векторы фрагментов из ChromaDB, считает
вектор документа тем же способом, что и назначение (среднее фрагментов), и
печатает, какую тему дала бы модель. Регистрация версии и переразметка — шаги
отдельные и совершаются осознанно.

    ./venv/bin/python topics_check.py
    ./venv/bin/python topics_check.py --model data/topics_tj/topic_model_tj.npz
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sqlalchemy import text  # noqa: E402

BACKEND_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = BACKEND_ROOT / "data" / "topics_tj" / "topic_model_tj.npz"


async def run(model_path: Path) -> int:
    from app.core.database import session_context
    from app.modules.topics.service import (
        TopicEmbeddingUnavailable,
        document_vector,
        fetch_chunk_vectors,
        load_artifact,
    )

    artifact = load_artifact(model_path)
    print(f"модель: {model_path}")
    print(f"  кластеров {artifact.cluster_count}, эмбеддинги {artifact.embedding_model}")
    print(f"  преобразование: {artifact.transform.description}")
    print(f"  порог уверенности: {artifact.margin_threshold}")
    print()

    async with session_context() as session:
        rows = (
            await session.execute(
                text(
                    """
                    SELECT d.id, d.name, d.language, d.topic_label_ru,
                           array_agg(c.id::text ORDER BY c.id) AS chunk_ids
                    FROM document d
                    JOIN chunk c ON c.doc_id = d.id
                    WHERE d.status = 'indexed'
                    GROUP BY d.id, d.name, d.language, d.topic_label_ru
                    ORDER BY d.id
                    """
                )
            )
        ).all()

    if not rows:
        print("проиндексированных документов нет")
        return 0

    print(f"{'документ':30s} {'было':38s} {'станет':38s} запас")
    print("-" * 118)
    for doc_id, name, language, was, chunk_ids in rows:
        vectors = fetch_chunk_vectors(list(chunk_ids))
        found = [vectors[cid] for cid in chunk_ids if cid in vectors]
        try:
            centre = document_vector(found)
        except TopicEmbeddingUnavailable as exc:
            print(f"{name[:29]:30s} {str(was)[:37]:38s} {'— ' + str(exc):38s}")
            continue
        if centre is None:
            print(f"{name[:29]:30s} {str(was)[:37]:38s} {'— векторов нет':38s}")
            continue
        cluster, margin = artifact.assign_with_margin(centre, group=language)
        if artifact.is_confident(margin):
            now = artifact.label_in(cluster, "ru") or artifact.label_of(cluster)
        else:
            now = "тема не определена"
        print(f"{name[:29]:30s} {str(was)[:37]:38s} {now[:37]:38s} {margin:.3f}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    args = parser.parse_args()
    return asyncio.run(run(Path(args.model)))


if __name__ == "__main__":
    raise SystemExit(main())
