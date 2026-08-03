"""
Сверка коллекции ChromaDB с таблицей chunk в PostgreSQL.

Ищет векторы, которым не соответствует ни одна строка chunk: такие остаются
после удаления документа или блокнота, если ChromaDB в этот момент была
недоступна. В поиске они видны как цитаты из несуществующих документов.

По умолчанию — только отчёт, ничего не удаляется:

    python reconcile_chroma.py

Удаление найденных сирот выполняется явным флагом:

    python reconcile_chroma.py --apply

Коллекция берётся та же, что использует приложение (её имя зависит от модели
эмбеддингов из runtime-настроек), подключение — через ChromaGateway.
"""

import argparse
import asyncio
import os
import sys

from sqlalchemy import func
from sqlmodel import select

# Add backend to path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from app.core.database import session_context
from app.models.models import Chunk
from app.services.rag_service import RAGService
from app.shared.settings.config import settings

SCAN_BATCH_SIZE = 1000
DELETE_BATCH_SIZE = 200
SAMPLE_SIZE = 10


def _describe(chunk_id: str, metadata: dict | None) -> str:
    metadata = metadata or {}
    doc_id = metadata.get("doc_id")
    doc_name = metadata.get("doc_name")
    page = metadata.get("page")
    return f"{chunk_id} (doc_id={doc_id}, doc={doc_name!r}, page={page})"


async def _existing_chunk_ids(numeric_ids: list[int]) -> set[int]:
    """Какие из id действительно есть в таблице chunk."""
    if not numeric_ids:
        return set()
    async with session_context() as session:
        result = await session.exec(select(Chunk.id).where(Chunk.id.in_(numeric_ids)))
        return {chunk_id for chunk_id in result.all() if chunk_id is not None}


async def _total_chunks() -> int:
    async with session_context() as session:
        result = await session.exec(select(func.count()).select_from(Chunk))
        return int(result.one())


async def scan(collection, batch_size: int) -> tuple[int, list[str], list[str]]:
    """Пройти коллекцию батчами и собрать id векторов без строки chunk.

    Всю коллекцию в память не тянем: на каждом шаге забираем batch_size id
    вместе с метаданными (без текстов и эмбеддингов) и сверяем их с БД одним
    SELECT по первичному ключу.
    """
    offset = 0
    scanned = 0
    orphans: list[str] = []
    samples: list[str] = []
    while True:
        batch = collection.get(
            limit=batch_size,
            offset=offset,
            include=["metadatas"],
        )
        ids = batch.get("ids") or []
        if not ids:
            break
        metadatas = batch.get("metadatas") or []
        scanned += len(ids)
        offset += len(ids)

        numeric: dict[int, str] = {}
        non_numeric: list[str] = []
        for chroma_id in ids:
            try:
                numeric[int(chroma_id)] = chroma_id
            except (TypeError, ValueError):
                # id вектора — это str(chunk.id); всё остальное строке chunk
                # соответствовать не может по определению.
                non_numeric.append(chroma_id)

        existing = await _existing_chunk_ids(list(numeric))
        missing = {
            chroma_id
            for chunk_id, chroma_id in numeric.items()
            if chunk_id not in existing
        }
        for index, chroma_id in enumerate(ids):
            if chroma_id in missing or chroma_id in non_numeric:
                orphans.append(chroma_id)
                if len(samples) < SAMPLE_SIZE:
                    metadata = metadatas[index] if index < len(metadatas) else None
                    samples.append(_describe(chroma_id, metadata))

        print(f"  ...просмотрено {scanned} векторов, сирот найдено {len(orphans)}")
        if len(ids) < batch_size:
            break
    return scanned, orphans, samples


def delete_orphans(collection, orphans: list[str]) -> int:
    deleted = 0
    for start in range(0, len(orphans), DELETE_BATCH_SIZE):
        batch = orphans[start : start + DELETE_BATCH_SIZE]
        collection.delete(ids=batch)
        deleted += len(batch)
        print(f"  удалено {deleted} из {len(orphans)}")
    return deleted


async def reconcile(apply: bool, batch_size: int, collection_name: str | None) -> int:
    rag = RAGService()
    if rag.collection is None:
        print(f"ChromaDB недоступна: {rag.chroma_error}")
        return 1

    collection = rag.collection
    if collection_name and collection_name != collection.name:
        # После смены модели эмбеддингов в системе остаются коллекции от
        # прежних моделей: приложение в них уже не ходит, а векторы лежат.
        try:
            collection = rag.chroma_client.get_collection(collection_name)
        except Exception as exc:
            print(f"Коллекция {collection_name!r} недоступна: {exc}")
            return 1
    # Имя коллекции выводится из OLLAMA_MODEL_EMBEDDING, у которого есть
    # дефолт в настройках. Запуск без окружения бэкенда молча уводит сверку
    # в чужую коллекцию, и отчёт "расхождений нет" оказывается ложным.
    # Поэтому показываем модель и все коллекции сервера, а удаление без
    # явного --collection не выполняем.
    print(f"Модель эмбеддингов: {settings.OLLAMA_MODEL_EMBEDDING} (OLLAMA_MODEL_EMBEDDING)")
    try:
        available = ", ".join(sorted(c.name for c in rag.chroma_client.list_collections()))
        print(f"Коллекции на сервере: {available or '(нет)'}")
    except Exception as exc:  # noqa: BLE001 - диагностика не должна ронять сверку
        print(f"Список коллекций недоступен: {exc}")
    if apply and not collection_name:
        print()
        print(
            "Отказ: удаление требует явного --collection. Проверьте, что имя ниже "
            "совпадает с коллекцией работающего бэкенда, и повторите с "
            f"--collection {collection.name}"
        )
        return 1

    total_vectors = collection.count()
    total_chunks = await _total_chunks()
    print(f"Коллекция ChromaDB: {collection.name}")
    print(f"Векторов в ChromaDB: {total_vectors}")
    print(f"Строк chunk в PostgreSQL: {total_chunks}")
    print("Сверка (режим: %s)..." % ("УДАЛЕНИЕ" if apply else "только отчёт"))

    scanned, orphans, samples = await scan(collection, batch_size)

    print()
    print(f"Просмотрено векторов: {scanned}")
    print(f"Висячих векторов (нет строки chunk): {len(orphans)}")
    if samples:
        print(f"Примеры (до {SAMPLE_SIZE}):")
        for sample in samples:
            print(f"  {sample}")

    if not orphans:
        print("Расхождений нет.")
        return 0
    if not apply:
        print()
        print("Ничего не удалено: это режим отчёта.")
        print("Для удаления запустите: python reconcile_chroma.py --apply")
        return 0

    print()
    print(f"Удаляю {len(orphans)} векторов...")
    deleted = delete_orphans(collection, orphans)
    print(f"Готово. Удалено векторов: {deleted}")
    print(f"Осталось в коллекции: {collection.count()}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Сверить коллекцию ChromaDB с таблицей chunk и найти висячие векторы."
        )
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="удалить найденные висячие векторы (без флага — только отчёт)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=SCAN_BATCH_SIZE,
        help=f"размер батча при обходе коллекции (по умолчанию {SCAN_BATCH_SIZE})",
    )
    parser.add_argument(
        "--collection",
        default=None,
        help=(
            "сверять другую коллекцию вместо рабочей "
            "(например оставшуюся от прежней модели эмбеддингов)"
        ),
    )
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size должен быть положительным")
    return asyncio.run(
        reconcile(
            apply=args.apply,
            batch_size=args.batch_size,
            collection_name=args.collection,
        )
    )


if __name__ == "__main__":
    sys.exit(main())
