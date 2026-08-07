#!/usr/bin/env python3
"""Векторы документов таджикского корпуса — досчёт недостающих в кэш.

Отдельный шаг, а не часть обучения, по двум причинам. Первая: эмбеддирование
идёт минуты и упирается в GPU, а обучение — секунды и упирается в решения;
смешивать их значило бы каждую попытку подобрать k оплачивать заново. Вторая:
корпус дособирается — агенты добирают государственные документы, — и кэш
ключуется по id документа (хэш от url). Значит повторный прогон считает ТОЛЬКО
новые записи, а уже посчитанные берёт с диска.

Модель берётся оттуда же, откуда её берёт продукт: runtime_settings.json, затем
OLLAMA_MODEL_EMBEDDING. Своего умолчания здесь нет намеренно — кэш, посчитанный
не той моделью, дал бы правдоподобные векторы из чужого пространства.

    ./venv/bin/python topics_embed.py
    ./venv/bin/python topics_embed.py --data data/topics_tj/data
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

from app.modules.topics.pipeline.dataset import load_full  # noqa: E402
from app.modules.topics.pipeline.embeddings import (  # noqa: E402
    embed_corpus,
    ollama_embed_fn,
)

BACKEND_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = BACKEND_ROOT / "data" / "topics_tj" / "data"
DEFAULT_CACHE = BACKEND_ROOT / "data" / "topics_tj" / "embeddings.npz"

# Предел длины куска. Взят не с потолка: индексация продукта режет документы
# HybridChunker'ом с target_tokens=450 при CHARS_PER_TOKEN=3.6, то есть на куски
# примерно по 1600 знаков (см. app/services/hybrid_chunker.py и
# app/modules/documents/service.py).
#
# Совпадение обязательно по двум причинам. Первая — верность: вектор боевого
# документа считается как СРЕДНЕЕ векторов его фрагментов
# (topics/service.py, document_vector), и обучать модель на векторе целого
# текста значило бы сравнивать при назначении разные величины. Вторая —
# исполнимость: в корпусе есть документ на 116 тысяч знаков, и попытка
# получить один вектор на весь текст упирается в предел контекста модели —
# первый прогон встал именно на этом.
CHUNK_CHARS = 1620

# ПОЧЕМУ ЗДЕСЬ НЕТ ПАРАЛЛЕЛЬНОСТИ — замер, чтобы её не пробовали заново.
#
# Прогон идёт около двух часов при 8% загрузки видеокарты, и напрашивается
# послать запросы одновременно. Проверено: не помогает, потому что Ollama
# СЕРИАЛИЗУЕТ эмбеддинг сама. Четыре одновременных запроса curl к /api/embed
# при OLLAMA_NUM_PARALLEL=4 легли в один и тот же слот — в журнале сервера все
# 101 задача с «id 0». Клиентская параллельность до раннера просто не доходит.
#
# OLLAMA_NUM_PARALLEL при этом трогать нельзя: на стенде он равен единице не
# случайно, а потому что при пяти слотах планировщик резервировал контекст
# сразу на все и вытеснял загруженную модель посреди генерации презентации
# (см. scripts/start_ollama.sh). Возвращать эту беду ради ускорения, которого
# всё равно не будет, незачем.


def embedding_model_name(explicit: str | None = None) -> str:
    """Имя embedding-модели так, как его выбирает продукт.

    Тем же порядком и тем же кодом, что в cluster_topics.py: своя копия разбора
    настроек означала бы, что эксперимент однажды посчитает векторы одной
    моделью, а продукт — другой. Несовпадение пространств не выдаёт ошибки, оно
    выдаёт неверные темы.
    """
    if explicit:
        return explicit.strip()
    from app.shared.settings.runtime_settings import RuntimeSettingsService

    resolved = RuntimeSettingsService.embedding_model().strip()
    if not resolved:
        raise SystemExit(
            "embedding-модель не задана: ни --embedding-model, ни "
            "runtime_settings.json, ни OLLAMA_MODEL_EMBEDDING"
        )
    return resolved


def split_text(text: str, limit: int = CHUNK_CHARS) -> list[str]:
    """Текст на куски не длиннее limit, по границам абзацев и предложений.

    Проще HybridChunker'а продукта намеренно: тому нужны заголовки, таблицы и
    сноски документа, а здесь на входе новостная статья одним куском текста,
    без структуры. Общее у них одно и важное — предел длины куска.

    Резать по границам, а не посимвольно: кусок, начинающийся с середины слова,
    даёт вектор, описывающий обрывок, а не тему.
    """
    text = str(text).strip()
    if len(text) <= limit:
        return [text] if text else []

    pieces: list[str] = []
    current = ""
    # Сначала абзацы, внутри длинного абзаца — предложения, а если и предложение
    # длиннее предела (сплошной текст без точек), только тогда посимвольно.
    for paragraph in re.split(r"\n\s*\n|\n", text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        units = [paragraph]
        if len(paragraph) > limit:
            units = re.split(r"(?<=[.!?…])\s+", paragraph)
        for unit in units:
            while len(unit) > limit:
                pieces.append(unit[:limit])
                unit = unit[limit:]
            if not unit:
                continue
            if len(current) + len(unit) + 1 <= limit:
                current = f"{current} {unit}".strip()
            else:
                if current:
                    pieces.append(current)
                current = unit
    if current:
        pieces.append(current)
    return pieces or [text[:limit]]


def chunked_embed_fn(base, limit: int = CHUNK_CHARS):
    """Вектор документа как среднее векторов его кусков — как в бою.

    Возвращает функцию того же вида «список текстов -> список векторов», что
    ждёт embed_corpus, поэтому кэш по-прежнему хранит ОДИН вектор на документ.
    Куски наружу не выходят: они нужны только чтобы вектор считался тем же
    способом, что и у документа в системе.

    Среднее без нормировки — тоже как в бою: document_vector возвращает
    обычное среднее, а к единичной длине вектор приводят уже потребители
    (обучение нормирует матрицу, боевое преобразование — полем unit_input).
    """

    def embed(texts):
        flat: list[str] = []
        spans: list[tuple[int, int]] = []
        for text in texts:
            pieces = split_text(text, limit) or [""]
            spans.append((len(flat), len(flat) + len(pieces)))
            flat.extend(pieces)
        vectors = base(flat)
        if len(vectors) != len(flat):
            raise ValueError(f"модель вернула {len(vectors)} векторов на {len(flat)} кусков")
        matrix = np.asarray(vectors, dtype=np.float64)
        return [matrix[start:stop].mean(axis=0) for start, stop in spans]

    return embed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--cache", default=str(DEFAULT_CACHE))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument("--chunk-chars", type=int, default=CHUNK_CHARS)
    args = parser.parse_args()

    model = embedding_model_name(args.embedding_model)
    corpus = load_full(args.data)
    print(f"корпус: {len(corpus)} документов, модель {model}", flush=True)
    Path(args.cache).parent.mkdir(parents=True, exist_ok=True)
    embed_corpus(
        corpus.documents,
        model=model,
        cache_path=args.cache,
        embed_fn=chunked_embed_fn(ollama_embed_fn(model), args.chunk_chars),
        batch_size=args.batch_size,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
