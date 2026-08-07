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
import json
import math
import os
import re
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Sequence

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

from app.modules.topics.pipeline.dataset import load_full  # noqa: E402
from app.modules.topics.pipeline.embeddings import embed_corpus  # noqa: E402

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

# КАК УСКОРИТЬ ПРОГОН, И ЧТО НЕ РАБОТАЕТ — три замера, чтобы их не повторяли.
#
# Прогон идёт около двух часов при 8% загрузки видеокарты: карта простаивает.
# Напрашиваются три способа, и работает только третий.
#
# 1. ПОСЛАТЬ ЗАПРОСЫ ОДНОВРЕМЕННО — не помогает. Ollama СЕРИАЛИЗУЕТ эмбеддинг
#    сама: четыре одновременных запроса curl к /api/embed при
#    OLLAMA_NUM_PARALLEL=4 легли в один и тот же слот, в журнале сервера все
#    101 задача с «id 0». Клиентская параллельность до раннера не доходит.
#
#    Поднимать OLLAMA_NUM_PARALLEL при этом нельзя и незачем: на стенде он равен
#    единице потому, что при пяти слотах планировщик резервировал контекст сразу
#    на все и вытеснял загруженную модель посреди генерации презентации (см.
#    scripts/start_ollama.sh).
#
# 2. СДЕЛАТЬ КОПИИ МОДЕЛИ ПОД ДРУГИМИ ИМЕНАМИ (`ollama cp`) — не помогает.
#    Ollama различает модели по содержимому, а не по имени: у четырёх копий один
#    дайджест, и на все четыре имени поднимается ОДИН раннер. Замер: 1.04x, то
#    есть ничего. Память видеокарты при этом остаётся 12 ГБ — вторая копия не
#    загружается вовсе.
#
# 3. ПОДНЯТЬ НЕСКОЛЬКО ПРОЦЕССОВ OLLAMA НА РАЗНЫХ ПОРТАХ — работает. У каждого
#    свой раннер и своя копия модели в памяти. Замер на 96 настоящих кусках
#    корпуса: один сервер 1.89 куска/с, четыре сервера 4.61 куска/с, то есть
#    **2.44x**. Не вчетверо, потому что на четырёх раннерах карта наконец
#    упирается в счёт, а не в накладные расходы.
#
#    Четыре, а не шесть: модель занимает 12 ГБ, четыре копии — 48 ГБ из 80, и
#    остатка хватает на gemma4 (20 ГБ). Забив память под завязку, мы уронили бы
#    чат и презентации на живом стенде ради ещё нескольких процентов.
#
# Раздельные серверы поднимаются РУКАМИ на время прогона и гасятся после, а не
# живут в scripts/start_all.sh: это разовая пакетная работа, а не часть системы.
#
#     for p in 11435 11436 11437; do
#       OLLAMA_FLASH_ATTENTION=1 OLLAMA_NUM_PARALLEL=1 OLLAMA_HOST=127.0.0.1:$p \
#         setsid ollama serve > /tmp/ollama-$p.log 2>&1 < /dev/null &
#     done
#     ./venv/bin/python topics_embed.py --hosts 11434,11435,11436,11437
DEFAULT_HOSTS = ("127.0.0.1:11434",)
EMBED_TIMEOUT_SECONDS = 900.0


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


def normalize_hosts(raw: str | None) -> tuple[str, ...]:
    """«11435,11436» и «127.0.0.1:11435» — обе записи, одна форма на выходе.

    Голый номер порта разворачивается в локальный адрес: команда с четырьмя
    портами читается глазом, а с четырьмя полными адресами — уже нет.
    """
    if not raw:
        return DEFAULT_HOSTS
    hosts = []
    for piece in str(raw).split(","):
        piece = piece.strip()
        if not piece:
            continue
        hosts.append(f"127.0.0.1:{piece}" if piece.isdigit() else piece)
    return tuple(hosts) or DEFAULT_HOSTS


def http_embed_fn(model: str, hosts: Sequence[str]):
    """Векторы через несколько серверов Ollama сразу.

    Свой HTTP вместо ModelManager: тот ходит по одному адресу — тому, на котором
    живёт продукт, — и это правильно для продукта, но здесь адресов несколько.
    Запрос простой (POST /api/embed со списком текстов), и заводить ради него
    второй ModelManager значило бы дублировать разбор настроек.

    Список делится на непрерывные отрезки по числу серверов, каждый считается
    своим потоком, ответы склеиваются В ИСХОДНОМ ПОРЯДКЕ. Перепутанный порядок
    означал бы вектор одного документа, приписанный другому, — ошибку, которую
    не видно ничем: номера кластеров всё равно посчитаются.

    Отказ любого сервера поднимается наружу, а не заменяется нулями: молча
    подставленный нулевой вектор дал бы документу тему, посчитанную ни по чему.
    """

    def one(host: str, texts: Sequence[str]) -> list[list[float]]:
        body = json.dumps({"model": model, "input": list(texts)}).encode("utf-8")
        request = urllib.request.Request(
            f"http://{host}/api/embed", data=body, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(request, timeout=EMBED_TIMEOUT_SECONDS) as response:
            payload = json.load(response)
        vectors = payload.get("embeddings")
        if not isinstance(vectors, list) or len(vectors) != len(texts):
            raise ValueError(
                f"{host}: вернул {len(vectors) if isinstance(vectors, list) else '?'} "
                f"векторов на {len(texts)} текстов"
            )
        return vectors

    def embed(texts: Sequence[str]) -> list[list[float]]:
        items = list(texts)
        if len(hosts) == 1 or len(items) <= 1:
            return one(hosts[0], items)
        size = math.ceil(len(items) / len(hosts))
        parts = [items[start : start + size] for start in range(0, len(items), size)]
        with ThreadPoolExecutor(max_workers=len(parts)) as pool:
            results = list(pool.map(lambda pair: one(*pair), zip(hosts, parts)))
        merged: list[list[float]] = []
        for vectors in results:
            merged.extend(vectors)
        return merged

    return embed


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
    parser.add_argument(
        "--hosts",
        default=None,
        help="адреса серверов Ollama через запятую; можно только порты: 11434,11435",
    )
    args = parser.parse_args()

    model = embedding_model_name(args.embedding_model)
    hosts = normalize_hosts(args.hosts)
    corpus = load_full(args.data)
    print(
        f"корпус: {len(corpus)} документов, модель {model}, "
        f"серверов {len(hosts)}: {', '.join(hosts)}",
        flush=True,
    )
    Path(args.cache).parent.mkdir(parents=True, exist_ok=True)
    embed_corpus(
        corpus.documents,
        model=model,
        cache_path=args.cache,
        embed_fn=chunked_embed_fn(http_embed_fn(model, hosts), args.chunk_chars),
        batch_size=args.batch_size,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
