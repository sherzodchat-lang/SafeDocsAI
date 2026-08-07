#!/usr/bin/env python3
"""Загрузка корпуса для проверки системы — тем же путём, каким ходит человек.

Через HTTP, а не прямой записью в базу, и это главное решение файла. Положить
строки в document и chunk напрямую было бы вдвое короче и проверило бы ровно
ничего: настоящий путь документа — разбор файла, нарезка на фрагменты, векторы,
ChromaDB, назначение темы, — и ошибки живут именно там. Скрипт, обходящий этот
путь, отвечает на вопрос «есть ли строки в базе», а спрашивают у него «работает
ли система».

Отсюда же ожидание индексации: загрузка отвечает сразу, а работу делает воркер, и
скрипт, отчитавшийся об успехе до её конца, соврал бы.

Каталог с файлами собирают topics-агенты, см. data/test_corpus/*/manifest.json —
там у каждого файла адрес источника, лицензия и предмет. Предмет нужен человеку:
по нему сверяют, куда модель отнесла документ.

    ./venv/bin/python load_test_corpus.py --dry-run        # что будет загружено
    ./venv/bin/python load_test_corpus.py                  # загрузить всё
    ./venv/bin/python load_test_corpus.py --limit 5        # только пять
    ./venv/bin/python load_test_corpus.py --notebook 3     # в конкретный блокнот
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests

BACKEND_ROOT = Path(__file__).resolve().parent
CORPUS_ROOT = BACKEND_ROOT / "data" / "test_corpus"
DEFAULT_API = "http://localhost:8001/api/v1"

# Предел загрузки у продукта — 50 МБ (MAX_UPLOAD_SIZE_MB). Файл крупнее сюда даже
# не отправляется: отказ придёт всё равно, а лишний запрос на пятьдесят мегабайт
# оплачивается временем.
MAX_UPLOAD_BYTES = 50 * 1024 * 1024

# Что вообще принимает продукт (DocumentService.ALLOWED_EXTENSIONS). Берём
# только это, а не «всё, кроме json»: в каталоге сборки остаётся служебное —
# скачанные страницы поиска, черновики, точечные файлы, — и отправлять их
# значило бы получить сотню честных отказов и потерять в них настоящие.
UPLOADABLE_SUFFIXES = {".pdf", ".docx", ".txt"}

# Сколько ждать, пока воркер доиндексирует документ. Крупный закон на две сотни
# страниц — это сотня фрагментов и столько же обращений к модели эмбеддингов.
INDEX_TIMEOUT_SECONDS = 900.0
POLL_SECONDS = 3.0

# Ограничитель загрузки: 30 запросов за 300 секунд (upload_limiter). Умолчание
# паузы — окно целиком: сервер обычно говорит Retry-After сам, а когда молчит,
# переждать окно надёжнее, чем угадывать остаток.
RATE_LIMIT_PAUSE_SECONDS = 300.0
RATE_LIMIT_RETRIES = 4


def say(message: str = "") -> None:
    print(message, flush=True)


def login(api: str, username: str, password: str) -> str:
    response = requests.post(
        f"{api}/auth/login", data={"username": username, "password": password}, timeout=30
    )
    response.raise_for_status()
    token = response.json().get("access_token")
    if not token:
        raise SystemExit("вход выполнен, но токена в ответе нет")
    return token


def collect(root: Path) -> list[dict]:
    """Файлы корпуса вместе с их записями из manifest.json.

    Файл без записи в манифесте берётся всё равно, но помечается: манифест
    пишет агент-сборщик, и рассинхронизация — его беда, а не повод потерять
    документ. Обратное — запись без файла — сообщается отдельно: это уже
    потерянная загрузка.
    """
    items: list[dict] = []
    for directory in sorted(p for p in root.iterdir() if p.is_dir()):
        manifest_path = directory / "manifest.json"
        described: dict[str, dict] = {}
        if manifest_path.is_file():
            try:
                raw = json.loads(manifest_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                say(f"  ! {manifest_path} не разбирается: {exc}")
                raw = []
            rows = raw if isinstance(raw, list) else raw.get("files", [])
            described = {str(row.get("file")): row for row in rows if isinstance(row, dict)}

        present = {
            p.name
            for p in directory.iterdir()
            if p.is_file() and p.suffix.lower() in UPLOADABLE_SUFFIXES
        }
        for missing in sorted(set(described) - present):
            say(f"  ! в манифесте есть {missing}, а файла нет")

        for path in sorted(directory.iterdir()):
            if not path.is_file() or path.suffix.lower() not in UPLOADABLE_SUFFIXES:
                continue
            if path.name.startswith("."):
                continue
            row = described.get(path.name, {})
            items.append(
                {
                    "path": path,
                    "group": directory.name,
                    "subject": row.get("subject", ""),
                    "title": row.get("title", ""),
                    "size": path.stat().st_size,
                }
            )
    return items


def upload(api: str, token: str, item: dict, notebook_id: int | None) -> dict | None:
    """Одна загрузка, с уважением к ограничителю частоты.

    У загрузки стоит предел 30 запросов за 300 секунд (upload_limiter в
    app/api/endpoints/documents.py), а корпус — под сотню файлов. Обходить
    ограничитель нельзя и не нужно: он часть системы, которую мы и проверяем.
    Поэтому 429 здесь — не отказ, а «подожди»: ждём и повторяем.

    Ждём столько, сколько сказал сервер в Retry-After, и только если он молчит —
    своё умолчание. Придумывать паузу за сервер значит либо долбить его, либо
    спать втрое дольше нужного.
    """
    data = {}
    if notebook_id is not None:
        data["notebook_id"] = str(notebook_id)

    for attempt in range(RATE_LIMIT_RETRIES):
        with open(item["path"], "rb") as handle:
            response = requests.post(
                f"{api}/sources/upload",
                headers={"Authorization": f"Bearer {token}"},
                files={"file": (item["path"].name, handle)},
                data=data,
                timeout=300,
            )
        if response.status_code != 429:
            break
        pause = float(response.headers.get("Retry-After") or RATE_LIMIT_PAUSE_SECONDS)
        say(f"  ограничитель частоты: ждём {pause:.0f} с (попытка {attempt + 1})")
        time.sleep(pause)

    if response.status_code >= 400:
        say(f"  ОТКАЗ {response.status_code}: {response.text[:200]}")
        return None
    return response.json()


def wait_indexed(api: str, token: str, doc_ids: set[int]) -> dict[int, str]:
    """Ждём, пока воркер доведёт документы до indexed или failed.

    Опрашиваем список, а не каждый документ по отдельности: у сотни документов
    это сотня запросов на каждый круг ожидания.
    """
    headers = {"Authorization": f"Bearer {token}"}
    started = time.monotonic()
    statuses: dict[int, str] = {}
    while doc_ids and time.monotonic() - started < INDEX_TIMEOUT_SECONDS:
        response = requests.get(f"{api}/sources/?limit=500", headers=headers, timeout=60)
        response.raise_for_status()
        for row in response.json():
            doc_id = int(row.get("id", 0))
            if doc_id in doc_ids and row.get("status") in ("indexed", "failed"):
                statuses[doc_id] = row["status"]
                doc_ids.discard(doc_id)
        if doc_ids:
            done = len(statuses)
            say(f"  ждём индексацию: готово {done}, осталось {len(doc_ids)}")
            time.sleep(POLL_SECONDS)
    for doc_id in doc_ids:
        statuses[doc_id] = "timeout"
    return statuses


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus", default=str(CORPUS_ROOT))
    parser.add_argument("--api", default=os.environ.get("SAFEDOCS_API", DEFAULT_API))
    parser.add_argument("--username", default=os.environ.get("SAFEDOCS_USER", "123"))
    parser.add_argument("--password", default=os.environ.get("SAFEDOCS_PASSWORD", "123"))
    parser.add_argument("--notebook", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = Path(args.corpus)
    if not root.is_dir():
        raise SystemExit(f"каталога {root} нет — сначала соберите корпус")

    items = collect(root)
    if args.limit:
        items = items[: args.limit]
    if not items:
        raise SystemExit(f"в {root} нет файлов")

    too_big = [item for item in items if item["size"] > MAX_UPLOAD_BYTES]
    items = [item for item in items if item["size"] <= MAX_UPLOAD_BYTES]
    for item in too_big:
        say(f"  ! {item['path'].name} — {item['size'] / 1e6:.1f} МБ, больше предела загрузки")

    by_group: dict[str, int] = {}
    for item in items:
        by_group[item["group"]] = by_group.get(item["group"], 0) + 1
    say(f"к загрузке: {len(items)} файлов — " + ", ".join(f"{k} {v}" for k, v in by_group.items()))
    total = sum(item["size"] for item in items)
    say(f"суммарно {total / 1e6:.1f} МБ")
    if args.dry_run:
        for item in items:
            say(f"  {item['group']:10s} {item['subject']:16s} {item['path'].name}")
        say("\n(--dry-run: ничего не загружено)")
        return 0

    token = login(args.api, args.username, args.password)
    uploaded: dict[int, dict] = {}
    failed = 0
    for index, item in enumerate(items, start=1):
        say(f"[{index}/{len(items)}] {item['path'].name}")
        result = upload(args.api, token, item, args.notebook)
        if result is None:
            failed += 1
            continue
        doc_id = int(result.get("id") or result.get("document", {}).get("id") or 0)
        if doc_id:
            uploaded[doc_id] = item

    say(f"\nзагружено {len(uploaded)}, отказов {failed}")
    if not uploaded:
        return 1

    statuses = wait_indexed(args.api, token, set(uploaded))
    counts: dict[str, int] = {}
    for status in statuses.values():
        counts[status] = counts.get(status, 0) + 1
    say("\nитог индексации: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    for doc_id, status in sorted(statuses.items()):
        if status != "indexed":
            say(f"  {status}: {uploaded[doc_id]['path'].name}")
    return 0 if counts.get("indexed") == len(uploaded) else 1


if __name__ == "__main__":
    raise SystemExit(main())
