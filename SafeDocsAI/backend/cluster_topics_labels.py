#!/usr/bin/env python3
"""Приведение подписей тем в уже обученном артефакте к языкам интерфейса.

Зачем отдельный скрипт, а не переобучение. Названия тем лежат в корпусе на трёх
языках, но обучающий код брал подпись кластера «у первого попавшегося документа
кластера» — а колонка topic в корпусе переведена ВМЕСТЕ с документом. Отсюда
две беды сразу:

  * в артефакт попадало имя на случайном языке (в боевом файле кластер 19 был
    подписан «Маориф» посреди английских подписей);
  * перевода не было вовсе, и пользователь с русским или таджикским интерфейсом
    видел «Economics and business analytics» — при том что английского
    интерфейса в продукте нет.

Переобучать ради этого нечего: подпись кластера не участвует ни в геометрии, ни
в назначении. Центроиды, преобразование, метрики и сама разметка «кластер ->
тема» остаются байт в байт теми же; меняется только meta.cluster_topics, где у
каждого кластера по его topic_id проставляются topic (английское, устойчивое),
topic_ru и topic_tg.

Три свойства, ради которых это скрипт, а не разовая правка руками:

**Повторяемость.** Следующий артефакт можно прогнать той же командой. При этом
обучающие скрипты (cluster_topics.py, cluster_topics_variants.py) с этого
момента пишут все три имени сами — этот скрипт нужен для файлов, обученных
раньше, и остаётся как способ починить артефакт, приехавший без подписей.

**Ничего не менять, когда менять нечего.** Файл перезаписывается ТОЛЬКО если
хоть одна подпись действительно изменилась. Причина не в экономии: приложение
узнаёт переобученную модель по sha256 артефакта, и перезапись «тем же самым»
завела бы новую версию модели, обесценив все назначения документов до
переразметки. Холостой прогон обязан быть бесплатным.

**Отказ вместо тихой потери.** Тема, для которой в корпусе не нашлось имени на
нужном языке, названа в выводе поимённо. Молча оставить её как есть значит
получить раздел, где девятнадцать тем на одном языке и одна на другом, — ровно
то, с чего всё началось.

Примеры:

    # посмотреть, что изменится, ничего не трогая
    ./venv/bin/python cluster_topics_labels.py --dry-run

    # привести подписи в боевом артефакте
    ./venv/bin/python cluster_topics_labels.py

    # другой файл
    ./venv/bin/python cluster_topics_labels.py --model data/.../topic_model.npz
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

from app.modules.topics.pipeline.dataset import FULL_FILE, load_jsonl  # noqa: E402

BACKEND_ROOT = Path(__file__).resolve().parent
DATASET_ROOT = BACKEND_ROOT / "data" / "task1_multilingual_dataset"
DEFAULT_DATA_DIR = DATASET_ROOT / "data"
# По умолчанию правится ИМЕННО боевой артефакт: тот, которым размечаются
# документы (см. DEFAULT_TOPIC_MODEL_PATH в app/modules/topics/service.py).
DEFAULT_MODEL = DATASET_ROOT / "topic_model_best.npz"

# Какое поле артефакта каким языком заполняется. Английское лежит в topic и
# служит ключом темы: по нему сходятся отчёты эксперимента и назначения прошлых
# версий, и в интерфейсе оно не показывается. Остальные два — языки экранов.
FIELD_LANGUAGE = (("topic", "en"), ("topic_ru", "ru"), ("topic_tg", "tg"))


def topic_names(data_dir: Path) -> dict[str, dict[str, str]]:
    """Названия тем по языкам: {"ru": {topic_id: имя}, ...}.

    Считается по всему корпусу (Corpus.topic_names), а не по первой встреченной
    записи каждой темы: расхождение внутри одной пары «тема + язык» означает,
    что корпус сам себе противоречит, и узнать об этом лучше здесь, чем по
    подписи, которая меняется от прогона к прогону.
    """
    corpus = load_jsonl(data_dir / FULL_FILE)
    return {language: corpus.topic_names(language) for _, language in FIELD_LANGUAGE}


def enrich(model_path: Path, names: dict[str, dict[str, str]], *, dry_run: bool = False) -> dict:
    """Проставить подписи в meta артефакта. Возвращает сводку изменений.

    Артефакт читается и пишется как СЫРОЙ npz, а не через
    pipeline.model_io.TopicModel: тот загрузчик проверяет версию формата и
    пересобирает файл из полей, которые знает, — то есть тихо потерял бы всё,
    чего в нём ещё нет. Здесь же меняется ровно одна строка метаданных, а все
    остальные массивы переносятся как есть.
    """
    with np.load(model_path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.keys() if name != "meta"}
        meta = json.loads(str(archive["meta"]))

    topics = meta.get("cluster_topics") or []
    changed: list[tuple[int, str, str, str]] = []
    unchanged = 0
    missing: list[tuple[int, str, str]] = []
    for item in topics:
        if not isinstance(item, dict):
            continue
        cluster = int(item.get("cluster"))
        topic_id = str(item.get("topic_id") or "")
        if not topic_id:
            # Пустой кластер обучающей выборки: темы у него нет вообще, и имени
            # ему взять неоткуда. Это не пропажа.
            continue
        for field, language in FIELD_LANGUAGE:
            name = names.get(language, {}).get(topic_id)
            if not name:
                missing.append((cluster, topic_id, language))
                continue
            current = str(item.get(field) or "")
            if current == name:
                unchanged += 1
                continue
            changed.append((cluster, field, current, name))
            item[field] = name

    summary = {
        "path": str(model_path),
        "clusters": len(topics),
        "changed": changed,
        "unchanged": unchanged,
        "missing": missing,
        "written": False,
    }
    if not changed or dry_run:
        return summary

    payload = dict(arrays)
    payload["meta"] = np.array(json.dumps(meta, ensure_ascii=False))
    # Через временный файл: артефакт читает работающий бэкенд, и оборванная
    # запись поверх него оставила бы раздел тем без модели.
    temporary = model_path.with_suffix(model_path.suffix + ".tmp")
    with open(temporary, "wb") as handle:
        np.savez_compressed(handle, **payload)
    temporary.replace(model_path)
    summary["written"] = True
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="cluster_topics_labels.py",
        description="привести подписи тем в обученном артефакте к языкам интерфейса",
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    if not args.model.exists():
        print(f"артефакта нет: {args.model}", file=sys.stderr)
        return 2

    names = topic_names(args.data_dir)
    for _, language in FIELD_LANGUAGE:
        print(f"названий на языке {language}: {len(names[language])}")

    summary = enrich(args.model, names, dry_run=args.dry_run)
    for cluster, field, before, after in summary["changed"]:
        print(f"  кластер {cluster:>2} {field}: {before or '—'} -> {after}")
    if summary["unchanged"]:
        print(f"подписей уже на месте: {summary['unchanged']}")
    for cluster, topic_id, language in summary["missing"]:
        print(
            f"  ВНИМАНИЕ: кластеру {cluster} (тема {topic_id}) не нашлось "
            f"названия на языке {language}"
        )

    if summary["written"]:
        print(f"записано: {summary['path']}")
        print(
            "sha256 артефакта изменился — бэкенд зарегистрирует новую версию модели. "
            "Пока не пройдёт переразметка (POST /api/v1/topics/reassign), "
            "распределение показывает нули: назначения сделаны прошлой версией."
        )
    elif summary["changed"]:
        print("--dry-run: файл не тронут")
    else:
        print("менять нечего, файл не тронут")
    # Ненайденное название — не повод для аварии (артефакт остаётся рабочим),
    # но и не «всё хорошо»: отличный от нуля код виден в CI.
    return 1 if summary["missing"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
