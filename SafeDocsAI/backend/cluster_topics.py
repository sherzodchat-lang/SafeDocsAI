#!/usr/bin/env python3
"""Кластеризация документов по темам: прогон эксперимента от корпуса до отчёта.

Что делает скрипт по шагам:

  1. читает готовые разбиения корпуса task1_multilingual_dataset;
  2. считает эмбеддинги той же моделью, которой работает продукт
     (qwen3-embedding:8b через Ollama), и кладёт их в кэш на диске — повторный
     запуск ничего не пересчитывает;
  3. перебирает k собственной реализацией choose_k и печатает таблицу
     «k / инерция / силуэт» для графика локтя;
  4. обучает K-means на train, назначает validation и test;
  5. считает метрики против ЧЕТЫРЁХ разметок сразу — тема, подтема, язык,
     происхождение — отдельно по слоям (синтетика / реальные / смесь);
  6. повторяет сравнение слоёв при ЗАКРЕПЛЁННОМ языке, по шести ячейкам
     «происхождение x язык»;
  7. пишет машиночитаемый отчёт и сохранённую модель.

Пункты 5 и 6 — главные, и порознь они обманывают. Ожидание было такое:
синтетическая половина сгенерирована по темам, разделима по построению, и
метрика на ней завышена, а разрыв между слоями — честный ответ на вопрос «а не
потому ли разделилось, что вы сами так и написали».

На этом корпусе пункт 5 такого разрыва не показывает, и не потому, что его
нет: кластеры внутри каждого слоя уходят на ЯЗЫК раньше, чем на тему, и оба
слоя выглядят одинаково плохо по причине, к слоям отношения не имеющей.
Разрыв проявляется только в пункте 6, где язык закреплён, — и оказывается
обратным ожидаемому. Поэтому в отчёте лежат оба числа, а не одно удобное.

Примеры:

    # полный прогон (первый раз считает эмбеддинги, минут десять)
    ./venv/bin/python cluster_topics.py

    # только эмбеддинги, без обучения — прогреть кэш
    ./venv/bin/python cluster_topics.py --embed-only

    # эксперимент на готовом кэше, свой набор k
    ./venv/bin/python cluster_topics.py --k-values 10,20,30 --no-embed
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

from app.modules.topics.pipeline.dataset import load_full, load_splits  # noqa: E402
from app.modules.topics.pipeline.embeddings import (  # noqa: E402
    DEFAULT_BATCH_SIZE,
    EmbeddingCache,
    embed_corpus,
    ollama_embed_fn,
)
from app.modules.topics.pipeline import experiment  # noqa: E402

BACKEND_ROOT = Path(__file__).resolve().parent
DATASET_ROOT = BACKEND_ROOT / "data" / "task1_multilingual_dataset"
DEFAULT_DATA_DIR = DATASET_ROOT / "data"
DEFAULT_CACHE = DATASET_ROOT / "embeddings.npz"
DEFAULT_REPORT = DATASET_ROOT / "clustering_report.json"
DEFAULT_MODEL_OUT = DATASET_ROOT / "topic_model.npz"
DEFAULT_KSEARCH_CSV = DATASET_ROOT / "choose_k.csv"

# Диапазон перебора k. Обязан накрывать 20 (столько настоящих тем) с запасом в
# обе стороны: если бы 20 стояло на краю, «выбрали 20, потому что дальше не
# смотрели» было бы справедливым упрёком. Верхний край 40 выбран как удвоенное
# число тем — там уже должно быть видно, что дробление не окупается; нижний
# край 2 показывает самое грубое деление, на котором обычно и видно, по какому
# признаку корпус распадается в первую очередь.
DEFAULT_K_VALUES = (2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 35, 40)

# Число настоящих тем. Прогон при этом k считается всегда, независимо от того,
# что выбрал choose_k: он отвечает на отдельный вопрос — «а если бы число тем
# было известно заранее, нашлись бы они?».
N_TRUE_TOPICS = 20


def resolve_embedding_model(explicit: str | None) -> str:
    """Имя embedding-модели тем же порядком, что и у продукта.

    Порядок (runtime_settings.json -> OLLAMA_MODEL_EMBEDDING -> отказ) взят не
    из головы: по нему модель выбирает ChromaGateway и воркер индексации.
    Собственное умолчание здесь означало бы, что эксперимент считает векторы
    одной моделью, а продукт — другой, и сравнивать их числа было бы нельзя.
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


def parse_k_values(raw: str | None) -> tuple[int, ...]:
    if not raw:
        return DEFAULT_K_VALUES
    values = tuple(sorted({int(part) for part in raw.replace(" ", "").split(",") if part}))
    if not values:
        raise SystemExit("--k-values не содержит ни одного числа")
    return values


def write_ksearch_csv(path: Path, table: list[dict]) -> None:
    """Таблица локтя отдельным файлом.

    В JSON-отчёте она тоже есть, но график рисуют не из JSON: csv открывается
    любым инструментом без единой строки кода, а спрашивают на защите именно
    график.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["k,inertia,silhouette,n_iter"]
    lines += [
        f"{row['k']},{row['inertia']:.6f},{row['silhouette']:.6f},{row['n_iter']}"
        for row in table
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def dataset_summary(splits, full) -> dict:
    return {
        "n_documents": len(full),
        "splits": {name: len(corpus) for name, corpus in splits.items()},
        "by_language": full.counts("language"),
        "by_origin": full.counts("dataset_origin"),
        "by_topic": full.counts("topic_id"),
        "n_topics": len(full.counts("topic_id")),
        "n_subtopics": len(full.counts("subtopic_id")),
        "origin_by_split": {
            name: corpus.counts("dataset_origin") for name, corpus in splits.items()
        },
    }


def headline(report: dict) -> dict:
    """Ответы на вопросы защиты, вынутые из отчёта отдельными числами.

    Вынуто наверх намеренно: если ARI по языку выше, чем ARI по теме, это
    результат работы, а не деталь, и он должен читаться первым, а не
    находиться перебором вложенных ключей.
    """
    main_test = report["main"]["splits"]["test"]["external"]
    reference_test = report["reference_true_k"]["splits"]["test"]["external"]
    layers = report["layers"]
    synthetic = layers["synthetic"]["splits"]["test"]["external"]["topic_id"]
    real = layers["real"]["splits"]["test"]["external"]["topic_id"]
    cells = report["layers_language_controlled"]
    geometry = report["embedding_geometry"]["same_label_as_nearest_neighbour"]
    return {
        "chosen_k": report["chosen_k"]["value"],
        "ari_topic_test": main_test["topic_id"]["ari"],
        "ari_language_test": main_test["language"]["ari"],
        "ari_origin_test": main_test["dataset_origin"]["ari"],
        "ari_subtopic_test": main_test["subtopic_id"]["ari"],
        "ari_topic_and_language_test": main_test["topic_and_language"]["ari"],
        "clusters_explained_better_by": max(
            ("topic_id", main_test["topic_id"]["ari"]),
            ("language", main_test["language"]["ari"]),
            ("dataset_origin", main_test["dataset_origin"]["ari"]),
            ("subtopic_id", main_test["subtopic_id"]["ari"]),
            key=lambda item: item[1],
        )[0],
        # Те же вопросы при k = числу настоящих тем. Нужны отдельно: если
        # choose_k выберет далёкое от 20 k, ответ «тема или язык» на нём будет
        # относиться к другому по грубости разбиению, и упрёк «вы просто взяли
        # не то k» останется без числа.
        "true_k": report["reference_true_k"]["k"],
        "ari_topic_test_at_true_k": reference_test["topic_id"]["ari"],
        "ari_language_test_at_true_k": reference_test["language"]["ari"],
        "ari_origin_test_at_true_k": reference_test["dataset_origin"]["ari"],
        "ari_subtopic_test_at_true_k": reference_test["subtopic_id"]["ari"],
        "purity_topic_test_at_true_k": reference_test["topic_id"]["purity"],
        "synthetic_vs_real_gap_ari": synthetic["ari"] - real["ari"],
        "synthetic_ari_test": synthetic["ari"],
        "real_ari_test": real["ari"],
        "synthetic_purity_test": synthetic["purity"],
        "real_purity_test": real["purity"],
        # Тот же разрыв, но при закреплённом языке. Именно это число отвечает
        # на упрёк «разделилось, потому что вы сами это написали»: разрыв выше
        # считается на трёх языках сразу, и кластеры там уходят на язык, из-за
        # чего оба слоя выглядят одинаково плохо и разрыв выходит около нуля.
        "synthetic_ari_test_per_language": cells["by_layer"]["synthetic"]["test"][
            "mean_ari_topic"
        ],
        "real_ari_test_per_language": cells["by_layer"]["real"]["test"]["mean_ari_topic"],
        "synthetic_vs_real_gap_ari_per_language": cells["gap_ari_topic"]["test"],
        # Согласие ближайших соседей: есть ли тематический сигнал в векторах
        # вообще, независимо от выбора k.
        "nearest_neighbour_same_topic": geometry["topic_id"],
        "nearest_neighbour_same_language": geometry["language"],
        "nearest_neighbour_same_origin": geometry["dataset_origin"],
    }


def print_summary(report: dict) -> None:
    head = report["headline"]
    print("\n=== коротко ===")
    print(f"k выбрано: {head['chosen_k']} ({report['chosen_k']['why']})")
    print(f"ARI на test против темы:           {head['ari_topic_test']:+.3f}")
    print(f"ARI на test против языка:          {head['ari_language_test']:+.3f}")
    print(f"ARI на test против происхождения:  {head['ari_origin_test']:+.3f}")
    print(f"ARI на test против подтемы:        {head['ari_subtopic_test']:+.3f}")
    print(f"кластеры лучше объясняются: {head['clusters_explained_better_by']}")
    print(
        f"при k={head['true_k']} (число настоящих тем): тема {head['ari_topic_test_at_true_k']:+.3f}, "
        f"язык {head['ari_language_test_at_true_k']:+.3f}, "
        f"происхождение {head['ari_origin_test_at_true_k']:+.3f}, "
        f"подтема {head['ari_subtopic_test_at_true_k']:+.3f}"
    )
    print(
        f"слои (ARI по теме, test): синтетика {head['synthetic_ari_test']:+.3f}, "
        f"реальные {head['real_ari_test']:+.3f}, "
        f"разрыв {head['synthetic_vs_real_gap_ari']:+.3f}"
    )
    print(
        f"слои при закреплённом языке: синтетика "
        f"{head['synthetic_ari_test_per_language']:+.3f}, реальные "
        f"{head['real_ari_test_per_language']:+.3f}, разрыв "
        f"{head['synthetic_vs_real_gap_ari_per_language']:+.3f}"
    )
    print(
        f"ближайший сосед той же метки: тема {head['nearest_neighbour_same_topic']:.3f}, "
        f"язык {head['nearest_neighbour_same_language']:.3f}, "
        f"происхождение {head['nearest_neighbour_same_origin']:.3f}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cluster_topics.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="каталог с full_*.jsonl (по умолчанию %(default)s)",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=DEFAULT_CACHE,
        help="файл кэша эмбеддингов; ключуется именем модели (по умолчанию %(default)s)",
    )
    parser.add_argument(
        "--report", type=Path, default=DEFAULT_REPORT, help="куда писать JSON-отчёт"
    )
    parser.add_argument(
        "--model-out", type=Path, default=DEFAULT_MODEL_OUT, help="куда писать модель"
    )
    parser.add_argument(
        "--ksearch-csv",
        type=Path,
        default=DEFAULT_KSEARCH_CSV,
        help="куда писать таблицу локтя в csv",
    )
    parser.add_argument(
        "--embedding-model",
        default=None,
        help="имя модели эмбеддингов; по умолчанию берётся так же, как берёт продукт",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="сколько текстов в одном обращении к Ollama (по умолчанию %(default)s)",
    )
    parser.add_argument(
        "--k-values",
        default=None,
        help=f"через запятую; по умолчанию {','.join(str(k) for k in DEFAULT_K_VALUES)}",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help="взять k напрямую, минуя выбор по таблице",
    )
    parser.add_argument(
        "--random-state", type=int, default=experiment.RANDOM_STATE, help="seed"
    )
    parser.add_argument(
        "--embed-only",
        action="store_true",
        help="только посчитать эмбеддинги и выйти",
    )
    parser.add_argument(
        "--no-embed",
        action="store_true",
        help="не ходить в Ollama; при неполном кэше — отказ вместо тихого пропуска",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    splits = load_splits(args.data_dir)
    full = load_full(args.data_dir)
    print(
        f"корпус: {len(full)} документов, "
        + ", ".join(f"{name} {len(corpus)}" for name, corpus in splits.items())
    )

    model_name = resolve_embedding_model(args.embedding_model)

    if args.no_embed:
        cache = EmbeddingCache.load(args.cache, model_name)
        absent = cache.missing(full.ids())
        if absent:
            # Тихо продолжить на неполном кэше нельзя: часть документов просто
            # выпала бы из выборки, и метрики описывали бы другой корпус.
            raise SystemExit(
                f"--no-embed, но в кэше нет {len(absent)} векторов "
                f"(модель {model_name}); уберите флаг или укажите другой --cache"
            )
    else:
        cache = embed_corpus(
            full.documents,
            model=model_name,
            cache_path=args.cache,
            embed_fn=ollama_embed_fn(model_name),
            batch_size=args.batch_size,
        )

    if args.embed_only:
        print(f"кэш: {len(cache)} векторов, {args.cache}")
        return 0

    data = experiment.matrices(splits, cache)
    print(f"матрицы собраны: train {data['train'].shape}, test {data['test'].shape}")

    k_values = parse_k_values(args.k_values)
    print(f"перебор k: {list(k_values)} (это долго — по одному K-means на каждое k)")
    started = time.monotonic()
    k_search = experiment.k_search_table(
        data["train"], k_values, random_state=args.random_state
    )
    print(f"перебор занял {k_search['seconds']:.0f} с")
    for row in k_search["table"]:
        print(
            f"  k={row['k']:>3}  инерция={row['inertia']:10.2f}  "
            f"силуэт={row['silhouette']:+.4f}  итераций={row['n_iter']}"
        )

    if args.k is not None:
        chosen = {"value": int(args.k), "why": "задано ключом --k"}
    else:
        chosen = {
            "value": int(k_search["best_k_by_silhouette"]),
            "why": (
                f"максимум силуэта в таблице; локоть инерции указывает на "
                f"k={k_search['elbow_k']}"
            ),
        }
    print(f"выбрано k={chosen['value']}: {chosen['why']}")

    fitted, main_report = experiment.run_full_model(
        splits, data, chosen["value"], random_state=args.random_state
    )
    if chosen["value"] == N_TRUE_TOPICS:
        reference = main_report
    else:
        _, reference = experiment.run_full_model(
            splits, data, N_TRUE_TOPICS, random_state=args.random_state
        )

    layers = experiment.run_layers(splits, data, random_state=args.random_state)
    cells = experiment.run_cells(splits, data, random_state=args.random_state)
    within = {
        name: experiment.layers_within_global(fitted, splits[name], data[name])
        for name in ("train", "validation", "test")
    }
    geometry = experiment.neighbour_agreement(full, cache.matrix(full.ids()))

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "embedding": {
            "model": model_name,
            "dim": cache.dim,
            "n_vectors": len(cache),
            "batch_size": args.batch_size,
            "seconds_spent_computing": round(float(cache.stats.get("seconds", 0.0)), 1),
            "vectors_computed": int(cache.stats.get("computed", 0)),
            "cache_path": str(args.cache),
        },
        "dataset": dataset_summary(splits, full),
        "random_state": args.random_state,
        "k_search": k_search,
        "chosen_k": chosen,
        "main": main_report,
        "reference_true_k": reference,
        "layers": layers,
        "layers_language_controlled": cells,
        "layers_within_global_model": within,
        "embedding_geometry": geometry,
        # Только счёт: эмбеддирование сюда не входит, его время лежит рядом в
        # embedding.seconds_spent_computing. Складывать их в одно число нельзя —
        # эмбеддинги считаются один раз, а эксперимент прогоняется десятки.
        "experiment_seconds": round(time.monotonic() - started, 1),
    }
    report["headline"] = headline(report)

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(jsonable(report), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    write_ksearch_csv(args.ksearch_csv, k_search["table"])

    topic_model = experiment.build_topic_model(
        fitted,
        model_name,
        params={
            "k": chosen["value"],
            "random_state": args.random_state,
            "trained_on": "train",
            "n_train": len(splits["train"]),
            "inertia": fitted.kmeans.inertia_,
            "n_iter": fitted.kmeans.n_iter_,
            "dataset": "task1_multilingual_dataset",
        },
    )
    topic_model.save(args.model_out)

    print(f"\nотчёт: {args.report}")
    print(f"таблица локтя: {args.ksearch_csv}")
    print(f"модель: {args.model_out}")
    print_summary(report)
    return 0


def jsonable(value):
    """Отчёт -> структура, которую JSON примет без оговорок.

    Две вещи, из-за которых обычный json.dumps здесь не годится.

    NaN. Силуэт не определён, когда кластер один (сравнивать не с чем), и
    метрики честно возвращают NaN. json.dumps по умолчанию пишет литерал NaN,
    которого в стандарте JSON нет: файл откроется питоном и не откроется ничем
    другим. Пишется null — «величины нет». Подменять нулём нельзя: ноль
    означал бы «плохое разбиение», а не «неприменимо».

    numpy-числа. np.int64 и np.float64 приезжают из меток и метрик и не
    сериализуются; default= до них не всегда доходит, потому что часть из них
    сидит ключами словарей.
    """
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return None if np.isnan(number) or np.isinf(number) else number
    if isinstance(value, np.bool_):
        return bool(value)
    return value


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
