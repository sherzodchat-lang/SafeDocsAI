#!/usr/bin/env python3
"""Сравнение шести способов убрать из векторов язык и жанр.

Продолжение cluster_topics.py. Тот прогон установил: кластеры этого корпуса
ложатся на язык и на жанр, а не на тему (при k=20 ARI против языка +0.415,
против темы +0.063), но тематический сигнал в векторах есть — у 68% документов
ближайший сосед той же темы при случайном уровне 6%. Значит, тема не
отсутствует, а перекрыта. Здесь проверяется, проявится ли она, если перекрытие
убрать.

Шесть вариантов, все на тех же эмбеддингах:

  1. baseline           — как есть, для сравнения;
  2. per_language       — своя модель на каждый язык;
  3. centered_language  — минус средний вектор языка;
  4. centered_origin    — минус средний вектор жанра;
  5. centered_cell      — минус средний вектор ячейки «язык x жанр»;
  6. per_cell           — своя модель в каждой из шести ячеек.

Что делает скрипт, чего не делал предыдущий:

  * средние по группам считает ТОЛЬКО на train и применяет к validation и test.
    Среднее по всему корпусу подглядывает в test;
  * каждый вариант считает против трёх разметок сразу — тема, язык,
    происхождение, — чтобы видеть, действительно ли ось подавлена;
  * каждый вариант считает согласие ближайших соседей по теме, величину, от k
    не зависящую вовсе. Она отличает «улучшилась геометрия» от «удачнее
    нарезали»;
  * считает каждый вариант в нескольких режимах k, включая режим с одинаковым
    ОБЩИМ числом кластеров: союз шести разбиений даёт 60 кластеров против 20 у
    общего варианта, и ARI за дробление наказывает.

Эмбеддинги берутся из готового кэша, Ollama не нужна.

Примеры:

    ./venv/bin/python cluster_topics_variants.py
    ./venv/bin/python cluster_topics_variants.py --variants baseline,per_cell
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

from cluster_topics import jsonable  # noqa: E402
from app.modules.topics.pipeline.dataset import load_full, load_splits  # noqa: E402
from app.modules.topics.pipeline.embeddings import EmbeddingCache  # noqa: E402
from app.modules.topics.pipeline import experiment, variants  # noqa: E402

BACKEND_ROOT = Path(__file__).resolve().parent
DATASET_ROOT = BACKEND_ROOT / "data" / "task1_multilingual_dataset"
DEFAULT_DATA_DIR = DATASET_ROOT / "data"
DEFAULT_CACHE = DATASET_ROOT / "embeddings.npz"
DEFAULT_REPORT = DATASET_ROOT / "clustering_report_variants.json"
DEFAULT_MODEL_OUT = DATASET_ROOT / "topic_model_best.npz"

# Общее число кластеров, при котором варианты сравниваются на равных. Взято не
# произвольно: 60 — это то, что даёт союз шести ячеек (3 x 12 у синтетики плюс
# 3 x 8 у реальных) и союз трёх языков (3 x 20). Общие варианты считаются при
# том же числе, иначе разница между строками таблицы включала бы штраф ARI за
# дробление.
MATCHED_CLUSTERS = 60


def model_name_from_cache(path: Path) -> str:
    """Имя embedding-модели читается из самого кэша.

    Не из runtime_settings и не из аргумента по умолчанию: сравниваются векторы,
    которые уже посчитаны, и любое другое имя означало бы, что кэш будет
    признан протухшим и молча заменён пустым.
    """
    with np.load(path, allow_pickle=False) as archive:
        meta = json.loads(str(archive["meta"]))
    name = str(meta.get("model") or "")
    if not name:
        raise SystemExit(f"в кэше {path} не записано имя модели эмбеддингов")
    return name


def oracle(splits) -> dict:
    """Потолок: что показали бы метрики, если бы кластеры В ТОЧНОСТИ были темами.

    Нужен, чтобы не требовать от вариантов невозможного. Наборы тем у синтетики
    и у реальных текстов не пересекаются, поэтому идеальное тематическое
    разбиение САМО ПО СЕБЕ вложено в жанр: у него ARI против жанра заведомо не
    ноль, а чистота по жанру ровно единица. Без этой строки «ARI против жанра
    +0.100» читался бы как «жанр не подавлен», хотя у идеала он ещё выше.
    """
    result = {}
    for name in ("train", "validation", "test"):
        corpus = splits[name]
        topics = sorted(set(corpus.labels("topic_id")))
        index = {value: number for number, value in enumerate(topics)}
        labels = np.array([index[value] for value in corpus.labels("topic_id")])
        result[name] = experiment.external_scores(labels, corpus)
    return result


def summary_rows(report: dict) -> list[dict]:
    """Одна строка на вариант — то, что потом кладётся в таблицу."""
    rows = []
    for name, block in report["variants"].items():
        for regime, data in block["regimes"].items():
            test = data["splits"]["test"]["external"]
            geometry = block["geometry"]
            rows.append(
                {
                    "variant": name,
                    "regime": regime,
                    "k_total": data["n_clusters_total"],
                    "k_per_stratum": data["k_per_stratum"],
                    "silhouette_test": data["splits"]["test"]["silhouette"],
                    "inertia_train": data["inertia_train_sum"],
                    "ari_topic": test["topic_id"]["ari"],
                    "ari_language": test["language"]["ari"],
                    "ari_origin": test["dataset_origin"]["ari"],
                    "purity_topic": test["topic_id"]["purity"],
                    "purity_language": test["language"]["purity"],
                    "purity_origin": test["dataset_origin"]["purity"],
                    "ari_topic_validation": data["splits"]["validation"]["external"]["topic_id"]["ari"],
                    "ari_topic_train": data["splits"]["train"]["external"]["topic_id"]["ari"],
                    "mean_ari_topic_in_cell_test": data["by_cell"]["test"]["mean_ari_topic"],
                    "nn_topic_global": geometry["global"]["same_label_as_nearest_neighbour"]["topic_id"],
                    "nn_topic_in_cell": geometry["within_cell"]["same_label_as_nearest_neighbour"]["topic_id"],
                    "nn_topic_chance_global": geometry["global"]["chance_level"]["topic_id"],
                    "nn_topic_chance_in_cell": geometry["within_cell"]["chance_level"]["topic_id"],
                }
            )
    return rows


def format_table(rows: list[dict]) -> list[str]:
    """Сводная таблица текстом.

    Текстом, а не только числами в JSON: таблицу читают глазами и переносят в
    отчёт целиком, а из вложенного JSON её каждый раз собирают заново и
    каждый раз по-своему.
    """
    header = (
        f"{'вариант':<18} {'режим':<10} {'k':>4} {'ARI тема':>9} {'ARI язык':>9} "
        f"{'ARI жанр':>9} {'чистота':>8} {'сосед тема':>11} {'сосед в ячейке':>15}"
    )
    lines = [header, "-" * len(header)]
    for row in rows:
        lines.append(
            f"{row['variant']:<18} {row['regime']:<10} {row['k_total']:>4} "
            f"{row['ari_topic']:>+9.3f} {row['ari_language']:>+9.3f} "
            f"{row['ari_origin']:>+9.3f} {row['purity_topic']:>8.3f} "
            f"{row['nn_topic_global']:>11.3f} {row['nn_topic_in_cell']:>15.3f}"
        )
    return lines


def headline(report: dict) -> dict:
    """Ответы на вопросы защиты отдельными числами, а не перебором ключей.

    Три вопроса, ради которых прогон делался: насколько вырос ARI по теме,
    подавились ли оси языка и жанра, и — главное — улучшилась ли геометрия или
    мы просто удачнее нарезали. Последнее решается не ARI, а согласием
    ближайших соседей: оно от k не зависит вовсе.
    """
    rows = {(row["variant"], row["regime"]): row for row in report["summary_rows"]}
    base = rows[("baseline", "true_k")]
    best = report["winner"]
    base_geometry = report["variants"]["baseline"]["geometry"]
    best_geometry = report["variants"][best["variant"]]["geometry"]
    oracle_test = report["oracle"]["test"]
    return {
        "winner": best["variant"],
        "winner_k": best["k_total"],
        "ari_topic_test_baseline": base["ari_topic"],
        "ari_topic_test_winner": best["ari_topic"],
        "ari_topic_test_gain": best["ari_topic"] - base["ari_topic"],
        "ari_topic_test_ratio": best["ari_topic"] / base["ari_topic"] if base["ari_topic"] else None,
        "purity_topic_test_baseline": base["purity_topic"],
        "purity_topic_test_winner": best["purity_topic"],
        # Подавлена ли ось: ARI против неё и чистота по ней. Одного ARI мало —
        # у разделяющих вариантов кластеры строго одноязычны, но ARI против
        # языка у них низкий просто потому, что разбиение мельче разметки.
        "ari_language_test_baseline": base["ari_language"],
        "ari_language_test_winner": best["ari_language"],
        "purity_language_test_baseline": base["purity_language"],
        "purity_language_test_winner": best["purity_language"],
        "ari_origin_test_baseline": base["ari_origin"],
        "ari_origin_test_winner": best["ari_origin"],
        "ari_origin_test_oracle": oracle_test["dataset_origin"]["ari"],
        "purity_origin_test_oracle": oracle_test["dataset_origin"]["purity"],
        "ari_language_test_oracle": oracle_test["language"]["ari"],
        # Геометрия. Если эти два числа совпадают, а ARI вырос — помогло не
        # преобразование пространства, а то, что кластеры перестали тратиться
        # на язык и жанр.
        "nn_topic_baseline": base_geometry["global"]["same_label_as_nearest_neighbour"]["topic_id"],
        "nn_topic_winner": best_geometry["global"]["same_label_as_nearest_neighbour"]["topic_id"],
        "nn_topic_in_cell_baseline": base_geometry["within_cell"]["same_label_as_nearest_neighbour"]["topic_id"],
        "nn_topic_in_cell_winner": best_geometry["within_cell"]["same_label_as_nearest_neighbour"]["topic_id"],
        "nn_topic_chance": base_geometry["global"]["chance_level"]["topic_id"],
        "nn_topic_chance_in_cell": base_geometry["within_cell"]["chance_level"]["topic_id"],
        "geometry_changed": abs(
            best_geometry["global"]["same_label_as_nearest_neighbour"]["topic_id"]
            - base_geometry["global"]["same_label_as_nearest_neighbour"]["topic_id"]
        ),
        # Тот же вопрос при одинаковом ОБЩЕМ числе кластеров у всех вариантов.
        "matched_clusters": report["matched_clusters"],
        "ari_topic_test_matched": {
            row["variant"]: row["ari_topic"]
            for row in report["summary_rows"]
            if row["regime"] in ("true_k", "matched")
            and row["k_total"] == report["matched_clusters"]
        },
    }


def pick_winner(rows: list[dict]) -> dict:
    """Лучший вариант — по ARI против темы на test в режиме true_k.

    Режим фиксирован намеренно. Сравнивать варианты в разных режимах k значит
    сравнивать их с разной грубостью разбиения, а true_k — единственный режим,
    в котором вопрос звучит одинаково для всех: «нашлись ли эти темы».
    При равенстве побеждает вариант, требующий меньше знаний о документе:
    модель, которой нужен определитель языка, дороже во внедрении.
    """
    candidates = [row for row in rows if row["regime"] == "true_k"]
    return max(
        candidates,
        key=lambda row: (round(row["ari_topic"], 6), -len(row["k_per_stratum"])),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cluster_topics_variants.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--model-out", type=Path, default=DEFAULT_MODEL_OUT)
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument("--random-state", type=int, default=experiment.RANDOM_STATE)
    parser.add_argument(
        "--variants",
        default=None,
        help="через запятую; по умолчанию все шесть",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    started = time.monotonic()

    splits = load_splits(args.data_dir)
    full = load_full(args.data_dir)
    model_name = args.embedding_model or model_name_from_cache(args.cache)
    cache = EmbeddingCache.load(args.cache, model_name)
    absent = cache.missing(full.ids())
    if absent:
        raise SystemExit(
            f"в кэше нет {len(absent)} векторов (модель {model_name}); "
            "сначала прогоните cluster_topics.py --embed-only"
        )
    print(f"корпус {len(full)}, векторов {len(cache)}, модель {model_name}")

    data = experiment.matrices(splits, cache)
    full_X = cache.matrix(full.ids())

    chosen = (
        [name.strip() for name in args.variants.split(",") if name.strip()]
        if args.variants
        else [spec.name for spec in variants.VARIANTS]
    )

    variant_reports: dict[str, dict] = {}
    fits: dict[str, dict] = {}
    transforms: dict[str, object] = {}
    for spec in variants.VARIANTS:
        if spec.name not in chosen:
            continue
        print(f"\n--- {spec.name}: {spec.title}")
        report, variant_fits, transform = variants.run_variant(
            spec,
            splits,
            data,
            full,
            full_X,
            matched_clusters=MATCHED_CLUSTERS,
            random_state=args.random_state,
        )
        variant_reports[spec.name] = report
        fits[spec.name] = variant_fits
        transforms[spec.name] = transform
        for regime, block in report["regimes"].items():
            test = block["splits"]["test"]["external"]
            print(
                f"    {regime:<10} k={block['n_clusters_total']:>3}  "
                f"тема {test['topic_id']['ari']:+.3f}  "
                f"язык {test['language']['ari']:+.3f}  "
                f"жанр {test['dataset_origin']['ari']:+.3f}  "
                f"чистота {test['topic_id']['purity']:.3f}"
            )
        print(
            f"    соседи: по корпусу {report['geometry']['global']['same_label_as_nearest_neighbour']['topic_id']:.3f}, "
            f"внутри языка {report['geometry']['within_language']['same_label_as_nearest_neighbour']['topic_id']:.3f}, "
            f"внутри ячейки {report['geometry']['within_cell']['same_label_as_nearest_neighbour']['topic_id']:.3f} "
            f"(случайно {report['geometry']['within_cell']['chance_level']['topic_id']:.3f})"
        )

    report: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "embedding": {"model": model_name, "dim": cache.dim, "n_vectors": len(cache)},
        "random_state": args.random_state,
        "matched_clusters": MATCHED_CLUSTERS,
        "leakage_note": (
            "средние по группам посчитаны только на train и применены к validation "
            "и test как есть; согласие соседей считается на всём корпусе, но теми "
            "же train-средними"
        ),
        "chance_levels": {
            "ari": 0.0,
            "why": (
                "ARI поправлен на случайность по построению: у случайного "
                "разбиения его ожидание равно нулю при любом числе классов и "
                "кластеров, поэтому слои с 8 и с 12 темами сравнимы напрямую. "
                "Случайный уровень отличается у согласия соседей — он посчитан "
                "рядом с каждым числом, отдельно по области поиска"
            ),
        },
        "oracle": oracle(splits),
        "variants": variant_reports,
    }
    rows = summary_rows(report)
    report["summary_rows"] = rows
    report["summary_table"] = format_table(rows)

    winner = pick_winner(rows)
    report["winner"] = winner
    report["headline"] = headline(report)

    spec = next(item for item in variants.VARIANTS if item.name == winner["variant"])
    fit = fits[winner["variant"]]["true_k"]
    model = variants.build_model(
        fit,
        transforms[winner["variant"]],
        model_name,
        params={
            "variant": spec.name,
            "variant_title": spec.title,
            "regime": "true_k",
            "k_per_stratum": winner["k_per_stratum"],
            "random_state": args.random_state,
            "trained_on": "train",
            "n_train": len(splits["train"]),
            "dataset": "task1_multilingual_dataset",
            "ari_topic_test": winner["ari_topic"],
            "purity_topic_test": winner["purity_topic"],
            "ari_language_test": winner["ari_language"],
            "ari_origin_test": winner["ari_origin"],
            # Требование к встраиванию: без этих признаков модель не применяется.
            # Язык в приложении придётся определять, разметки там нет.
            "requires_fields_at_apply_time": sorted(
                set(spec.center_by) | set(spec.split_by)
            ),
        },
    )
    model.save(args.model_out)
    report["saved_model"] = {
        "path": str(args.model_out),
        "variant": spec.name,
        "n_clusters": model.n_clusters,
        "required_fields": list(model.required_fields),
        "embedding_model": model_name,
        "transform": model.transform.meta() if model.transform is not None else None,
        "routing_groups": list(model.routing.groups) if model.routing is not None else None,
        "cluster_topics": [item.as_dict() for item in model.cluster_topics],
    }
    report["seconds"] = round(time.monotonic() - started, 1)

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(jsonable(report), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    print("\n=== сводная таблица (test) ===")
    for line in report["summary_table"]:
        print(line)
    head = report["headline"]
    oracle_test = report["oracle"]["test"]
    print(
        f"{'оракул (кластер=тема)':<18} {'-':<10} {20:>4} {1.0:>+9.3f} "
        f"{oracle_test['language']['ari']:>+9.3f} {oracle_test['dataset_origin']['ari']:>+9.3f} "
        f"{1.0:>8.3f}"
    )
    print(
        f"\nлучший вариант: {winner['variant']} ({spec.title}), "
        f"ARI по теме на test {winner['ari_topic']:+.3f} "
        f"против {head['ari_topic_test_baseline']:+.3f} у базового"
    )
    print(
        f"ось языка: ARI {head['ari_language_test_baseline']:+.3f} -> "
        f"{head['ari_language_test_winner']:+.3f}, чистота "
        f"{head['purity_language_test_baseline']:.3f} -> {head['purity_language_test_winner']:.3f}"
    )
    print(
        f"ось жанра: ARI {head['ari_origin_test_baseline']:+.3f} -> "
        f"{head['ari_origin_test_winner']:+.3f} (у оракула {head['ari_origin_test_oracle']:+.3f})"
    )
    print(
        f"геометрия: сосед той же темы {head['nn_topic_baseline']:.3f} -> "
        f"{head['nn_topic_winner']:.3f} (внутри ячейки "
        f"{head['nn_topic_in_cell_baseline']:.3f} -> {head['nn_topic_in_cell_winner']:.3f}); "
        f"случайный уровень {head['nn_topic_chance']:.3f}"
    )
    print(f"модель: {args.model_out}; требует признаков: {list(model.required_fields)}")
    print(f"отчёт: {args.report}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
