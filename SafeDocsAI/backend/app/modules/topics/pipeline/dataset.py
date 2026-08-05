"""Чтение размеченного корпуса task1_multilingual_dataset.

Отдельный модуль, а не разбор json прямо в скрипте эксперимента, по одной
причине: разметка здесь не одна, а четыре сразу — тема (topic_id), подтема
(subtopic_id), язык (language) и происхождение (dataset_origin). Весь смысл
работы в том, чтобы сравнить кластеры со ВСЕМИ четырьмя и увидеть, какой из
них кластеризация на самом деле воспроизводит. Значит, читать корпус должен
один код, отдающий все четыре колонки одинаково выровненными по документам;
иначе рано или поздно ARI посчитается по языку в одном порядке, а по теме — в
другом, и расхождение никто не заметит.

Готовые разбиения (full_train / full_validation / full_test) берутся как есть.
Резать full_multilingual самому нельзя: в файлах уже сбалансированы язык и
тема, а случайная нарезка сломала бы этот баланс молча.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

# Имена файлов разбиения. Собраны в одном месте, потому что рядом лежат
# похожие train.jsonl / test.jsonl от ПРЕДЫДУЩЕЙ версии корпуса (только
# реальные тексты, без синтетики) — перепутать их легко, а разница в 840
# документов обнаружилась бы только по метрикам.
SPLIT_FILES = {
    "train": "full_train.jsonl",
    "validation": "full_validation.jsonl",
    "test": "full_test.jsonl",
}

FULL_FILE = "full_multilingual.jsonl"

SYNTHETIC = "synthetic"
REAL = "real"


@dataclass(frozen=True)
class Document:
    """Документ корпуса — только то, что нужно кластеризации и метрикам.

    Остальные поля исходного jsonl (title, source_url, format, style, ...) в
    структуру не переносятся намеренно: они есть не у всех записей — у реальных
    выдержек нет format/style, у синтетики нет source_url, — и код, читающий их
    через .get, начал бы молча работать с None там, где ждал строку.
    """

    id: str
    text: str
    language: str
    topic_id: str
    topic: str
    subtopic_id: str
    dataset_origin: str
    split: str
    word_count: int

    @property
    def is_synthetic(self) -> bool:
        return self.dataset_origin == SYNTHETIC


@dataclass(frozen=True)
class Corpus:
    """Набор документов плюс выровненные по ним колонки разметки.

    Колонки отдаются методами, а не хранятся: список документов — единственный
    источник порядка, и любая колонка строится из него на месте. Хранить их
    рядом означало бы завести второй источник истины и следить за его
    синхронностью при каждом subset().
    """

    documents: tuple[Document, ...]

    def __len__(self) -> int:
        return len(self.documents)

    def __iter__(self):
        return iter(self.documents)

    def ids(self) -> list[str]:
        return [document.id for document in self.documents]

    def texts(self) -> list[str]:
        return [document.text for document in self.documents]

    def labels(self, field: str) -> list[str]:
        """Колонка разметки по имени поля — topic_id, language, ...

        Именно по имени, а не отдельным методом на каждое поле: сравнение
        «кластеры против темы / против языка / против жанра» — это один и тот
        же расчёт, отличающийся только колонкой, и три копии расчёта разошлись
        бы при первой же правке.
        """
        return [str(getattr(document, field)) for document in self.documents]

    def subset(self, predicate) -> "Corpus":
        return Corpus(tuple(d for d in self.documents if predicate(d)))

    def synthetic(self) -> "Corpus":
        return self.subset(lambda d: d.dataset_origin == SYNTHETIC)

    def real(self) -> "Corpus":
        return self.subset(lambda d: d.dataset_origin == REAL)

    def counts(self, field: str) -> dict[str, int]:
        result: dict[str, int] = {}
        for value in self.labels(field):
            result[value] = result.get(value, 0) + 1
        return dict(sorted(result.items()))


def _document_from_record(record: dict) -> Document:
    missing = [
        name
        for name in (
            "id",
            "text",
            "language",
            "topic_id",
            "topic",
            "subtopic_id",
            "dataset_origin",
            "split",
        )
        if not record.get(name)
    ]
    if missing:
        # Падать здесь, а не подставлять пустую строку: документ без topic_id
        # попал бы в метрики как отдельная «тема», состоящая из мусора, и
        # занизил бы ARI по причине, которую потом не найти.
        raise ValueError(f"в записи нет обязательных полей {missing}: {record.get('id')!r}")
    return Document(
        id=str(record["id"]),
        text=str(record["text"]),
        language=str(record["language"]),
        topic_id=str(record["topic_id"]),
        topic=str(record["topic"]),
        subtopic_id=str(record["subtopic_id"]),
        dataset_origin=str(record["dataset_origin"]),
        split=str(record["split"]),
        word_count=int(record.get("word_count") or 0),
    )


def load_jsonl(path: str | Path) -> Corpus:
    """Читает один jsonl-файл корпуса."""
    documents: list[Document] = []
    with open(path, encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{number}: не разбирается как JSON") from exc
            documents.append(_document_from_record(record))
    return Corpus(tuple(documents))


def load_splits(data_dir: str | Path) -> dict[str, Corpus]:
    """Три готовых разбиения плюс проверка, что они не пересекаются.

    Пересечение train и test — самая дорогая ошибка в такой работе: метрики на
    test окажутся завышены, и заметить это по самим числам невозможно, они
    просто будут «хорошими». Поэтому проверка здесь, а не в отчёте.
    """
    directory = Path(data_dir)
    splits = {name: load_jsonl(directory / file) for name, file in SPLIT_FILES.items()}

    seen: dict[str, str] = {}
    for name, corpus in splits.items():
        for document in corpus:
            if document.split != name:
                raise ValueError(
                    f"документ {document.id} лежит в {name}, "
                    f"а его поле split = {document.split!r}"
                )
            if document.id in seen:
                raise ValueError(
                    f"документ {document.id} встречается и в {seen[document.id]}, и в {name}"
                )
            seen[document.id] = name
    return splits


def load_full(data_dir: str | Path) -> Corpus:
    """Весь корпус одним куском — для слоёв и для эмбеддирования.

    Эмбеддинги считаются один раз для всех 2278 документов, а не по разбиениям:
    вектор документа от разбиения не зависит, а три отдельных прогона стоили бы
    трёх ожиданий Ollama.
    """
    return load_jsonl(Path(data_dir) / FULL_FILE)
