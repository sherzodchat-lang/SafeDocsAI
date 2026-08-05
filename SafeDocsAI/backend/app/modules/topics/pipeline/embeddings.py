"""Мост «тексты корпуса -> матрица чисел» с кэшем на диске.

Кластеризация в этой работе — предмет защиты, а эмбеддинги — вход к ней, и
считаются они той же моделью, которой работает продукт (qwen3-embedding:8b
через Ollama). Собственного HTTP-клиента здесь нет намеренно: обращение к
Ollama живёт в ModelManager вместе с разбором отказов и запасными путями для
старых серверов, и вторая реализация рядом означала бы, что эксперимент и
продукт считают векторы по-разному.

Полный прогон корпуса занимает минуты, а эксперимент запускается десятки раз,
поэтому векторы кладутся на диск. Кэш ключуется парой (имя модели, id
документа): вектор от другой модели лежит в другом пространстве другой
размерности, и подставить его вместо нужного — это не «немного неточные
числа», а другая работа с прежними выводами в отчёте. Отсюда правило: при
несовпадении имени модели кэш считается пустым целиком, а не дополняется.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np

# Формат файла кэша. Версия записана внутрь, чтобы смена раскладки массивов
# ловилась так же, как смена модели, — отказом, а не молчаливым мусором.
CACHE_FORMAT_VERSION = 1

# Размер пачки для одного обращения к Ollama. /api/embed принимает массив, и
# ModelManager.embed уже передаёт список целиком, так что пачка — это один
# запрос, а не n запросов подряд.
#
# Замер на прогретой модели (qwen3-embedding:8b, тексты корпуса, медиана 171
# слово): 16 текстов — 4.5 с, 32 — 9.5 с, 64 — 17.3 с, то есть 3.4-3.7 док/с
# при любом размере пачки. Пропускная способность упирается в GPU, а не в
# раунд-трипы, поэтому укрупнять пачку дальше бессмысленно: выигрыша нет, а
# цена сорванного запроса растёт линейно (теряется вся пачка) и при большой
# пачке появляется риск упереться в контекстное окно. Параллельные запросы
# тоже не помогают: модель эмбеддингов поднимается с одним слотом независимо
# от OLLAMA_NUM_PARALLEL, и восемь одновременных вызовов оказались медленнее
# восьми последовательных. 32 выбрано как компромисс: запрос в десяток секунд
# даёт достаточно частый прогресс и не теряет много при обрыве.
DEFAULT_BATCH_SIZE = 32

# Как часто сбрасывать кэш на диск, в пачках. Прогон корпуса — минуты, и
# прерывание на середине не должно означать «начать сначала».
DEFAULT_FLUSH_EVERY = 10

EmbedFn = Callable[[Sequence[str]], list[list[float]]]


@dataclass
class EmbeddingCache:
    """Векторы документов, привязанные к имени модели.

    Хранится в .npz: массив id (строки), матрица векторов и строка метаданных.
    Порядок строк матрицы отвечает порядку id — это единственная связь между
    ними, поэтому обе величины всегда пишутся и читаются вместе.

    stats хранит, сколько векторов и за сколько секунд было посчитано. Число
    живёт в файле, а не в памяти прогона, потому что отвечает на вопрос
    защиты «сколько времени заняло эмбеддирование», а задают его на десятом
    запуске, когда всё уже читается из кэша и прогон длится ноль секунд.
    """

    model: str
    _vectors: dict[str, np.ndarray]
    stats: dict[str, float] = field(default_factory=lambda: {"computed": 0, "seconds": 0.0})

    @classmethod
    def empty(cls, model: str) -> "EmbeddingCache":
        return cls(model=model, _vectors={})

    @classmethod
    def load(cls, path: str | Path, model: str) -> "EmbeddingCache":
        """Читает кэш; при несовпадении модели или версии — возвращает пустой.

        Именно пустой, а не отказ: пересчитать векторы новой моделью — обычный
        сценарий, ради которого эксперимент и запускают. Отказ заставил бы
        удалять файл руками, а руками рано или поздно удалят не тот.
        """
        file = Path(path)
        if not file.exists():
            return cls.empty(model)

        with np.load(file, allow_pickle=False) as archive:
            meta = json.loads(str(archive["meta"]))
            if meta.get("version") != CACHE_FORMAT_VERSION:
                return cls.empty(model)
            if meta.get("model") != model:
                return cls.empty(model)
            ids = [str(value) for value in archive["ids"]]
            vectors = np.asarray(archive["vectors"], dtype=np.float32)

        if len(ids) != vectors.shape[0]:
            # Рассинхрон id и строк матрицы означает, что связь «id -> вектор»
            # потеряна и восстановить её нечем. Молча брать первые len(ids)
            # строк нельзя: это выдало бы чужие векторы за нужные.
            raise ValueError(
                f"кэш {file} повреждён: {len(ids)} id против {vectors.shape[0]} векторов"
            )
        stats = dict(meta.get("stats") or {})
        return cls(
            model=model,
            _vectors=dict(zip(ids, vectors)),
            stats={
                "computed": float(stats.get("computed", 0)),
                "seconds": float(stats.get("seconds", 0.0)),
            },
        )

    def __len__(self) -> int:
        return len(self._vectors)

    def __contains__(self, document_id: str) -> bool:
        return document_id in self._vectors

    def get(self, document_id: str) -> np.ndarray | None:
        return self._vectors.get(document_id)

    def put(self, document_id: str, vector: Sequence[float]) -> None:
        array = np.asarray(vector, dtype=np.float32)
        if array.ndim != 1:
            raise ValueError("вектор должен быть одномерным")
        if not np.all(np.isfinite(array)):
            # NaN отравил бы и нормировку, и центроиды; K-means на такой матрице
            # откажется работать, но уже после часа ожидания.
            raise ValueError(f"вектор документа {document_id} содержит NaN или inf")
        known = self.dim
        if known is not None and array.shape[0] != known:
            raise ValueError(
                f"размерность вектора {array.shape[0]} не совпадает с {known} "
                f"у остальных документов"
            )
        self._vectors[document_id] = array

    @property
    def dim(self) -> int | None:
        for vector in self._vectors.values():
            return int(vector.shape[0])
        return None

    def missing(self, document_ids: Iterable[str]) -> list[str]:
        return [value for value in document_ids if value not in self._vectors]

    def save(self, path: str | Path) -> None:
        """Атомарная запись: сначала во временный файл, потом переименование.

        Прогон длинный, и сброс кэша идёт по ходу дела. Оборванная запись
        поверх готового файла стоила бы всех уже посчитанных векторов.
        """
        file = Path(path)
        file.parent.mkdir(parents=True, exist_ok=True)
        ids = sorted(self._vectors)
        if ids:
            vectors = np.vstack([self._vectors[value] for value in ids])
        else:
            vectors = np.zeros((0, 0), dtype=np.float32)
        meta = json.dumps(
            {
                "version": CACHE_FORMAT_VERSION,
                "model": self.model,
                "count": len(ids),
                "dim": self.dim,
                "stats": self.stats,
            },
            ensure_ascii=False,
        )
        temporary = file.with_suffix(file.suffix + ".tmp")
        with open(temporary, "wb") as handle:
            np.savez_compressed(
                handle,
                # dtype="U" — фиксированная ширина юникода. allow_pickle=False
                # при чтении не пустит object-массив, а без него в npz строки
                # не кладутся вовсе.
                ids=np.array(ids, dtype="U") if ids else np.zeros(0, dtype="U1"),
                vectors=vectors,
                meta=np.array(meta),
            )
        temporary.replace(file)

    def matrix(self, document_ids: Sequence[str]) -> np.ndarray:
        """Матрица (len(document_ids), dim) в порядке переданных id.

        Порядок задаётся снаружи и повторяет порядок документов, потому что
        рядом с матрицей всегда идут колонки разметки. Взять строки в порядке
        словаря значило бы сдвинуть разметку относительно векторов — метрики
        при этом посчитаются, просто ответят про случайное разбиение.
        """
        absent = self.missing(document_ids)
        if absent:
            raise KeyError(
                f"в кэше нет векторов для {len(absent)} документов, "
                f"например {absent[:3]}"
            )
        if not document_ids:
            raise ValueError("пустой список документов")
        return np.vstack([self._vectors[value] for value in document_ids]).astype(np.float64)


def _format_eta(seconds: float) -> str:
    minutes, secs = divmod(int(max(seconds, 0.0)), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}ч {minutes:02d}м"
    return f"{minutes}м {secs:02d}с"


def embed_corpus(
    documents: Sequence,
    *,
    model: str,
    cache_path: str | Path,
    embed_fn: EmbedFn,
    batch_size: int = DEFAULT_BATCH_SIZE,
    flush_every: int = DEFAULT_FLUSH_EVERY,
    progress: Callable[[str], None] | None = print,
) -> EmbeddingCache:
    """Досчитывает недостающие векторы и возвращает кэш целиком.

    documents — любые объекты с полями .id и .text (в работе это
    dataset.Document; тесты передают что угодно с теми же полями, чтобы не
    поднимать Ollama ради проверки кэша).

    embed_fn — функция «список текстов -> список векторов». Передаётся снаружи,
    а не создаётся здесь, ровно ради тестов: живая модель отвечает минуты, и
    проверять на ней логику кэша невозможно.
    """
    if batch_size < 1:
        raise ValueError("batch_size должен быть >= 1")

    cache = EmbeddingCache.load(cache_path, model)
    known_before = len(cache)
    pending = [document for document in documents if document.id not in cache]

    def say(message: str) -> None:
        if progress is not None:
            progress(message)

    say(
        f"эмбеддинги: модель {model}, в кэше {known_before} из {len(documents)}, "
        f"считать {len(pending)}"
    )
    if not pending:
        return cache

    started = time.monotonic()
    done = 0
    batches = 0
    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]
        vectors = embed_fn([document.text for document in batch])
        if len(vectors) != len(batch):
            # Молчаливое расхождение длин сдвинуло бы соответствие «документ ->
            # вектор» на всю оставшуюся пачку.
            raise ValueError(
                f"модель вернула {len(vectors)} векторов на {len(batch)} текстов"
            )
        for document, vector in zip(batch, vectors):
            cache.put(document.id, vector)

        done += len(batch)
        batches += 1
        elapsed = time.monotonic() - started
        rate = done / elapsed if elapsed > 0 else 0.0
        remaining = (len(pending) - done) / rate if rate > 0 else 0.0
        say(
            f"  {done}/{len(pending)} за {_format_eta(elapsed)}, "
            f"{rate:.1f} док/с, осталось ~{_format_eta(remaining)}"
        )
        if batches % flush_every == 0:
            # Промежуточный сброс идёт без обновления stats: незавершённый
            # прогон ещё не знает своего времени, и записать половину значило бы
            # занизить ответ на вопрос «сколько заняло эмбеддирование».
            cache.save(cache_path)

    elapsed = time.monotonic() - started
    cache.stats = {
        "computed": float(cache.stats.get("computed", 0)) + float(done),
        "seconds": float(cache.stats.get("seconds", 0.0)) + float(elapsed),
    }
    cache.save(cache_path)
    say(
        f"эмбеддинги готовы: {len(cache)} векторов размерности {cache.dim}, "
        f"посчитано {done} за {_format_eta(elapsed)} ({done / elapsed:.2f} док/с)"
    )
    return cache


def ollama_embed_fn(model: str, timeout: float = 600.0) -> EmbedFn:
    """Готовая embed_fn поверх ModelManager — та же дорога, что и у продукта.

    Импорт внутри функции намеренно: модуль эмбеддингов читают и тесты, и
    сохранение модели, а ModelManager тянет за собой ollama и httpx. Держать
    эту зависимость на уровне модуля значило бы требовать HTTP-клиент там, где
    нужен один np.load.
    """
    from app.modules.rag.model_manager import ModelManager

    manager = ModelManager(timeout=timeout)

    def embed(texts: Sequence[str]) -> list[list[float]]:
        return manager.embed(texts, model=model)

    return embed
