"""Прикладной слой раздела тем: артефакт модели, назначение, реестр, очередь.

Здесь заканчивается обученная модель и начинается продукт. Файл отвечает на
четыре вопроса, и каждый из них решён так, а не иначе, по своей причине.

**Как применяется обученная модель.** Не «сравнить вектор с центроидами», а
«ПРЕОБРАЗОВАТЬ вектор так же, как это сделали при обучении, и только потом
сравнить». На этом корпусе кластеризация в лоб делит документы по языку и
жанру, а не по теме, поэтому победивший вариант вычитает из вектора среднее по
его языку. Слой назначения, забывший про преобразование, не упал бы и не
пожаловался: он сравнил бы сырой вектор с центроидами другого пространства и
выдал бы правдоподобные номера кластеров. Поэтому преобразование читается из
артефакта, а НЕИЗВЕСТНОЕ преобразование — это отказ применять модель вовсе
(TopicModelUnusable), а не тихий переход к сравнению как есть.

Победивший артефакт центрирует по ячейке «язык × происхождение», а у боевого
документа происхождения нет: синтетика существует только внутри обучающего
корпуса. Отсюда единственное допущение этого файла — dataset_origin=real, — из-за
которого шесть средних превращаются в три; оно названо и объяснено у
PRODUCTION_FIXED_FIELDS. Поле, за которым такого постоянного значения не
закреплено, по-прежнему означает отказ.

**Где живёт модель и где реестр.** Центроиды — в файле-артефакте: матрица
4096-мерных векторов нужна целиком и только слою назначения, в PostgreSQL ей
делать нечего. В базе лежит реестр версий (TopicModelVersion): он отвечает,
какая модель активна, когда её обучили и — главное — какой версией размечен
конкретный документ. Артефакт перезаписывается по одному и тому же пути, и без
реестра переобучение молча выдавало бы старые назначения за новые.

**Почему отказ в теме не отказ в индексации.** Тема — украшение документа;
поиск по документу — основная функция продукта. Недоступная ChromaDB,
незаданная embedding-модель, необученная модель тем не имеют права уронить
загрузку источника, поэтому назначение после индексации не бросает наружу
НИЧЕГО (assign_after_indexing ловит даже то, чего не ждёт).

**Почему переразметка — задача в общей таблице job.** Ей нужно ровно то же,
что очереди индексации: пережить перезапуск, захватываться атомарно при
uvicorn --workers 2, показывать прогресс. Всё это уже сделано в JobsService
(claim_next с FOR UPDATE SKIP LOCKED, heartbeat, аренда, бюджет попыток), и
вторая реализация того же разошлась бы с первой на первой же правке. Свой у
раздела только цикл-воркер: переразметка ходит по всем документам сразу, и в
одной очереди с индексацией она задержала бы загрузку файлов на своё время —
тот же довод, по которому отдельный воркер получили презентации.

sklearn здесь запрещён так же, как в kmeans.py: сторож
tests/test_topics_no_sklearn_guard.py разбирает все файлы этого каталога.
Предсказание идёт через ту же KMeans, что и обучение, — своя копия «найти
ближайший центроид» разошлась бы с оригиналом на первой правке нормировки.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from fastapi.concurrency import run_in_threadpool
from sqlalchemy import text
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.core.database import (
    TOPIC_MODEL_ACTIVE_INDEX,
    TOPIC_REASSIGN_JOB_TYPE,
    session_context,
)
from app.core.exceptions import TopicErrors
from app.modules.topics.kmeans import KMeans, l2_normalize
from app.shared.models import Chunk, Document, TopicModelVersion, utcnow

logger = logging.getLogger(__name__)


# --- Где лежит артефакт ---------------------------------------------------
#
# Путь относительный, как у хранилища презентаций (PRESENTATION_STORAGE_DIR):
# он разрешается от рабочего каталога процесса, а тот у бэкенда — backend/.
# Значение по умолчанию указывает туда, куда пишет обучающий скрипт
# (backend/cluster_topics.py, --model-out), поэтому на рабочем стенде ничего
# настраивать не нужно.
#
# Переменная окружения нужна не ради гибкости, а ради тестов и ради второго
# стенда: подменять её приходится в каждом тесте назначения тем.
TOPIC_MODEL_PATH_ENV = "TOPIC_MODEL_PATH"

# Умолчание указывает на ПОБЕДИВШУЮ модель, а не на первую обученную.
#
# topic_model.npz — базовый прогон без преобразования, k выбран по максимуму
# силуэта. На этом корпусе он даёт k=4 и делит документы не по темам, а по языку
# и жанру: ARI против языка +0.415 против +0.063 по теме, а само решение при k=4
# буквально «реальные-en, реальные-ru, реальные-tg, вся синтетика».
#
# topic_model_best.npz — центрирование по ячейке «язык × жанр», k=20: тема
# +0.191 (втрое выше), ось языка исчезает (+0.002). Разбор — docs/kmeans-topics.md.
#
# Держать умолчанием базовую модель значило бы раздавать пользователям темы,
# которые на самом деле не темы. Базовый артефакт остаётся на диске: он нужен
# для сравнения в отчёте, но не для боевой разметки.
DEFAULT_TOPIC_MODEL_PATH = "data/task1_multilingual_dataset/topic_model_best.npz"


def topic_model_path() -> Path:
    """Путь к артефакту. Читается на каждый вызов, а не на импорт.

    Значение, снятое на импорте, невозможно подменить ни в тесте, ни на
    работающем стенде: модуль импортируется один раз за жизнь процесса.
    """
    return Path(os.environ.get(TOPIC_MODEL_PATH_ENV) or DEFAULT_TOPIC_MODEL_PATH)


# --- Преобразование вектора ----------------------------------------------
#
# Четыре вида, и каждый обязан быть назван в артефакте явно. Умолчания «если не
# написано — значит ничего не делаем» здесь нет намеренно: артефакт, обученный
# с вычитанием среднего, но забывший про это написать, получил бы молчаливое
# сравнение сырого вектора с чужими центроидами. Отсутствие ключа считается
# отсутствием преобразования только тогда, когда это единственное прочтение
# (см. parse_transform).

TRANSFORM_NONE = "none"
# Вычесть одно и то же среднее из любого вектора. Убирает общую составляющую
# корпуса — то, что одинаково во всех документах и потому ничего не различает.
TRANSFORM_MEAN_SHIFT = "mean_shift"
# Вычесть среднее СВОЕЙ группы (у нас — своего языка). Ради этого вида всё и
# затевалось: без него кластеры совпадают с языками, а не с темами.
#
# Своего производителя у этого вида в проекте нет: обучающая сторона пишет
# group_centering (см. ниже). Он остаётся читаемым, потому что артефакт кладут
# на диск отдельно от выкладки бэкенда, и уже лежащий файл не должен погаснуть
# от обновления кода.
TRANSFORM_GROUP_MEAN_SHIFT = "group_mean_shift"
# То же вычитание среднего группы, но так, как его называет и сохраняет
# обучающая сторона (pipeline/transforms.py, GroupCentering): группа задаётся
# СПИСКОМ полей, ключи составные («en|real»), и после вычитания вектор
# нормируется заново.
TRANSFORM_GROUP_CENTERING = "group_centering"

KNOWN_TRANSFORMS = (
    TRANSFORM_NONE,
    TRANSFORM_MEAN_SHIFT,
    TRANSFORM_GROUP_MEAN_SHIFT,
    TRANSFORM_GROUP_CENTERING,
)

# Единственный признак, по которому слой назначения умеет выбрать группу
# ПО ДОКУМЕНТУ. Строка, а не молчаливое допущение: артефакт, группирующий по
# чему-то ещё, должен получить внятный отказ, а не среднее по языку.
GROUP_FIELD_LANGUAGE = "language"

# Разделитель составного ключа группы. То же значение, что KEY_SEPARATOR в
# pipeline/transforms.py, и разъехаться им нельзя: этим символом обучающая
# сторона склеивает «en|real», а этот файл — разбирает.
GROUP_KEY_SEPARATOR = "|"

# --- Допущение о боевых значениях полей группы -----------------------------
#
# СУЩЕСТВЕННОЕ ДОПУЩЕНИЕ, и оно здесь одно.
#
# Победивший вариант центрирует вектор по ячейке «язык × происхождение» и
# сохраняет шесть средних: en|real, en|synthetic, ru|real, ru|synthetic,
# tg|real, tg|synthetic. Языку документа соответствие есть — Document.language.
# Происхождения у боевого документа нет и быть не может: dataset_origin — это
# признак ОБУЧАЮЩЕГО корпуса, где половину текстов сгенерировали, чтобы добрать
# объём. Документ, который пользователь загрузил в продукт, синтетическим не
# бывает по определению: он пришёл из внешнего мира, а не из генератора.
#
# Поэтому при загрузке артефакта шесть средних проецируются на три: остаются
# ячейки с dataset_origin=real, и ключом становится один язык (en|real -> en).
# Средние ветки |synthetic в бою не нужны — применить их не к чему.
#
# Словарь, а не константа в коде: поле, которого здесь нет, зафиксировать
# нечем, и артефакт, группирующий по нему, получает честный отказ (см.
# _parse_group_centering). «Возьмём что-нибудь похожее» здесь запрещено ровно
# по той же причине, по которой запрещено неизвестное преобразование: молчаливо
# выбранное чужое среднее даёт правдоподобный номер кластера и неверную тему.
PRODUCTION_FIXED_FIELDS: dict[str, str] = {"dataset_origin": "real"}

# Код таджикского языка внутри проекта — "tj" (определитель языка, доменные
# профили, колоды презентаций), а в датасете и, значит, в артефакте — "tg"
# (ISO 639-1). Расхождение известное и уже описанное в presentations/constants.py
# (HTML_LANG), новым решением здесь оно не является.
#
# Без этой строки таджикский документ не находил бы своего среднего и тихо
# получал бы запасное (общее по корпусу): преобразование выродилось бы в
# обычное глобальное центрирование ровно для того языка, ради которого продукт
# и делается, и заметить это было бы нечем — номер кластера всё равно
# посчитался бы.
LANGUAGE_GROUP_ALIASES: dict[str, str] = {"tj": "tg"}


def unit_vector(vector: np.ndarray) -> np.ndarray:
    """Привести вектор к единичной длине тем же кодом, что и обучение.

    l2_normalize из kmeans.py, а не своя строка с делением на норму: у неё
    внутри обработка нулевого вектора, и вторая реализация разошлась бы с
    первой ровно там, где расхождение не видно.
    """
    return l2_normalize(np.asarray(vector, dtype=np.float64).reshape(1, -1))[0]


def _group_candidates(group: str | None) -> tuple[str, ...]:
    """Ключи, под которыми среднее этой группы может лежать в артефакте."""
    key = str(group or "")
    alias = LANGUAGE_GROUP_ALIASES.get(key)
    return (key,) if alias is None else (key, alias)


class TopicModelUnusable(Exception):
    """Артефакт прочитан, но применить его нельзя.

    Отдельно от «файла нет»: пустой раздел и раздел с неверными темами — это
    разные беды. Первая видна пользователю как отсутствие функции, вторая — как
    работающая функция, которая врёт.
    """


class TopicEmbeddingUnavailable(Exception):
    """Вектор документа получить не удалось.

    Несёт машинный код, потому что причина попадает в результат фоновой задачи
    и в журнал: «пропущено 240 документов» без причины неотличимо от «модель
    считает их бестемными».
    """

    error_code = TopicErrors.EMBEDDING_UNAVAILABLE


@dataclass(frozen=True)
class TopicTransform:
    """Преобразование вектора перед сравнением с центроидами."""

    kind: str
    # mean_shift: вычитаемое. group_mean_shift и group_centering: запасное
    # вычитаемое для группы, которой в модели нет (документ на четвёртом языке).
    mean: np.ndarray | None = None
    group_field: str = GROUP_FIELD_LANGUAGE
    group_means: dict[str, np.ndarray] | None = None
    # Привести вход к единичной длине ПЕРЕД вычитанием. Не украшение: средние
    # групп посчитаны на нормированных эмбеддингах (обучающий кэш хранит
    # векторы единичной длины), а вектор документа в бою — СРЕДНЕЕ векторов его
    # фрагментов, и длина у него меньше единицы тем сильнее, чем разнороднее
    # документ. Вычесть среднее единичных векторов из вектора длины 0.8 —
    # значит сдвинуть его дальше, чем сдвигало обучение, и получить другое
    # направление, то есть другой кластер. Ошибка молчаливая: номер всё равно
    # посчитается.
    unit_input: bool = False
    # Нормировать заново ПОСЛЕ вычитания — так делает обучающая сторона
    # (GroupCentering.apply_to_keys), и артефакт говорит об этом полем
    # renormalize. Само назначение от этого не меняется, пока сравнение
    # косинусное (KMeans(normalize=True) нормирует вход predict'а сам), но
    # соблюдать объявленное преобразование обязан читатель, а не случайное
    # совпадение двух нормировок: артефакт с normalize=false тем же кодом
    # молча уехал бы в другую геометрию.
    renormalize: bool = False
    # Поля группы, зафиксированные боевым значением (dataset_origin=real).
    # Хранятся ради описания: пользователь и журнал должны видеть, что модель
    # группировала не только по языку.
    fixed_fields: tuple[tuple[str, str], ...] = ()

    @property
    def description(self) -> str:
        """Строка для API и для журнала: что именно делает преобразование."""
        if self.kind in (TRANSFORM_GROUP_MEAN_SHIFT, TRANSFORM_GROUP_CENTERING):
            # Зафиксированные поля названы прямо в подписи: «по языку» и «по
            # языку, при dataset_origin=real» — это разные преобразования, и
            # различать их в реестре надо без чтения кода.
            fixed = "".join(
                f"; {field}={value}" for field, value in self.fixed_fields
            )
            return f"{self.kind}({self.group_field}{fixed})"
        return self.kind

    def apply(self, vector: np.ndarray, *, group: str | None) -> np.ndarray:
        prepared = np.asarray(vector, dtype=np.float64)
        if self.unit_input:
            prepared = unit_vector(prepared)
        if self.kind == TRANSFORM_NONE:
            return prepared
        if self.kind == TRANSFORM_MEAN_SHIFT:
            shift = self.mean
        else:
            # Группировка: незнакомая группа — не повод отказаться от темы,
            # но и вычитать чужое среднее нельзя. Берём общее, если оно есть,
            # иначе оставляем вектор как есть: это худшее из двух назначений,
            # но всё же назначение, а документ на неизвестном языке — редкость,
            # ради которой не стоит гасить функцию целиком.
            means = self.group_means or {}
            shift = None
            for candidate in _group_candidates(group):
                shift = means.get(candidate)
                if shift is not None:
                    break
            if shift is None:
                shift = self.mean
        if shift is not None:
            prepared = prepared - shift
        return unit_vector(prepared) if self.renormalize else prepared


def _as_vector(value: Any) -> np.ndarray | None:
    array = np.asarray(value, dtype=np.float64)
    return None if array.ndim != 1 or array.size == 0 else array


def _fixed_fields_text() -> str:
    return ", ".join(f"{field}={value}" for field, value in PRODUCTION_FIXED_FIELDS.items())


def _parse_group_centering(
    raw: dict[str, Any], arrays: dict[str, np.ndarray], *, fallback: np.ndarray | None
) -> TopicTransform:
    """Разобрать преобразование обучающей стороны и спроецировать его на бой.

    Артефакт группирует по СПИСКУ полей («язык × происхождение»), а слой
    назначения знает у документа одно — язык. Разница закрывается не догадкой, а
    определением: остальные поля обязаны иметь в бою постоянное значение
    (PRODUCTION_FIXED_FIELDS), и тогда средние по ячейкам с этим значением —
    ровно те, которые к боевому документу применимы. Все прочие ячейки
    выбрасываются: применить их не к чему.

    Поле, которого в PRODUCTION_FIXED_FIELDS нет, — отказ. Такой артефакт
    группировал по признаку, которого у боевого документа не определить и не
    зафиксировать, и любое среднее для него было бы выбрано наугад.
    """
    fields = [str(value) for value in (raw.get("fields") or [])]
    if not fields:
        raise TopicModelUnusable(
            "преобразование group_centering объявлено без transform.fields: "
            "по какому признаку выбирать среднее — неизвестно"
        )
    if GROUP_FIELD_LANGUAGE not in fields:
        raise TopicModelUnusable(
            f"группировка по {fields} не поддерживается: у документа слой "
            f"назначения знает только {GROUP_FIELD_LANGUAGE!r}, а его среди "
            "полей группы нет"
        )
    unsupported = [
        field
        for field in fields
        if field != GROUP_FIELD_LANGUAGE and field not in PRODUCTION_FIXED_FIELDS
    ]
    if unsupported:
        raise TopicModelUnusable(
            f"группировка по {unsupported} не поддерживается: в бою этих полей "
            "у документа нет и постоянного значения за ними не закреплено "
            f"(известны {GROUP_FIELD_LANGUAGE!r} и {_fixed_fields_text()})"
        )

    keys = [str(value) for value in (raw.get("keys") or [])]
    means: dict[str, np.ndarray] = {}
    inline = raw.get("means")
    if isinstance(inline, dict):
        for key, value in inline.items():
            vector = _as_vector(value)
            if vector is not None:
                means[str(key)] = vector
    matrix = arrays.get("transform_means")
    if matrix is not None:
        if matrix.ndim != 2 or matrix.shape[0] != len(keys):
            raise TopicModelUnusable(
                "transform_means не согласован с transform.keys: "
                f"{getattr(matrix, 'shape', None)} против {len(keys)} групп"
            )
        for index, key in enumerate(keys):
            means[key] = np.asarray(matrix[index], dtype=np.float64)
    if not means:
        raise TopicModelUnusable(
            "преобразование group_centering объявлено, но средних по группам "
            "нет ни в метаданных (transform.means), ни в архиве "
            "(transform_means + transform.keys)"
        )

    language_at = fields.index(GROUP_FIELD_LANGUAGE)
    projected: dict[str, np.ndarray] = {}
    dropped: list[str] = []
    for key, vector in means.items():
        parts = key.split(GROUP_KEY_SEPARATOR)
        if len(parts) != len(fields):
            raise TopicModelUnusable(
                f"ключ группы {key!r} не разбирается по полям {fields}: "
                f"частей {len(parts)}, а полей {len(fields)}"
            )
        cell = dict(zip(fields, parts))
        # Все поля, кроме языка, проверены выше: у каждого есть боевое значение.
        production = all(
            cell[field] == PRODUCTION_FIXED_FIELDS[field]
            for field in fields
            if field != GROUP_FIELD_LANGUAGE
        )
        if production:
            projected[parts[language_at]] = vector
        else:
            dropped.append(key)
    if not projected:
        raise TopicModelUnusable(
            f"ни одна группа артефакта не описывает боевой документ: ключи "
            f"{sorted(means)} при боевых значениях {_fixed_fields_text()}"
        )

    if dropped:
        # Громко, потому что допущение существенное: из шести средних боевыми
        # оказываются три, и человек, читающий журнал, обязан узнать об этом
        # отсюда, а не из чужой головы.
        logger.info(
            "Артефакт тем группирует по %s. В бою %s (синтетики среди "
            "загруженных документов не бывает), поэтому из %s средних "
            "оставлены %s и ключуются языком: %s. Не нужны в бою: %s.",
            ", ".join(fields),
            _fixed_fields_text(),
            len(means),
            len(projected),
            ", ".join(sorted(projected)),
            ", ".join(sorted(dropped)),
        )

    fallback_mean = arrays.get("transform_fallback")
    if fallback_mean is None:
        fallback_mean = _as_vector(raw["fallback"]) if raw.get("fallback") is not None else None
    if fallback_mean is None:
        fallback_mean = fallback
    return TopicTransform(
        kind=TRANSFORM_GROUP_CENTERING,
        mean=fallback_mean,
        group_field=GROUP_FIELD_LANGUAGE,
        group_means=projected,
        # Средние посчитаны на нормированных векторах — см. unit_input.
        unit_input=True,
        # Умолчание True: обучающая сторона считает перенормировку частью
        # преобразования (GroupCentering.renormalize=True по умолчанию), и
        # артефакт без этого ключа обучен ею же.
        renormalize=bool(raw.get("renormalize", True)),
        fixed_fields=tuple(
            (field, PRODUCTION_FIXED_FIELDS[field])
            for field in fields
            if field in PRODUCTION_FIXED_FIELDS
        ),
    )


def parse_transform(meta: dict[str, Any], arrays: dict[str, np.ndarray]) -> TopicTransform:
    """Собрать преобразование из метаданных артефакта.

    Числа принимаются двумя путями сразу — прямо в JSON и отдельными массивами
    в том же .npz. Не из любви к вариантам: JSON проще всего написать обучающей
    стороне, а массив на 4096 чисел в JSON — это сотни килобайт текста, который
    ещё и теряет точность. Читатель обязан понимать оба, потому что выбирает
    формат не он.

    Неизвестный вид преобразования — TopicModelUnusable, а не «сделаем ничего».
    Это единственное место файла, где ошибка предпочтительнее работы.
    """
    raw = meta.get("transform")
    if raw is None:
        raw = (meta.get("params") or {}).get("transform")
    if raw is None:
        # Ключа нет вовсе — артефакт обучен без преобразования. Это честное
        # прочтение, а не догадка: вид преобразования называется явно, и
        # отсутствие названия означает отсутствие преобразования.
        return TopicTransform(kind=TRANSFORM_NONE)
    if isinstance(raw, str):
        raw = {"kind": raw}
    if not isinstance(raw, dict):
        raise TopicModelUnusable(f"описание преобразования не разобрано: {raw!r}")

    kind = str(raw.get("kind") or TRANSFORM_NONE)
    if kind not in KNOWN_TRANSFORMS:
        raise TopicModelUnusable(
            f"неизвестное преобразование {kind!r}: применить модель нечем "
            f"(известны {', '.join(KNOWN_TRANSFORMS)})"
        )

    mean = _as_vector(raw["mean"]) if raw.get("mean") is not None else None
    if mean is None:
        mean = arrays.get("transform_mean")

    if kind == TRANSFORM_MEAN_SHIFT and mean is None:
        raise TopicModelUnusable(
            "преобразование mean_shift объявлено, но вычитаемого вектора нет "
            "ни в метаданных (transform.mean), ни в архиве (transform_mean)"
        )

    if kind == TRANSFORM_GROUP_CENTERING:
        return _parse_group_centering(raw, arrays, fallback=mean)

    if kind != TRANSFORM_GROUP_MEAN_SHIFT:
        return TopicTransform(kind=kind, mean=mean)

    group_field = str(raw.get("group_field") or GROUP_FIELD_LANGUAGE)
    if group_field != GROUP_FIELD_LANGUAGE:
        raise TopicModelUnusable(
            f"группировка по {group_field!r} не поддерживается: слой назначения "
            f"знает у документа только {GROUP_FIELD_LANGUAGE!r}"
        )

    group_means: dict[str, np.ndarray] = {}
    inline = raw.get("means")
    if isinstance(inline, dict):
        for key, value in inline.items():
            vector = _as_vector(value)
            if vector is not None:
                group_means[str(key)] = vector
    matrix = arrays.get("transform_group_means")
    groups = raw.get("groups")
    if matrix is not None and isinstance(groups, (list, tuple)):
        if matrix.ndim != 2 or matrix.shape[0] != len(groups):
            raise TopicModelUnusable(
                "transform_group_means не согласован с transform.groups: "
                f"{getattr(matrix, 'shape', None)} против {len(groups)} групп"
            )
        for index, key in enumerate(groups):
            group_means[str(key)] = np.asarray(matrix[index], dtype=np.float64)

    if not group_means:
        raise TopicModelUnusable(
            "преобразование group_mean_shift объявлено, но средних по группам "
            "нет ни в метаданных (transform.means), ни в архиве "
            "(transform_group_means + transform.groups)"
        )
    return TopicTransform(
        kind=kind, mean=mean, group_field=group_field, group_means=group_means
    )


# --- Артефакт --------------------------------------------------------------


@dataclass(frozen=True)
class TopicArtifact:
    """Всё, что нужно, чтобы назначить документу тему."""

    centroids: np.ndarray
    embedding_model: str
    normalize: bool
    transform: TopicTransform
    labels: dict[int, str]
    # Подписи тех же кластеров на языках интерфейса: {"ru": {0: "Налоги"}, ...}.
    # Отдельно от labels, а не вместо: английская подпись устойчива (она же
    # лежит в артефакте, в отчётах эксперимента и в назначениях прошлых
    # версий), переводы же есть не у каждого артефакта — старые файлы их просто
    # не содержат. Кластер без перевода в словарь НЕ попадает: пустая строка
    # тут означала бы «подпись есть, но пустая», и клиент показал бы безымянную
    # строку вместо отката.
    labels_localized: dict[str, dict[int, str]]
    k: int
    metrics: dict[str, float | None]
    trained_at: datetime
    digest: str
    path: str

    @property
    def cluster_count(self) -> int:
        return int(self.centroids.shape[0])

    @property
    def dim(self) -> int:
        return int(self.centroids.shape[1])

    def label_of(self, cluster: int) -> str:
        return self.labels.get(int(cluster)) or default_label(cluster)

    def label_in(self, cluster: int, language: str) -> str | None:
        """Подпись кластера на языке интерфейса или None, если её в артефакте нет.

        None, а не default_label: «Кластер 7» уже отдаётся основной подписью, и
        вторая такая же строка не добавила бы ничего, зато лишила бы клиента
        возможности отличить «перевода нет» от «он совпал с оригиналом».
        """
        return (self.labels_localized.get(language) or {}).get(int(cluster)) or None

    def assign(self, vector: np.ndarray, *, group: str | None) -> int:
        """Номер ближайшего кластера для одного вектора документа.

        Предсказание идёт через ту же KMeans, что и обучение: у неё внутри и
        нормировка, и проверка размерности. Своя копия «argmin по расстояниям»
        разошлась бы с оригиналом на первой же правке геометрии, и расхождение
        проявилось бы только в продакшене — метками, которые вроде бы
        посчитались.
        """
        prepared = self.transform.apply(
            np.asarray(vector, dtype=np.float64), group=group
        )
        model = KMeans(n_clusters=self.cluster_count, normalize=self.normalize)
        model.centroids_ = self.centroids
        return int(model.predict(prepared.reshape(1, -1))[0])


def default_label(cluster: int) -> str:
    """Подпись кластера, у которого её нет в артефакте.

    Пустая строка здесь недопустима: пользователь увидел бы безымянную строку
    распределения и не смог бы отличить её от соседней. Номер — плохая тема, но
    честная.
    """
    return f"Кластер {int(cluster)}"


def _extract_labels(meta: dict[str, Any], cluster_count: int) -> dict[int, str]:
    """Подписи кластеров из артефакта.

    Читается ровно то, что кладёт обучающая сторона (cluster_topics: cluster,
    topic). Кластер без имени получает номер, а не пропускается: раздел обязан
    уметь подписать ЛЮБОЙ номер, который может вернуть predict, — в том числе
    пустой кластер, который на обучающей выборке не набрал ни одного документа.
    """
    labels: dict[int, str] = {}
    for item in meta.get("cluster_topics") or []:
        if not isinstance(item, dict):
            continue
        try:
            cluster = int(item.get("cluster"))
        except (TypeError, ValueError):
            continue
        name = str(item.get("topic") or item.get("topic_id") or "").strip()
        if name:
            labels[cluster] = name
    return {index: labels.get(index, default_label(index)) for index in range(cluster_count)}


# Языки интерфейса, для которых у темы бывает своя подпись. Список закрыт не
# здесь, а в продукте: экраны переведены на русский и таджикский, английского
# интерфейса нет вовсе. Ключ каждой темы (поле topic артефакта) при этом
# остаётся английским — он не для показа, а для ссылок и сверок.
#
# Порядок значения не имеет, но кортеж, а не множество: он ходит в JSON
# метаданных и в ответ API, и переменный порядок ключей мешал бы сравнивать
# ответы глазами.
LOCALIZED_LABEL_LANGUAGES = ("ru", "tg")

# Как поле перевода называется в артефакте: topic + суффикс языка.
_LOCALIZED_LABEL_FIELDS = {language: f"topic_{language}" for language in LOCALIZED_LABEL_LANGUAGES}


def _extract_labels_localized(
    meta: dict[str, Any], cluster_count: int
) -> dict[str, dict[int, str]]:
    """Переводы подписей из артефакта — только те, что там есть.

    Отсутствующие НЕ добираются номером кластера, как это делает
    _extract_labels: там подпись обязана быть у любого номера, потому что она
    единственная, а здесь она вторая. Кластер без перевода просто выпадает из
    словаря, и клиент показывает основную подпись — это и есть честный откат, а
    не подмена.

    Переводов нет ни у одного кластера, если артефакт обучен кодом до их
    появления. Раздел от этого не ломается: он возвращается ровно в то
    состояние, в котором был.
    """
    labels: dict[str, dict[int, str]] = {language: {} for language in LOCALIZED_LABEL_LANGUAGES}
    for item in meta.get("cluster_topics") or []:
        if not isinstance(item, dict):
            continue
        try:
            cluster = int(item.get("cluster"))
        except (TypeError, ValueError):
            continue
        if not 0 <= cluster < cluster_count:
            continue
        for language, field in _LOCALIZED_LABEL_FIELDS.items():
            name = str(item.get(field) or "").strip()
            if name:
                labels[language][cluster] = name
    return labels


_METRIC_ALIASES = {
    # Ключ ответа API -> имена, под которыми число может лежать в артефакте.
    # Псевдонимы нужны потому, что метрики считает эксперимент, а не этот слой:
    # он называет их так, как удобно отчёту, и требовать от него единственного
    # написания значило бы ронять регистрацию модели из-за имени ключа.
    #
    # Имена с суффиксом _test — то, как называет свои числа эксперимент с
    # вариантами (pipeline/variants.py): метрика победителя меряется на
    # ОТЛОЖЕННОЙ выборке, и именно её честно показывать в карточке модели.
    "ari_topic": ("ari_topic", "ari_topic_test", "ari", "adjusted_rand_index"),
    "purity": ("purity", "purity_topic_test"),
    "silhouette": ("silhouette", "silhouette_test", "silhouette_score"),
}


def _extract_metrics(meta: dict[str, Any]) -> dict[str, float | None]:
    sources: list[dict[str, Any]] = []
    for candidate in (meta.get("metrics"), (meta.get("params") or {}).get("metrics"), meta.get("params")):
        if isinstance(candidate, dict):
            sources.append(candidate)
    result: dict[str, float | None] = {}
    for key, aliases in _METRIC_ALIASES.items():
        value: float | None = None
        for source in sources:
            for alias in aliases:
                raw = source.get(alias)
                if isinstance(raw, (int, float)) and not isinstance(raw, bool):
                    number = float(raw)
                    # NaN в JSON доезжает как float('nan') и в ответе API
                    # превращается в невалидный JSON. Метрика, которую не
                    # посчитали, — это null, а не «нечисло».
                    value = None if number != number else number
                    break
            if value is not None:
                break
        result[key] = value
    return result


def _extract_trained_at(meta: dict[str, Any], path: Path) -> datetime:
    raw = meta.get("trained_at") or (meta.get("params") or {}).get("trained_at")
    if isinstance(raw, str):
        with contextlib.suppress(ValueError):
            parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            if parsed.tzinfo is not None:
                parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
            return parsed
    # Даты обучения в артефакте нет — берём время файла. Это не то же самое,
    # но это честное «когда модель здесь появилась», и оно всё равно точнее
    # текущего момента: тот сдвигался бы на каждом перезапуске.
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).replace(
        tzinfo=None
    )


def file_digest(path: Path) -> str:
    """sha256 артефакта.

    По содержимому, а не по mtime: обычное копирование файла меняет время, а
    перерегистрация модели обесценивает ВСЕ назначения разом — они начинают
    считаться сделанными прошлой версией. Платить за это стоит только тогда,
    когда модель действительно другая.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_artifact(path: Path | None = None) -> TopicArtifact:
    """Прочитать артефакт с диска.

    Читается npz напрямую, а не через
    app.modules.topics.pipeline.model_io.TopicModel.load. Тот загрузчик —
    контракт ОБУЧАЮЩЕЙ стороны и версионирован вместе с ней: он отказывается
    читать формат другой версии, потому что при обучении это правильно
    (несовпадение версий там означает, что эксперимент воспроизводится не тем
    кодом). У слоя выдачи задача обратная: артефакт кладут отдельно от
    выкладки бэкенда, и поднятая версия формата не должна гасить раздел,
    который умеет прочитать нужные ему поля.

    Строгость при этом не потеряна, а перенесена туда, где ошибка тихая: на
    преобразование (parse_transform). Незнакомая версия формата — повод для
    предупреждения, незнакомое преобразование — повод для отказа.
    """
    file = Path(path) if path is not None else topic_model_path()
    with np.load(file, allow_pickle=False) as archive:
        keys = list(archive.keys())
        if "centroids" not in keys or "meta" not in keys:
            raise TopicModelUnusable(
                f"в артефакте {file} нет обязательных ключей centroids и meta "
                f"(есть: {', '.join(keys) or 'ничего'})"
            )
        centroids = np.asarray(archive["centroids"], dtype=np.float64)
        meta_raw = str(archive["meta"])
        arrays = {
            name: np.asarray(archive[name], dtype=np.float64)
            for name in keys
            if name not in ("centroids", "meta")
        }

    try:
        meta = json.loads(meta_raw)
    except ValueError as exc:
        raise TopicModelUnusable(f"метаданные артефакта {file} не разобраны: {exc}") from exc
    if not isinstance(meta, dict):
        raise TopicModelUnusable(f"метаданные артефакта {file} не объект")
    if centroids.ndim != 2 or centroids.size == 0:
        raise TopicModelUnusable(
            "центроиды должны быть непустой матрицей (n_clusters, n_features)"
        )

    embedding_model = str(meta.get("embedding_model") or "").strip()
    if not embedding_model:
        # Без имени модели эмбеддингов сверять пространство не с чем, а вектор
        # от чужой модели даёт правдоподобный, но бессмысленный номер кластера.
        raise TopicModelUnusable(
            f"в артефакте {file} не указана embedding-модель: назначать темы "
            "по векторам неизвестного происхождения нельзя"
        )

    transform = parse_transform(meta, arrays)
    cluster_count = int(centroids.shape[0])
    params = meta.get("params") if isinstance(meta.get("params"), dict) else {}
    raw_k = params.get("k", meta.get("n_clusters", cluster_count))
    try:
        k = int(raw_k)
    except (TypeError, ValueError):
        k = cluster_count

    return TopicArtifact(
        centroids=centroids,
        embedding_model=embedding_model,
        # normalize отсутствующим считается ИСТИНОЙ, а не ложью: так обучают
        # эмбеддинги во всём проекте (KMeans(normalize=True) в experiment.py),
        # и умолчание False молча сменило бы геометрию сравнения.
        normalize=bool(meta.get("normalize", True)),
        transform=transform,
        labels=_extract_labels(meta, cluster_count),
        labels_localized=_extract_labels_localized(meta, cluster_count),
        k=k,
        metrics=_extract_metrics(meta),
        trained_at=_extract_trained_at(meta, file),
        digest=file_digest(file),
        path=str(file),
    )


# Артефакт читается с диска один раз на версию: переразметка зовёт назначение
# для каждого документа, и чтение npz на каждый из них означало бы тысячи
# лишних открытий файла. Ключ — дайджест: перезаписанный файл получает другой
# ключ и перечитывается сам.
_ARTIFACT_CACHE: dict[str, TopicArtifact] = {}


def load_artifact_cached(path: Path | None = None) -> TopicArtifact:
    file = Path(path) if path is not None else topic_model_path()
    key = f"{file}:{file.stat().st_mtime_ns}:{file.stat().st_size}"
    cached = _ARTIFACT_CACHE.get(key)
    if cached is not None:
        return cached
    artifact = load_artifact(file)
    _ARTIFACT_CACHE.clear()
    _ARTIFACT_CACHE[key] = artifact
    return artifact


def forget_cached_artifacts() -> None:
    """Забыть прочитанные артефакты. Нужно тестам и смене пути на ходу."""
    _ARTIFACT_CACHE.clear()


# --- Векторы документа ----------------------------------------------------

# Сколько id спрашивать у ChromaDB за один вызов. Значение того же порядка, что
# ADD_BATCH_SIZE при записи, но больше: чтение не считает эмбеддинги, оно
# только достаёт готовые.
CHROMA_GET_BATCH = 200


def _collection():
    """Активная коллекция ChromaDB или отказ с машинным кодом.

    Импорт локальный: модуль тем не должен тянуть ChromaDB и Ollama при
    импорте — этим он ломал бы сторожа, который проверяет, что раздел тем
    ничего тяжёлого за собой не тащит, и удорожал бы старт.
    """
    from app.core.exceptions import EmbeddingModelNotConfigured
    from app.modules.rag.service import RAGService

    try:
        service = RAGService()
    except EmbeddingModelNotConfigured as exc:
        raise TopicEmbeddingUnavailable(
            "embedding-модель не задана: коллекции с векторами не существует"
        ) from exc
    if service.collection is None:
        raise TopicEmbeddingUnavailable(
            f"ChromaDB недоступна: {service.chroma_error}"
        )
    return service.collection


def fetch_chunk_vectors(chunk_ids: Sequence[str]) -> dict[str, np.ndarray]:
    """Векторы перечисленных фрагментов из активной коллекции.

    Синхронный вызов: у ChromaDB клиент синхронный. Вызывающий обязан унести
    его в run_in_threadpool — иначе на время запроса встаёт весь цикл событий
    вместе с очередью индексации.

    Отсутствующие id молча пропускаются: коллекция могла быть пересобрана под
    другую embedding-модель, и это не ошибка чтения, а отсутствие данных.
    Решение «что делать без векторов» принимает вызывающий.
    """
    if not chunk_ids:
        return {}
    collection = _collection()
    vectors: dict[str, np.ndarray] = {}
    for start in range(0, len(chunk_ids), CHROMA_GET_BATCH):
        batch = [str(value) for value in chunk_ids[start : start + CHROMA_GET_BATCH]]
        try:
            found = collection.get(ids=batch, include=["embeddings"])
        except Exception as exc:  # noqa: BLE001 - причина ниже
            # Любой отказ чтения — это «векторов нет», а не «документов нет».
            # Разделять их здесь нечем, а вызывающему нужен один ответ.
            raise TopicEmbeddingUnavailable(f"чтение векторов не удалось: {exc}") from exc
        found_ids = found.get("ids") or []
        found_vectors = found.get("embeddings")
        if found_vectors is None:
            continue
        for chunk_id, vector in zip(found_ids, found_vectors):
            if vector is None:
                continue
            array = np.asarray(vector, dtype=np.float64)
            if array.ndim == 1 and array.size:
                vectors[str(chunk_id)] = array
    return vectors


def document_vector(vectors: Iterable[np.ndarray]) -> np.ndarray | None:
    """Вектор документа — среднее векторов его фрагментов.

    Среднее, а не вектор первого фрагмента и не отдельный эмбеддинг всего
    текста: фрагменты уже посчитаны при индексации (второй проход по Ollama
    стоил бы минуты на документ), а первый фрагмент документа — это титул и
    оглавление, то есть ровно та его часть, которая про тему говорит меньше
    всего.
    """
    stacked = [np.asarray(vector, dtype=np.float64) for vector in vectors]
    if not stacked:
        return None
    width = stacked[0].shape[0]
    # Фрагменты разной длины означают смесь двух коллекций в одном документе.
    # Усреднять их нельзя, и «привести к общей длине» тоже: обе интерпретации
    # выдумывают данные.
    if any(vector.shape[0] != width for vector in stacked):
        raise TopicEmbeddingUnavailable(
            "векторы фрагментов документа имеют разную размерность"
        )
    return np.mean(np.vstack(stacked), axis=0)


# --- Реестр версий --------------------------------------------------------


@dataclass(frozen=True)
class AssignmentOutcome:
    """Сколько документов пачки получили тему и сколько остались без неё.

    Два числа, а не одно: «размечено 800» без «пропущено 200» невозможно
    объяснить, а именно объяснение и нужно администратору, который смотрит на
    результат переразметки. Причина пропуска у всех одна и та же
    (topic.embedding_unavailable) — векторов документа в активной коллекции
    не нашлось.
    """

    assigned: int = 0
    skipped: int = 0

    def __add__(self, other: "AssignmentOutcome") -> "AssignmentOutcome":
        return AssignmentOutcome(
            assigned=self.assigned + other.assigned, skipped=self.skipped + other.skipped
        )


@dataclass(frozen=True)
class TopicShare:
    """Строка распределения тем."""

    cluster_index: int
    label: str
    # Та же подпись на языках интерфейса: {"ru": "Налоги", "tg": "Андозҳо"}.
    # Языка в словаре нет — перевода нет, показывать надо label. Решение «что
    # показать» остаётся за клиентом: интерфейс переведён на ru и tg, но фильтр
    # по теме и ссылки живут по устойчивому английскому имени, поэтому обрезать
    # что-то из двух здесь было бы решением за него.
    labels_localized: dict[str, str]
    document_count: int
    share: float


class TopicsService:
    # -- реестр ----------------------------------------------------------
    @staticmethod
    async def active_model(session: AsyncSession) -> TopicModelVersion | None:
        result = await session.exec(
            select(TopicModelVersion)
            .where(TopicModelVersion.is_active == True)  # noqa: E712 - SQL, не Python
            .order_by(TopicModelVersion.version.desc())
            .limit(1)
        )
        return result.first()

    @staticmethod
    async def sync_active_model(session: AsyncSession) -> TopicModelVersion | None:
        """Свести реестр с тем, что лежит на диске.

        Регистрация автоматическая, потому что в контракте раздела нет ручки
        «зарегистрировать модель»: артефакт кладёт обучающая сторона, и второй
        обязательный шаг (сходить в API и сказать «я обучил») однажды забыли бы,
        а раздел молча показывал бы прошлую модель.

        Зовут её три места, и ни одно из них не является обычным чтением: старт
        приложения, начало переразметки и цикл воркера раз в
        MODEL_SYNC_INTERVAL_SECONDS. Обычные GET реестр НЕ трогают: чтение,
        которое пишет в базу, гоняется само с собой при двух процессах uvicorn и
        превращает случайный запрос пользователя в миграцию.

        Новая версия заводится только при ДРУГОМ содержимом файла (sha256).
        Перезапуск, копирование и восстановление из бэкапа версию не двигают:
        новая версия обесценивает все назначения разом.
        """
        file = topic_model_path()
        if not file.exists():
            return await TopicsService.active_model(session)
        try:
            artifact = load_artifact_cached(file)
        except TopicModelUnusable as exc:
            # Громко: артефакт есть, но применить его нельзя. Молча оставить
            # прошлую активную модель — значит показывать пользователю темы от
            # модели, которую уже переобучили.
            logger.error(
                "Артефакт тем %s не пригоден (%s). Раздел тем продолжает "
                "работать на прошлой версии, если она была.", file, exc
            )
            return await TopicsService.active_model(session)
        except Exception as exc:  # noqa: BLE001 - файл читается с диска
            logger.error("Артефакт тем %s не прочитан: %s", file, exc)
            return await TopicsService.active_model(session)

        current = await TopicsService.active_model(session)
        if current is not None and current.artifact_digest == artifact.digest:
            return current

        version = int(
            (
                await session.execute(
                    text("SELECT COALESCE(MAX(version), 0) FROM topicmodelversion")
                )
            ).scalar_one()
        ) + 1
        # Снятие флага и вставка — одна транзакция: частичный уникальный индекс
        # TOPIC_MODEL_ACTIVE_INDEX не даст им разъехаться даже при гонке двух
        # процессов, но разъехавшаяся пара всё равно откатилась бы целиком.
        await session.execute(
            text("UPDATE topicmodelversion SET is_active = FALSE WHERE is_active")
        )
        row = TopicModelVersion(
            version=version,
            k=artifact.k,
            cluster_count=artifact.cluster_count,
            embedding_model=artifact.embedding_model,
            transform=artifact.transform.description,
            metrics_json=json.dumps(artifact.metrics, ensure_ascii=False),
            labels_json=json.dumps(
                {str(index): label for index, label in artifact.labels.items()},
                ensure_ascii=False,
            ),
            # Переводы копируются в реестр по той же причине, что и основные
            # подписи: список тем обязан отвечать и тогда, когда файл артефакта
            # уже перезаписан следующим обучением. Одной колонкой на все языки,
            # а не колонкой на язык: набор языков принадлежит продукту, а не
            # схеме БД, и новый перевод не должен стоить миграции. Кластеров без
            # перевода здесь нет — их отсутствие и есть ответ.
            labels_localized_json=json.dumps(
                {
                    language: {str(index): label for index, label in names.items()}
                    for language, names in artifact.labels_localized.items()
                },
                ensure_ascii=False,
            ),
            artifact_path=artifact.path,
            artifact_digest=artifact.digest,
            trained_at=artifact.trained_at,
            is_active=True,
        )
        session.add(row)
        try:
            await session.commit()
        except Exception as exc:  # noqa: BLE001 - гонка двух процессов
            await session.rollback()
            logger.warning(
                "Регистрация модели тем версии %s не прошла (%s: %s); скорее "
                "всего её уже зарегистрировал соседний процесс — индекс %s.",
                version,
                type(exc).__name__,
                exc,
                TOPIC_MODEL_ACTIVE_INDEX,
            )
            return await TopicsService.active_model(session)
        await session.refresh(row)
        logger.info(
            "Модель тем версии %s зарегистрирована: k=%s, кластеров %s, "
            "эмбеддинги %s, преобразование %s. Документы, размеченные прошлой "
            "версией, в распределение не попадают до переразметки.",
            row.version,
            row.k,
            row.cluster_count,
            row.embedding_model,
            row.transform,
        )
        return row

    @staticmethod
    def labels_of(model: TopicModelVersion) -> dict[int, str]:
        """Подписи кластеров из реестра.

        Из реестра, а не из файла: список тем обязан отвечать и тогда, когда
        артефакт недоступен, — иначе пропавший файл гасил бы весь раздел, а не
        только назначение новых тем.
        """
        try:
            raw = json.loads(model.labels_json or "{}")
        except ValueError:
            raw = {}
        labels: dict[int, str] = {}
        if isinstance(raw, dict):
            for key, value in raw.items():
                with contextlib.suppress(TypeError, ValueError):
                    labels[int(key)] = str(value)
        return {
            index: labels.get(index) or default_label(index)
            for index in range(int(model.cluster_count or 0))
        }

    @staticmethod
    def labels_localized_of(model: TopicModelVersion) -> dict[str, dict[int, str]]:
        """Переводы подписей из реестра — только те, что там есть.

        Отсутствующие не подменяются номером кластера (см.
        _extract_labels_localized): пустое место здесь — сигнал клиенту
        откатиться к основной подписи, а не повод показать «Кластер 7» дважды.
        """
        try:
            raw = json.loads(model.labels_localized_json or "{}")
        except ValueError:
            raw = {}
        labels: dict[str, dict[int, str]] = {
            language: {} for language in LOCALIZED_LABEL_LANGUAGES
        }
        if not isinstance(raw, dict):
            return labels
        for language, names in raw.items():
            if language not in labels or not isinstance(names, dict):
                continue
            for key, value in names.items():
                with contextlib.suppress(TypeError, ValueError):
                    name = str(value).strip()
                    if name:
                        labels[language][int(key)] = name
        return labels

    @staticmethod
    def metrics_of(model: TopicModelVersion) -> dict[str, float | None]:
        try:
            raw = json.loads(model.metrics_json or "{}")
        except ValueError:
            raw = {}
        if not isinstance(raw, dict):
            raw = {}
        return {key: raw.get(key) for key in _METRIC_ALIASES}

    # -- распределение ---------------------------------------------------
    @staticmethod
    async def distribution(
        session: AsyncSession,
        model: TopicModelVersion,
        *,
        notebook_id: int | None = None,
        owner_id: int | None = None,
    ) -> list[TopicShare]:
        """Сколько документов в каждой теме активной модели.

        Считаются ТОЛЬКО назначения активной версии. Смешивать их с прошлыми
        нельзя: после переобучения номер 3 у старой модели и номер 3 у новой —
        разные темы, и сумма по ним была бы числом без смысла. Пока
        переразметка не прошла, распределение честно показывает нули — это
        видимое состояние «модель сменилась», а не тихая подмена.

        owner_id=None — вызов от админа, выборка не сужается (то же правило,
        что у _owner_filter в разделе источников). Владение блокнотом
        проверяет HTTP-слой до вызова.
        """
        conditions = [
            "topic_model_version = :version",
            "topic_cluster_index IS NOT NULL",
        ]
        params: dict[str, Any] = {"version": model.version}
        if notebook_id is not None:
            conditions.append("notebook_id = :notebook_id")
            params["notebook_id"] = notebook_id
        if owner_id is not None:
            conditions.append("owner_id = :owner_id")
            params["owner_id"] = owner_id
        result = await session.execute(
            text(
                f"""
                SELECT topic_cluster_index, COUNT(*) AS documents
                FROM document
                WHERE {' AND '.join(conditions)}
                GROUP BY topic_cluster_index
                """
            ),
            params,
        )
        counts = {int(row[0]): int(row[1]) for row in result.all()}
        total = sum(counts.values())
        labels = TopicsService.labels_of(model)
        localized = TopicsService.labels_localized_of(model)
        # Кластеры без документов остаются в ответе с нулём. Пропускать их
        # нельзя: пользователь, сравнивающий распределение по двум блокнотам,
        # иначе видит два разных набора строк и не может их сопоставить. То же
        # относится к администратору: полный список — это ответ на вопрос «что
        # модель вообще умеет различать», и он теряется, если строки с нулём
        # отфильтровать на сервере. Прятать их от ОБЫЧНОГО пользователя —
        # работа экрана, а не выдачи: см. TopicsPage, где пустые темы свёрнуты
        # под раскрытие.
        rows = [
            TopicShare(
                cluster_index=index,
                label=labels.get(index) or default_label(index),
                labels_localized={
                    language: names[index]
                    for language, names in localized.items()
                    if index in names
                },
                document_count=counts.get(index, 0),
                share=(counts.get(index, 0) / total) if total else 0.0,
            )
            for index in sorted(labels)
        ]
        # Порядок — по убыванию числа документов: это распределение, и главное
        # в нём то, чего больше. Номер кластера вторым ключом, чтобы порядок
        # был устойчив при равных числах (иначе список прыгал бы между
        # обновлениями страницы).
        rows.sort(key=lambda row: (-row.document_count, row.cluster_index))
        return rows

    # -- назначение ------------------------------------------------------
    @staticmethod
    async def assign_after_indexing(session: AsyncSession, doc_id: int) -> bool:
        """Назначить тему только что проиндексированному документу.

        НЕ БРОСАЕТ НИЧЕГО. Тема — украшение документа, поиск по нему — основная
        функция: необученная модель, недоступная ChromaDB и любая неожиданная
        беда обязаны оставить документ проиндексированным и работающим. Отказ в
        теме заметен только по журналу и лечится переразметкой, а отказ в
        индексации виден пользователю как несработавшая загрузка файла.

        Возвращает True, если тема назначена, — ради тестов и журнала.
        """
        try:
            model = await TopicsService.active_model(session)
            if model is None:
                return False
            outcome = await TopicsService.assign_documents(session, model, [doc_id])
            return bool(outcome.assigned)
        except Exception as exc:  # noqa: BLE001 - см. докстринг
            logger.info(
                "Документ %s остался без темы (%s: %s). Индексация не "
                "затронута; тема появится после переразметки.",
                doc_id,
                type(exc).__name__,
                exc,
            )
            return False

    @staticmethod
    async def assign_documents(
        session: AsyncSession,
        model: TopicModelVersion,
        doc_ids: Sequence[int],
    ) -> AssignmentOutcome:
        """Назначить темы пачке документов.

        Пачкой, а не по одному: у переразметки это единственная разница между
        одним запросом в ChromaDB и тысячей. Отсюда же и порядок шагов —
        сначала собрать все id фрагментов, потом один поход за векторами, потом
        арифметика в памяти.

        Разница между «беда общая» и «беда одного документа» здесь проведена
        руками, и она принципиальна. Отсутствие векторов У ЭТОГО документа —
        это пропуск: он считается в AssignmentOutcome.skipped, а работа идёт
        дальше. Отказ, общий для всех (не задана embedding-модель, ChromaDB
        лежит, артефакт не пригоден), поднимается наружу: продолжать после него
        значит тысячу раз получить один и тот же отказ.
        """
        if not doc_ids:
            return AssignmentOutcome()
        artifact = await run_in_threadpool(load_artifact_cached, Path(model.artifact_path))
        TopicsService._assert_same_space(model, artifact)

        documents = (
            await session.exec(select(Document).where(Document.id.in_(list(doc_ids))))
        ).all()
        if not documents:
            return AssignmentOutcome()
        chunk_rows = (
            await session.exec(
                select(Chunk.id, Chunk.doc_id)
                .where(Chunk.doc_id.in_([int(document.id) for document in documents]))
                .order_by(Chunk.doc_id, Chunk.id)
            )
        ).all()
        by_document: dict[int, list[str]] = {}
        for chunk_id, doc_id in chunk_rows:
            by_document.setdefault(int(doc_id), []).append(str(chunk_id))

        all_chunk_ids = [cid for ids in by_document.values() for cid in ids]
        vectors = await run_in_threadpool(fetch_chunk_vectors, all_chunk_ids)

        assigned = 0
        skipped = 0
        for document in documents:
            own = [vectors[cid] for cid in by_document.get(int(document.id), []) if cid in vectors]
            try:
                centre = document_vector(own)
            except TopicEmbeddingUnavailable:
                # Беда ЭТОГО документа (фрагменты разной размерности — смесь
                # двух коллекций), а не всей работы: остальные документы пачки
                # к ней отношения не имеют.
                centre = None
            if centre is None:
                # Документ без единого вектора в активной коллекции: его либо
                # индексировали под другую embedding-модель, либо векторы
                # потеряны. Тему ему не назначаем и НЕ трогаем прежнюю: старая
                # подпись с прошлой версией — это история, и стирать её ради
                # неудачной попытки незачем.
                logger.info(
                    "Документ %s: векторов в активной коллекции нет (%s), тема "
                    "не назначена",
                    document.id,
                    TopicErrors.EMBEDDING_UNAVAILABLE,
                )
                skipped += 1
                continue
            cluster = artifact.assign(centre, group=document.language)
            document.topic_cluster_index = cluster
            document.topic_label = artifact.label_of(cluster)
            # Переводы ХРАНЯТСЯ рядом с основной подписью и по тому же правилу:
            # все пишутся в момент назначения и остаются такими, какими их
            # сделала эта версия модели. Собирать перевод на лету по
            # topic_cluster_index из АКТИВНОЙ модели было бы дешевле на две
            # колонки, но у документа, размеченного прошлой версией, номер 7
            # означает уже другую тему: рядом с исторической английской
            # подписью встало бы название от чужой модели, и документ показывал
            # бы два разных имени одной темы.
            #
            # Документы, размеченные ДО появления колонок, остаются с None:
            # переводы у них появятся при первой же переразметке — той самой,
            # которую всё равно приходится запускать после смены артефакта.
            document.topic_label_ru = artifact.label_in(cluster, "ru")
            document.topic_label_tg = artifact.label_in(cluster, "tg")
            document.topic_model_version = model.version
            session.add(document)
            assigned += 1
        await session.commit()
        return AssignmentOutcome(assigned=assigned, skipped=skipped)

    @staticmethod
    def _assert_same_space(model: TopicModelVersion, artifact: TopicArtifact) -> None:
        """Векторы и центроиды обязаны быть из одного пространства.

        Проверка стоит здесь, а не при регистрации модели: embedding-модель
        меняют мышкой в админ-панели в любой момент, в том числе между
        регистрацией и назначением. Вектор от другой модели даёт правдоподобный
        номер кластера — то есть неверную тему без единой ошибки в журнале.
        """
        from app.shared.settings.runtime_settings import RuntimeSettingsService

        active = (RuntimeSettingsService.embedding_model() or "").strip()
        if not active:
            raise TopicEmbeddingUnavailable("embedding-модель не задана")
        if active != artifact.embedding_model:
            raise TopicEmbeddingUnavailable(
                f"модель тем обучена на эмбеддингах {artifact.embedding_model!r}, "
                f"а система считает векторы моделью {active!r}: пространства "
                "разные, назначать темы нечем"
            )
        if artifact.digest != model.artifact_digest:
            raise TopicModelUnusable(
                f"артефакт по пути {artifact.path} изменился с момента "
                f"регистрации версии {model.version}"
            )


# --- Переразметка ---------------------------------------------------------

# Сколько документов размечается за один заход. Пачка задаёт и число походов в
# ChromaDB, и частоту heartbeat'а: слишком большая рискует арендой задачи
# (LEASE_SECONDS у JobsService), слишком маленькая упирается в накладные
# расходы на запрос.
REASSIGN_BATCH = 50

# Как часто воркер заглядывает в очередь, когда его никто не будил. То же
# значение, что у воркеров индексации и презентаций.
POLL_INTERVAL_SECONDS = 2.0
ERROR_BACKOFF_SECONDS = 5.0
STOP_TIMEOUT_SECONDS = 30.0

# Как часто воркер сверяет реестр с артефактом на диске.
#
# Обучение идёт офлайн-скриптом на GPU и кладёт новый файл по тому же пути, ни о
# чём не спрашивая бэкенд. Узнавать об этом только на перезапуске нельзя: до
# него новые документы вообще не получают тем (артефакт не совпадает с
# зарегистрированным — см. _assert_same_space), и починка выглядит как
# «перезагрузите сервер».
#
# Минута, а не секунды: сверка дешёвая (два stat и один SELECT, sha256 считается
# только у изменившегося файла), но и спешить некуда — переобучение происходит
# раз в дни, а не раз в минуту.
MODEL_SYNC_INTERVAL_SECONDS = 60.0

# Мгновенная побудка воркера после постановки задачи в том же процессе. При
# uvicorn --workers 2 соседний процесс события не увидит и подберёт задачу
# следующим опросом — очередь живёт в БД, а не в памяти.
_QUEUE_WAKEUP = asyncio.Event()


def queue_wakeup() -> asyncio.Event:
    return _QUEUE_WAKEUP


async def run_reassign(job_id: int) -> dict[str, Any]:
    """Переразметить все проиндексированные документы активной моделью.

    Идёт пачками по возрастанию id и обновляет прогресс на каждой: это ещё и
    аренда задачи (JobsService.heartbeat), по которой согласование отличает
    живого воркера от мёртвого. Без неё длинная переразметка выглядела бы как
    зависшая и возвращалась бы в очередь на каждом reap_stale.

    Документ, у которого не нашлось векторов, ПРОПУСКАЕТСЯ, а работа идёт
    дальше: одна недостающая пачка векторов не повод оставить остальные
    документы без тем. А вот отказ, общий для всех (не задана
    embedding-модель, ChromaDB лежит, артефакт не пригоден), останавливает
    задачу — продолжать её значит тысячу раз получить один и тот же отказ.
    """
    from app.modules.jobs.service import JobsService

    async with session_context() as session:
        model = await TopicsService.sync_active_model(session)
        if model is None:
            raise TopicModelUnusable("активной обученной модели тем нет")
        total = int(
            (
                await session.execute(
                    text("SELECT COUNT(*) FROM document WHERE status = 'indexed'")
                )
            ).scalar_one()
        )

    outcome = AssignmentOutcome()
    processed = 0
    last_id = 0
    while True:
        async with session_context() as session:
            rows = (
                await session.execute(
                    text(
                        "SELECT id FROM document WHERE status = 'indexed' AND id > :last "
                        "ORDER BY id LIMIT :limit"
                    ),
                    {"last": last_id, "limit": REASSIGN_BATCH},
                )
            ).all()
            doc_ids = [int(row[0]) for row in rows]
            if not doc_ids:
                break
            last_id = doc_ids[-1]
            # Отказ отсюда НЕ ловится намеренно. Пропуск отдельного документа
            # (векторов в активной коллекции нет) assign_documents считает сам
            # и не бросает; а всё, что всё-таки долетает сюда, — общее для всей
            # работы: ChromaDB не отвечает, embedding-модель не та, артефакт не
            # пригоден. Продолжать после такого значит тысячу раз получить один
            # и тот же отказ и закончить бодрым «выполнено» с нулём разметки.
            outcome += await TopicsService.assign_documents(session, model, doc_ids)
            processed += len(doc_ids)

        # Прогресс и продление аренды — своей сессией и вне разметки пачки:
        # держать её открытой на время запроса в ChromaDB незачем.
        async with session_context() as session:
            await JobsService.heartbeat(
                session,
                job_id,
                progress=min(99, int(processed * 100 / total)) if total else 99,
            )

    # Документы, которые размечены прошлой версией и не попали в переразметку
    # (не 'indexed'), остаются с прежней подписью. Число называется в
    # результате: без него «размечено 800 из 1000» невозможно объяснить.
    stale = 0
    async with session_context() as session:
        stale = int(
            (
                await session.execute(
                    text(
                        "SELECT COUNT(*) FROM document "
                        "WHERE topic_model_version IS NOT NULL "
                        "AND topic_model_version <> :version"
                    ),
                    {"version": model.version},
                )
            ).scalar_one()
        )
    result = {
        "model_version": model.version,
        "documents": total,
        "assigned": outcome.assigned,
        "skipped": outcome.skipped,
        "stale": stale,
    }
    if outcome.skipped:
        result["skipped_reason"] = TopicErrors.EMBEDDING_UNAVAILABLE
    return result


class TopicsWorker:
    """Цикл переразметки: тот же скелет, что у воркеров индексации и презентаций.

    Свой воркер, а не задача в очереди индексации: переразметка проходит по
    всем документам сразу и надолго занимает ChromaDB, и в общей очереди она
    задержала бы загрузку файлов ровно на своё время. Очередь при этом ОБЩАЯ —
    таблица job, — потому что захват, аренда и бюджет попыток уже написаны и
    проверены там.

    Главное свойство цикла: исключение внутри задачи его не роняет. Упавшая
    переразметка — это job со status='failed', а не остановленная очередь:
    перезапустить воркер до рестарта сервера некому.
    """

    def __init__(self, poll_interval: float = POLL_INTERVAL_SECONDS) -> None:
        self._poll_interval = poll_interval
        self._task: asyncio.Task | None = None
        # Отрицательная бесконечность, а не 0.0: первая же итерация обязана
        # сверить реестр с диском, каким бы ни было loop.time() на старте.
        self._model_synced_at = float("-inf")

    def start(self) -> None:
        if self._task is not None and not self._task.done():
            return
        self._task = asyncio.create_task(self._run(), name="topics-worker")
        logger.info("Topics worker started")

    async def stop(self) -> None:
        task, self._task = self._task, None
        if task is None:
            return
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError, asyncio.TimeoutError):
            await asyncio.wait_for(task, timeout=STOP_TIMEOUT_SECONDS)
        logger.info("Topics worker stopped")

    async def _run(self) -> None:
        wakeup = queue_wakeup()
        while True:
            try:
                await self.sync_model_if_due()
                claimed = await self.claim_and_process()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Topics worker iteration failed")
                await asyncio.sleep(ERROR_BACKOFF_SECONDS)
                continue
            if claimed:
                continue
            wakeup.clear()
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(wakeup.wait(), timeout=self._poll_interval)

    async def sync_model_if_due(self) -> None:
        """Заметить переобученную модель, не дожидаясь перезапуска.

        Модель обучается офлайн-скриптом и кладётся на диск мимо приложения.
        Пока новый артефакт не зарегистрирован, назначение тем отказывается
        работать (артефакт не совпадает с зарегистрированной версией), то есть
        новые документы молча остаются без тем.

        Регистрация новой версии сама по себе НЕ переразмечает документы: она
        обнуляет распределение, потому что прежние назначения принадлежат
        прошлой версии. Это видимое состояние «модель сменилась, нужна
        переразметка», а не тихая подмена — номер кластера у новой модели
        означает другую тему, и показывать старые назначения под новыми
        подписями было бы враньём.

        Отказ здесь не роняет цикл: очередь переразметки должна работать и
        тогда, когда артефакт унесли с диска.
        """
        loop = asyncio.get_running_loop()
        if loop.time() - self._model_synced_at < MODEL_SYNC_INTERVAL_SECONDS:
            return
        self._model_synced_at = loop.time()
        try:
            async with session_context() as session:
                await TopicsService.sync_active_model(session)
        except Exception as exc:  # noqa: BLE001 - см. докстринг
            logger.warning("Сверка реестра моделей тем с диском не удалась: %s", exc)

    async def claim_and_process(self) -> bool:
        from app.modules.jobs.service import JobsService

        async with session_context() as session:
            job = await JobsService.claim_next(session, TOPIC_REASSIGN_JOB_TYPE)
            if job is None:
                return False
            job_id = job.id
        await self._process(job_id)
        return True

    async def _process(self, job_id: int) -> None:
        from app.modules.jobs.service import JobsService

        started = utcnow()
        try:
            result = await run_reassign(job_id)
        except asyncio.CancelledError:
            # CancelledError наследуется от BaseException, и except Exception
            # его не видит. Без этой ветки штатный SIGTERM оставил бы задачу в
            # 'running' до следующего reap_stale, то есть на всю аренду.
            await asyncio.shield(self._release(job_id))
            raise
        except Exception as exc:  # noqa: BLE001 - терминальное состояние задачи
            logger.warning("Переразметка тем %s не удалась: %s", job_id, exc, exc_info=True)
            # Машинный код в начале текста, если он у отказа есть. У job нет
            # колонки error_code (в отличие от document и presentation), а
            # причина отказа переразметки обязана быть различима машинно:
            # «недоступны векторы» и «модель не пригодна» лечатся разным.
            error_code = getattr(exc, "error_code", None)
            prefix = f"{error_code}: " if error_code else ""
            async with session_context() as session:
                await JobsService.finish(
                    session,
                    job_id,
                    error_text=f"{prefix}{type(exc).__name__}: {exc}"[:500],
                )
            return
        async with session_context() as session:
            await JobsService.finish(session, job_id, result=result)
        logger.info(
            "Переразметка тем %s завершена за %.1fс: %s",
            job_id,
            (utcnow() - started).total_seconds(),
            result,
        )

    async def _release(self, job_id: int) -> None:
        """Прерванная задача возвращается в очередь, а не падает в ошибку.

        Остановка сервера — не вина задачи: переразметка идемпотентна (тот же
        артефакт даст те же кластеры), и при следующем старте она честно
        отработает с начала.
        """
        from app.modules.jobs.service import JobsService

        try:
            async with session_context() as session:
                await JobsService.requeue(
                    session, job_id, error_text="Переразметка прервана остановкой сервера"
                )
        except Exception:
            logger.exception("Не удалось вернуть переразметку %s в очередь", job_id)
