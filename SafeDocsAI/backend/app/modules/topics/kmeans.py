"""Собственная реализация K-means для кластеризации документов по темам.

Библиотечной кластеризации здесь нет и не будет: sklearn в окружении есть
(приехал транзитивно), но в этой работе алгоритм — предмет защиты, а не
инструмент. numpy используется как векторная арифметика; всё, что составляет
K-means — инициализация, назначение, пересчёт центроидов, критерий остановки,
выбор k, — написано в этом файле. Запрет проверяется тестом-сторожом
tests/test_topics_no_sklearn_guard.py, а не обещанием.

Модуль ни к чему не подключён и ничего не знает про документы: на вход
приходит матрица чисел (n_samples, n_features), на выход — метки кластеров.
Такая развязка позволяет разбирать алгоритм на синтетике, где правильный
ответ известен заранее.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.modules.topics.metrics import silhouette_score

# Значения по умолчанию собраны в одном месте: их называют на защите, и они же
# подставляются в choose_k, поэтому расходиться им нельзя.
DEFAULT_N_INIT = 10
DEFAULT_MAX_ITER = 300
DEFAULT_TOL = 1e-4


def l2_normalize(X: np.ndarray) -> np.ndarray:
    """Приводит каждую строку к единичной длине.

    Нужна для косинусной близости. Для векторов единичной длины
    ||x - c||^2 = ||x||^2 - 2*x·c + ||c||^2 = 2 - 2*cos(x, c) при ||c|| = 1,
    то есть «ближайший по евклиду» и «ближайший по косинусу» — это один и тот
    же центроид. Поэтому отдельный «косинусный K-means» писать не нужно:
    достаточно нормировать вход и запустить обычный евклидов.

    Нулевые строки оставляем как есть: делить на ноль нельзя, а выбрасывать
    документ молча — хуже, чем оставить его в начале координат, где он честно
    окажется одинаково далёк от всех центроидов.
    """
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    safe = np.where(norms == 0.0, 1.0, norms)
    return X / safe


def squared_distances(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Матрица квадратов расстояний: (n_samples, n_clusters).

    Наивная запись — двойной цикл по точкам и центроидам — читается лучше, но
    на эмбеддингах qwen3 (размерность порядка тысяч) она неприемлемо медленна,
    поэтому здесь раскрытие квадрата разности:

        ||x - c||^2 = ||x||^2 - 2*(x·c) + ||c||^2

    Каждое слагаемое считается сразу для всех пар: скалярные произведения —
    одним матричным умножением, квадраты норм — по одному вектору на сторону,
    сложение растягивает их broadcast'ом по нужной оси. Результат тот же, что у
    цикла, но без цикла.

    Квадратный корень не берём намеренно: и назначение точки ближайшему
    центроиду, и инерция монотонны по квадрату расстояния, а корень — лишняя
    работа и лишняя потеря точности.
    """
    x_sq = np.einsum("ij,ij->i", X, X)[:, np.newaxis]
    c_sq = np.einsum("ij,ij->i", centroids, centroids)[np.newaxis, :]
    distances = x_sq - 2.0 * (X @ centroids.T) + c_sq
    # Разность больших близких чисел даёт малые отрицательные значения порядка
    # -1e-13 там, где точное расстояние равно нулю. На argmin это не влияет, но
    # инерция из-за них становится отрицательной, а корень — NaN.
    return np.maximum(distances, 0.0)


def kmeans_plus_plus_init(
    X: np.ndarray, n_clusters: int, rng: np.random.Generator
) -> np.ndarray:
    """Выбор стартовых центроидов по схеме k-means++.

    Почему не случайные точки. Случайный старт даёт разброс результата от
    запуска к запуску и регулярно сажает алгоритм в плохой локальный минимум:
    два центра попадают внутрь одного плотного кластера, а соседний кластер
    остаётся без центра и «съедается». k-means++ выбирает каждый следующий
    центр с вероятностью, пропорциональной квадрату расстояния до ближайшего
    уже выбранного, — то есть тем охотнее, чем хуже точка покрыта текущими
    центрами. Далёкие области перестают оставаться без центра.

    Первый центр берётся равновероятно из самих точек (а не из воздуха), чтобы
    старт заведомо лежал внутри облака данных.
    """
    n_samples = X.shape[0]
    centroids = np.empty((n_clusters, X.shape[1]), dtype=X.dtype)

    first = int(rng.integers(n_samples))
    centroids[0] = X[first]

    # closest_sq[i] — квадрат расстояния от точки i до ближайшего уже выбранного
    # центра. Пересчитывать его целиком на каждом шаге не нужно: добавление
    # нового центра может только уменьшить минимум, поэтому хватает поэлементного
    # минимума со свежим столбцом расстояний.
    closest_sq = squared_distances(X, centroids[:1]).ravel()

    for index in range(1, n_clusters):
        total = float(closest_sq.sum())
        if total <= 0.0:
            # Все оставшиеся точки совпадают с уже выбранными центрами
            # (дубликаты документов, либо n_clusters близко к числу различных
            # точек). Вероятности выродились в нули — берём точку равновероятно,
            # иначе делить не на что.
            chosen = int(rng.integers(n_samples))
        else:
            # Розыгрыш по вероятности D(x)^2 / sum(D^2) без нормировки массива:
            # бросаем точку на отрезок длины total и ищем, в чей интервал попали.
            threshold = float(rng.random()) * total
            chosen = int(np.searchsorted(np.cumsum(closest_sq), threshold))
            chosen = min(chosen, n_samples - 1)

        centroids[index] = X[chosen]
        closest_sq = np.minimum(
            closest_sq, squared_distances(X, centroids[index : index + 1]).ravel()
        )

    return centroids


@dataclass
class _SingleRun:
    """Результат одного перезапуска — чтобы сравнивать их по инерции целиком."""

    centroids: np.ndarray
    labels: np.ndarray
    inertia: float
    n_iter: int


class KMeans:
    """K-means с инициализацией k-means++ и несколькими перезапусками.

    Параметры
    ---------
    n_clusters : число кластеров k.
    n_init : сколько раз запустить алгоритм с разных стартов. K-means сходится
        к локальному минимуму, и один запуск — лотерея: тот же вход при другом
        старте даёт другое разбиение и другую инерцию. Из n_init запусков
        берётся результат с наименьшей инерцией.
    max_iter : потолок числа итераций Ллойда.
    tol : порог сходимости по сдвигу центроидов.
    random_state : seed. Без него повторить число, названное на защите, нельзя.
    normalize : L2-нормировать вход, то есть кластеризовать по косинусу.
        Для эмбеддингов текстов включено по умолчанию — см. l2_normalize.

    Атрибуты после fit
    ------------------
    centroids_ : (n_clusters, n_features) — в том же пространстве, в котором
        шла кластеризация: при normalize=True это нормированное пространство.
    labels_, inertia_, n_iter_.
    """

    def __init__(
        self,
        n_clusters: int = 8,
        *,
        n_init: int = DEFAULT_N_INIT,
        max_iter: int = DEFAULT_MAX_ITER,
        tol: float = DEFAULT_TOL,
        random_state: int | None = None,
        normalize: bool = True,
    ) -> None:
        if n_clusters < 1:
            raise ValueError("n_clusters должно быть >= 1")
        if n_init < 1:
            raise ValueError("n_init должно быть >= 1")
        if max_iter < 1:
            raise ValueError("max_iter должно быть >= 1")
        if tol < 0:
            raise ValueError("tol не может быть отрицательным")

        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.normalize = normalize

        self.centroids_: np.ndarray | None = None
        self.labels_: np.ndarray | None = None
        self.inertia_: float | None = None
        self.n_iter_: int | None = None

    # --- публичный интерфейс -------------------------------------------------

    def fit(self, X: np.ndarray) -> "KMeans":
        data = self._prepare(X)
        if data.shape[0] < self.n_clusters:
            raise ValueError(
                f"точек ({data.shape[0]}) меньше, чем кластеров ({self.n_clusters})"
            )

        # Один генератор на все перезапуски, а не n_init одинаковых: иначе
        # перезапуски стартовали бы из одной точки и n_init потерял бы смысл.
        # Последовательность розыгрышей при заданном random_state
        # воспроизводима целиком, поэтому и итог воспроизводим.
        rng = np.random.default_rng(self.random_state)

        best: _SingleRun | None = None
        for _ in range(self.n_init):
            run = self._run_once(data, rng)
            if best is None or run.inertia < best.inertia:
                best = run

        assert best is not None  # n_init >= 1 проверен в __init__
        self.centroids_ = best.centroids
        self.labels_ = best.labels
        self.inertia_ = best.inertia
        self.n_iter_ = best.n_iter
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Метка ближайшего центроида для новых точек.

        Центроиды не пересчитываются: предсказание не должно менять модель,
        иначе два одинаковых вызова дали бы разные ответы.
        """
        if self.centroids_ is None:
            raise ValueError("модель не обучена: сначала fit()")
        data = self._prepare(X)
        if data.shape[1] != self.centroids_.shape[1]:
            raise ValueError(
                f"ожидалось {self.centroids_.shape[1]} признаков, "
                f"получено {data.shape[1]}"
            )
        return np.argmin(squared_distances(data, self.centroids_), axis=1)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        assert self.labels_ is not None
        return self.labels_

    # --- внутренности --------------------------------------------------------

    def _prepare(self, X: np.ndarray) -> np.ndarray:
        data = np.asarray(X, dtype=np.float64)
        if data.ndim != 2:
            raise ValueError(f"ожидалась матрица (n_samples, n_features), а не {data.ndim}D")
        if data.shape[0] == 0:
            raise ValueError("пустая выборка")
        if not np.all(np.isfinite(data)):
            # NaN отравляет и argmin, и среднее: кластеризация не упадёт, а
            # тихо выдаст мусор. Лучше остановиться здесь.
            raise ValueError("в данных есть NaN или inf")
        return l2_normalize(data) if self.normalize else data

    def _run_once(self, X: np.ndarray, rng: np.random.Generator) -> _SingleRun:
        """Один запуск: инициализация плюс итерации Ллойда."""
        centroids = kmeans_plus_plus_init(X, self.n_clusters, rng)

        # Первое назначение — до цикла, чтобы внутри цикла всегда сохранялся
        # инвариант «labels отвечают текущим centroids». Благодаря ему выход из
        # цикла в любой момент оставляет согласованный результат.
        distances = squared_distances(X, centroids)
        labels = np.argmin(distances, axis=1)

        n_iter = 0
        for iteration in range(1, self.max_iter + 1):
            # Шаг 1 Ллойда — пересчёт центроидов как среднего своего кластера.
            new_centroids = self._update_centroids(X, labels, centroids, distances)

            # Критерий сходимости — сдвиг центроидов. Смотрим на максимальный
            # сдвиг, а не на сумму: сумма по k кластерам растёт вместе с k, и
            # один и тот же tol означал бы при разных k разную строгость.
            shift = float(np.max(np.linalg.norm(new_centroids - centroids, axis=1)))
            centroids = new_centroids

            # Шаг 2 Ллойда — назначение точек ближайшему центроиду.
            distances = squared_distances(X, centroids)
            labels = np.argmin(distances, axis=1)
            n_iter = iteration

            if shift <= self.tol:
                break
            # Потолка max_iter одного не хватило бы: на хороших данных он
            # заставлял бы крутить сотни бесполезных итераций после сходимости.
            # Одного tol тоже не хватает — на вырожденных данных (много
            # дубликатов, перенос центроидов из пустых кластеров) центроиды
            # могут ходить по циклу, и порог не сработает никогда. Нужны оба.

        inertia = float(np.sum(distances[np.arange(X.shape[0]), labels]))
        return _SingleRun(centroids, labels, inertia, n_iter)

    def _update_centroids(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        centroids: np.ndarray,
        distances: np.ndarray,
    ) -> np.ndarray:
        """Новые центроиды: среднее своего кластера плюс лечение пустых.

        Здесь единственный цикл во всём алгоритме, и он оставлен намеренно: это
        буквальная запись шага Ллойда — «взять точки кластера и усреднить», —
        которую разбирают глазами. Векторная замена (np.add.at по номерам
        кластеров) выглядела бы учёнее, но на n=5000, d=1024, k=20 она
        оказалась втрое медленнее этого цикла: np.add.at работает без
        буферизации, поэлементно. Читаемость здесь ничего не стоила.

        Цикл идёт по k кластерам, а не по n точкам, и k на два-три порядка
        меньше n — стоимость шага всё равно определяется усреднением, которое
        внутри цикла векторное.
        """
        new_centroids = centroids.copy()
        empty: list[int] = []
        for cluster in range(self.n_clusters):
            members = labels == cluster
            if not members.any():
                # Центроид пустого кластера пока оставляем на месте: чем его
                # заменить, видно только после того, как посчитаны остальные.
                empty.append(cluster)
                continue
            new_centroids[cluster] = X[members].mean(axis=0)

        if empty:
            self._revive_empty_clusters(
                X, labels, distances, new_centroids, np.array(empty, dtype=np.int64)
            )
        return new_centroids

    @staticmethod
    def _revive_empty_clusters(
        X: np.ndarray,
        labels: np.ndarray,
        distances: np.ndarray,
        new_centroids: np.ndarray,
        empty: np.ndarray,
    ) -> None:
        """Пустой кластер: переносим центроид в самую «дорогую» точку.

        Кластер пустеет, когда ни одна точка не выбрала его центроид ближайшим.
        Оставить центроид на месте — значит вернуть на выходе k-1 кластер там,
        где просили k, причём молча. Взять случайную точку — значит внести в
        детерминированный шаг лишний источник разброса.

        Выбрана точка с наибольшим квадратом расстояния до СВОЕГО центроида:
        именно она вносит наибольший вклад в инерцию, и именно она с наибольшим
        основанием претендует на собственный кластер. После переноса эта точка
        окажется на нулевом расстоянии от нового центроида, то есть её вклад в
        инерцию обнулится. Строгой монотонности убывания инерции этот шаг не
        доказывает (остальные центроиды на той же итерации тоже сдвинулись),
        поэтому перезапуски сравниваются по фактически посчитанной инерции, а
        не по предположению.

        Если пустых кластеров несколько, точки берутся по убыванию вклада и
        каждая используется один раз — иначе два центроида встали бы в одну
        точку и один из них снова опустел бы на следующей итерации.
        """
        point_cost = distances[np.arange(X.shape[0]), labels]
        # Сортировка по убыванию: argsort по возрастанию и разворот — так
        # порядок остаётся детерминированным при равных стоимостях.
        by_cost = np.argsort(point_cost, kind="stable")[::-1]
        for offset, cluster in enumerate(empty):
            new_centroids[cluster] = X[by_cost[offset]]


# --- выбор k -----------------------------------------------------------------


@dataclass(frozen=True)
class KScore:
    """Строка таблицы перебора k — то, что показывается на графике."""

    k: int
    inertia: float
    silhouette: float
    n_iter: int


@dataclass(frozen=True)
class KSearchResult:
    """Итог перебора k: и автоматический выбор, и данные, по которым он сделан.

    scores — таблица целиком, по ней строится график «локтя» и график силуэта.
    best_k — выбор по максимуму силуэта.
    elbow_k — выбор по «локтю» кривой инерции.

    Оба числа отдаются вместе намеренно. Инерция монотонно убывает с ростом k
    (при k = n она равна нулю), поэтому по одной инерции «лучшее» k выбрать
    нельзя — можно только увидеть, где падение перестаёт окупаться. Силуэт
    такого перекоса не имеет, но описывает только геометрию. Совпали два
    критерия или разошлись — это и есть содержание разговора о выборе k.
    """

    scores: tuple[KScore, ...]
    best_k: int
    elbow_k: int


def choose_k(
    X: np.ndarray,
    k_values,
    *,
    n_init: int = DEFAULT_N_INIT,
    max_iter: int = DEFAULT_MAX_ITER,
    tol: float = DEFAULT_TOL,
    random_state: int | None = None,
    normalize: bool = True,
    silhouette_sample_size: int | None = 2000,
) -> KSearchResult:
    """Перебирает k и возвращает таблицу для решения плюс автоматический выбор.

    Нормировка делается здесь один раз, а самому KMeans передаётся
    normalize=False. Причина не в скорости: силуэт обязан считаться в том же
    пространстве, в котором работал алгоритм. Посчитанный по ненормированным
    векторам, он описывал бы другую геометрию, и таблица сравнивала бы k по
    величине, которую алгоритм не оптимизировал.

    Каждое k получает один и тот же random_state — иначе разница между строками
    таблицы включала бы разницу стартов, и сравнивать их было бы нельзя.

    silhouette_sample_size ограничивает выборку для силуэта: точный силуэт
    требует матрицы расстояний n x n. None — считать точно.
    """
    data = np.asarray(X, dtype=np.float64)
    if normalize:
        data = l2_normalize(data)

    n_samples = data.shape[0]
    scores: list[KScore] = []
    for k in sorted({int(value) for value in k_values}):
        if k < 1 or k > n_samples:
            raise ValueError(f"k={k} вне допустимого диапазона 1..{n_samples}")

        model = KMeans(
            n_clusters=k,
            n_init=n_init,
            max_iter=max_iter,
            tol=tol,
            random_state=random_state,
            normalize=False,
        ).fit(data)
        assert model.labels_ is not None and model.inertia_ is not None

        # Силуэт не определён, когда кластер один или когда каждая точка сама
        # себе кластер: сравнивать не с чем. NaN здесь честнее нуля — ноль
        # означал бы «плохое разбиение», а не «величина неприменима».
        distinct = len(np.unique(model.labels_))
        if distinct < 2 or distinct >= n_samples:
            silhouette = float("nan")
        else:
            silhouette = silhouette_score(
                data,
                model.labels_,
                sample_size=silhouette_sample_size,
                random_state=random_state,
            )
        scores.append(KScore(k, float(model.inertia_), silhouette, int(model.n_iter_ or 0)))

    return KSearchResult(
        scores=tuple(scores),
        best_k=_best_by_silhouette(scores),
        elbow_k=_elbow(scores),
    )


def _best_by_silhouette(scores: list[KScore]) -> int:
    """Максимум силуэта; при равенстве — меньшее k как более простая модель."""
    usable = [score for score in scores if not np.isnan(score.silhouette)]
    if not usable:
        return min(score.k for score in scores)
    best = max(usable, key=lambda score: (score.silhouette, -score.k))
    return best.k


def _elbow(scores: list[KScore]) -> int:
    """«Локоть» кривой инерции — точка максимального отклонения от хорды.

    Глазами локоть ищут как место, где кривая резче всего перегибается.
    Формализация та же, только измеримая: проводим прямую через крайние точки
    кривой (k_min, inertia_min) и (k_max, inertia_max) и берём k с наибольшим
    расстоянием до этой прямой. На выпуклой убывающей кривой это ровно то
    место, после которого добавление кластера почти перестаёт снижать инерцию.

    Меньше трёх точек — перегибу неоткуда взяться, возвращаем меньшее k.
    """
    if len(scores) < 3:
        return min(score.k for score in scores)

    ks = np.array([score.k for score in scores], dtype=np.float64)
    inertias = np.array([score.inertia for score in scores], dtype=np.float64)

    # Оси несравнимы (k — единицы, инерция — тысячи), поэтому обе шкалируются в
    # [0, 1]: иначе «расстояние до хорды» определялось бы только той осью, у
    # которой крупнее числа.
    ks = _rescale(ks)
    inertias = _rescale(inertias)

    start = np.array([ks[0], inertias[0]])
    end = np.array([ks[-1], inertias[-1]])
    chord = end - start
    chord_len = float(np.linalg.norm(chord))
    if chord_len == 0.0:
        return int(scores[0].k)

    points = np.column_stack([ks, inertias]) - start
    # Модуль векторного произведения в 2D, делённый на длину хорды, — это и
    # есть расстояние от точки до прямой.
    deviation = np.abs(points[:, 0] * chord[1] - points[:, 1] * chord[0]) / chord_len
    return int(scores[int(np.argmax(deviation))].k)


def _rescale(values: np.ndarray) -> np.ndarray:
    spread = float(values.max() - values.min())
    if spread == 0.0:
        return np.zeros_like(values)
    return (values - values.min()) / spread
