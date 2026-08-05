"""Метрики качества кластеризации — собственные реализации, без sklearn.

Метрики делятся на две группы, и разница между ними важнее формул.

Внутренние (инерция, силуэт) смотрят только на геометрию: насколько точки
внутри кластера ближе друг к другу, чем к соседним кластерам. Разметка им не
нужна — но и сказать, попал ли алгоритм в настоящие темы, они не могут.
Компактное разбиение может резать одну тему пополам и получать хороший силуэт.

Внешние (purity, ARI) сравнивают разбиение с известными настоящими темами —
у нас это разделы сайтов. Они сильнее, потому что проверяют совпадение с
реальностью, а не с представлением алгоритма о красоте; но требуют разметки,
которой на боевых данных обычно нет.
"""

from __future__ import annotations

import numpy as np


def _as_labels(labels) -> np.ndarray:
    array = np.asarray(labels)
    if array.ndim != 1:
        raise ValueError("метки должны быть одномерным массивом")
    if array.size == 0:
        raise ValueError("пустой массив меток")
    return array


def _pairwise_distances(X: np.ndarray) -> np.ndarray:
    """Полная матрица евклидовых расстояний (n, n).

    Раскрытие квадрата разности, как и в kmeans.squared_distances: сумма
    квадратов норм минус удвоенные скалярные произведения. Здесь корень нужен —
    силуэт складывает расстояния, а не их квадраты, и на квадратах отношение
    (b - a) / max(a, b) дало бы другое число.
    """
    sq_norms = np.einsum("ij,ij->i", X, X)
    distances = sq_norms[:, np.newaxis] - 2.0 * (X @ X.T) + sq_norms[np.newaxis, :]
    np.maximum(distances, 0.0, out=distances)
    distances = np.sqrt(distances)
    # Диагональ обязана быть строго нулевой: округление даёт там 1e-8, и это
    # значение попало бы в среднее расстояние точки до своих соседей.
    np.fill_diagonal(distances, 0.0)
    return distances


# --- внутренние метрики ------------------------------------------------------


def inertia(X: np.ndarray, labels, centroids: np.ndarray) -> float:
    """Сумма квадратов расстояний от точек до своего центроида.

    Ровно та величина, которую минимизирует K-means, — поэтому она годится
    сравнивать перезапуски между собой, но не годится сравнивать разные k:
    инерция монотонно убывает с ростом k и при k = n равна нулю.
    """
    data = np.asarray(X, dtype=np.float64)
    centers = np.asarray(centroids, dtype=np.float64)
    label_array = _as_labels(labels)
    if label_array.shape[0] != data.shape[0]:
        raise ValueError("число меток не совпадает с числом точек")
    if label_array.max() >= centers.shape[0] or label_array.min() < 0:
        raise ValueError("метка выходит за пределы списка центроидов")

    diff = data - centers[label_array]
    return float(np.einsum("ij,ij->", diff, diff))


def silhouette_samples(X: np.ndarray, labels) -> np.ndarray:
    """Силуэт каждой точки: (b - a) / max(a, b) в диапазоне [-1, 1].

    a — среднее расстояние до остальных точек СВОЕГО кластера (насколько точке
    там тесно), b — минимальное по чужим кластерам среднее расстояние до
    кластера целиком (насколько близок ближайший сосед). Значение около 1 —
    точка уверенно своя; около 0 — лежит на границе; отрицательное — ей
    ближе чужой кластер, чем свой.

    Одиночному кластеру приписывается 0: величины a у него нет (не с кем
    усреднять), а выдавать 1 значило бы награждать разбиение за выброс,
    отсаженный в собственный кластер.

    Считается точно, по полной матрице расстояний: память O(n^2). Для больших
    выборок берите silhouette_score с sample_size.
    """
    data = np.asarray(X, dtype=np.float64)
    label_array = _as_labels(labels)
    if label_array.shape[0] != data.shape[0]:
        raise ValueError("число меток не совпадает с числом точек")

    unique = np.unique(label_array)
    if unique.size < 2:
        raise ValueError("силуэт не определён при одном кластере")
    if unique.size >= data.shape[0]:
        raise ValueError("силуэт не определён, когда каждая точка — свой кластер")

    distances = _pairwise_distances(data)

    # cluster_sums[i, c] — суммарное расстояние от точки i до всех точек
    # кластера c. Одна матрица вместо двух вложенных циклов по точкам и
    # кластерам; из неё делением на размеры получаются и a, и все кандидаты в b.
    n_samples = data.shape[0]
    cluster_sums = np.zeros((n_samples, unique.size), dtype=np.float64)
    sizes = np.zeros(unique.size, dtype=np.int64)
    for position, cluster in enumerate(unique):
        members = label_array == cluster
        sizes[position] = int(members.sum())
        cluster_sums[:, position] = distances[:, members].sum(axis=1)

    # Позиция своего кластера в unique — не сама метка: метки могут быть любыми
    # числами, а колонки нумеруются подряд.
    own = np.searchsorted(unique, label_array)
    rows = np.arange(n_samples)

    own_sizes = sizes[own]
    # Расстояние точки до самой себя равно нулю и в сумму ничего не вносит, но
    # делить надо на размер кластера без неё самой.
    with np.errstate(divide="ignore", invalid="ignore"):
        a = np.where(own_sizes > 1, cluster_sums[rows, own] / (own_sizes - 1), 0.0)

    foreign = cluster_sums / sizes[np.newaxis, :]
    foreign[rows, own] = np.inf  # свой кластер в минимум по чужим не попадает
    b = foreign.min(axis=1)

    # Знаменатель равен нулю, когда точка совпадает со всеми соседями сразу
    # (полные дубликаты документов): и a, и b равны нулю, силуэт — ноль.
    denominator = np.maximum(a, b)
    silhouettes = np.divide(
        b - a, denominator, out=np.zeros_like(a), where=denominator > 0
    )
    # Одиночки: a не определено, поэтому формула к ним не применяется.
    return np.where(own_sizes > 1, silhouettes, 0.0)


def _stratified_sample(
    labels: np.ndarray, sample_size: int, random_state: int | None
) -> np.ndarray:
    """Индексы подвыборки, сохраняющей все кластеры.

    Равномерная случайная выборка может целиком потерять мелкий кластер, и
    тогда оценка описывала бы уже другое разбиение. Поэтому доля берётся из
    каждого кластера отдельно, и не меньше двух точек: при одной точке в
    кластере силуэт вырождается в ноль по определению.
    """
    rng = np.random.default_rng(random_state)
    unique = np.unique(labels)
    n_samples = labels.shape[0]

    chosen: list[np.ndarray] = []
    for cluster in unique:
        members = np.flatnonzero(labels == cluster)
        share = int(round(sample_size * members.size / n_samples))
        take = min(members.size, max(2, share))
        # replace=False — точка не должна попасть в выборку дважды: дубликат
        # даёт нулевое расстояние сам до себя и завышает плотность кластера.
        chosen.append(rng.choice(members, size=take, replace=False))

    # Сортировка нужна не для алгоритма, а для воспроизводимости порядка строк.
    return np.sort(np.concatenate(chosen))


def silhouette_score(
    X: np.ndarray,
    labels,
    *,
    sample_size: int | None = None,
    random_state: int | None = None,
) -> float:
    """Средний силуэт по всем точкам.

    ВНИМАНИЕ: при заданном sample_size это ОЦЕНКА, а не точное значение.
    Точный силуэт требует матрицы расстояний n x n, что на десятках тысяч
    документов упирается в память; поэтому силуэт считается по подвыборке из
    примерно sample_size точек, и результат зависит от того, какие точки в неё
    попали. Разброс между разными подвыборками того же размера — обычное дело,
    сравнивать такие числа между собой можно только при одинаковых sample_size
    и random_state. random_state фиксирует подвыборку, чтобы повторный запуск
    дал то же число; он не делает оценку точной.

    sample_size=None или выборка меньше него — считается точно.
    """
    data = np.asarray(X, dtype=np.float64)
    label_array = _as_labels(labels)

    if sample_size is not None and data.shape[0] > sample_size:
        indices = _stratified_sample(label_array, sample_size, random_state)
        data = data[indices]
        label_array = label_array[indices]

    return float(np.mean(silhouette_samples(data, label_array)))


def silhouette_per_cluster(
    X: np.ndarray,
    labels,
    *,
    sample_size: int | None = None,
    random_state: int | None = None,
) -> dict[int, float]:
    """Средний силуэт внутри каждого кластера.

    Общее среднее прячет главное: обычно один-два кластера собраны хорошо, а
    остальное — свалка. Разбивка по кластерам показывает, какую именно тему
    алгоритм не разделил. Про оценочный характер sample_size — см.
    silhouette_score.
    """
    data = np.asarray(X, dtype=np.float64)
    label_array = _as_labels(labels)

    if sample_size is not None and data.shape[0] > sample_size:
        indices = _stratified_sample(label_array, sample_size, random_state)
        data = data[indices]
        label_array = label_array[indices]

    values = silhouette_samples(data, label_array)
    return {
        int(cluster): float(values[label_array == cluster].mean())
        for cluster in np.unique(label_array)
    }


# --- внешние метрики (нужна разметка) ----------------------------------------


def contingency_matrix(labels_true, labels_pred) -> np.ndarray:
    """Таблица сопряжённости: строки — настоящие темы, столбцы — кластеры.

    Общая заготовка для purity и ARI: обе метрики — это разные свёртки одной и
    той же таблицы, и считать её дважды незачем.
    """
    true_array = _as_labels(labels_true)
    pred_array = _as_labels(labels_pred)
    if true_array.shape[0] != pred_array.shape[0]:
        raise ValueError("длины разметки и предсказания не совпадают")

    # Метки могут быть строками ('Новости', 'Спорт') или произвольными числами,
    # поэтому обе стороны переводятся в подряд идущие индексы.
    _, true_index = np.unique(true_array, return_inverse=True)
    _, pred_index = np.unique(pred_array, return_inverse=True)

    matrix = np.zeros((true_index.max() + 1, pred_index.max() + 1), dtype=np.int64)
    np.add.at(matrix, (true_index, pred_index), 1)
    return matrix


def purity(labels_true, labels_pred) -> float:
    """Чистота: доля точек, попавших в кластер вместе со своим большинством.

    Каждому кластеру приписывается самая частая в нём настоящая тема; чистота —
    доля точек, у которых тема совпала с этой. Диапазон (0, 1], 1 — ни один
    кластер не смешал две темы.

    Слабое место, которое обязательно спросят: чистота растёт с ростом числа
    кластеров и равна единице, когда каждая точка — свой кластер. Поэтому она
    осмысленна только при сравнении разбиений с одинаковым k, а как
    самостоятельный критерий выбора k не годится — для этого нужен ARI, который
    за дробление наказывает.
    """
    matrix = contingency_matrix(labels_true, labels_pred)
    return float(matrix.max(axis=0).sum() / matrix.sum())


def adjusted_rand_index(labels_true, labels_pred) -> float:
    """Скорректированный индекс Рэнда (ARI).

    Индекс Рэнда считает пары точек: сколько пар согласованы (лежат вместе в
    обоих разбиениях либо порознь в обоих). Беда обычного индекса Рэнда в том,
    что случайное разбиение получает у него не ноль, а вполне приличные 0.6-0.7
    — просто потому, что пар «порознь» всегда много. Поправка вычитает
    ожидание при случайном разбиении с теми же размерами кластеров:

        ARI = (index - expected) / (max_index - expected)

    Отсюда шкала: 1 — полное совпадение с разметкой (нумерация кластеров при
    этом может быть любой, метрика сравнивает разбиения, а не номера), около 0
    — совпадение на уровне случайного угадывания, отрицательное — согласие хуже
    случайного.
    """
    matrix = contingency_matrix(labels_true, labels_pred)
    n_samples = int(matrix.sum())

    # Вырожденные случаи, где знаменатель обращается в ноль: одно разбиение на
    # всех либо каждая точка сама по себе. Пар для сравнения нет, и оба
    # разбиения тривиально совпадают, поэтому 1.0, а не деление на ноль.
    n_true, n_pred = matrix.shape
    if n_true == n_pred == 1 or n_true == n_pred == n_samples:
        return 1.0

    def pairs(counts: np.ndarray) -> float:
        """Число пар внутри группы: C(n, 2) = n * (n - 1) / 2."""
        counts = counts.astype(np.float64)
        return float(np.sum(counts * (counts - 1.0) / 2.0))

    index = pairs(matrix)
    true_pairs = pairs(matrix.sum(axis=1))
    pred_pairs = pairs(matrix.sum(axis=0))
    total_pairs = n_samples * (n_samples - 1) / 2.0

    expected = true_pairs * pred_pairs / total_pairs
    max_index = (true_pairs + pred_pairs) / 2.0
    if max_index == expected:
        return 1.0
    return float((index - expected) / (max_index - expected))
