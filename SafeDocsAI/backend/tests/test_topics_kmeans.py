"""K-means собственной реализации: находит ли он кластеры и воспроизводим ли.

Все данные синтетические и порождаются с фиксированным seed: только там
правильный ответ известен заранее, и «алгоритм нашёл кластеры» — проверяемое
утверждение, а не впечатление от картинки.

Сквозное правило файла: разбиение сравнивается как разбиение, а не по номерам
меток. Номера кластеров произвольны — тот же результат при другом порядке
инициализации получит метки (1, 2, 0) вместо (0, 1, 2), и посимвольное
сравнение краснело бы на верном ответе.
"""

import os
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.modules.topics.kmeans import (  # noqa: E402
    KMeans,
    choose_k,
    kmeans_plus_plus_init,
    l2_normalize,
    squared_distances,
)
from app.modules.topics.metrics import adjusted_rand_index  # noqa: E402

WELL_SEPARATED_CENTERS = np.array([[5.0, 5.0], [-5.0, 5.0], [0.0, -6.0]])


def make_blobs(centers=WELL_SEPARATED_CENTERS, per_center=60, spread=0.5, seed=0):
    """Хорошо разделимые облака: расстояние между центрами на порядок больше
    разброса внутри облака, поэтому верное разбиение единственно."""
    rng = np.random.default_rng(seed)
    points = np.vstack(
        [center + rng.normal(0.0, spread, (per_center, centers.shape[1])) for center in centers]
    )
    labels = np.repeat(np.arange(centers.shape[0]), per_center)
    return points, labels


def partition_of(labels):
    """Разбиение как множество групп индексов — форма, не зависящая от номеров."""
    labels = np.asarray(labels)
    return {frozenset(np.flatnonzero(labels == value).tolist()) for value in np.unique(labels)}


class FindsSyntheticClustersTests(unittest.TestCase):
    def test_recovers_the_generated_partition(self):
        X, true_labels = make_blobs()
        model = KMeans(n_clusters=3, random_state=42, normalize=False).fit(X)
        self.assertEqual(partition_of(model.labels_), partition_of(true_labels))

    def test_ari_is_exactly_one_on_separable_data(self):
        """ARI = 1 — то же утверждение в виде числа, которое можно назвать вслух."""
        X, true_labels = make_blobs(seed=1)
        model = KMeans(n_clusters=3, random_state=42, normalize=False).fit(X)
        self.assertEqual(adjusted_rand_index(true_labels, model.labels_), 1.0)

    def test_survives_high_dimensional_data(self):
        """Размерность эмбеддингов qwen3 — тысячи; на двумерной игрушке
        проверка ничего не сказала бы про боевой режим."""
        rng = np.random.default_rng(4)
        centers = rng.normal(0.0, 6.0, (4, 512))
        X, true_labels = make_blobs(centers=centers, per_center=25, spread=0.4, seed=5)
        model = KMeans(n_clusters=4, random_state=7, normalize=False).fit(X)
        self.assertEqual(adjusted_rand_index(true_labels, model.labels_), 1.0)

    def test_attributes_are_filled_after_fit(self):
        X, _ = make_blobs()
        model = KMeans(n_clusters=3, random_state=0, normalize=False).fit(X)
        self.assertEqual(model.centroids_.shape, (3, 2))
        self.assertEqual(model.labels_.shape, (X.shape[0],))
        self.assertGreater(model.inertia_, 0.0)
        self.assertGreaterEqual(model.n_iter_, 1)

    def test_inertia_attribute_matches_the_definition(self):
        """inertia_ обязана быть суммой квадратов до своего центроида, а не
        каким-то внутренним накопителем, разошедшимся с определением."""
        X, _ = make_blobs(seed=2)
        model = KMeans(n_clusters=3, random_state=0, normalize=False).fit(X)
        difference = X - model.centroids_[model.labels_]
        self.assertAlmostEqual(model.inertia_, float(np.sum(difference**2)), places=8)


class DeterminismTests(unittest.TestCase):
    """Число, названное на защите, должно воспроизводиться."""

    def test_same_seed_gives_identical_result(self):
        X, _ = make_blobs(seed=3)
        first = KMeans(n_clusters=4, random_state=123, normalize=False).fit(X)
        second = KMeans(n_clusters=4, random_state=123, normalize=False).fit(X)
        np.testing.assert_array_equal(first.labels_, second.labels_)
        np.testing.assert_allclose(first.centroids_, second.centroids_)
        self.assertEqual(first.inertia_, second.inertia_)
        self.assertEqual(first.n_iter_, second.n_iter_)

    def test_the_same_model_refitted_repeats_itself(self):
        X, _ = make_blobs(seed=3)
        model = KMeans(n_clusters=4, random_state=123, normalize=False)
        first = model.fit_predict(X).copy()
        second = model.fit_predict(X)
        np.testing.assert_array_equal(first, second)

    def test_without_a_seed_runs_are_free_to_differ(self):
        """Обратная сторона: детерминизм даёт именно random_state, а не удача.

        На данных, где локальных минимумов много (равномерный шум без структуры),
        разные старты дают разную инерцию. Тест не требует, чтобы они разошлись
        обязательно, — он требует, чтобы seed при этом всё равно фиксировал итог.
        """
        rng = np.random.default_rng(9)
        X = rng.normal(size=(120, 8))
        seeded = {
            KMeans(n_clusters=6, random_state=17, n_init=1, normalize=False).fit(X).inertia_
            for _ in range(3)
        }
        self.assertEqual(len(seeded), 1)


class MultipleRestartsTests(unittest.TestCase):
    def test_more_restarts_never_lose_to_one(self):
        """n_init перебирает старты и берёт лучший, поэтому проигрыш одному
        запуску означал бы ошибку в выборе минимума.

        Оба прогона делят один seed, значит первый старт у них общий: инерция
        best-of-10 не может оказаться хуже инерции этого первого старта.
        """
        rng = np.random.default_rng(21)
        X = rng.normal(size=(150, 6))
        single = KMeans(n_clusters=7, n_init=1, random_state=5, normalize=False).fit(X)
        many = KMeans(n_clusters=7, n_init=10, random_state=5, normalize=False).fit(X)
        self.assertLessEqual(many.inertia_, single.inertia_)

    def test_restarts_actually_help_somewhere(self):
        """Проверка не только на «не хуже», но и на «иногда лучше»: условие
        «не хуже» выполнил бы и код, который n_init молча игнорирует."""
        rng = np.random.default_rng(33)
        X = rng.normal(size=(200, 4))
        improved = 0
        for seed in range(12):
            single = KMeans(n_clusters=8, n_init=1, random_state=seed, normalize=False).fit(X)
            many = KMeans(n_clusters=8, n_init=10, random_state=seed, normalize=False).fit(X)
            if many.inertia_ < single.inertia_ - 1e-9:
                improved += 1
        self.assertGreater(improved, 0)


class KMeansPlusPlusTests(unittest.TestCase):
    def test_initial_centroids_are_real_points(self):
        """Стартовые центры берутся из данных, а не из воздуха: иначе первый
        же шаг мог бы стартовать в пустой области пространства."""
        X, _ = make_blobs(seed=6)
        centroids = kmeans_plus_plus_init(X, 3, np.random.default_rng(0))
        for centroid in centroids:
            self.assertTrue(np.any(np.all(np.isclose(X, centroid), axis=1)))

    def test_init_is_deterministic_for_a_given_generator(self):
        X, _ = make_blobs(seed=6)
        first = kmeans_plus_plus_init(X, 4, np.random.default_rng(11))
        second = kmeans_plus_plus_init(X, 4, np.random.default_rng(11))
        np.testing.assert_array_equal(first, second)

    def test_it_beats_a_uniformly_random_start(self):
        """Главный довод в пользу k-means++ — измеримый.

        Данные: три облака сильно разного размера. Равновероятный выбор точек
        сажает два центра в самое многочисленное облако тем чаще, чем сильнее
        перекос размеров, и маленькое облако остаётся без центра. k-means++
        берёт следующий центр пропорционально квадрату расстояния, поэтому
        далёкое облако получает центр почти всегда.

        Сравнивается доля стартов, накрывших все три облака.
        """
        centers = np.array([[0.0, 0.0], [25.0, 0.0], [0.0, 25.0]])
        rng = np.random.default_rng(2)
        X = np.vstack(
            [
                centers[0] + rng.normal(0.0, 1.0, (400, 2)),
                centers[1] + rng.normal(0.0, 1.0, (20, 2)),
                centers[2] + rng.normal(0.0, 1.0, (20, 2)),
            ]
        )

        def covers_all_blobs(centroids):
            nearest_blob = np.argmin(squared_distances(centroids, centers), axis=1)
            return len(np.unique(nearest_blob)) == 3

        trials = 200
        plus_plus = sum(
            covers_all_blobs(kmeans_plus_plus_init(X, 3, np.random.default_rng(seed)))
            for seed in range(trials)
        )
        uniform = 0
        for seed in range(trials):
            picker = np.random.default_rng(seed)
            uniform += covers_all_blobs(X[picker.choice(X.shape[0], size=3, replace=False)])

        # Наблюдаемые доли: k-means++ около 0.86, равновероятный старт около
        # 0.015. Пороги взяты с большим запасом — тест про разницу на порядок,
        # а не про конкретную цифру.
        self.assertGreaterEqual(plus_plus / trials, 0.7)
        self.assertLess(uniform / trials, 0.1)
        self.assertGreater(plus_plus, 5 * uniform)


class ConvergenceTests(unittest.TestCase):
    def test_max_iter_is_a_hard_ceiling(self):
        X, _ = make_blobs(seed=8)
        model = KMeans(n_clusters=3, max_iter=1, random_state=0, normalize=False).fit(X)
        self.assertEqual(model.n_iter_, 1)

    def test_a_loose_tol_stops_immediately(self):
        """tol — второй критерий, и он должен действительно останавливать."""
        X, _ = make_blobs(seed=8)
        model = KMeans(n_clusters=3, tol=1e9, random_state=0, normalize=False).fit(X)
        self.assertEqual(model.n_iter_, 1)

    def test_a_strict_tol_still_terminates(self):
        """Без потолка max_iter нулевой tol крутил бы цикл бесконечно."""
        X, _ = make_blobs(seed=8)
        model = KMeans(
            n_clusters=3, tol=0.0, max_iter=50, random_state=0, normalize=False
        ).fit(X)
        self.assertLessEqual(model.n_iter_, 50)

    def test_converged_run_stops_early(self):
        """На разделимых данных сходимость наступает задолго до max_iter —
        иначе tol не работает, а работает только потолок."""
        X, _ = make_blobs(seed=8)
        model = KMeans(n_clusters=3, max_iter=300, random_state=0, normalize=False).fit(X)
        self.assertLess(model.n_iter_, 20)

    def test_labels_match_the_final_centroids(self):
        """Инвариант «labels отвечают centroids» обязан пережить любой выход из
        цикла, в том числе обрыв по max_iter."""
        X, _ = make_blobs(seed=8)
        model = KMeans(n_clusters=3, max_iter=1, random_state=0, normalize=False).fit(X)
        expected = np.argmin(squared_distances(X, model.centroids_), axis=1)
        np.testing.assert_array_equal(model.labels_, expected)


class EmptyClusterTests(unittest.TestCase):
    """Пустой кластер — крайний случай, который любят спрашивать.

    Найти его на обычных данных трудно: k-means++ разносит старты, и при
    осмысленном k кластер не пустеет. Надёжный источник — данные, где различных
    точек меньше, чем запрошено кластеров: дубликаты документов, которых в базе
    сколько угодно.
    """

    DUPLICATES = np.vstack([np.zeros((20, 2)), np.array([[100.0, 100.0]])])

    def _fit_watching_revival(self, X, n_clusters, seed=0):
        original = KMeans._revive_empty_clusters
        seen = []

        def spy(points, labels, distances, new_centroids, empty):
            seen.append(tuple(empty.tolist()))
            return original(points, labels, distances, new_centroids, empty)

        with patch.object(KMeans, "_revive_empty_clusters", staticmethod(spy)):
            model = KMeans(n_clusters, random_state=seed, n_init=1, normalize=False).fit(X)
        return model, seen

    def test_the_constructed_input_really_empties_a_cluster(self):
        """Сторож на сам тест: если вход перестанет вызывать пустой кластер,
        остальные проверки этого класса станут проверять пустоту."""
        _, seen = self._fit_watching_revival(self.DUPLICATES, 3)
        self.assertTrue(seen)

    def test_k_centroids_are_returned_even_so(self):
        """Молча вернуть k-1 кластер там, где просили k, — худший исход:
        вызывающий код рассчитывает на k тем и не узнает, что их стало меньше."""
        model, _ = self._fit_watching_revival(self.DUPLICATES, 3)
        self.assertEqual(model.centroids_.shape[0], 3)
        self.assertTrue(np.all(np.isfinite(model.centroids_)))

    def test_result_stays_valid(self):
        model, _ = self._fit_watching_revival(self.DUPLICATES, 3)
        self.assertTrue(np.all(model.labels_ >= 0))
        self.assertTrue(np.all(model.labels_ < 3))
        self.assertTrue(np.isfinite(model.inertia_))

    def test_the_revived_centroid_takes_the_costliest_point(self):
        """Принятое решение проверяется буквально: центроид пустого кластера
        переезжает в точку с наибольшим квадратом расстояния до СВОЕГО
        центроида — ту, что вносит наибольший вклад в инерцию."""
        X = np.array([[0.0, 0.0], [0.0, 1.0], [0.0, 2.0], [0.0, 30.0]])
        labels = np.array([0, 0, 0, 0])  # кластер 1 пуст
        centroids = np.array([[0.0, 0.0], [50.0, 50.0]])
        model = KMeans(n_clusters=2, normalize=False)
        updated = model._update_centroids(X, labels, centroids, squared_distances(X, centroids))
        np.testing.assert_allclose(updated[1], X[3])

    def test_two_empty_clusters_take_two_different_points(self):
        """Иначе оба центроида встали бы в одну точку и один снова опустел бы."""
        X = np.array([[0.0, 0.0], [0.0, 1.0], [0.0, 20.0], [0.0, 30.0]])
        labels = np.zeros(4, dtype=int)
        centroids = np.array([[0.0, 0.0], [50.0, 50.0], [60.0, 60.0]])
        model = KMeans(n_clusters=3, normalize=False)
        updated = model._update_centroids(X, labels, centroids, squared_distances(X, centroids))
        self.assertFalse(np.allclose(updated[1], updated[2]))
        np.testing.assert_allclose(updated[1], X[3])
        np.testing.assert_allclose(updated[2], X[2])

    def test_non_empty_clusters_are_left_as_means(self):
        """Лечение пустых не должно трогать остальные центроиды."""
        X = np.array([[0.0, 0.0], [0.0, 2.0], [10.0, 10.0]])
        labels = np.array([0, 0, 0])
        centroids = np.array([[0.0, 0.0], [99.0, 99.0]])
        model = KMeans(n_clusters=2, normalize=False)
        updated = model._update_centroids(X, labels, centroids, squared_distances(X, centroids))
        np.testing.assert_allclose(updated[0], X.mean(axis=0))


class DegenerateInputTests(unittest.TestCase):
    def test_single_point(self):
        model = KMeans(n_clusters=1, random_state=0, normalize=False).fit(np.array([[1.0, 2.0]]))
        np.testing.assert_array_equal(model.labels_, [0])
        self.assertEqual(model.inertia_, 0.0)
        np.testing.assert_allclose(model.centroids_, [[1.0, 2.0]])

    def test_k_equals_number_of_points(self):
        """Каждая точка — свой кластер, инерция строго ноль."""
        rng = np.random.default_rng(15)
        X = rng.normal(size=(6, 3))
        model = KMeans(n_clusters=6, random_state=0, normalize=False).fit(X)
        self.assertEqual(len(np.unique(model.labels_)), 6)
        self.assertAlmostEqual(model.inertia_, 0.0, places=10)

    def test_all_points_identical(self):
        """Разделять нечего: инерция ноль, зацикливания нет, k центроидов есть."""
        X = np.ones((10, 4))
        model = KMeans(n_clusters=3, random_state=0, normalize=False).fit(X)
        self.assertAlmostEqual(model.inertia_, 0.0, places=10)
        self.assertEqual(model.centroids_.shape[0], 3)
        self.assertLessEqual(model.n_iter_, 2)

    def test_k_larger_than_sample_is_refused(self):
        """Молчать нельзя: k кластеров из трёх точек не получится никак, и
        вернуть меньше — значит соврать вызывающему коду."""
        with self.assertRaises(ValueError):
            KMeans(n_clusters=5, normalize=False).fit(np.zeros((3, 2)))

    def test_empty_input_is_refused(self):
        with self.assertRaises(ValueError):
            KMeans(n_clusters=1, normalize=False).fit(np.zeros((0, 2)))

    def test_non_finite_values_are_refused(self):
        """NaN не роняет argmin и среднее — он тихо портит результат."""
        X = np.array([[1.0, 2.0], [np.nan, 1.0]])
        with self.assertRaises(ValueError):
            KMeans(n_clusters=1, normalize=False).fit(X)

    def test_one_dimensional_input_is_refused(self):
        with self.assertRaises(ValueError):
            KMeans(n_clusters=1, normalize=False).fit(np.array([1.0, 2.0, 3.0]))

    def test_bad_hyperparameters_are_refused_at_construction(self):
        for kwargs in ({"n_clusters": 0}, {"n_init": 0}, {"max_iter": 0}, {"tol": -1.0}):
            with self.subTest(**kwargs):
                with self.assertRaises(ValueError):
                    KMeans(**{"n_clusters": 3, **kwargs})


class CosineTests(unittest.TestCase):
    """Косинус против евклида на текстовых данных.

    Данные подобраны так, чтобы разница была видна: три группы различаются
    направлением, но длина вектора внутри группы гуляет в 400 раз. Для текстов
    это норма — длина эмбеддинга зависит от длины документа, а не от темы.
    Евклид в такой ситуации кластеризует по длине, косинус — по направлению.
    """

    @staticmethod
    def directional_blobs():
        rng = np.random.default_rng(11)
        directions = np.eye(3)
        rows, labels = [], []
        for index, direction in enumerate(directions):
            noisy = direction + rng.normal(0.0, 0.08, (40, 3))
            noisy /= np.linalg.norm(noisy, axis=1, keepdims=True)
            rows.append(noisy * rng.uniform(0.05, 20.0, (40, 1)))
            labels += [index] * 40
        return np.vstack(rows), np.array(labels)

    def test_normalization_recovers_the_topics(self):
        X, true_labels = self.directional_blobs()
        model = KMeans(n_clusters=3, random_state=0, normalize=True).fit(X)
        self.assertEqual(adjusted_rand_index(true_labels, model.labels_), 1.0)

    def test_euclidean_on_raw_vectors_does_worse(self):
        """Довод в пользу нормировки — измеримый, а не рассуждение."""
        X, true_labels = self.directional_blobs()
        cosine = KMeans(n_clusters=3, random_state=0, normalize=True).fit(X)
        euclidean = KMeans(n_clusters=3, random_state=0, normalize=False).fit(X)
        self.assertGreater(
            adjusted_rand_index(true_labels, cosine.labels_),
            adjusted_rand_index(true_labels, euclidean.labels_) + 0.3,
        )

    def test_normalization_ignores_vector_length(self):
        """Проверка самой эквивалентности: удлинение вектора не меняет темы,
        потому что косинус от длины не зависит."""
        X, _ = self.directional_blobs()
        scaled = X * np.linspace(0.1, 50.0, X.shape[0])[:, np.newaxis]
        first = KMeans(n_clusters=3, random_state=0, normalize=True).fit_predict(X)
        second = KMeans(n_clusters=3, random_state=0, normalize=True).fit_predict(scaled)
        self.assertEqual(partition_of(first), partition_of(second))

    def test_l2_normalize_survives_zero_rows(self):
        """Нулевой вектор нормировать нечем; деления на ноль быть не должно."""
        normalized = l2_normalize(np.array([[3.0, 4.0], [0.0, 0.0]]))
        np.testing.assert_allclose(normalized[0], [0.6, 0.8])
        np.testing.assert_allclose(normalized[1], [0.0, 0.0])

    def test_centroids_live_in_the_normalized_space(self):
        """Документированное поведение: при normalize=True centroids_ заданы в
        нормированном пространстве, поэтому их длина не больше единицы."""
        X, _ = self.directional_blobs()
        model = KMeans(n_clusters=3, random_state=0, normalize=True).fit(X)
        self.assertTrue(np.all(np.linalg.norm(model.centroids_, axis=1) <= 1.0 + 1e-9))


class PredictTests(unittest.TestCase):
    def test_predict_on_training_data_repeats_labels(self):
        X, _ = make_blobs(seed=12)
        model = KMeans(n_clusters=3, random_state=0, normalize=False).fit(X)
        np.testing.assert_array_equal(model.predict(X), model.labels_)

    def test_predict_assigns_new_points_to_the_nearest_centroid(self):
        X, _ = make_blobs(seed=12)
        model = KMeans(n_clusters=3, random_state=0, normalize=False).fit(X)
        for center in WELL_SEPARATED_CENTERS:
            expected = int(np.argmin(np.sum((model.centroids_ - center) ** 2, axis=1)))
            self.assertEqual(int(model.predict(center[np.newaxis, :])[0]), expected)

    def test_predict_does_not_change_the_model(self):
        X, _ = make_blobs(seed=12)
        model = KMeans(n_clusters=3, random_state=0, normalize=False).fit(X)
        before = model.centroids_.copy()
        model.predict(np.zeros((5, 2)))
        np.testing.assert_array_equal(model.centroids_, before)

    def test_predict_before_fit_is_refused(self):
        with self.assertRaises(ValueError):
            KMeans(n_clusters=2).predict(np.zeros((3, 2)))

    def test_wrong_feature_count_is_refused(self):
        """Иначе broadcast дал бы не ошибку, а бессмысленные метки."""
        X, _ = make_blobs(seed=12)
        model = KMeans(n_clusters=3, random_state=0, normalize=False).fit(X)
        with self.assertRaises(ValueError):
            model.predict(np.zeros((4, 5)))

    def test_fit_predict_matches_fit_then_labels(self):
        X, _ = make_blobs(seed=12)
        by_fit = KMeans(n_clusters=3, random_state=0, normalize=False).fit(X).labels_
        by_shortcut = KMeans(n_clusters=3, random_state=0, normalize=False).fit_predict(X)
        np.testing.assert_array_equal(by_fit, by_shortcut)


class SquaredDistancesTests(unittest.TestCase):
    """Матричная формула должна давать ровно то же, что честный цикл."""

    def test_matches_the_naive_loop(self):
        rng = np.random.default_rng(19)
        X = rng.normal(size=(30, 7))
        centroids = rng.normal(size=(4, 7))
        expected = np.array(
            [[float(np.sum((point - centroid) ** 2)) for centroid in centroids] for point in X]
        )
        np.testing.assert_allclose(squared_distances(X, centroids), expected, atol=1e-9)

    def test_never_returns_a_negative_distance(self):
        """Разность больших близких чисел даёт -1e-13 вместо нуля, и инерция
        становится отрицательной. Отсечение по нулю обязано быть."""
        X = np.full((5, 3), 1e6)
        self.assertTrue(np.all(squared_distances(X, X[:1]) >= 0.0))


class ChooseKTests(unittest.TestCase):
    def test_finds_the_generated_number_of_clusters(self):
        X, _ = make_blobs(seed=20)
        result = choose_k(X, range(2, 8), random_state=42, normalize=False)
        self.assertEqual(result.best_k, 3)

    def test_elbow_agrees_on_clean_data(self):
        """На разделимых данных два независимых критерия обязаны сойтись;
        расхождение — это повод для разговора, а не молчаливый выбор."""
        X, _ = make_blobs(seed=20)
        result = choose_k(X, range(2, 8), random_state=42, normalize=False)
        self.assertEqual(result.elbow_k, 3)

    def test_the_table_is_returned_whole(self):
        """Ради графика на защите: автоматический выбор ничего не прячет."""
        X, _ = make_blobs(seed=20)
        result = choose_k(X, range(2, 8), random_state=42, normalize=False)
        self.assertEqual([score.k for score in result.scores], list(range(2, 8)))
        for score in result.scores:
            self.assertGreater(score.inertia, 0.0)
            self.assertGreaterEqual(score.n_iter, 1)

    def test_inertia_falls_as_k_grows(self):
        """Ровно та причина, по которой по одной инерции k выбрать нельзя."""
        X, _ = make_blobs(seed=20)
        result = choose_k(X, range(2, 9), random_state=42, normalize=False)
        inertias = [score.inertia for score in result.scores]
        for previous, current in zip(inertias, inertias[1:]):
            self.assertLessEqual(current, previous + 1e-9)

    def test_silhouette_peaks_at_the_true_k(self):
        X, _ = make_blobs(seed=20)
        result = choose_k(X, range(2, 8), random_state=42, normalize=False)
        best = max(result.scores, key=lambda score: score.silhouette)
        self.assertEqual(best.k, 3)
        self.assertGreater(best.silhouette, 0.8)

    def test_k_equal_one_gets_no_silhouette(self):
        """NaN, а не ноль: величина неприменима, а не «плохая»."""
        X, _ = make_blobs(seed=20)
        result = choose_k(X, [1, 2, 3], random_state=42, normalize=False)
        by_k = {score.k: score for score in result.scores}
        self.assertTrue(np.isnan(by_k[1].silhouette))
        self.assertFalse(np.isnan(by_k[3].silhouette))
        self.assertEqual(result.best_k, 3)

    def test_choose_k_is_deterministic(self):
        X, _ = make_blobs(seed=20)
        first = choose_k(X, range(2, 6), random_state=1, normalize=False)
        second = choose_k(X, range(2, 6), random_state=1, normalize=False)
        self.assertEqual(first, second)

    def test_out_of_range_k_is_refused(self):
        X, _ = make_blobs(per_center=3, seed=20)
        with self.assertRaises(ValueError):
            choose_k(X, [2, 1000], random_state=0, normalize=False)

    def test_normalized_search_works_on_directional_data(self):
        X, _ = CosineTests.directional_blobs()
        result = choose_k(X, range(2, 7), random_state=0, normalize=True)
        self.assertEqual(result.best_k, 3)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
