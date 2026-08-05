"""Метрики качества кластеризации: значения на заведомо известных разбиениях.

Метрику нельзя проверить «на глаз по реальным данным» — там неизвестен верный
ответ, и любое число выглядит правдоподобно. Поэтому здесь только разбиения,
для которых результат посчитан руками или следует из определения: полное
совпадение, случайное угадывание, вырожденные случаи.
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.modules.topics.kmeans import KMeans  # noqa: E402
from app.modules.topics.metrics import (  # noqa: E402
    adjusted_rand_index,
    contingency_matrix,
    inertia,
    purity,
    silhouette_per_cluster,
    silhouette_samples,
    silhouette_score,
)


def make_blobs(per_center=40, spread=0.4, seed=0):
    rng = np.random.default_rng(seed)
    centers = np.array([[6.0, 6.0], [-6.0, 6.0], [0.0, -7.0]])
    points = np.vstack([center + rng.normal(0.0, spread, (per_center, 2)) for center in centers])
    return points, np.repeat(np.arange(3), per_center)


class InertiaTests(unittest.TestCase):
    def test_matches_a_hand_computed_value(self):
        X = np.array([[0.0, 0.0], [2.0, 0.0], [10.0, 0.0], [12.0, 0.0]])
        centroids = np.array([[1.0, 0.0], [11.0, 0.0]])
        labels = np.array([0, 0, 1, 1])
        # Четыре точки на расстоянии 1 от своего центроида: 4 * 1^2.
        self.assertAlmostEqual(inertia(X, labels, centroids), 4.0)

    def test_is_zero_when_points_sit_on_their_centroids(self):
        X = np.array([[1.0, 1.0], [5.0, 5.0]])
        self.assertAlmostEqual(inertia(X, np.array([0, 1]), X), 0.0)

    def test_agrees_with_the_model_attribute(self):
        """Метрика и алгоритм считают одну величину — расходиться им нельзя."""
        X, _ = make_blobs(seed=1)
        model = KMeans(n_clusters=3, random_state=0, normalize=False).fit(X)
        self.assertAlmostEqual(
            inertia(X, model.labels_, model.centroids_), model.inertia_, places=8
        )

    def test_mismatched_lengths_are_refused(self):
        with self.assertRaises(ValueError):
            inertia(np.zeros((3, 2)), np.array([0, 1]), np.zeros((2, 2)))

    def test_label_outside_the_centroid_list_is_refused(self):
        with self.assertRaises(ValueError):
            inertia(np.zeros((2, 2)), np.array([0, 5]), np.zeros((2, 2)))


class SilhouetteTests(unittest.TestCase):
    def test_separated_clusters_score_near_one(self):
        X, true_labels = make_blobs(seed=2)
        self.assertGreater(silhouette_score(X, true_labels), 0.9)

    def test_random_labels_score_near_zero_or_below(self):
        """Силуэт на случайной разметке не обязан быть ровно нулём, но точно
        не должен быть высоким — иначе метрика не различает разбиения."""
        rng = np.random.default_rng(3)
        X = rng.normal(size=(150, 5))
        random_labels = rng.integers(0, 3, 150)
        self.assertLess(silhouette_score(X, random_labels), 0.1)

    def test_a_wrong_split_scores_worse_than_the_right_one(self):
        X, true_labels = make_blobs(seed=2)
        wrong = np.arange(X.shape[0]) % 3  # режет каждое облако на три части
        self.assertGreater(silhouette_score(X, true_labels), silhouette_score(X, wrong))

    def test_matches_the_definition_on_a_hand_example(self):
        """Три точки, две в одном кластере: a и b считаются в уме.

        Точки 0 и 1 стоят на расстоянии 2, точка 2 — на расстоянии 10 и 8.
        Для точки 0: a = 2, b = 10, силуэт = (10 - 2) / 10 = 0.8.
        Для точки 1: a = 2, b = 8, силуэт = (8 - 2) / 8 = 0.75.
        Точка 2 одна в своём кластере — по соглашению 0.
        """
        X = np.array([[0.0], [2.0], [10.0]])
        values = silhouette_samples(X, np.array([0, 0, 1]))
        np.testing.assert_allclose(values, [0.8, 0.75, 0.0])

    def test_a_singleton_cluster_gets_zero(self):
        """Не единицу: иначе выброс, отсаженный в свой кластер, награждался бы."""
        X = np.array([[0.0], [1.0], [2.0], [100.0]])
        values = silhouette_samples(X, np.array([0, 0, 0, 1]))
        self.assertEqual(values[3], 0.0)

    def test_point_in_the_wrong_cluster_goes_negative(self):
        """Отрицательный силуэт — рабочий сигнал «точке ближе чужие»."""
        X = np.array([[0.0], [1.0], [2.0], [20.0], [21.0]])
        values = silhouette_samples(X, np.array([0, 0, 1, 1, 1]))
        self.assertLess(values[2], 0.0)

    def test_identical_points_score_zero(self):
        """Полные дубликаты: и a, и b равны нулю — деления на ноль быть не должно."""
        X = np.ones((6, 3))
        values = silhouette_samples(X, np.array([0, 0, 0, 1, 1, 1]))
        np.testing.assert_allclose(values, np.zeros(6))

    def test_one_cluster_is_refused(self):
        """Силуэт сравнивает со СОСЕДНИМ кластером; одного кластера мало."""
        with self.assertRaises(ValueError):
            silhouette_samples(np.zeros((5, 2)), np.zeros(5, dtype=int))

    def test_all_singletons_is_refused(self):
        with self.assertRaises(ValueError):
            silhouette_samples(np.eye(4), np.arange(4))

    def test_arbitrary_label_values_work(self):
        """Метки — не индексы колонок: K-means выдаёт 0..k-1, но разметка
        разделов сайта может прийти любыми числами."""
        X = np.array([[0.0], [2.0], [10.0]])
        values = silhouette_samples(X, np.array([70, 70, 900]))
        np.testing.assert_allclose(values, [0.8, 0.75, 0.0])


class SilhouetteSamplingTests(unittest.TestCase):
    """Выборочная оценка: воспроизводима и близка к точному значению."""

    def test_same_seed_gives_the_same_estimate(self):
        X, labels = make_blobs(per_center=200, seed=4)
        first = silhouette_score(X, labels, sample_size=60, random_state=1)
        second = silhouette_score(X, labels, sample_size=60, random_state=1)
        self.assertEqual(first, second)

    def test_different_seeds_may_give_different_estimates(self):
        """Именно поэтому в докстринге написано «оценка»: без фиксированного
        seed два запуска дадут разные числа на одних и тех же данных."""
        rng = np.random.default_rng(5)
        X = rng.normal(size=(400, 6))
        labels = rng.integers(0, 4, 400)
        estimates = {
            silhouette_score(X, labels, sample_size=40, random_state=seed) for seed in range(6)
        }
        self.assertGreater(len(estimates), 1)

    def test_the_estimate_is_close_to_the_exact_value(self):
        X, labels = make_blobs(per_center=200, seed=4)
        exact = silhouette_score(X, labels)
        estimate = silhouette_score(X, labels, sample_size=100, random_state=0)
        self.assertAlmostEqual(estimate, exact, places=2)

    def test_a_sample_smaller_than_the_data_is_ignored(self):
        """Считать точно, когда точного значения хватает по памяти."""
        X, labels = make_blobs(per_center=10, seed=4)
        self.assertEqual(
            silhouette_score(X, labels, sample_size=10_000, random_state=0),
            silhouette_score(X, labels),
        )

    def test_sampling_keeps_every_cluster(self):
        """Равномерная выборка потеряла бы мелкий кластер целиком, и оценка
        описывала бы уже другое разбиение."""
        rng = np.random.default_rng(6)
        X = np.vstack([rng.normal(0.0, 1.0, (500, 3)), rng.normal(20.0, 1.0, (4, 3))])
        labels = np.array([0] * 500 + [1] * 4)
        per_cluster = silhouette_per_cluster(X, labels, sample_size=30, random_state=0)
        self.assertEqual(set(per_cluster), {0, 1})


class SilhouettePerClusterTests(unittest.TestCase):
    def test_reports_every_cluster(self):
        X, labels = make_blobs(seed=7)
        per_cluster = silhouette_per_cluster(X, labels)
        self.assertEqual(set(per_cluster), {0, 1, 2})
        for value in per_cluster.values():
            self.assertGreater(value, 0.9)

    def test_the_mean_of_the_parts_is_the_whole(self):
        """Разбивка обязана быть разбивкой того же числа, а не другой метрики."""
        X, labels = make_blobs(seed=7)
        per_cluster = silhouette_per_cluster(X, labels)
        # Кластеры одного размера, поэтому простое среднее совпадает с общим.
        self.assertAlmostEqual(
            float(np.mean(list(per_cluster.values()))), silhouette_score(X, labels), places=8
        )

    def test_it_shows_which_cluster_is_the_bad_one(self):
        """Ради чего разбивка и нужна: общий силуэт приличный, но один кластер
        — свалка из двух перемешанных облаков."""
        rng = np.random.default_rng(8)
        good = rng.normal(0.0, 0.3, (60, 2)) + np.array([30.0, 30.0])
        mixed = rng.normal(0.0, 0.3, (60, 2))
        labels = np.array([0] * 60 + [1] * 30 + [2] * 30)
        X = np.vstack([good, mixed])
        per_cluster = silhouette_per_cluster(X, labels)
        self.assertGreater(per_cluster[0], 0.9)
        self.assertLess(per_cluster[1], 0.5)
        self.assertLess(per_cluster[2], 0.5)


class ContingencyMatrixTests(unittest.TestCase):
    def test_counts_pairs_of_labels(self):
        matrix = contingency_matrix([0, 0, 1, 1, 1], [0, 1, 1, 1, 1])
        np.testing.assert_array_equal(matrix, [[1, 1], [0, 3]])

    def test_string_labels_work(self):
        """Настоящие темы приходят названиями разделов, а не числами."""
        matrix = contingency_matrix(["Новости", "Новости", "Спорт"], [1, 1, 0])
        self.assertEqual(matrix.sum(), 3)
        self.assertEqual(matrix.shape, (2, 2))

    def test_mismatched_lengths_are_refused(self):
        with self.assertRaises(ValueError):
            contingency_matrix([0, 1, 2], [0, 1])


class PurityTests(unittest.TestCase):
    def test_perfect_match_is_one(self):
        self.assertEqual(purity([0, 0, 1, 1, 2, 2], [0, 0, 1, 1, 2, 2]), 1.0)

    def test_renumbering_clusters_changes_nothing(self):
        """Номера кластеров произвольны; метрика сравнивает разбиения."""
        self.assertEqual(purity([0, 0, 1, 1], [7, 7, 3, 3]), 1.0)

    def test_hand_computed_example(self):
        """Шесть точек, два кластера. В первом четыре точки: три темы «0» и
        одна «1» — большинство даёт 3. Во втором две точки темы «1» — 2.
        Чистота = (3 + 2) / 6."""
        true_labels = [0, 0, 0, 1, 1, 1]
        predicted = [0, 0, 0, 0, 1, 1]
        self.assertAlmostEqual(purity(true_labels, predicted), 5 / 6)

    def test_a_single_cluster_gives_the_largest_class_share(self):
        self.assertAlmostEqual(purity([0, 0, 0, 1], [0, 0, 0, 0]), 0.75)

    def test_splitting_everything_gives_one(self):
        """Слабое место чистоты, названное в докстринге: каждая точка своим
        кластером даёт идеальную единицу при совершенно бесполезном разбиении.
        Поэтому одной чистоты для выбора k мало."""
        self.assertEqual(purity([0, 0, 1, 1], [0, 1, 2, 3]), 1.0)

    def test_kmeans_on_separable_data_is_pure(self):
        X, true_labels = make_blobs(seed=9)
        model = KMeans(n_clusters=3, random_state=0, normalize=False).fit(X)
        self.assertEqual(purity(true_labels, model.labels_), 1.0)


class AdjustedRandIndexTests(unittest.TestCase):
    def test_identical_partitions_give_one(self):
        self.assertEqual(adjusted_rand_index([0, 0, 1, 1, 2, 2], [0, 0, 1, 1, 2, 2]), 1.0)

    def test_renumbering_clusters_changes_nothing(self):
        self.assertEqual(adjusted_rand_index([0, 0, 1, 1], [5, 5, 9, 9]), 1.0)

    def test_random_labels_hover_around_zero(self):
        """Ради этого поправка и вводится: некорректированный индекс Рэнда на
        тех же данных дал бы уверенные 0.6, будто разбиение осмысленно."""
        rng = np.random.default_rng(10)
        values = [
            adjusted_rand_index(rng.integers(0, 4, 300), rng.integers(0, 4, 300))
            for _ in range(20)
        ]
        self.assertLess(abs(float(np.mean(values))), 0.05)
        for value in values:
            self.assertLess(abs(value), 0.2)

    def test_hand_checked_textbook_value(self):
        """Хрестоматийный пример: одно разбиение расщепляет пару последней
        группы. Формула даёт 4/7."""
        self.assertAlmostEqual(adjusted_rand_index([0, 0, 1, 1], [0, 0, 1, 2]), 4 / 7)

    def test_orthogonal_partition_goes_negative(self):
        """Разбиение, систематически расходящееся с разметкой, хуже случайного."""
        self.assertAlmostEqual(adjusted_rand_index([0, 0, 1, 1], [0, 1, 0, 1]), -0.5)

    def test_one_cluster_against_many_is_zero(self):
        """Ни пары не совпало сверх ожидания — ровно ноль."""
        self.assertAlmostEqual(adjusted_rand_index([0, 0, 0, 0], [0, 1, 2, 3]), 0.0)

    def test_degenerate_single_cluster_case(self):
        """Оба разбиения — «всё в одну кучу»: пар нет, знаменатель ноль.
        Совпадение тривиальное, но полное, поэтому 1.0, а не деление на ноль."""
        self.assertEqual(adjusted_rand_index([0, 0, 0], [0, 0, 0]), 1.0)

    def test_degenerate_all_singletons_case(self):
        self.assertEqual(adjusted_rand_index([0, 1, 2], [5, 6, 7]), 1.0)

    def test_string_labels_work(self):
        self.assertEqual(adjusted_rand_index(["Спорт", "Спорт", "Новости"], [1, 1, 0]), 1.0)

    def test_kmeans_on_separable_data_scores_one(self):
        X, true_labels = make_blobs(seed=9)
        model = KMeans(n_clusters=3, random_state=0, normalize=False).fit(X)
        self.assertEqual(adjusted_rand_index(true_labels, model.labels_), 1.0)

    def test_ari_punishes_the_split_that_purity_rewards(self):
        """Пара метрик показана вместе намеренно: там, где чистота даёт 1.0 за
        разбиение каждой точки в свой кластер, ARI даёт около нуля."""
        true_labels = [0, 0, 0, 0, 1, 1, 1, 1]
        everything_split = list(range(8))
        self.assertEqual(purity(true_labels, everything_split), 1.0)
        self.assertLess(adjusted_rand_index(true_labels, everything_split), 0.01)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
