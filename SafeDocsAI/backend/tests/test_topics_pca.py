"""Собственный PCA: находит ли он оси, которые в данные заложены, и воспроизводим ли.

Данные синтетические и порождаются с фиксированным seed: только там известно,
какие направления в выборке главные, и «PCA нашёл ось» — проверяемое
утверждение, а не впечатление.

Сквозное правило файла: направление сравнивается с точностью до знака.
Собственный вектор определён с точностью до умножения на -1, и сравнение
«в лоб» краснело бы на верном ответе. Знак закрепляется отдельным правилом (см.
_fixed_signs) — и это проверяется прицельно, а не подразумевается.
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.modules.topics.pca import PrincipalAxes, project_onto  # noqa: E402


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Косинус между направлениями по модулю: знак оси не значим."""
    return float(abs(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def anisotropic_cloud(n=200, dim=12, spreads=(20.0, 8.0, 1.0), seed=0):
    """Облако с ЗАРАНЕЕ ИЗВЕСТНЫМИ главными осями.

    Строится в базисе из случайного ортогонального поворота, по первым осям
    разброс задан явно и убывает, по остальным — маленький и одинаковый.
    Значит первые главные компоненты обязаны совпасть со столбцами поворота,
    и совпадение можно проверить числом, а не глазом.
    """
    rng = np.random.default_rng(seed)
    rotation, _ = np.linalg.qr(rng.normal(size=(dim, dim)))
    scales = np.full(dim, 0.1)
    scales[: len(spreads)] = spreads
    coordinates = rng.normal(size=(n, dim)) * scales
    return coordinates @ rotation.T, rotation


class PrincipalAxesFindTheAxesThatWerePutInTests(unittest.TestCase):
    def test_first_components_match_the_axes_the_data_was_built_on(self):
        X, rotation = anisotropic_cloud()
        axes = PrincipalAxes.fit(X, n_components=3)
        for index in range(3):
            with self.subTest(component=index):
                self.assertGreater(
                    cosine(axes.components[:, index], rotation[:, index]), 0.99
                )

    def test_explained_variance_matches_the_spreads_that_were_set(self):
        """Дисперсия вдоль оси — не украшение отчёта: по ней решают, сколько
        компонент снимать. Если она врёт, врёт и обоснование выбора."""
        X, _ = anisotropic_cloud(n=4000, spreads=(20.0, 8.0, 1.0), seed=3)
        axes = PrincipalAxes.fit(X, n_components=3)
        for index, spread in enumerate((20.0, 8.0, 1.0)):
            with self.subTest(component=index):
                self.assertAlmostEqual(
                    float(np.sqrt(axes.explained_variance[index])), spread, delta=0.05 * spread
                )

    def test_components_are_ordered_by_decreasing_spread(self):
        """Порядок — весь смысл слова «первые». Без него drop=3 снимал бы
        произвольные три оси."""
        X, _ = anisotropic_cloud(dim=20, spreads=(30.0, 12.0, 5.0, 2.0), seed=7)
        axes = PrincipalAxes.fit(X, n_components=10)
        spreads = axes.explained_variance
        self.assertTrue(np.all(np.diff(spreads) <= 1e-9), spreads)

    def test_basis_is_orthonormal(self):
        X, _ = anisotropic_cloud(dim=15, seed=11)
        axes = PrincipalAxes.fit(X, n_components=8)
        gram = axes.components.T @ axes.components
        np.testing.assert_allclose(gram, np.eye(8), atol=1e-9)

    def test_the_tall_branch_and_the_wide_branch_agree(self):
        """Две ветки разложения (грамм-матрица при n<d, ковариация при n>=d) —
        это одна и та же математика, выбранная по размеру. Расхождение между
        ними означало бы, что результат зависит от того, сколько документов
        принесли, — а он не должен."""
        rng = np.random.default_rng(5)
        rotation, _ = np.linalg.qr(rng.normal(size=(40, 40)))
        scales = np.full(40, 0.2)
        scales[:4] = (9.0, 6.0, 3.0, 1.5)
        # 30 точек в 40 измерениях — узкая ветка; те же оси, но 400 точек —
        # широкая. Сравниваем направления, а не координаты.
        narrow = (rng.normal(size=(30, 40)) * scales) @ rotation.T
        wide = (rng.normal(size=(4000, 40)) * scales) @ rotation.T
        by_gram = PrincipalAxes.fit(narrow, n_components=4)
        by_covariance = PrincipalAxes.fit(wide, n_components=4)
        self.assertLess(narrow.shape[0], narrow.shape[1])
        self.assertGreater(wide.shape[0], wide.shape[1])
        for index in range(3):
            with self.subTest(component=index):
                self.assertGreater(
                    cosine(by_gram.components[:, index], by_covariance.components[:, index]),
                    0.9,
                )


class ProjectionTests(unittest.TestCase):
    def test_dropping_then_projecting_equals_projecting(self):
        """ТОЖДЕСТВО, НА КОТОРОМ СТОИТ ФОРМАТ АРТЕФАКТА.

        Модель хранит один базис и одно среднее — не два базиса, — потому что
        «снять первые d компонент, затем спроецировать на следующие k» равно
        «спроецировать на следующие k». Если тождество когда-нибудь перестанет
        выполняться, сохранённая модель начнёт считать не то, что обучали, и
        заметить это будет нечем: номер кластера посчитается в обоих случаях.
        """
        X, _ = anisotropic_cloud(n=120, dim=30, spreads=(15.0, 9.0, 4.0), seed=13)
        axes = PrincipalAxes.fit(X, n_components=20)
        centered = X - axes.mean
        dropped_basis = axes.basis(drop=0, keep=3)
        kept_basis = axes.basis(drop=3, keep=10)

        stepwise = (centered - (centered @ dropped_basis) @ dropped_basis.T) @ kept_basis
        direct = centered @ kept_basis
        np.testing.assert_allclose(stepwise, direct, atol=1e-9)

    def test_projection_removes_the_axis_it_was_told_to_remove(self):
        """Смысл drop: заданная ось после проекции не различает документы.

        Строим облако, где первая ось — это «язык» (две группы разъехались по
        ней далеко), а остальные — тема. После снятия первой компоненты
        координаты двух групп обязаны перемешаться.
        """
        rng = np.random.default_rng(17)
        topic = rng.normal(size=(300, 10))
        language = np.repeat([[-30.0], [30.0]], 150, axis=0)
        X = np.hstack([language, topic])
        axes = PrincipalAxes.fit(X, n_components=6)

        raw_gap = abs(X[:150, 0].mean() - X[150:, 0].mean())
        projected = axes.project(X, drop=1, keep=4, normalize=False)
        left, right = projected[:150], projected[150:]
        residual_gap = float(np.linalg.norm(left.mean(axis=0) - right.mean(axis=0)))
        self.assertGreater(raw_gap, 50.0)
        self.assertLess(residual_gap, 1.0)

    def test_single_vector_gets_the_same_answer_as_the_matrix(self):
        """Боевой путь подаёт документы по одному. Своя ветка для одного
        вектора разошлась бы с обучающей на первой же правке геометрии."""
        X, _ = anisotropic_cloud(n=80, dim=16, seed=19)
        axes = PrincipalAxes.fit(X, n_components=10)
        batch = axes.project(X, drop=2, keep=5)
        for index in (0, 7, 40):
            with self.subTest(row=index):
                one = axes.project(X[index], drop=2, keep=5)
                self.assertEqual(one.shape, (5,))
                np.testing.assert_allclose(one, batch[index], atol=1e-12)

    def test_projected_rows_are_unit_length_when_normalising(self):
        X, _ = anisotropic_cloud(n=60, dim=14, seed=23)
        axes = PrincipalAxes.fit(X, n_components=9)
        projected = axes.project(X, drop=1, keep=6)
        np.testing.assert_allclose(np.linalg.norm(projected, axis=1), 1.0, atol=1e-12)

    def test_project_onto_is_the_same_code_the_artifact_will_run(self):
        """Артефакт хранит среднее и базис и применяет их project_onto.
        Расхождение с методом означало бы, что обучение и назначение считают
        проекцию по-разному."""
        X, _ = anisotropic_cloud(n=90, dim=18, seed=29)
        axes = PrincipalAxes.fit(X, n_components=12)
        np.testing.assert_allclose(
            axes.project(X, drop=3, keep=6),
            project_onto(X, mean=axes.mean, basis=axes.basis(drop=3, keep=6)),
            atol=1e-12,
        )


class DeterminismTests(unittest.TestCase):
    def test_two_fits_on_the_same_data_give_a_byte_identical_basis(self):
        """Приложение узнаёт переобученную модель по sha256 артефакта. Базис,
        отличающийся знаками столбцов, — это геометрически та же модель и
        побайтно другой файл, то есть новая версия и обесценивание всех
        назначений до переразметки. Холостой пересчёт обязан быть бесплатным.
        """
        X, _ = anisotropic_cloud(n=150, dim=25, seed=31)
        first = PrincipalAxes.fit(X, n_components=10)
        second = PrincipalAxes.fit(X, n_components=10)
        np.testing.assert_array_equal(first.components, second.components)
        np.testing.assert_array_equal(first.mean, second.mean)

    def test_sign_rule_is_actually_applied(self):
        """Само правило знака: наибольшая по модулю координата положительна.
        Без прицельной проверки правило можно потерять, а тест на
        воспроизводимость останется зелёным — два запуска `eigh` на одних
        данных и так обычно совпадают."""
        X, _ = anisotropic_cloud(n=100, dim=20, seed=37)
        axes = PrincipalAxes.fit(X, n_components=8)
        for index in range(8):
            column = axes.components[:, index]
            with self.subTest(component=index):
                self.assertGreater(column[np.argmax(np.abs(column))], 0.0)


class RefusalTests(unittest.TestCase):
    """Отказ с внятной причиной вместо базиса, у которого последние столбцы —
    шум округления. Молча посчитанная проекция дала бы правдоподобный номер
    кластера и неверную тему."""

    def test_more_components_than_the_sample_can_give(self):
        X, _ = anisotropic_cloud(n=12, dim=40, seed=41)
        # 12 точек после центрирования лежат в 11 измерениях, не в 12.
        PrincipalAxes.fit(X, n_components=11)
        with self.assertRaises(ValueError) as caught:
            PrincipalAxes.fit(X, n_components=12)
        self.assertIn("11", str(caught.exception))

    def test_degenerate_sample_is_refused_by_name(self):
        """Все точки на одной прямой: второй оси не существует, и `eigh`
        вернул бы для неё направление, определённое шумом."""
        direction = np.array([1.0, 2.0, 3.0, 4.0])
        X = np.outer(np.linspace(-1.0, 1.0, 50), direction)
        PrincipalAxes.fit(X, n_components=1)
        with self.assertRaises(ValueError) as caught:
            PrincipalAxes.fit(X, n_components=2)
        self.assertIn("вырожден", str(caught.exception))

    def test_zero_or_negative_components(self):
        X, _ = anisotropic_cloud(n=30, dim=10, seed=43)
        for value in (0, -1):
            with self.subTest(n_components=value):
                with self.assertRaises(ValueError):
                    PrincipalAxes.fit(X, n_components=value)

    def test_single_document_has_no_axes_at_all(self):
        with self.assertRaises(ValueError):
            PrincipalAxes.fit(np.array([[1.0, 2.0, 3.0]]), n_components=1)

    def test_one_dimensional_input_is_not_silently_treated_as_a_corpus(self):
        with self.assertRaises(ValueError):
            PrincipalAxes.fit(np.array([1.0, 2.0, 3.0]), n_components=1)

    def test_keep_beyond_what_was_built(self):
        X, _ = anisotropic_cloud(n=60, dim=20, seed=47)
        axes = PrincipalAxes.fit(X, n_components=6)
        axes.basis(drop=2, keep=4)
        with self.assertRaises(ValueError):
            axes.basis(drop=2, keep=5)
        with self.assertRaises(ValueError):
            axes.basis(drop=6, keep=1)
        with self.assertRaises(ValueError):
            axes.basis(drop=-1)

    def test_vector_from_another_space_is_refused(self):
        """Вектор чужой размерности — это чужая embedding-модель. Дополнить
        нулями или обрезать значило бы выдумать данные."""
        X, _ = anisotropic_cloud(n=50, dim=12, seed=53)
        axes = PrincipalAxes.fit(X, n_components=5)
        with self.assertRaises(ValueError):
            axes.project(np.zeros((3, 11)), drop=1, keep=3)
        with self.assertRaises(ValueError):
            axes.project(np.zeros(13), drop=1, keep=3)


if __name__ == "__main__":
    unittest.main()
