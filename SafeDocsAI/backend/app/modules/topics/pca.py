"""Собственный метод главных компонент: главные оси корпуса и проекция на них.

Зачем он понадобился. Кластеризация этого корпуса «в лоб» ложится не на тему, а
на язык и жанр: при k=20 ARI против языка +0.415, против темы +0.063.
Предыдущее решение вычитало среднее ГРУППЫ документа (язык × жанр) и тем самым
требовало эту группу знать — у обучающего корпуса она в разметке, а у боевого
документа жанра нет вовсе, и его приходилось назначать допущением. Допущение
записано в отчёте как непроверенный риск, и оно же — самое хрупкое место
модели.

Главные компоненты снимают ту же ось, ничего не зная о документе. Первые
несколько направлений наибольшего разброса многоязычного эмбеддинга — это и
есть «на каком языке написано» и «в каком регистре»: они различают документы
сильнее всего, потому что тема на их фоне — эффект второго порядка. Убрать их —
значит убрать конкурирующие оси, не спрашивая у документа ни языка, ни жанра.

Замер на размеченном корпусе (k=20, test), прежний способ против этого:

    центрирование по ячейке   ARI тема +0.191   ось языка +0.002   силуэт 0.018
    снять 3 компоненты, 32    ARI тема +0.196   ось языка +0.090   силуэт 0.095

Качество то же, требование знать язык и жанр исчезает, силуэт вырос в пять раз —
и выбор k, который на сыром пространстве был плоским (0.033–0.071 на всём
диапазоне 2..40), снова стало чем измерять.

ТОЖДЕСТВО, НА КОТОРОМ ДЕРЖИТСЯ ФОРМАТ АРТЕФАКТА. «Убрать первые d компонент, а
потом спроецировать на следующие k» даёт ровно то же, что «спроецировать на
следующие k»: компоненты попарно ортогональны, поэтому вычитание проекции на
первые d ничего не меняет в координатах по остальным. Значит хранить нужен один
базис (d_модели, keep) и одно среднее, а не два базиса и шесть групповых
средних. Тождество не постулируется, а проверяется тестом
(test_topics_pca.py), потому что на нём стоит весь размер и вся простота
сохранённой модели.

Библиотечного PCA здесь нет по той же причине, по которой нет библиотечного
K-means: в этой работе алгоритм — предмет защиты. numpy используется как
линейная алгебра (`eigh` — разложение симметричной матрицы, не метод главных
компонент). Запрет проверяется tests/test_topics_no_sklearn_guard.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.modules.topics.kmeans import l2_normalize

# Порог, ниже которого собственное значение считается нулевым, — доля от самого
# большого. Направление с нулевым разбросом не определено: любая ось внутри
# вырожденного подпространства одинаково хороша, и `eigh` вернёт ту, которую
# подскажет шум округления. Проецировать на такую ось — значит подмешать в
# модель случайное число, устойчивое ровно до следующего пересчёта.
RANK_TOLERANCE = 1e-10


def _fixed_signs(components: np.ndarray) -> np.ndarray:
    """Знак каждой компоненты закрепляется: наибольшая координата — положительна.

    Собственный вектор определён с точностью до знака, и `eigh` выбирает его
    произвольно. Без закрепления два запуска на одних и тех же данных дали бы
    базисы, отличающиеся знаками столбцов: геометрия та же, а сохранённый
    артефакт — побайтно другой. Приложение узнаёт переобученную модель по
    sha256 файла, поэтому «побайтно другой» означает новую версию модели и
    обесценивание всех назначений до переразметки. Холостой пересчёт обязан
    давать тот же файл.
    """
    leading = np.argmax(np.abs(components), axis=0)
    signs = np.sign(components[leading, np.arange(components.shape[1])])
    # Ровно нулевой столбец знака не имеет — оставляем как есть, а не делаем 0.
    signs[signs == 0.0] = 1.0
    return components * signs


@dataclass(frozen=True)
class PrincipalAxes:
    """Главные оси корпуса: среднее, ортонормированный базис, разбросы по осям.

    Столбцы `components` упорядочены по убыванию `explained_variance` — это и
    есть смысл «первых компонент»: первая объясняет больше всех, и снимают
    именно её.
    """

    mean: np.ndarray
    components: np.ndarray
    explained_variance: np.ndarray

    @property
    def dim(self) -> int:
        """Размерность исходного пространства (сколько признаков ждём на вход)."""
        return int(self.components.shape[0])

    @property
    def n_components(self) -> int:
        return int(self.components.shape[1])

    @classmethod
    def fit(cls, X: np.ndarray, n_components: int) -> "PrincipalAxes":
        """Главные оси по обучающей выборке.

        Разложение считается через ту из двух матриц, которая меньше: при
        n < d — грамм-матрица X·Xᵀ размера n×n, иначе ковариация XᵀX размера
        d×d. На нашем корпусе это 1799×1799 против 4096×4096, то есть впятеро
        меньше элементов и вшестеро меньше работы у `eigh`; собственные же
        значения у этих двух матриц одинаковы, а собственные векторы связаны
        умножением на Xᵀ. Выбор ветки — арифметика, а не приближение: обе дают
        один и тот же ответ.
        """
        data = np.asarray(X, dtype=np.float64)
        if data.ndim != 2:
            raise ValueError(f"ожидалась матрица (n_samples, n_features), получено {data.shape}")
        n_samples, n_features = data.shape
        if n_components < 1:
            raise ValueError("n_components должно быть >= 1")

        # Потолок — min(n_samples - 1, n_features), а не min(n_samples,
        # n_features). Центрирование забирает одну степень свободы: n точек
        # после вычитания их собственного среднего лежат в подпространстве
        # размерности n-1, и n-я компонента у них — уже чистый шум округления.
        max_components = min(n_samples - 1, n_features)
        if max_components < 1:
            raise ValueError(
                f"из {n_samples} документов главных осей не построить: "
                "после центрирования не остаётся ни одного направления"
            )
        if n_components > max_components:
            raise ValueError(
                f"запрошено {n_components} компонент, а из выборки "
                f"({n_samples} документов, {n_features} признаков) их можно "
                f"получить не больше {max_components}"
            )

        mean = data.mean(axis=0)
        centered = data - mean

        if n_samples < n_features:
            gram = centered @ centered.T
            values, vectors = np.linalg.eigh(gram)
            order = np.argsort(values)[::-1][:n_components]
            values = values[order]
            # Собственный вектор ковариации восстанавливается из собственного
            # вектора грамм-матрицы: если G·v = λ·v, то (XᵀX)·(Xᵀv) = λ·(Xᵀv).
            # Длина при этом не сохраняется, поэтому столбцы нормируются.
            directions = centered.T @ vectors[:, order]
        else:
            covariance = centered.T @ centered
            values, vectors = np.linalg.eigh(covariance)
            order = np.argsort(values)[::-1][:n_components]
            values = values[order]
            directions = vectors[:, order]

        norms = np.linalg.norm(directions, axis=0)
        # Вырожденное направление ловится здесь, а не «как-нибудь потом»:
        # нулевая длина означает, что ранг выборки исчерпан, и следующая ось
        # определена шумом. Отказ с числом доступных осей полезнее, чем базис,
        # у которого последние столбцы случайны.
        largest = float(values[0]) if values.size else 0.0
        usable = int(np.sum(values > RANK_TOLERANCE * max(largest, 1.0)))
        if usable < n_components or np.any(norms <= 0.0):
            raise ValueError(
                f"запрошено {n_components} компонент, но выборка вырождена: "
                f"направлений с ненулевым разбросом всего {usable}"
            )

        components = _fixed_signs(directions / norms)
        # Разброс вдоль оси — несмещённая оценка дисперсии. Хранится не ради
        # проекции (ей нужны только направления), а ради отчёта: доля
        # объяснённого разброса — это и есть ответ на вопрос «почему снимаем
        # именно три компоненты, а не одну и не десять».
        explained = values / max(n_samples - 1, 1)
        return cls(mean=mean, components=components, explained_variance=explained)

    def basis(self, *, drop: int = 0, keep: int | None = None) -> np.ndarray:
        """Столбцы [drop, drop + keep) — то, на что проецируем.

        Первые `drop` компонент выбрасываются как оси языка и регистра,
        следующие `keep` оставляются как рабочее пространство модели.
        """
        if drop < 0:
            raise ValueError("drop не может быть отрицательным")
        available = self.n_components - drop
        if available < 1:
            raise ValueError(
                f"после отбрасывания {drop} компонент из {self.n_components} "
                "не остаётся ни одной"
            )
        if keep is None:
            keep = available
        if keep < 1:
            raise ValueError("keep должно быть >= 1")
        if keep > available:
            raise ValueError(
                f"запрошено keep={keep} компонент после drop={drop}, "
                f"а всего построено {self.n_components}"
            )
        return self.components[:, drop : drop + keep]

    def to_projection(
        self, *, drop: int = 0, keep: int | None = None, renormalize: bool = True
    ) -> "Projection":
        """Готовое к сохранению преобразование: среднее и один базис.

        Ровно то, что уезжает в артефакт. Обучающая сторона строит его отсюда,
        боевая — читает из файла, и обе применяют одним и тем же кодом (см.
        Projection.apply): своя копия проекции на боевой стороне разошлась бы с
        обучающей на первой же правке, а расхождение проявилось бы номерами
        кластеров, которые вроде бы посчитались.
        """
        return Projection(
            mean=self.mean,
            basis=self.basis(drop=drop, keep=keep),
            dropped=int(drop),
            renormalize=bool(renormalize),
        )

    def project(
        self,
        X: np.ndarray,
        *,
        drop: int = 0,
        keep: int | None = None,
        normalize: bool = True,
    ) -> np.ndarray:
        """Координаты в пространстве модели: (n_samples, keep).

        Принимает и одиночный вектор (d,) — тогда и возвращает (keep,). Это не
        удобство, а требование боевого пути: там документ приходит по одному, и
        своя ветка «а для одного вектора сделаем то же самое руками» разошлась
        бы с обучающей на первой же правке.
        """
        return project_onto(
            X, mean=self.mean, basis=self.basis(drop=drop, keep=keep), normalize=normalize
        )


# Имя вида преобразования в сохранённой модели. Одна строка на весь проект:
# её пишет обучающая сторона и по ней же узнаёт боевая, и разъехаться им нельзя.
PROJECTION_KIND = "pca_projection"


@dataclass(frozen=True)
class Projection:
    """Преобразование, которое уезжает вместе с моделью: среднее и базис.

    Чем оно лучше прежнего центрирования по группе. Прежнее вычитало среднее
    ГРУППЫ документа и потому требовало эту группу знать: язык — определителем,
    жанр — допущением «боевой документ всегда real», причём среднее ветки real
    посчитано на выдержках Википедии, а боевой корпус — законы и паёмы. Проекция
    не спрашивает у документа ничего: те же оси снимаются по геометрии корпуса.

    dropped хранится не для расчёта — базис уже содержит только оставленные
    компоненты, — а для отчёта и для чтения человеком: «сняты первые три оси»
    иначе не восстановить из файла никак.
    """

    mean: np.ndarray
    basis: np.ndarray
    dropped: int = 0
    renormalize: bool = True

    @property
    def dim(self) -> int:
        """Размерность входа: столько признаков ждёт преобразование."""
        return int(self.basis.shape[0])

    @property
    def out_dim(self) -> int:
        """Размерность выхода: в этом пространстве лежат центроиды."""
        return int(self.basis.shape[1])

    def apply(self, X: np.ndarray) -> np.ndarray:
        return project_onto(X, mean=self.mean, basis=self.basis, normalize=self.renormalize)

    def meta(self) -> dict:
        return {
            "kind": PROJECTION_KIND,
            "dim": self.dim,
            "components": self.out_dim,
            "dropped": int(self.dropped),
            "renormalize": bool(self.renormalize),
        }

    @classmethod
    def from_saved(cls, meta: dict, mean: np.ndarray, basis: np.ndarray) -> "Projection":
        """Восстановление из артефакта с проверкой согласованности.

        Проверки не формальность: несовпадение размеров здесь — единственное
        место, где ошибку ещё можно заметить. Дальше проекция посчитается на
        чём угодно подходящей длины и выдаст правдоподобный номер кластера.
        """
        kind = str((meta or {}).get("kind") or "")
        if kind != PROJECTION_KIND:
            raise ValueError(f"ожидалось преобразование {PROJECTION_KIND!r}, в файле {kind!r}")
        mean = np.asarray(mean, dtype=np.float64)
        basis = np.asarray(basis, dtype=np.float64)
        if basis.ndim != 2:
            raise ValueError("базис проекции должен быть матрицей (dim, components)")
        if mean.ndim != 1 or mean.shape[0] != basis.shape[0]:
            raise ValueError(
                f"среднее ({mean.shape}) и базис ({basis.shape}) из разных пространств"
            )
        declared_dim = int(meta.get("dim") or basis.shape[0])
        declared_components = int(meta.get("components") or basis.shape[1])
        if (declared_dim, declared_components) != basis.shape:
            raise ValueError(
                f"в метаданных проекция {declared_dim}x{declared_components}, "
                f"а базис {basis.shape[0]}x{basis.shape[1]}"
            )
        return cls(
            mean=mean,
            basis=basis,
            dropped=int(meta.get("dropped") or 0),
            renormalize=bool(meta.get("renormalize", True)),
        )


def project_onto(
    X: np.ndarray, *, mean: np.ndarray, basis: np.ndarray, normalize: bool = True
) -> np.ndarray:
    """Вычесть среднее, спроецировать на базис, при необходимости нормировать.

    Отдельная функция, а не только метод: сохранённый артефакт хранит ровно эти
    два массива — среднее и базис, — и применяет их этим же кодом. Иначе
    обучение и назначение считали бы проекцию двумя разными реализациями, а
    расхождение между ними проявилось бы только в бою, номерами кластеров,
    которые вроде бы посчитались.
    """
    data = np.asarray(X, dtype=np.float64)
    single = data.ndim == 1
    if single:
        data = data.reshape(1, -1)
    if data.ndim != 2:
        raise ValueError(f"ожидалась матрица или вектор, получено {np.asarray(X).shape}")
    if data.shape[1] != basis.shape[0]:
        raise ValueError(
            f"ожидалось {basis.shape[0]} признаков, получено {data.shape[1]}"
        )
    if mean.shape[0] != basis.shape[0]:
        raise ValueError(
            f"среднее ({mean.shape[0]}) и базис ({basis.shape[0]}) из разных пространств"
        )
    projected = (data - mean) @ basis
    if normalize:
        projected = l2_normalize(projected)
    return projected[0] if single else projected
