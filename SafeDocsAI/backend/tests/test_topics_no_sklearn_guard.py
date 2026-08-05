"""Сторож: в app/modules/topics не должно быть библиотечной кластеризации.

Зачем машинная проверка, а не договорённость. sklearn 1.9.0 в окружении
установлен — приехал транзитивно, и убрать его нельзя. Значит, заменить
собственный K-means одной строкой `from sklearn.cluster import KMeans`
физически возможно в любой момент: при рефакторинге, при «оптимизации», по
невнимательности. Это учебная работа, и алгоритм в ней — предмет защиты, а не
деталь реализации, поэтому подмену должен ловить тест, а не совесть.

Что именно запрещено:
  * sklearn / scikit-learn целиком — библиотечный K-means и библиотечные
    метрики (silhouette_score, adjusted_rand_score) лежат там же;
  * scipy.cluster — сам scipy разрешён как векторная арифметика, но
    scipy.cluster.vq.kmeans2 — это ровно тот алгоритм, который пишется руками.

Проверка идёт по дереву разбора, а не поиском подстроки: комментарии и
докстринги про sklearn писать можно и нужно (в них объясняется, почему его
здесь нет), а вот `import sklearn` и обращение `scipy.cluster.vq` — нельзя.
Разбор ловит и то, что подстрочный поиск пропустил бы: псевдонимы
(`import sklearn.cluster as c`) и динамический импорт по строке.
"""

import ast
import os
import subprocess
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BACKEND_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOPICS_DIR = os.path.join(BACKEND_ROOT, "app", "modules", "topics")

# Префиксы полных путей модулей. Проверяется именно префикс, а не корень
# пакета: scipy разрешён, а scipy.cluster внутри него — нет.
BANNED_PREFIXES = ("sklearn", "scikit", "scipy.cluster")

DYNAMIC_IMPORT_CALLS = ("__import__", "import_module")


def topics_source_files():
    return sorted(
        os.path.join(TOPICS_DIR, name)
        for name in os.listdir(TOPICS_DIR)
        if name.endswith(".py")
    )


def is_banned(dotted_name: str) -> bool:
    """Запрещён сам модуль и всё, что лежит внутри него."""
    return any(
        dotted_name == prefix or dotted_name.startswith(prefix + ".")
        for prefix in BANNED_PREFIXES
    )


def dotted_name(node: ast.AST) -> str:
    """Собирает `scipy.cluster.vq` обратно из вложенных ast.Attribute."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return ""
    parts.append(node.id)
    return ".".join(reversed(parts))


def banned_usages(source: str) -> list[str]:
    """Все запрещённые обращения в одном файле — список нарушений."""
    found: list[str] = []
    for node in ast.walk(ast.parse(source)):
        # import sklearn / import sklearn.cluster as c
        if isinstance(node, ast.Import):
            found += [alias.name for alias in node.names if is_banned(alias.name)]

        # from sklearn.cluster import KMeans
        elif isinstance(node, ast.ImportFrom):
            if node.module and is_banned(node.module):
                found.append(node.module)

        # Обращение через уже импортированный корень: scipy.cluster.vq.kmeans2
        elif isinstance(node, ast.Attribute):
            name = dotted_name(node)
            if name and is_banned(name):
                found.append(name)

        # importlib.import_module("sklearn.cluster") — импорт, которого нет в
        # дереве импортов и который подстрочный поиск по слову "import" тоже
        # не поймал бы.
        elif isinstance(node, ast.Call):
            callee = node.func
            name = callee.attr if isinstance(callee, ast.Attribute) else getattr(callee, "id", "")
            if name in DYNAMIC_IMPORT_CALLS:
                for argument in node.args:
                    if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
                        if is_banned(argument.value):
                            found.append(argument.value)
    return found


class TopicsSourceIsFreeOfLibraryClusteringTests(unittest.TestCase):
    def test_the_guard_actually_sees_the_module(self):
        """Сторож на сторожа.

        Если каталог переедет или расширение файлов сменится, обход вернёт
        пустой список и все остальные проверки станут зелёными, ничего не
        проверяя. Молчаливо зелёный сторож хуже отсутствующего.
        """
        files = topics_source_files()
        self.assertTrue(files, f"не найдено ни одного исходника в {TOPICS_DIR}")
        names = {os.path.basename(path) for path in files}
        self.assertLessEqual({"__init__.py", "kmeans.py", "metrics.py"}, names)

    def test_no_banned_imports_in_any_topics_file(self):
        for path in topics_source_files():
            with self.subTest(file=os.path.basename(path)):
                with open(path, encoding="utf-8") as handle:
                    usages = banned_usages(handle.read())
                self.assertEqual(
                    usages,
                    [],
                    f"{os.path.basename(path)} использует библиотечную кластеризацию: {usages}. "
                    "K-means в этой работе пишется руками.",
                )

    def test_importing_topics_does_not_pull_sklearn_in(self):
        """Вторая половина сторожа: разбор исходников видит только эти файлы.

        Если topics начнёт импортировать соседний модуль проекта, а тот —
        sklearn, дерево разбора останется чистым, а библиотека всё равно
        окажется в процессе. Поэтому импорт проверяется ещё и на живом
        процессе: отдельный интерпретатор, чистый sys.modules.
        """
        script = (
            "import sys; import app.modules.topics; "
            "print(','.join(sorted(name for name in sys.modules "
            "if name.split('.')[0] in ('sklearn', 'scikit') "
            "or name.startswith('scipy.cluster'))))"
        )
        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=BACKEND_ROOT,
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(
            completed.stdout.strip(),
            "",
            "импорт app.modules.topics затянул библиотечную кластеризацию",
        )


class TheGuardItselfCatchesViolationsTests(unittest.TestCase):
    """Краснота сторожа доказывается здесь же, а не только руками при сдаче.

    Каждая строка — форма подмены, которую сторож обязан поймать. Без этого
    класса «сторож зелёный» означало бы всего лишь «сторож ничего не умеет».
    """

    VIOLATIONS = (
        "import sklearn",
        "import sklearn.cluster",
        "import sklearn.cluster as c",
        "from sklearn.cluster import KMeans",
        "from sklearn.metrics import silhouette_score",
        "from scipy.cluster.vq import kmeans2",
        "import scipy\nscipy.cluster.vq.kmeans2(X, 3)",
        "import importlib\nimportlib.import_module('sklearn.cluster')",
        "m = __import__('sklearn')",
    )

    ALLOWED = (
        "import numpy as np",
        "import scipy\nscipy.linalg.norm(X)",
        "from scipy.spatial.distance import cdist",
        # Слово в тексте — не импорт: комментарии про запрет должны жить.
        "'''sklearn здесь запрещён, см. tests/test_topics_no_sklearn_guard.py'''",
        "# scikit-learn не используется\nimport numpy",
    )

    def test_every_known_form_of_substitution_is_caught(self):
        for source in self.VIOLATIONS:
            with self.subTest(source=source):
                self.assertTrue(banned_usages(source))

    def test_allowed_code_is_not_flagged(self):
        """Сторож, ругающийся на numpy и на комментарии, отключат первым же
        коммитом — и он перестанет ловить то, ради чего написан."""
        for source in self.ALLOWED:
            with self.subTest(source=source):
                self.assertEqual(banned_usages(source), [])

    def test_the_real_module_would_fail_with_a_line_added(self):
        """Тот же прогон, что и в основной проверке, но по исходнику с
        подмешанной строкой: краснота показана на настоящем файле, а не только
        на выдуманных примерах."""
        path = os.path.join(TOPICS_DIR, "kmeans.py")
        with open(path, encoding="utf-8") as handle:
            source = handle.read()
        self.assertEqual(banned_usages(source), [])
        self.assertEqual(
            banned_usages("from sklearn.cluster import KMeans\n" + source),
            ["sklearn.cluster"],
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
