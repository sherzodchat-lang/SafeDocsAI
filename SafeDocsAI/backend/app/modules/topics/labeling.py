"""Подписи кластеров: рубрика плюс уточнение по характерным словам (c-TF-IDF).

Зачем уточнение. Подпись кластера бралась раньше как «преобладающая тема его
документов». На двадцати кластерах это дало двенадцать названий: «Образование»
встречалось трижды, «Экономика» трижды, «История» дважды. Пользователь видел
список, в котором три строки называются одинаково и различить их нечем, — а
различие между ними есть, просто оно не названо.

Что здесь считается. Для каждого кластера — слова, которые встречаются в нём
чаще, чем в остальных кластерах вместе взятых. Мера — c-TF-IDF: частота слова
внутри кластера, приглушённая тем, насколько это слово распространено по всем
кластерам. Кластер считается одним большим документом, а «коллекцией» выступает
набор кластеров; отсюда и буква c.

    tf(t, c)  = сколько раз t встретилось в кластере c / всего слов в c
    idf(t)    = log(1 + A / f(t)),  A — среднее число слов в кластере,
                                    f(t) — сколько раз t встретилось везде
    score     = tf * idf

Слово вроде «Тоҷикистон» встречается почти в каждом документе, поэтому его f(t)
велико, idf мал, и в подпись оно не попадёт само собой — списком запрещённых
слов это давить не нужно.

ГЛАВНОЕ РЕШЕНИЕ ФАЙЛА. Уточнение считается не по всему корпусу, а ТОЛЬКО СРЕДИ
КЛАСТЕРОВ ОДНОЙ РУБРИКИ. Иначе два кластера «Иқтисод» получили бы обоим
одинаково характерные слова («иқтисод», «сомонӣ») и остались бы неразличимы —
то есть уточнение не уточняло бы ничего. Считая меру внутри семьи, мы получаем
ровно то, чем эти кластеры отличаются ДРУГ ОТ ДРУГА.

Языковых моделей здесь нет намеренно. Подпись обязана быть воспроизводимой и
объяснимой: на защите на вопрос «почему кластер называется так» ответом должен
быть счётчик, а не «так сказала модель».
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Hashable, Iterable, Mapping, Sequence

# Слово — буквы кириллицы (включая таджикские ӣ ӯ ҳ қ ҷ ғ) или латиницы, не
# короче трёх букв. Цифры и даты отброшены целиком: «2024» характерно для
# кластера ровно до следующего года.
#
# Таджикские буквы перечислены явно, а не взяты диапазоном \w: они лежат в
# кириллическом блоке Unicode вразнобой (Ғ U+0492, Ӣ U+04E2, Қ U+049A,
# Ӯ U+04EE, Ҳ U+04B2, Ҷ U+04B6), и «а-я» их не покрывает. Без них слово
# «Ҷумҳурӣ» распалось бы на «ум», «ур» — то есть на мусор.
WORD = re.compile(r"[а-яёa-zӣӯҳқҷғ]{3,}", re.IGNORECASE)


def _default_stopwords() -> frozenset[str]:
    """Служебные слова таджикского и русского.

    Список явный и короткий: c-TF-IDF и так давит распространённое, стоп-слова
    нужны лишь для того, что распространено НЕРАВНОМЕРНО — предлоги и связки
    встречаются чаще в длинных официальных текстах, чем в коротких спортивных
    заметках, и без списка «мебошад» стало бы характерным словом кластера
    законов.
    """
    tajik = """
    ва дар ба аз бо ки ин он барои аст буд бош шуд шудааст мешавад мебошад
    гардид гардад кард кардан мекунад намуд гуфт худ ҳам низ як ду се чор панҷ
    бисёр чун то агар аммо вале ҳар ҳамчунин инчунин тавассути назди бояд оид
    тибқи роҷеъ зеро чӣ кӣ куҷо чанд ҳоло боз дигар ҳамин ҳамон яъне ё аз рӯи
    аллакай бештар камтар ҳамаи тамоми доир мазкур мебошанд шуданд карданд
    доранд дорад дошт будаанд ҳастанд бошад бошанд метавонад метавонанд
    """.split()
    russian = """
    и в на с по для что это как не от к о из за у при а но же то все был была
    было были есть быть его ее их том так уже еще или если чтобы также между
    после более менее который которая которые которого которых он она они мы
    вы я мне нам них ним тем этом этой этих один два три года году лет также
    свои свое своих под над про без через до во со об бы ли да нет
    """.split()
    return frozenset(tajik + russian)


STOPWORDS: frozenset[str] = _default_stopwords()


def tokenize(text: str, stopwords: frozenset[str] = STOPWORDS) -> list[str]:
    """Слова документа в нижнем регистре, без служебных и без цифр."""
    return [word for word in WORD.findall(str(text).lower()) if word not in stopwords]


@dataclass(frozen=True)
class Term:
    term: str
    score: float


# Какую долю всех своих употреблений внутри семьи слово должно набрать в
# кластере, чтобы считаться его отличительным.
#
# Само по себе c-TF-IDF слово, поделённое между близнецами поровну, только
# приглушает — но не отбрасывает. У двух кластеров «Иқтисод» слово «иқтисод»
# набирает половину частоты каждому и всё равно всплывает вторым в обоих
# уточнениях: «Иқтисод — барқ, иқтисод» и «Иқтисод — пахта, иқтисод». Формально
# имена разные, по существу вторая половина обоих не значит ничего.
#
# Поэтому отдельное требование: слово, которым один близнец отличается от
# другого, должно в этом близнеце и жить. 0.6, а не 0.5, чтобы слово с
# перевесом в один документ не объявлялось отличительным.
DOMINANCE = 0.6

# В скольких кластерах слово может встречаться, чтобы ещё считаться характерным.
#
# Одного c-TF-IDF мало, и это видно на настоящем корпусе: «тоҷикистон» стоит во
# ВСЕХ кластерах, но в кластере законов занимает такую долю слов, что даже
# приглушённое idf произведение выносит его наверх. Так в подпись и попало
# «Законодательство и право — ҷумҳурии, тоҷикистон» — два слова, не сказавшие
# ничего.
#
# Мера частоты этого не ловит принципиально: она отвечает на вопрос «часто ли
# слово встречается», а нужен ответ на «встречается ли оно ГДЕ-ТО ЕЩЁ». Поэтому
# отдельный потолок: слово, попавшее больше чем в половину кластеров, — это
# лексика корпуса, а не примета кластера.
MAX_CLUSTER_SHARE = 0.5

# Сколько первых букв должно совпасть, чтобы два слова считались одним и тем же.
#
# Таджикский маркирует изафет суффиксом, и «конфронс» с «конфронси» — одно слово
# в двух формах. Токенизатор их не сводит (нормализатора форм у нас нет и он был
# бы отдельной работой), поэтому оба попадали в подпись: «Послания и выступления
# Президента — конфронси, конфронс». Половина уточнения не сказала ничего.
#
# Отсечение по общему началу — грубая замена приведению к начальной форме, и это
# сказано прямо. Она ловит именно тот случай, который встречается: одно и то же
# слово с суффиксом и без.
SAME_STEM_PREFIX = 5

# Ниже какой доли преобладающая рубрика перестаёт быть именем кластера.
#
# Кластер, где рубрика большинства набирает столько же, сколько она занимает во
# всём корпусе, о ней не говорит ничего: такую долю она получила бы и в куче,
# набранной наугад. На нашем корпусе самая крупная рубрика — 19% (517 документов
# из 2741), и это и есть уровень случайности.
#
# Порог поставлен чуть выше него. Не выше: сначала он стоял на 0.35, и под него
# попал кластер, куда легли паёмы президента — рубрика там 0.30, то есть
# в полтора раза больше случайного, — и вместо findable «Послания и выступления
# Президента» пользователь получил «Тематическая группа». Имя, по которому
# документ не найти, вредит больше, чем неточная доля: саму долю видно в
# карточке модели.
MIN_RUBRIC_SHARE = 0.25


def c_tf_idf(
    groups: Mapping[Hashable, Sequence[Sequence[str]]],
    *,
    top_n: int = 5,
    min_count: int = 2,
    idf_groups: Mapping[Hashable, Sequence[Sequence[str]]] | None = None,
) -> dict[Hashable, list[Term]]:
    """Характерные слова каждой группы: {ключ группы -> список Term}.

    На вход — уже разобранные на слова документы, сложенные по группам. Своего
    разбора здесь нет намеренно: одна и та же токенизация нужна и здесь, и при
    отборе внутри семьи, и повторять её двумя вызовами значило бы получить два
    разных набора слов от одного текста.

    min_count отсекает слова, встретившиеся в группе один-два раза: у редкого
    слова tf мал, зато idf огромен, и произведение легко выносит наверх
    опечатку или имя собственное из единственной статьи.

    Порядок при равных значениях — по слову. Без этого две сборки модели с
    одинаковой геометрией дали бы разные подписи, а подпись входит в артефакт,
    который приложение опознаёт по sha256.

    idf_groups — по чему считать РАСПРОСТРАНЁННОСТЬ слова, если это не сами
    groups. Нужно ровно для одного случая, и случай этот важный: уточнение
    близнецов считается внутри семьи из двух-трёх кластеров, и слово, которое
    есть во всём корпусе, но поделено между ними 70 на 30, внутри семьи
    выглядит различающим. Так в подпись и попало «Законодательство и право —
    ҷумҳурии, тоҷикистон»: «Таджикистан» стоит в каждом документе корпуса.
    Считая tf по семье, а idf по всем кластерам, мы спрашиваем «чем этот
    кластер отличается от брата» и «редкое ли это слово вообще» — а это два
    разных вопроса, и мерить их по одному набору нельзя.
    """
    if top_n < 1:
        raise ValueError("top_n должно быть >= 1")

    counts: dict[Hashable, Counter] = {}
    for key, documents in groups.items():
        counter: Counter = Counter()
        for tokens in documents:
            counter.update(tokens)
        counts[key] = counter

    reference = counts
    if idf_groups is not None:
        reference = {}
        for key, documents in idf_groups.items():
            counter = Counter()
            for tokens in documents:
                counter.update(tokens)
            reference[key] = counter

    total_across: Counter = Counter()
    for counter in reference.values():
        total_across.update(counter)
    if not total_across:
        return {key: [] for key in groups}

    sizes = [sum(counter.values()) for counter in reference.values()]
    non_empty = [size for size in sizes if size]
    average_size = (sum(non_empty) / len(non_empty)) if non_empty else 0.0

    # Во скольких кластерах слово вообще встречается. Считается по тому же
    # набору, что и idf: вопрос «есть ли это слово где-то ещё» имеет смысл
    # только относительно всех кластеров, а не двух братьев.
    spread = Counter()
    for counter in reference.values():
        spread.update(set(counter))
    ceiling = max(1, int(len(reference) * MAX_CLUSTER_SHARE))

    result: dict[Hashable, list[Term]] = {}
    for key, counter in counts.items():
        size = sum(counter.values())
        if not size:
            result[key] = []
            continue
        scored = []
        for term, count in counter.items():
            if count < min_count:
                continue
            # Слова, которого нет в наборе для idf, быть не может, когда набор
            # тот же; когда он шире — тем более. Но полагаться на это нельзя:
            # деление на ноль дало бы бесконечность, то есть слово-победитель,
            # взявшееся из ниоткуда.
            seen = total_across.get(term, 0)
            if seen <= 0:
                continue
            # Слово из большинства кластеров — лексика корпуса, а не примета
            # одного из них. Потолок не применяется, когда кластер всего один:
            # там сравнивать не с чем, и запрет отобрал бы все слова разом.
            if len(reference) > 1 and spread[term] > ceiling:
                continue
            tf = count / size
            idf = math.log(1.0 + average_size / seen)
            scored.append(Term(term=term, score=tf * idf))
        scored.sort(key=lambda item: (-item.score, item.term))
        result[key] = scored[:top_n]
    return result


def _only_dominant(
    terms: Mapping[Hashable, Sequence[Term]],
    groups: Mapping[Hashable, Sequence[Sequence[str]]],
) -> dict[Hashable, list[Term]]:
    """Оставляет у каждой группы слова, которые в ней преимущественно и живут.

    Порог — DOMINANCE, доля от всех употреблений слова внутри переданного
    набора групп. Слово, размазанное по группам ровно, отличать их не может, и
    место в подписи занимать не должно.
    """
    counts: dict[Hashable, Counter] = {}
    for key, documents in groups.items():
        counter: Counter = Counter()
        for tokens in documents:
            counter.update(tokens)
        counts[key] = counter
    total: Counter = Counter()
    for counter in counts.values():
        total.update(counter)

    kept: dict[Hashable, list[Term]] = {}
    for key, candidates in terms.items():
        here = counts.get(key, Counter())
        kept[key] = [
            term
            for term in candidates
            if total[term.term] and here[term.term] / total[term.term] >= DOMINANCE
        ]
    return kept


def compose_labels(
    rubric_of_cluster: Mapping[int, str],
    tokens_of_cluster: Mapping[int, Sequence[Sequence[str]]],
    rubric_names: Mapping[str, str],
    *,
    terms_in_label: int = 2,
    separator: str = " — ",
    share_of_cluster: Mapping[int, float] | None = None,
    mixed_name: str = "",
) -> dict[int, str]:
    """Подпись каждого кластера: имя рубрики, а при совпадении — с уточнением.

    Кластер, единственный у своей рубрики, называется просто рубрикой: лишнее
    уточнение у однозначного имени только мешает читать.

    Кластеры-близнецы получают уточнение, посчитанное ВНУТРИ ИХ СЕМЬИ (см.
    объяснение в начале файла). Если у семьи характерных слов не нашлось вовсе
    (короткие тексты, всё отсеял min_count), близнецы получают номер кластера —
    два одинаковых имени хуже некрасивого, потому что по ним нельзя перейти к
    документам.
    """
    families: dict[str, list[int]] = {}
    for cluster, rubric in sorted(rubric_of_cluster.items()):
        families.setdefault(rubric, []).append(cluster)

    # Запасные слова — характерные для кластера на фоне ВСЕГО корпуса, а не
    # только своей семьи. Нужны там, где семейный признак не дал ничего: у
    # близнецов, поделивших словарь поровну, отличительных слов между собой нет,
    # но сказать про кластер что-то осмысленное всё равно можно. Номер кластера
    # остаётся последним средством, потому что «Экономика #4» пользователю не
    # сообщает ничего.
    all_tokens = {cluster: tokens_of_cluster.get(cluster, ()) for cluster in rubric_of_cluster}
    corpus_terms = c_tf_idf(all_tokens, top_n=terms_in_label * 4)

    # Кластеры, где преобладающая рубрика слишком слаба, из семей вынимаются:
    # их имя строится не от рубрики, и делить с ней уточнения им незачем.
    weak = {
        cluster
        for cluster, share in (share_of_cluster or {}).items()
        if float(share) < MIN_RUBRIC_SHARE
    }
    labels: dict[int, str] = {}
    if weak:
        weak_terms = c_tf_idf(
            {cluster: tokens_of_cluster.get(cluster, ()) for cluster in sorted(weak)},
            top_n=terms_in_label * 4,
            idf_groups=all_tokens,
        )
        taken_weak: set[str] = set()
        for cluster in sorted(weak):
            chosen = _pick(weak_terms.get(cluster, ()), taken_weak, terms_in_label)
            taken_weak.update(chosen)
            base = mixed_name or rubric_names.get(rubric_of_cluster.get(cluster, "")) or ""
            if chosen and base:
                labels[cluster] = base + separator + ", ".join(chosen)
            elif chosen:
                labels[cluster] = ", ".join(chosen)
            else:
                labels[cluster] = f"{base or 'Кластер'} #{cluster}"
        families = {
            rubric: [cluster for cluster in clusters if cluster not in weak]
            for rubric, clusters in families.items()
        }
        families = {rubric: clusters for rubric, clusters in families.items() if clusters}

    for rubric, clusters in families.items():
        base = rubric_names.get(rubric) or rubric
        if len(clusters) == 1:
            labels[clusters[0]] = base
            continue
        family_tokens = {cluster: tokens_of_cluster.get(cluster, ()) for cluster in clusters}
        # Берём с запасом: часть верхушки отсеет требование преобладания, и без
        # запаса у кластера не осталось бы ни одного слова.
        family_terms = c_tf_idf(
            family_tokens, top_n=terms_in_label * 4, idf_groups=all_tokens
        )
        family_terms = _only_dominant(family_terms, family_tokens)
        # Одно и то же слово не должно оказаться уточнением у двух близнецов:
        # «Иқтисод — сомонӣ» и «Иқтисод — сомонӣ» — это снова два одинаковых
        # имени. Слово занимается первым кластером, которому оно досталось; при
        # обходе в порядке номеров это правило воспроизводимо.
        taken: set[str] = set()
        for cluster in clusters:
            chosen = _pick(family_terms.get(cluster, ()), taken, terms_in_label)
            if not chosen:
                chosen = _pick(corpus_terms.get(cluster, ()), taken, terms_in_label)
            taken.update(chosen)
            if chosen:
                labels[cluster] = base + separator + ", ".join(chosen)
            else:
                labels[cluster] = f"{base} #{cluster}"
    return labels


def _same_stem(word: str, other: str) -> bool:
    """Одно ли это слово в двух формах — по общему началу (см. SAME_STEM_PREFIX)."""
    shortest = min(len(word), len(other))
    if shortest < SAME_STEM_PREFIX:
        return word == other
    return word[:SAME_STEM_PREFIX] == other[:SAME_STEM_PREFIX]


def _pick(terms: Sequence[Term], taken: set[str], limit: int) -> list[str]:
    """Первые limit слов, ещё не занятых соседом по рубрике и не повторяющих
    друг друга формой.

    Ограничение по числу обязательно: без него в подпись уходили все уцелевшие
    после отбора слова, и вместо «Иқтисод — барқ, нерӯгоҳ» получалось
    перечисление в полстроки. Обрезка на выходе, а не при подсчёте меры, потому
    что часть верхушки отсеивают требование преобладания и занятость соседом.
    """
    chosen: list[str] = []
    for term in terms:
        if any(_same_stem(term.term, word) for word in taken):
            continue
        if any(_same_stem(term.term, word) for word in chosen):
            continue
        chosen.append(term.term)
        if len(chosen) >= limit:
            break
    return chosen


def tokenize_all(texts: Iterable[str]) -> list[list[str]]:
    """Разбор пачки текстов — тем же tokenize, одной строкой у вызывающего."""
    return [tokenize(text) for text in texts]
