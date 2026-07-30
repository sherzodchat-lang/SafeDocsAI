import re

from app.modules.rag.chunker_config import (
    REASONING_MARKERS,
    RU_TJ_STOPWORDS,
    TAJIK_TO_RU_HINTS,
)


def normalize_query(query_text: str) -> str:
    normalized = (query_text or "").strip().lower()
    return re.sub(r"\s+", " ", normalized)


def detect_language(query_text: str) -> str:
    sample = (query_text or "").lower()
    tajik_chars = ("ӯ", "қ", "ҳ", "ҷ", "ғ", "ӣ")
    return "tj" if any(char in sample for char in tajik_chars) else "ru"


def is_prompt_injection_attempt(query_text: str) -> bool:
    lowered = (query_text or "").lower()
    patterns = (
        "ignore previous instructions",
        "forget all instructions",
        "system prompt",
        "developer message",
        "reveal prompt",
        "bypass",
        "jailbreak",
        "act as",
        "disregard above",
        "игнорируй предыдущие",
        "раскрой системный",
        "обойди ограничения",
        "фаромӯш кун дастур",
        "дастурҳоро нодида гир",
    )
    return any(pattern in lowered for pattern in patterns)


def looks_like_no_data(answer_text: str) -> bool:
    normalized = " ".join((answer_text or "").lower().split())
    return (
        "ответ не найден в базе" in normalized
        or "маълумот дар база мавҷуд нест" in normalized
        or "ответ не найден в выбранных источниках" in normalized
        or "маълумот дар манбаъҳои интихобшуда мавҷуд нест" in normalized
    )


_RU_SUFFIX_RE = re.compile(
    r"(ого|его|ому|ему|ыми|ими|ых|их|ая|яя|ое|ее|ый|ий|ой"
    r"|а|я|о|е|ы|и|у|ю|ь|ом|ем|ам|ям|ах|ях|ов|ев|ей)$"
)
_TJ_SUFFIX_RE = re.compile(r"(ҳоро|ҳои|ҳое|ҳо|ро|онӣ|он|ӣ|и)$")

_MIN_STEM_LEN = 3
# «он»/«онӣ» — таджикское множественное число (одамон, шахсон), но это же
# окончание частотных русских слов (закон, регион, миллион). Требуем остаток
# подлиннее: «шахсон» -> «шахс» режется, «закон» остаётся «закон».
_LONG_STEM_SUFFIXES = ("он", "онӣ")
_MIN_LONG_STEM_LEN = 4


def _strip_suffix(word: str, pattern: re.Pattern[str]) -> str:
    match = pattern.search(word)
    if not match:
        return word
    stem = word[: match.start()]
    min_len = (
        _MIN_LONG_STEM_LEN
        if match.group(1) in _LONG_STEM_SUFFIXES
        else _MIN_STEM_LEN
    )
    return stem if len(stem) >= min_len else word


def stem_simple(word: str) -> str:
    """Лёгкий ru/tj-стеммер: словоформы одного слова должны дать одну основу.

    Порядок правил критичен. Таджикские суффиксы снимаются первыми и до упора:
    в формах вида «андозҳоро» их два подряд, а русское правило, применённое
    раньше, съедало бы конечную гласную таджикского суффикса («ҳо» -> «ҳ»,
    «ро» -> «р»), после чего таджикское правило своего суффикса уже не находит
    и «андозҳо» расходится с «андоз». Русское правило идёт последним и в общем
    цикле — так «моддаҳо» -> «модда» -> «модд» сходится с «модда» -> «модд»,
    а «закона» -> «закон» успевает пройти таджикскую проверку следующим кругом.
    """
    if len(word) <= _MIN_STEM_LEN:
        return word
    for _ in range(4):
        stripped = word
        while True:
            shorter = _strip_suffix(stripped, _TJ_SUFFIX_RE)
            if shorter == stripped:
                break
            stripped = shorter
        stripped = _strip_suffix(stripped, _RU_SUFFIX_RE)
        if stripped == word:
            break
        word = stripped
    return word


def _char_ngrams(word: str, n: int = 3) -> set[str]:
    """Generate character n-grams for fuzzy matching (typo tolerance)."""
    if len(word) <= n:
        return {word}
    return {word[i:i + n] for i in range(len(word) - n + 1)}


def tokenize(text: str, ngram_size: int = 3) -> list[str]:
    """Токены документа со СЛУЧАЯМИ ПОВТОРОВ.

    Возвращает список, а не множество: BM25 считает частоту термина, и на
    множестве tf всегда равнялся бы 1 — чанк с пятью упоминаниями «ставка
    налога» ранжировался бы наравне с чанком с одним случайным упоминанием.
    Для стороны запроса, где повторы не нужны, есть query_tokens().
    """
    raw_tokens = re.findall(r"[а-яёА-ЯЁa-zA-Z0-9ӯқҳҷғӣӮҚҲҶҒӢ\-]+", (text or "").lower())
    normalized: list[str] = []
    for token in raw_tokens:
        if token not in RU_TJ_STOPWORDS:
            stemmed = stem_simple(token)
            if len(stemmed) >= 2 or stemmed.isdigit():
                normalized.append(stemmed)
                # Add character n-grams for words long enough to have typos
                if len(stemmed) >= ngram_size + 1:
                    normalized.extend(sorted(_char_ngrams(stemmed, ngram_size)))
        if "-" in token:
            for piece in token.split("-"):
                if piece not in RU_TJ_STOPWORDS:
                    stemmed_piece = stem_simple(piece)
                    if len(stemmed_piece) >= 2 or stemmed_piece.isdigit():
                        normalized.append(stemmed_piece)
    return normalized


def query_tokens(text: str) -> set[str]:
    return set(tokenize(text))


def is_reasoning_question(query: str) -> bool:
    query_norm = (query or "").lower()
    return bool(
        re.search(
            r"\b(почему|зачем|на каком основании|чаро|барои чӣ|бо кадом асос)\b",
            query_norm,
        )
    )


def has_reasoning_markers(text: str) -> bool:
    lowered = (text or "").lower()
    return any(marker in lowered for marker in REASONING_MARKERS)


def tajik_query_to_russian_hint(query_text: str) -> str:
    hinted = normalize_query(query_text)
    for pattern, replacement in TAJIK_TO_RU_HINTS:
        hinted = re.sub(pattern, replacement, hinted, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", hinted).strip()


def sanitize_answer_text(answer_text: str) -> str:
    text = (answer_text or "").strip()
    if not text or looks_like_no_data(text):
        return text
    heading = re.search(
        r"(?im)^\s*(legal\s+sources(?:\s*&\s*references)?|references)\s*:?\s*$", text
    )
    if heading:
        text = text[: heading.start()].strip()
    return re.sub(r"(?i)^\s*(answer|ответ|ҷавоб)\s*:\s*", "", text).strip()


def is_numeric_question(query: str) -> bool:
    query_norm = (query or "").lower()
    return bool(
        re.search(
            r"\b(сколько|каков|какая|какой|размер|ставк|процент|фоиз|чанд|андоза|қадар)\b",
            query_norm,
        )
    )


def detect_article_reference(query: str) -> str | None:
    q = query.lower()
    match_ru_prefix = re.search(r"(?:стать[а-яё]*|ст\.?)\s*(\d+)", q)
    if match_ru_prefix:
        return f"статья {match_ru_prefix.group(1)}"
    match_ru_suffix = re.search(r"(\d+)\s*-?\s*(?:стать[а-яё]*|ст\.?)", q)
    if match_ru_suffix:
        return f"статья {match_ru_suffix.group(1)}"
    match_tj_prefix = re.search(r"(?:моддаи?|мод\.?)\s*(\d+)", q)
    if match_tj_prefix:
        return f"моддаи {match_tj_prefix.group(1)}"
    match_tj_suffix = re.search(r"(\d+)\s*(?:мақола|моддаи?)", q)
    if match_tj_suffix:
        return f"моддаи {match_tj_suffix.group(1)}"
    match_law_suffix = re.search(r"(\d+)\s*-?(?:й|ый|ого)?\s*закон", q)
    if match_law_suffix:
        return f"закон {match_law_suffix.group(1)}"
    match_law_prefix = re.search(r"\bзакон[а-яё]*\s*(\d+)", q)
    if match_law_prefix:
        return f"закон {match_law_prefix.group(1)}"
    match_punkt = re.search(r"(?:пункт[а-яё]*|п\.?)\s*(\d+)", q)
    if match_punkt:
        return f"пункт {match_punkt.group(1)}"
    match_punkt_suffix = re.search(r"(\d+)\s*-?\s*(?:пункт[а-яё]*)", q)
    if match_punkt_suffix:
        return f"пункт {match_punkt_suffix.group(1)}"
    return None


def boost_article_chunks(results: dict, article_ref: str) -> dict:
    if not results.get("documents") or not results["documents"][0]:
        return results
    docs = list(results["documents"][0])
    ids = list(results["ids"][0])
    metas = list(results["metadatas"][0])
    dists = list(results["distances"][0])
    ref_lower = article_ref.lower()
    article_number = re.search(r"\d+", article_ref)
    number_str = article_number.group(0) if article_number else ""
    is_list_item_ref = ref_lower.startswith(("закон ", "пункт "))
    boosted = []
    normal = []
    for i, doc_text in enumerate(docs):
        text_lower = (doc_text or "").lower()
        contains_ref = ref_lower in text_lower
        if not contains_ref and number_str:
            keyword = ref_lower.replace(number_str, "").strip()
            pattern = re.escape(keyword) + r"\s+" + re.escape(number_str) + r"\b"
            contains_ref = bool(re.search(pattern, text_lower))
        if not contains_ref and is_list_item_ref and number_str:
            list_pattern = r"(?:^|\n)" + re.escape(number_str) + r"[.\s]"
            contains_ref = bool(re.search(list_pattern, text_lower))
        entry = (docs[i], ids[i], metas[i], dists[i])
        (boosted if contains_ref else normal).append(entry)
    reordered = boosted + normal
    if not reordered:
        return results
    return {
        **results,
        "documents": [[e[0] for e in reordered]],
        "ids": [[e[1] for e in reordered]],
        "metadatas": [[e[2] for e in reordered]],
        "distances": [
            [e[3] * 0.5 if i < len(boosted) else e[3] for i, e in enumerate(reordered)]
        ],
    }
