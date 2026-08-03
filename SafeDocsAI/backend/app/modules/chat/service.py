import hashlib
import json
import logging
import math
import re
from collections import Counter
from dataclasses import dataclass
from time import perf_counter, time as time_now
from typing import Any, AsyncIterator

from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.api.deps import assert_owns_notebook
from app.core.database import session_context
from app.core.exceptions import ApiError, AuthErrors
from app.models.models import Chunk, Document, Log, Notebook, User
from app.modules.chat.schemas import (
    ChatRequest,
    ChatResponse,
    RetrievalChunkItem,
    RetrievalRequest,
    RetrievalResponse,
    SourceItem,
)
from app.modules.rag.constants import DEFAULT_CHAT_MODEL
from app.services.profile_resolver import resolve_profile
from app.services.rag_service import RAGService, RELEVANCE_DISTANCE_THRESHOLD
from app.services.runtime_settings_service import RuntimeSettingsService

logger = logging.getLogger(__name__)

LEXICAL_BM25_K1 = 1.5
LEXICAL_BM25_B = 0.75
# Токенизатор подмешивает к каждому слову длиннее трёх букв все его символьные
# 3-граммы (страховка от опечаток). Из-за этого «процентов» даёт восемь
# «терминов», а «ндс» — один, и без понижающего веса BM25 ранжирует по длине
# слова, а не по значимости.
LEXICAL_NGRAM_SIZE = 3
LEXICAL_NGRAM_WEIGHT = 0.25
# Отсечка относительно лучшего чанка: кандидат, у которого вся лексическая
# «улика» — один общий низкоidf-стемминг («...ставки» на запрос про ставку НДС),
# набирает считанные проценты от лидера. В RRF значим только ранг, поэтому такой
# шум на второй строке даёт почти тот же вклад, что и настоящее совпадение на
# первой, и вытесняет из слияния векторные попадания. Порог относительный, так
# что лидер выживает всегда.
LEXICAL_MIN_SCORE_RATIO = 0.15
RRF_K = 60
# Сколько вариантов запроса (исходный + расширения доменного профиля) реально
# уходит в векторный поиск: каждый вариант — отдельный эмбеддинг и отдельный
# запрос в ChromaDB.
MAX_SEARCH_QUERY_VARIANTS = 3
# Пул для cross-encoder'а: эвристический реранкер отдаёт ему столько кандидатов,
# и только он режет список до final_top_k.
RERANKER_CANDIDATE_POOL = 45

# TTL cache for hybrid retrieval results (notebook_id, query) → results
_RETRIEVAL_CACHE: dict[str, tuple[float, dict[str, list[dict[str, Any]]]]] = {}
_RETRIEVAL_CACHE_TTL = 300  # 5 minutes
_RETRIEVAL_CACHE_MAX_SIZE = 200


@dataclass
class StreamChatPreparation:
    normalized_question: str
    language: str
    model: str
    assistant_name: str
    answer_rules: str | None
    no_data_answer: str
    chat_history: list[dict[str, str]]
    context: list[str]
    context_metadata: list[dict[str, Any]]
    sources: list[SourceItem]
    notebook_id: int | None
    domain_profile: str
    immediate_answer: str | None = None


def _retrieval_cache_key(
    notebook_id: int | None,
    query: str,
    allowed_doc_ids: set[int] | None = None,
    limits: tuple[int, int] | None = None,
    language: str = "",
) -> str:
    # Набор доступных документов входит в ключ: без блокнота он зависит от
    # пользователя, и общий ключ отдавал бы одному пользователю чужие чанки
    # из кэша другого. Лимиты и язык — потому что от них зависит сам результат:
    # язык выбирает варианты запроса у профиля, top-k — длину выдачи.
    if allowed_doc_ids is None:
        scope = "all"
    else:
        scope = hashlib.sha1(
            ",".join(str(doc_id) for doc_id in sorted(allowed_doc_ids)).encode()
        ).hexdigest()[:16]
    limits_part = f"{limits[0]}:{limits[1]}" if limits else "default"
    return (
        f"{notebook_id or 0}:{scope}:{limits_part}:{language}:"
        f"{RAGService.normalize_query(query)}"
    )


def _retrieval_cache_get(key: str) -> dict[str, list[dict[str, Any]]] | None:
    entry = _RETRIEVAL_CACHE.get(key)
    if entry is None:
        return None
    ts, result = entry
    if time_now() - ts > _RETRIEVAL_CACHE_TTL:
        _RETRIEVAL_CACHE.pop(key, None)
        return None
    return result


def _retrieval_cache_put(key: str, result: dict[str, list[dict[str, Any]]]) -> None:
    if len(_RETRIEVAL_CACHE) >= _RETRIEVAL_CACHE_MAX_SIZE:
        oldest_key = min(_RETRIEVAL_CACHE, key=lambda k: _RETRIEVAL_CACHE[k][0])
        _RETRIEVAL_CACHE.pop(oldest_key, None)
    _RETRIEVAL_CACHE[key] = (time_now(), result)


def safe_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(result) or math.isinf(result):
        return None
    return result


def stream_event(event: str, payload: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def require_log_author(current_user: User) -> int:
    """id автора новой записи журнала. Без автора запись не создаётся.

    log.user_id остаётся nullable осознанно: журнал — запись о событии, и
    «у события нет автора» бывает правдой. Такие строки в колонке уже есть
    (legacy), их никто не переписывает и не удаляет, поэтому NOT NULL на
    колонку не ставится — он запретил бы и их.

    Но сегодня безавторского писателя нет ни одного: журнал пишут только
    chat_request, chat_request_stream и handle_ask_request, то есть
    обработчики HTTP-запроса аутентифицированного пользователя. Фоновый
    воркер индексации (app/modules/jobs/worker.py) в журнал не пишет вовсе.
    Значит, новая строка без user_id — не системное событие, а потерянный
    автор, и ловить её надо здесь, до вставки, а не разбираться потом,
    почему у записи нет владельца.

    Если системный писатель когда-нибудь появится, он не должен звать эту
    функцию: он создаёт Log с user_id=None осознанно, и правило «ничьи
    записи видит только админ» (deps.user_owns) уже его накрывает.
    """
    user_id = current_user.id
    if user_id is None:
        raise ApiError(401, AuthErrors.INVALID_TOKEN, "Could not validate credentials")
    return user_id


async def persist_chat_log_short_lived(
    *,
    question: str,
    answer: str,
    sources: list[SourceItem],
    started: float,
    user_id: int,
    notebook_id: int | None,
    domain_profile: str,
) -> tuple[int | None, str | None]:
    try:
        async with session_context() as session:
            log_entry = Log(
                question=question,
                answer=answer,
                sources=json.dumps(
                    [item.model_dump() for item in sources], ensure_ascii=False
                ),
                time_ms=int((perf_counter() - started) * 1000),
                user_id=user_id,
                notebook_id=notebook_id,
                domain_profile=domain_profile,
            )
            session.add(log_entry)
            await session.commit()
            await session.refresh(log_entry)
            return log_entry.id, None
    except Exception:
        logger.exception("Failed to persist streaming chat log")
        return None, "log_persistence_failed"


def is_no_data_answer(answer: str) -> bool:
    normalized = " ".join((answer or "").lower().split())
    return (
        "ответ не найден в базе" in normalized
        or "маълумот дар база мавҷуд нест" in normalized
        or "ответ не найден в выбранных источниках" in normalized
        or "маълумот дар манбаъҳои интихобшуда мавҷуд нест" in normalized
    )


async def expand_with_neighbors(
    selected_chunks: list[dict[str, Any]], session: AsyncSession
) -> list[str]:
    if not selected_chunks:
        return []
    seen_ids = {item["chunk_id"] for item in selected_chunks}
    neighbor_queries: list[tuple[int, int]] = []
    for item in selected_chunks:
        meta = item.get("metadata", {})
        doc_id = meta.get("doc_id")
        chunk_idx = meta.get("chunk_index")
        if doc_id is None or chunk_idx is None:
            continue
        for offset in (-1, 1):
            neighbor_queries.append((doc_id, chunk_idx + offset))
    neighbor_texts: dict[str, str] = {}
    if neighbor_queries:
        from sqlalchemy import and_, or_

        conditions = [
            and_(Chunk.doc_id == did, Chunk.chunk_index == cidx)
            for did, cidx in neighbor_queries
        ]
        result = await session.exec(select(Chunk).where(or_(*conditions)))
        for chunk in result.all():
            cid = str(chunk.id)
            if cid not in seen_ids:
                neighbor_texts[cid] = chunk.text
    expanded = [item["text"] for item in selected_chunks]
    expanded.extend(neighbor_texts.values())
    return expanded


async def load_chunk_texts(
    session: AsyncSession, candidates: list[dict[str, Any]]
) -> dict[str, str]:
    """Тексты чанков из PostgreSQL по chunk_id кандидатов, одним запросом.

    Текст кандидата векторного поиска приходит из ChromaDB, а туда он попадает
    обогащённым: _build_embedding_text (app/modules/documents/service.py)
    приписывает к чанку «[имя документа | раздел | стр. N] » ради качества
    поиска. Для показа такой текст не годится, поэтому цитата берётся из
    таблицы chunk — источника истины. Срезать префикс регулярным выражением
    нельзя: его формат задаётся индексацией и меняется вместе с ней, а чистка
    сломалась бы молча и служебная строка снова уехала бы в ответ.

    Выборка одна на всю выдачу, а не по SELECT на цитату: кандидатов в ответе
    до retrieval_top_k на каждый список, и запрос на каждого — тот самый N+1,
    от которого lexical_retrieve_chunks_batch уходит одним проходом.
    """
    chunk_ids: set[int] = set()
    for item in candidates:
        # chunk_id кандидата — строка: идентификатором вектора в ChromaDB
        # служит str(chunk.id), и лексический поиск отдаёт его в том же виде.
        try:
            chunk_ids.add(int(item.get("chunk_id")))
        except (TypeError, ValueError):
            continue
    if not chunk_ids:
        return {}
    result = await session.exec(select(Chunk).where(Chunk.id.in_(chunk_ids)))
    return {str(chunk.id): chunk.text for chunk in result.all()}


def build_quote(chunk_id: Any, chunk_texts: dict[str, str]) -> str | None:
    """Цитата для ответа: начало chunk.text, а не текста из индекса.

    Строки в chunk_texts может не оказаться: между поиском и отрисовкой чанк
    успели удалить — та же гонка, ради которой существует
    drop_deleted_document_candidates. Тогда цитаты нет (None), но источник из
    ответа не выбрасывается: doc_id, имя документа и страница у него уже есть,
    и ссылка по ним работает. Подставлять сюда текст кандидата как запасной
    вариант нельзя — это вернуло бы служебный префикс в ответ, молча и только
    на редком пути, где этого никто не заметит.
    """
    quote = (chunk_texts.get(str(chunk_id)) or "").strip().replace("\n", " ")
    if len(quote) > 240:
        quote = quote[:240].rstrip() + "..."
    return quote or None


def is_greeting(text: str) -> bool:
    lowered = text.lower()
    patterns = [r"\b(салом|привет|здравствуйте|добрый\s+(день|вечер|утро))\b"]
    for pattern in patterns:
        if re.search(pattern, lowered) and len(lowered.split()) <= 3:
            return True
    return False


def candidate_identity(item: dict[str, Any]) -> str:
    metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
    doc_id = metadata.get("doc_id")
    chunk_index = metadata.get("chunk_index")
    if doc_id is not None and chunk_index is not None:
        return f"doc:{doc_id}:chunk:{chunk_index}"
    chunk_id = item.get("chunk_id")
    if chunk_id:
        return f"chunk:{chunk_id}"
    page = metadata.get("page")
    text = " ".join(str(item.get("text") or "").split())
    if doc_id is not None:
        return f"doc:{doc_id}:page:{page}:text:{text[:160]}"
    return text[:160]


def _merge_candidate_data(*items: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    merged_metadata: dict[str, Any] = {}
    for item in items:
        if not item:
            continue
        if not merged:
            merged = dict(item)
        else:
            for key, value in item.items():
                if key == "metadata":
                    continue
                if merged.get(key) is None and value is not None:
                    merged[key] = value
        metadata = item.get("metadata")
        if isinstance(metadata, dict):
            for key, value in metadata.items():
                if merged_metadata.get(key) is None and value is not None:
                    merged_metadata[key] = value
    merged["metadata"] = merged_metadata
    return merged


def select_relevant_chunks(
    context: list[str],
    context_chunk_ids: list[str],
    context_metadatas: list[dict],
    context_distances: list[Any],
    allowed_doc_ids: set[int] | None = None,
    query_text: str = "",
    final_top_k: int = 5,
    distance_threshold: float = RELEVANCE_DISTANCE_THRESHOLD,
) -> list[dict[str, Any]]:
    return rerank_retrieval_candidates(
        collect_chunk_candidates(
            context=context,
            context_chunk_ids=context_chunk_ids,
            context_metadatas=context_metadatas,
            context_distances=context_distances,
            allowed_doc_ids=allowed_doc_ids,
        ),
        query_text=query_text,
        final_top_k=final_top_k,
        distance_threshold=distance_threshold,
    )


def collect_chunk_candidates(
    context: list[str],
    context_chunk_ids: list[str],
    context_metadatas: list[dict],
    context_distances: list[Any],
    allowed_doc_ids: set[int] | None = None,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for idx, chunk_text in enumerate(context):
        if not chunk_text:
            continue
        metadata = (
            context_metadatas[idx]
            if idx < len(context_metadatas) and isinstance(context_metadatas[idx], dict)
            else {}
        )
        doc_id = metadata.get("doc_id")
        if allowed_doc_ids is not None and doc_id not in allowed_doc_ids:
            continue
        distance = safe_float(
            context_distances[idx] if idx < len(context_distances) else None
        )
        candidates.append(
            {
                "idx": idx,
                "text": chunk_text,
                "metadata": metadata,
                "chunk_id": context_chunk_ids[idx]
                if idx < len(context_chunk_ids)
                else None,
                "distance": distance,
            }
        )
    return candidates


async def drop_deleted_document_candidates(
    session: AsyncSession,
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Выбросить из выдачи ChromaDB чанки документов, которых нет в БД.

    Нужно там, где allowed_doc_ids = None, то есть «без фильтра» (админ без
    блокнота): у всех остальных множество построено запросом к document, и
    висячий вектор отсекается ещё в collect_chunk_candidates. Коллекция и
    таблица chunk расходятся, если удаление векторов не прошло (см. задачу
    cleanup_embeddings), и тогда админ получает цитату из документа, которого
    больше нет.

    Проверяем существование найденных документов, а не подставляем все id в
    where-фильтр Chroma: у админа список доступных документов — это вся
    коллекция, такой $in ничего не отсекает, но растёт вместе с базой и уходит
    в каждый запрос. Кандидатов же не больше retrieval_top_k на вариант
    запроса, поэтому проверка — один короткий SELECT по первичному ключу.
    Фильтр в chroma_doc_filter при этом не трогаем: для блокнота и обычного
    пользователя предварительный отбор по doc_id по-прежнему обязателен.
    """
    doc_ids = {
        candidate["metadata"].get("doc_id")
        for candidate in candidates
        if isinstance(candidate.get("metadata"), dict)
    }
    doc_ids = {doc_id for doc_id in doc_ids if isinstance(doc_id, int)}
    if not doc_ids:
        return candidates
    result = await session.exec(select(Document.id).where(Document.id.in_(doc_ids)))
    existing = {doc_id for doc_id in result.all() if doc_id is not None}
    missing = doc_ids - existing
    if not missing:
        return candidates
    kept = [
        candidate
        for candidate in candidates
        if not (
            isinstance(candidate.get("metadata"), dict)
            and candidate["metadata"].get("doc_id") in missing
        )
    ]
    logger.warning(
        "Dropped %d chunk(s) of %d deleted document(s) %s from vector results: "
        "ChromaDB still holds their vectors, run reconcile_chroma.py",
        len(candidates) - len(kept),
        len(missing),
        sorted(missing),
    )
    return kept


def rank_vector_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best_by_key: dict[str, dict[str, Any]] = {}
    for item in candidates:
        key = candidate_identity(item)
        current = best_by_key.get(key)
        if current is None:
            best_by_key[key] = dict(item)
            continue
        current_distance = current.get("distance")
        item_distance = item.get("distance")
        if current_distance is None or (
            item_distance is not None and item_distance < current_distance
        ):
            best_by_key[key] = _merge_candidate_data(item, current)
        else:
            best_by_key[key] = _merge_candidate_data(current, item)

    ranked = list(best_by_key.values())
    ranked.sort(
        key=lambda item: (
            item.get("distance") if item.get("distance") is not None else float("inf"),
            item.get("idx", 0),
        )
    )
    for rank, item in enumerate(ranked, start=1):
        item["rank"] = rank
        item["retrieval_method"] = "vector"
    return ranked


def rank_lexical_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best_by_key: dict[str, dict[str, Any]] = {}
    for item in candidates:
        key = candidate_identity(item)
        current = best_by_key.get(key)
        if current is None:
            best_by_key[key] = dict(item)
            continue
        current_score = current.get("lexical_score") or 0.0
        item_score = item.get("lexical_score") or 0.0
        if item_score > current_score:
            best_by_key[key] = _merge_candidate_data(item, current)
        else:
            best_by_key[key] = _merge_candidate_data(current, item)

    ranked = list(best_by_key.values())
    ranked.sort(
        key=lambda item: (
            -(item.get("lexical_score") or 0.0),
            item.get("idx", 0),
            item.get("metadata", {}).get("page") or 0,
        )
    )
    for rank, item in enumerate(ranked, start=1):
        item["rank"] = rank
        item["retrieval_method"] = "lexical"
    return ranked


async def lexical_retrieve_chunks(
    *,
    session: AsyncSession,
    query_text: str,
    allowed_doc_ids: set[int] | None,
    retrieval_top_k: int,
) -> list[dict[str, Any]]:
    return await lexical_retrieve_chunks_batch(
        session=session,
        query_texts=[query_text],
        allowed_doc_ids=allowed_doc_ids,
        retrieval_top_k=retrieval_top_k,
    )


def lexical_token_weights(query_tokens: set[str]) -> dict[str, float]:
    """Вес каждого терма запроса в BM25.

    Символьные 3-граммы (см. LEXICAL_NGRAM_WEIGHT) — это не самостоятельные
    термины, а размазанное по подстрокам повторное совпадение с тем же словом,
    поэтому они идут с понижающим весом. Опознаём их по длине и вхождению в
    более длинный токен того же запроса: список токенов плоский, и другого
    признака у нас нет.
    """
    long_tokens = [token for token in query_tokens if len(token) > LEXICAL_NGRAM_SIZE]
    weights: dict[str, float] = {}
    for token in query_tokens:
        is_ngram = len(token) == LEXICAL_NGRAM_SIZE and any(
            token in word for word in long_tokens
        )
        weights[token] = LEXICAL_NGRAM_WEIGHT if is_ngram else 1.0
    return weights


async def lexical_retrieve_chunks_batch(
    *,
    session: AsyncSession,
    query_texts: list[str],
    allowed_doc_ids: set[int] | None,
    retrieval_top_k: int,
) -> list[dict[str, Any]]:
    """BM25 retrieval with merged tokens from multiple query variants (single DB pass)."""
    merged_query_tokens: set[str] = set()
    for qt in query_texts:
        merged_query_tokens.update(RAGService._query_tokens(qt))
    if not merged_query_tokens:
        return []
    token_weights = lexical_token_weights(merged_query_tokens)
    if allowed_doc_ids is not None and not allowed_doc_ids:
        return []

    statement = select(Chunk, Document).join(Document, Document.id == Chunk.doc_id)
    if allowed_doc_ids is not None:
        statement = statement.where(Chunk.doc_id.in_(allowed_doc_ids))

    rows = (await session.exec(statement)).all()
    if not rows:
        return []

    # Merge article refs, years, normalized queries from all query variants
    article_refs: set[str] = set()
    query_years: set[str] = set()
    normalized_queries: list[str] = []
    for qt in query_texts:
        nq = RAGService.normalize_query(qt)
        normalized_queries.append(nq)
        ref = RAGService._detect_article_reference(nq)
        if ref:
            article_refs.add(ref)
        query_years.update(re.findall(r"\b(?:19|20)\d{2}\b", qt))

    chunk_records: list[dict[str, Any]] = []
    doc_frequency: Counter[str] = Counter()
    total_length = 0
    corpus_size = 0

    for chunk, document in rows:
        if allowed_doc_ids is not None and chunk.doc_id not in allowed_doc_ids:
            continue
        text = str(chunk.text or "")
        # tokenize() отдаёт токены с повторами: и tf, и длина чанка считаются по
        # одному и тому же потоку токенов — иначе нормировка B * (len / avg_len)
        # сравнивала бы несопоставимые величины.
        token_list = RAGService.tokenize(text)
        if not token_list:
            continue
        corpus_size += 1
        total_length += len(token_list)
        token_counts = Counter(token_list)
        overlap = [token for token in merged_query_tokens if token_counts.get(token)]
        for token in set(overlap):
            doc_frequency[token] += 1
        if not overlap:
            continue
        chunk_records.append(
            {
                "chunk": chunk,
                "document": document,
                "token_counts": token_counts,
                "chunk_length": len(token_list),
            }
        )

    if not chunk_records:
        return []

    avg_chunk_length = total_length / max(corpus_size, 1)
    ranked: list[dict[str, Any]] = []

    for record in chunk_records:
        chunk = record["chunk"]
        document = record["document"]
        token_counts: Counter[str] = record["token_counts"]
        chunk_length = record["chunk_length"]
        score = 0.0
        for token in merged_query_tokens:
            term_frequency = token_counts.get(token, 0)
            if term_frequency <= 0:
                continue
            frequency = doc_frequency.get(token, 0)
            idf = math.log(1.0 + ((corpus_size - frequency + 0.5) / (frequency + 0.5)))
            denominator = term_frequency + LEXICAL_BM25_K1 * (
                1.0
                - LEXICAL_BM25_B
                + LEXICAL_BM25_B * (chunk_length / avg_chunk_length)
            )
            score += (
                token_weights.get(token, 1.0)
                * idf
                * ((term_frequency * (LEXICAL_BM25_K1 + 1.0)) / denominator)
            )

        normalized_text = RAGService.normalize_query(chunk.text)
        # Boost for exact phrase match in any query variant
        if any(nq and nq in normalized_text for nq in normalized_queries):
            score += 0.25
        # Boost for article references
        if any(ref in normalized_text for ref in article_refs):
            score += 0.35
        if query_years and any(year in (document.name or "") for year in query_years):
            score += 0.3

        ranked.append(
            {
                "idx": chunk.chunk_index if chunk.chunk_index is not None else 0,
                "text": chunk.text,
                "metadata": {
                    "doc_id": chunk.doc_id,
                    "doc_name": document.name,
                    "page": chunk.page,
                    "chunk_index": chunk.chunk_index,
                    "section": chunk.section,
                },
                "chunk_id": str(chunk.id) if chunk.id is not None else None,
                "distance": None,
                "lexical_score": round(score, 6),
                "retrieval_method": "lexical",
            }
        )

    ranked.sort(
        key=lambda item: (
            -(item.get("lexical_score") or 0.0),
            item.get("idx", 0),
            item.get("metadata", {}).get("page") or 0,
        )
    )
    best_score = ranked[0].get("lexical_score") or 0.0
    if best_score > 0.0:
        cutoff = best_score * LEXICAL_MIN_SCORE_RATIO
        ranked = [
            item for item in ranked if (item.get("lexical_score") or 0.0) >= cutoff
        ]
    limited = ranked[:retrieval_top_k]
    for rank, item in enumerate(limited, start=1):
        item["rank"] = rank
    return limited


def fuse_candidates_with_rrf(
    vector_candidates: list[dict[str, Any]], lexical_candidates: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    fused_by_key: dict[str, dict[str, Any]] = {}

    def add_ranked(items: list[dict[str, Any]], rank_field: str) -> None:
        for rank, item in enumerate(items, start=1):
            key = candidate_identity(item)
            fused = fused_by_key.get(key)
            if fused is None:
                fused = dict(item)
                fused["modalities"] = set()
                fused["rrf_score"] = 0.0
                fused_by_key[key] = fused
            else:
                fused = _merge_candidate_data(fused, item)
                fused.setdefault("modalities", set())
                fused.setdefault("rrf_score", 0.0)
                fused_by_key[key] = fused
            fused[rank_field] = rank
            fused["rrf_score"] += 1.0 / (RRF_K + rank)
            method = item.get("retrieval_method")
            if method:
                fused["modalities"].add(method)

    add_ranked(vector_candidates, "vector_rank")
    add_ranked(lexical_candidates, "lexical_rank")

    fused = list(fused_by_key.values())
    fused.sort(
        key=lambda item: (
            -(item.get("rrf_score") or 0.0),
            item.get("vector_rank") or float("inf"),
            item.get("lexical_rank") or float("inf"),
            item.get("idx", 0),
        )
    )
    for rank, item in enumerate(fused, start=1):
        item["rank"] = rank
        modalities = sorted(item.pop("modalities", set()))
        item["retrieval_method"] = "+".join(modalities) if modalities else "hybrid"
        item["rrf_score"] = round(item.get("rrf_score") or 0.0, 6)
    return fused


def resolve_retrieval_limits(
    runtime_settings: dict[str, Any],
    requested_top_k: int | None = None,
    requested_retrieval_top_k: int | None = None,
) -> tuple[int, int]:
    final_top_k = safe_int(
        requested_top_k
        if requested_top_k is not None
        else runtime_settings.get("top_k"),
        default=5,
        min_value=1,
        max_value=20,
    )
    retrieval_top_k = safe_int(
        (
            requested_retrieval_top_k
            if requested_retrieval_top_k is not None
            else runtime_settings.get("retrieval_top_k")
        ),
        default=20,
        min_value=final_top_k,
        max_value=50,
    )
    return retrieval_top_k, final_top_k


def safe_int(
    value: Any, default: int, min_value: int | None = None, max_value: int | None = None
) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError):
        result = default
    if min_value is not None:
        result = max(min_value, result)
    if max_value is not None:
        result = min(max_value, result)
    return result


def chroma_doc_filter(allowed_doc_ids: set[int] | None) -> dict[str, Any] | None:
    """Фильтр по документам для самого векторного поиска, а не после него.

    Коллекция в ChromaDB одна на всю систему: без where топ-N глобального поиска
    может целиком состоять из документов чужих блокнотов, и питоновский
    пост-фильтр оставит от него ноль кандидатов — именно так «работавший на одном
    блокноте» поиск ломался после загрузки соседних баз.
    Фильтруем по doc_id, а не по notebook_id: notebook_id в метаданных чанка не
    обновляется при переносе документа между блокнотами и потому не надёжен.
    """
    if not allowed_doc_ids:
        return None
    doc_ids = sorted(allowed_doc_ids)
    if len(doc_ids) == 1:
        return {"doc_id": doc_ids[0]}
    return {"doc_id": {"$in": doc_ids}}


def expand_search_queries(profile: Any, search_query: str, language: str) -> list[str]:
    """Исходный запрос плюс варианты доменного профиля (тадж. → рус. синонимы).

    Профиль умеет расширять запрос, но раньше его об этом никто не спрашивал:
    список вариантов собирался прямо здесь из одного исходного запроса.
    """
    if not search_query:
        return []
    try:
        profile_variants = profile.search_queries(search_query, language)
    except Exception:
        logger.exception("Domain profile search_queries failed, using original query")
        profile_variants = None
    ordered: list[str] = []
    for variant in [search_query, *(profile_variants or [])]:
        normalized = (variant or "").strip()
        if normalized and normalized not in ordered:
            ordered.append(normalized)
    return ordered[:MAX_SEARCH_QUERY_VARIANTS]


def rerank_retrieval_candidates(
    candidates: list[dict[str, Any]],
    query_text: str,
    final_top_k: int,
    distance_threshold: float = RELEVANCE_DISTANCE_THRESHOLD,
) -> list[dict[str, Any]]:
    if not candidates:
        return []

    deduplicated: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    for item in candidates:
        key = candidate_identity(item)
        if not key or key in seen_keys:
            continue
        seen_keys.add(key)
        deduplicated.append(item)

    filtered = [
        item
        for item in deduplicated
        if item.get("distance") is None or item["distance"] <= distance_threshold
    ]
    if not filtered:
        filtered = list(deduplicated)
    # Guarantee minimum 3 chunks
    MIN_CHUNKS = 3
    if len(filtered) < MIN_CHUNKS and len(deduplicated) > len(filtered):
        existing_keys = {candidate_identity(i) for i in filtered}
        for item in deduplicated:
            if len(filtered) >= MIN_CHUNKS:
                break
            if candidate_identity(item) not in existing_keys:
                filtered.append(item)

    normalized_query = RAGService.normalize_query(query_text)
    query_tokens = set(RAGService._query_tokens(query_text))
    article_ref = RAGService._detect_article_reference(normalized_query)

    for item in filtered:
        item["rerank_score"] = _score_retrieval_candidate(
            item,
            normalized_query=normalized_query,
            query_tokens=query_tokens,
            article_ref=article_ref,
        )

    filtered.sort(
        key=lambda item: (
            -(item.get("rerank_score") or 0.0),
            item.get("distance") if item.get("distance") is not None else float("inf"),
            item.get("idx", 0),
        )
    )
    return filtered[:final_top_k]


def _score_retrieval_candidate(
    item: dict[str, Any],
    normalized_query: str,
    query_tokens: set[str],
    article_ref: str | None,
) -> float:
    text = str(item.get("text") or "")
    metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
    distance = item.get("distance")
    query_years = set(re.findall(r"\b(?:19|20)\d{2}\b", normalized_query))

    body_tokens = set(RAGService.tokenize(text))
    title_tokens: set[str] = set()
    metadata_text_parts: list[str] = []
    for key in (
        "title",
        "heading",
        "section_title",
        "section",
        "doc_name",
        "category",
        "article",
    ):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            metadata_text_parts.append(value)
            title_tokens.update(RAGService.tokenize(value))

    overlap_ratio = 0.0
    title_overlap_ratio = 0.0
    if query_tokens:
        overlap_ratio = len(query_tokens & body_tokens) / len(query_tokens)
        title_overlap_ratio = len(query_tokens & title_tokens) / len(query_tokens)

    combined_text = RAGService.normalize_query(" ".join([text, *metadata_text_parts]))
    exact_phrase_boost = (
        0.25 if normalized_query and normalized_query in combined_text else 0.0
    )
    article_boost = 0.35 if article_ref and article_ref in combined_text else 0.0
    best_rank = min(
        [
            rank
            for rank in [
                item.get("rank"),
                item.get("vector_rank"),
                item.get("lexical_rank"),
            ]
            if isinstance(rank, int) and rank > 0
        ],
        default=10,
    )
    rank_boost = max(0.0, 0.08 - ((best_rank - 1) * 0.01))
    year_boost = (
        0.25
        if query_years
        and any(year in (metadata.get("doc_name") or "") for year in query_years)
        else 0.0
    )
    distance_score = 0.5 if distance is None else 1.0 / (1.0 + max(distance, 0.0))

    return round(
        (distance_score * 0.45)
        + (overlap_ratio * 0.3)
        + (title_overlap_ratio * 0.12)
        + exact_phrase_boost
        + article_boost
        + year_boost
        + rank_boost,
        6,
    )


async def run_retrieval(
    *,
    rag_service: RAGService,
    session: AsyncSession,
    profile: Any,
    language: str,
    search_query: str,
    allowed_doc_ids: set[int] | None,
    retrieval_top_k: int,
    final_top_k: int,
    original_query: str | None = None,
    notebook_id: int | None = None,
) -> list[dict[str, Any]]:
    retrieval_result = await run_hybrid_retrieval(
        rag_service=rag_service,
        session=session,
        profile=profile,
        language=language,
        search_query=search_query,
        original_query=original_query,
        allowed_doc_ids=allowed_doc_ids,
        retrieval_top_k=retrieval_top_k,
        final_top_k=final_top_k,
        notebook_id=notebook_id,
    )
    return retrieval_result["final_chunks"]


async def run_hybrid_retrieval(
    *,
    rag_service: RAGService,
    session: AsyncSession,
    profile: Any,
    language: str,
    search_query: str,
    original_query: str | None,
    allowed_doc_ids: set[int] | None,
    retrieval_top_k: int,
    final_top_k: int,
    notebook_id: int | None = None,
) -> dict[str, list[dict[str, Any]]]:
    # Check cache
    cache_key = _retrieval_cache_key(
        notebook_id,
        search_query,
        allowed_doc_ids,
        limits=(retrieval_top_k, final_top_k),
        language=language,
    )
    cached = _retrieval_cache_get(cache_key)
    if cached is not None:
        logger.debug("Retrieval cache hit for: %s", search_query)
        return cached

    pooled_vector_candidates: list[dict[str, Any]] = []
    search_queries = expand_search_queries(profile, search_query, language)
    doc_filter = chroma_doc_filter(allowed_doc_ids)
    # Пустое (а не None) множество — «доступных документов нет вообще»:
    # обходим векторный поиск, иначе запрос без where вернул бы чужие чанки.
    scope_is_empty = allowed_doc_ids is not None and not allowed_doc_ids

    # Vector retrieval: still per-query (different embeddings)
    for candidate_query in [] if scope_is_empty else search_queries:
        logger.debug(
            "Querying ChromaDB with: %s, retrieval_top_k=%s, doc_filter=%s",
            candidate_query,
            retrieval_top_k,
            doc_filter,
        )
        results = rag_service.query_documents(
            candidate_query, n_results=retrieval_top_k, where=doc_filter
        )
        results = profile.rerank_results(candidate_query, results)
        documents = results.get("documents", [])
        chunk_ids = results.get("ids", [])
        metadatas = results.get("metadatas", [])
        distances = results.get("distances", [])
        query_candidates = collect_chunk_candidates(
            context=documents[0] if documents else [],
            context_chunk_ids=chunk_ids[0] if chunk_ids else [],
            context_metadatas=metadatas[0] if metadatas else [],
            context_distances=distances[0] if distances else [],
            allowed_doc_ids=allowed_doc_ids,
        )
        for item in query_candidates:
            item["retrieval_method"] = "vector"
        pooled_vector_candidates.extend(query_candidates)

    # Без allowed_doc_ids пост-фильтр в collect_chunk_candidates выключен, и
    # висячие векторы удалённых документов доходят до ответа как цитаты.
    # Проверяем их существование по БД (см. drop_deleted_document_candidates).
    if allowed_doc_ids is None and pooled_vector_candidates:
        pooled_vector_candidates = await drop_deleted_document_candidates(
            session, pooled_vector_candidates
        )

    # Lexical retrieval: single pass with merged tokens from all query variants
    pooled_lexical_candidates = await lexical_retrieve_chunks_batch(
        session=session,
        query_texts=search_queries,
        allowed_doc_ids=allowed_doc_ids,
        retrieval_top_k=retrieval_top_k,
    )

    vector_candidates = rank_vector_candidates(pooled_vector_candidates)[
        :retrieval_top_k
    ]
    lexical_candidates = rank_lexical_candidates(pooled_lexical_candidates)[
        :retrieval_top_k
    ]

    if not vector_candidates:
        logger.warning(
            "run_hybrid_retrieval: ChromaDB returned 0 candidates for query=%r (allowed_doc_ids=%s)",
            search_query,
            allowed_doc_ids,
        )
    if not lexical_candidates:
        logger.warning(
            "run_hybrid_retrieval: BM25 returned 0 candidates for query=%r (allowed_doc_ids=%s)",
            search_query,
            allowed_doc_ids,
        )

    fused_candidates = fuse_candidates_with_rrf(vector_candidates, lexical_candidates)

    runtime_settings = RuntimeSettingsService.get_settings()
    reranker_enabled = bool(runtime_settings.get("reranker_enabled"))
    # С включённым cross-encoder'ом эвристика работает как фильтр грубой очистки
    # и отдаёт ему расширенный пул: если срезать список до final_top_k здесь,
    # реранкер уже не сможет поднять чанк, отброшенный эвристикой.
    heuristic_top_k = (
        max(final_top_k, RERANKER_CANDIDATE_POOL) if reranker_enabled else final_top_k
    )
    final_chunks = rerank_retrieval_candidates(
        fused_candidates,
        query_text=search_query,
        final_top_k=heuristic_top_k,
    )

    if not final_chunks and not scope_is_empty:
        logger.warning(
            "run_hybrid_retrieval: final result is empty after RRF fusion "
            "(vector=%d, lexical=%d, fused=%d) for query=%r — triggering top-3 fallback",
            len(vector_candidates),
            len(lexical_candidates),
            len(fused_candidates),
            search_query,
        )
        # Hard fallback: get any 3 chunks from ChromaDB ignoring relevance
        try:
            fallback_results = rag_service.query_documents(
                search_query,
                n_results=3,
                where=doc_filter,
            )
            fb_docs = fallback_results.get("documents", [[]])[0]
            fb_ids = fallback_results.get("ids", [[]])[0]
            fb_metas = fallback_results.get("metadatas", [[]])[0]
            fb_dists = fallback_results.get("distances", [[]])[0]
            final_chunks = collect_chunk_candidates(
                context=fb_docs,
                context_chunk_ids=fb_ids,
                context_metadatas=fb_metas,
                context_distances=fb_dists,
                allowed_doc_ids=allowed_doc_ids,
            )
            for item in final_chunks:
                item["retrieval_method"] = "fallback"
            if allowed_doc_ids is None and final_chunks:
                final_chunks = await drop_deleted_document_candidates(
                    session, final_chunks
                )
            logger.warning(
                "run_hybrid_retrieval: fallback returned %d chunks", len(final_chunks)
            )
        except Exception as fb_exc:
            logger.error("run_hybrid_retrieval: fallback also failed: %s", fb_exc)

    # Optional reranker pass: именно он режет пул до final_top_k
    if reranker_enabled and final_chunks:
        # reranker_model из runtime-настроек передаётся только для логов: модель
        # зашита в reranker_service (см. комментарий к rerank_candidates).
        reranker_model = runtime_settings.get("reranker_model", "")
        try:
            from app.modules.rag.reranker_service import rerank_candidates
            from app.modules.rag.model_manager import ModelManager
            logger.debug(
                "Reranker: scoring %d candidates down to %d",
                len(final_chunks),
                final_top_k,
            )
            final_chunks = await rerank_candidates(
                candidates=final_chunks,
                query=search_query,
                model=reranker_model,
                model_manager=ModelManager(),
                top_k=final_top_k,
            )
        except Exception as rr_exc:
            logger.warning("Reranker failed, keeping original order: %s", rr_exc)
    # Страховка: при отказе реранкера пул мог остаться шире запрошенного top-k.
    final_chunks = final_chunks[:final_top_k]

    result = {
        "vector_candidates": vector_candidates,
        "lexical_candidates": lexical_candidates,
        "fused_candidates": fused_candidates,
        "final_chunks": final_chunks,
    }

    # Store in cache
    _retrieval_cache_put(cache_key, result)

    return result


async def retrieve_year_targeted_chunks(
    *,
    session: AsyncSession,
    question: str,
    allowed_doc_ids: set[int] | None,
    final_top_k: int,
) -> list[dict[str, Any]]:
    year_match = re.search(r"\b(?:19|20)\d{2}\b", question)
    if not year_match or not allowed_doc_ids:
        return []

    target_year = year_match.group(0)
    docs_result = await session.exec(
        select(Document).where(Document.id.in_(allowed_doc_ids))
    )
    target_docs = [
        doc for doc in docs_result.all() if target_year in str(doc.name or "")
    ]
    if not target_docs:
        return []

    target_doc = target_docs[0]

    # Use ChromaDB directly with a doc_id filter instead of re-embedding all chunks
    rag_service = RAGService()
    try:
        results = rag_service.query_documents(
            question,
            n_results=final_top_k,
            where={"doc_id": target_doc.id},
        )
    except Exception:
        return []

    documents = results.get("documents", [[]])[0]
    chunk_ids = results.get("ids", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    ranked: list[dict[str, Any]] = []
    for i, doc_text in enumerate(documents):
        if not doc_text:
            continue
        meta = (
            metadatas[i]
            if i < len(metadatas) and isinstance(metadatas[i], dict)
            else {}
        )
        distance = safe_float(distances[i] if i < len(distances) else None)
        ranked.append(
            {
                "idx": meta.get("chunk_index") or i,
                "text": doc_text,
                "metadata": {
                    "doc_id": target_doc.id,
                    "doc_name": target_doc.name,
                    "page": meta.get("page"),
                    "chunk_index": meta.get("chunk_index"),
                    "section": meta.get("section"),
                },
                "chunk_id": chunk_ids[i] if i < len(chunk_ids) else None,
                "distance": distance,
                "retrieval_method": "year_fallback",
                "rerank_score": 1.0 / (1.0 + max(distance, 0.0))
                if distance is not None
                else 0.15,
            }
        )

    ranked.sort(
        key=lambda item: (-(item.get("rerank_score") or 0.0), item.get("idx", 0))
    )
    return ranked[:final_top_k]


async def resolve_notebook_scope(
    *,
    notebook_id: int | None,
    session: AsyncSession,
    current_user: User,
) -> tuple[Notebook | None, set[int] | None]:
    """Блокнот из тела запроса и документы, доступные текущему пользователю.

    Раньше чужой или отсутствующий notebook_id молча заменялся первым блокнотом
    в системе, и ответ строился поверх чужих документов. Теперь чужой блокнот —
    404, а без блокнота поиск ограничен документами самого пользователя.
    None вместо множества возвращается только админу и означает «без фильтра».
    """
    notebook: Notebook | None = None
    if notebook_id is not None:
        await assert_owns_notebook(notebook_id, session, current_user)
        notebook = await session.get(Notebook, notebook_id)

    is_admin = current_user.role == "admin"
    if notebook is None and is_admin:
        return None, None

    statement = select(Document.id)
    if notebook is not None and notebook.id is not None:
        statement = statement.where(Document.notebook_id == notebook.id)
    if not is_admin:
        statement = statement.where(Document.owner_id == current_user.id)
    result = await session.exec(statement)
    return notebook, {doc_id for doc_id in result.all() if doc_id is not None}


async def retrieve_chunks(
    retrieval_request: RetrievalRequest,
    current_user: User,
    session: AsyncSession,
) -> RetrievalResponse:
    rag_service = RAGService()
    normalized_question = rag_service.normalize_query(retrieval_request.question)
    language = rag_service.detect_language(normalized_question)
    runtime_settings = RuntimeSettingsService.get_settings()
    retrieval_top_k, final_top_k = resolve_retrieval_limits(
        runtime_settings,
        requested_top_k=retrieval_request.top_k,
        requested_retrieval_top_k=retrieval_request.retrieval_top_k,
    )
    model = runtime_settings.get("chat_model") or runtime_settings.get(
        "model", DEFAULT_CHAT_MODEL
    )

    notebook, allowed_doc_ids = await resolve_notebook_scope(
        notebook_id=retrieval_request.notebook_id,
        session=session,
        current_user=current_user,
    )

    profile = resolve_profile(
        notebook=notebook, requested=retrieval_request.domain_profile
    )

    history_result = await session.exec(
        select(Log)
        .where(Log.user_id == current_user.id)
        .where(Log.notebook_id == (notebook.id if notebook else None))
        .order_by(Log.created_at.desc())
        .limit(5)
    )
    history_logs = sorted(history_result.all(), key=lambda x: x.created_at)
    chat_history = []
    for log in history_logs:
        chat_history.append({"role": "user", "content": log.question})
        chat_history.append({"role": "assistant", "content": log.answer})

    article_ref = rag_service._detect_article_reference(normalized_question)
    if article_ref:
        search_query = normalized_question
    else:
        search_query = await rag_service.condense_query(
            normalized_question, chat_history, model=model
        )

    retrieval_result = await run_hybrid_retrieval(
        rag_service=rag_service,
        session=session,
        profile=profile,
        language=language,
        search_query=search_query,
        original_query=normalized_question,
        allowed_doc_ids=allowed_doc_ids,
        retrieval_top_k=retrieval_top_k,
        final_top_k=final_top_k,
        notebook_id=notebook.id if notebook else None,
    )
    selected_chunks = retrieval_result["final_chunks"]

    reported_items = [
        *retrieval_result["vector_candidates"],
        *retrieval_result["lexical_candidates"],
        *retrieval_result["fused_candidates"],
        *selected_chunks,
    ]
    doc_id_set = {
        item["metadata"].get("doc_id")
        for item in reported_items
        if item["metadata"].get("doc_id") is not None
    }
    doc_name_map: dict[int, str] = {}
    if doc_id_set:
        docs_result = await session.exec(
            select(Document).where(Document.id.in_(doc_id_set))
        )
        for doc in docs_result.all():
            doc_name_map[doc.id] = doc.name
    # Один запрос на все четыре списка сразу: кандидаты в них по большей части
    # одни и те же, и отдельная выборка на список била бы по базе впустую.
    chunk_texts = await load_chunk_texts(session, reported_items)

    def to_retrieval_chunk_item(item: dict[str, Any]) -> RetrievalChunkItem:
        metadata = item["metadata"]
        doc_id = metadata.get("doc_id")
        return RetrievalChunkItem(
            rank=item.get("rank"),
            retrieval_method=item.get("retrieval_method"),
            doc_id=doc_id,
            doc_name=metadata.get("doc_name") or doc_name_map.get(doc_id),
            page=metadata.get("page"),
            chunk_id=item.get("chunk_id"),
            quote=build_quote(item.get("chunk_id"), chunk_texts),
            distance=item.get("distance"),
            lexical_score=item.get("lexical_score"),
            rrf_score=item.get("rrf_score"),
            rerank_score=item.get("rerank_score"),
        )

    chunks = [to_retrieval_chunk_item(item) for item in selected_chunks]

    return RetrievalResponse(
        question=retrieval_request.question,
        search_query=search_query,
        retrieval_top_k=retrieval_top_k,
        top_k=final_top_k,
        vector_candidates=[
            to_retrieval_chunk_item(item)
            for item in retrieval_result["vector_candidates"]
        ],
        lexical_candidates=[
            to_retrieval_chunk_item(item)
            for item in retrieval_result["lexical_candidates"]
        ],
        fused_candidates=[
            to_retrieval_chunk_item(item)
            for item in retrieval_result["fused_candidates"]
        ],
        chunks=chunks,
    )


async def chat_request(
    chat_request: ChatRequest,
    current_user: User,
    session: AsyncSession,
) -> ChatResponse:
    started = perf_counter()
    author_id = require_log_author(current_user)
    rag_service = RAGService()
    normalized_question = rag_service.normalize_query(chat_request.question)
    language = rag_service.detect_language(normalized_question)
    runtime_settings = RuntimeSettingsService.get_settings()
    retrieval_top_k, top_k = resolve_retrieval_limits(runtime_settings)
    enable_condense_query = bool(runtime_settings.get("enable_condense_query", True))
    model = runtime_settings.get("chat_model") or runtime_settings.get(
        "model", DEFAULT_CHAT_MODEL
    )

    notebook, allowed_doc_ids = await resolve_notebook_scope(
        notebook_id=chat_request.notebook_id,
        session=session,
        current_user=current_user,
    )

    profile = resolve_profile(notebook=notebook, requested=chat_request.domain_profile)
    no_data_answer = profile.no_data_answer(language)

    if is_greeting(chat_request.question):
        greeting_answer = profile.greeting(language)
        empty_sources: list[SourceItem] = []
        log_entry = Log(
            question=chat_request.question,
            answer=greeting_answer,
            sources=json.dumps(
                [item.model_dump() for item in empty_sources], ensure_ascii=False
            ),
            time_ms=int((perf_counter() - started) * 1000),
            user_id=author_id,
            notebook_id=notebook.id if notebook else None,
            domain_profile=profile.name,
        )
        session.add(log_entry)
        await session.commit()
        await session.refresh(log_entry)
        return ChatResponse(
            answer=greeting_answer, sources=empty_sources, log_id=log_entry.id
        )

    if rag_service.is_prompt_injection_attempt(normalized_question):
        safe_answer = profile.prompt_injection_message(language)
        empty_sources: list[SourceItem] = []
        log_entry = Log(
            question=chat_request.question,
            answer=safe_answer,
            sources=json.dumps(
                [item.model_dump() for item in empty_sources], ensure_ascii=False
            ),
            time_ms=int((perf_counter() - started) * 1000),
            user_id=author_id,
            notebook_id=notebook.id if notebook else None,
            domain_profile=profile.name,
        )
        session.add(log_entry)
        await session.commit()
        await session.refresh(log_entry)
        return ChatResponse(
            answer=safe_answer, sources=empty_sources, log_id=log_entry.id
        )

    history_result = await session.exec(
        select(Log)
        .where(Log.user_id == current_user.id)
        .where(Log.notebook_id == (notebook.id if notebook else None))
        .order_by(Log.created_at.desc())
        .limit(5)
    )
    history_logs = sorted(history_result.all(), key=lambda x: x.created_at)
    chat_history = []
    for log in history_logs:
        chat_history.append({"role": "user", "content": log.question})
        chat_history.append({"role": "assistant", "content": log.answer})

    article_ref = rag_service._detect_article_reference(normalized_question)
    if article_ref or not enable_condense_query:
        search_query = normalized_question
        logger.debug("Skipping condensation for search query")
    else:
        search_query = await rag_service.condense_query(
            normalized_question, chat_history, model=model
        )
        logger.debug(f"Condensed Search Query: {search_query}")

    selected_chunks: list[dict[str, Any]] = []
    selected_chunks = await run_retrieval(
        rag_service=rag_service,
        session=session,
        profile=profile,
        language=language,
        search_query=search_query,
        original_query=normalized_question,
        allowed_doc_ids=allowed_doc_ids,
        retrieval_top_k=retrieval_top_k,
        final_top_k=top_k,
        notebook_id=notebook.id if notebook else None,
    )

    if not selected_chunks:
        answer_text = no_data_answer
        empty_sources: list[SourceItem] = []
        log_entry = Log(
            question=chat_request.question,
            answer=answer_text,
            sources=json.dumps(
                [item.model_dump() for item in empty_sources], ensure_ascii=False
            ),
            time_ms=int((perf_counter() - started) * 1000),
            user_id=author_id,
            notebook_id=notebook.id if notebook else None,
            domain_profile=profile.name,
        )
        session.add(log_entry)
        await session.commit()
        await session.refresh(log_entry)
        return ChatResponse(
            answer=answer_text, sources=empty_sources, log_id=log_entry.id
        )

    doc_id_set = {
        item["metadata"].get("doc_id")
        for item in selected_chunks
        if item["metadata"].get("doc_id") is not None
    }
    doc_name_map: dict[int, str] = {}
    if doc_id_set:
        docs_result = await session.exec(
            select(Document).where(Document.id.in_(doc_id_set))
        )
        for doc in docs_result.all():
            doc_name_map[doc.id] = doc.name
    chunk_texts = await load_chunk_texts(session, selected_chunks)

    sources: list[SourceItem] = []
    expanded_context = await expand_with_neighbors(selected_chunks, session)
    # Build text→metadata map from selected chunks so neighbors also get doc_name
    text_to_meta: dict[str, dict] = {}
    for item in selected_chunks:
        _meta = item["metadata"]
        _doc_id = _meta.get("doc_id")
        text_to_meta[item["text"]] = {
            "doc_name": _meta.get("doc_name") or doc_name_map.get(_doc_id),
            "page": _meta.get("page"),
        }

    filtered_context: list[str] = list(expanded_context)
    context_metadata: list[dict[str, Any]] = [
        text_to_meta.get(t, {}) for t in filtered_context
    ]
    for item in selected_chunks:
        chunk_text = item["text"]
        meta = item["metadata"]
        doc_id = meta.get("doc_id")
        doc_name = meta.get("doc_name") or doc_name_map.get(doc_id)
        page = meta.get("page")
        chunk_id = item["chunk_id"]
        # В контекст для модели идёт текст кандидата (обогащённый в индексе), в
        # цитату — chunk.text: первое помогает генерации, второе видит человек.
        if chunk_text not in filtered_context:
            filtered_context.append(chunk_text)
            context_metadata.append({"doc_name": doc_name, "page": page})
        sources.append(
            SourceItem(
                source_type="source",
                doc_id=doc_id,
                doc_name=doc_name,
                page=page,
                chunk_id=chunk_id,
                quote=build_quote(chunk_id, chunk_texts),
            )
        )

    answer = await rag_service.generate_answer(
        query=normalized_question,
        context=filtered_context,
        chat_history=chat_history,
        language=language,
        model=model,
        assistant_name=profile.assistant_name,
        answer_rules=profile.answer_rules(language),
        no_data_answer=no_data_answer,
        context_metadata=context_metadata,
    )
    if is_no_data_answer(answer):
        sources = []
    log_entry = Log(
        question=chat_request.question,
        answer=answer,
        sources=json.dumps([item.model_dump() for item in sources], ensure_ascii=False),
        time_ms=int((perf_counter() - started) * 1000),
        user_id=author_id,
        notebook_id=notebook.id if notebook else None,
        domain_profile=profile.name,
    )
    session.add(log_entry)
    await session.commit()
    await session.refresh(log_entry)
    return ChatResponse(answer=answer, sources=sources, log_id=log_entry.id)


async def chat_request_stream(
    chat_request: ChatRequest,
    current_user: User,
) -> AsyncIterator[str]:
    started = perf_counter()
    rag_service = RAGService()
    normalized_question = rag_service.normalize_query(chat_request.question)
    language = rag_service.detect_language(normalized_question)
    runtime_settings = RuntimeSettingsService.get_settings()
    retrieval_top_k, top_k = resolve_retrieval_limits(runtime_settings)
    enable_condense_query = bool(runtime_settings.get("enable_condense_query", True))
    model = runtime_settings.get("chat_model") or runtime_settings.get(
        "model", DEFAULT_CHAT_MODEL
    )
    # То же правило, что в require_log_author («новая запись журнала всегда с
    # автором»), но своей веткой: генератор уже отдан StreamingResponse, и
    # исключение отсюда клиент увидел бы оборванным потоком, а не отказом.
    user_id = current_user.id
    if user_id is None:
        yield stream_event(
            "error",
            {"detail": "Could not validate credentials", "status": 403},
        )
        return

    preparation: StreamChatPreparation
    try:
        async with session_context() as session:
            # HTTPException отсюда перехватывается общим except ниже и уходит
            # клиенту как SSE-событие error со status из исключения.
            notebook, allowed_doc_ids = await resolve_notebook_scope(
                notebook_id=chat_request.notebook_id,
                session=session,
                current_user=current_user,
            )

            notebook_id = notebook.id if notebook else None
            profile = resolve_profile(
                notebook=notebook, requested=chat_request.domain_profile
            )
            no_data_answer = profile.no_data_answer(language)
            base_preparation = {
                "normalized_question": normalized_question,
                "language": language,
                "model": model,
                "assistant_name": profile.assistant_name,
                "answer_rules": profile.answer_rules(language),
                "no_data_answer": no_data_answer,
                "notebook_id": notebook_id,
                "domain_profile": profile.name,
            }

            if is_greeting(chat_request.question):
                preparation = StreamChatPreparation(
                    **base_preparation,
                    chat_history=[],
                    context=[],
                    context_metadata=[],
                    sources=[],
                    immediate_answer=profile.greeting(language),
                )
            elif rag_service.is_prompt_injection_attempt(normalized_question):
                preparation = StreamChatPreparation(
                    **base_preparation,
                    chat_history=[],
                    context=[],
                    context_metadata=[],
                    sources=[],
                    immediate_answer=profile.prompt_injection_message(language),
                )
            else:
                yield stream_event("status", {"stage": "retrieval"})

                history_result = await session.exec(
                    select(Log)
                    .where(Log.user_id == user_id)
                    .where(Log.notebook_id == notebook_id)
                    .order_by(Log.created_at.desc())
                    .limit(5)
                )
                history_logs = sorted(history_result.all(), key=lambda x: x.created_at)
                chat_history = []
                for log in history_logs:
                    chat_history.append({"role": "user", "content": log.question})
                    chat_history.append({"role": "assistant", "content": log.answer})

                article_ref = rag_service._detect_article_reference(
                    normalized_question
                )
                if article_ref or not enable_condense_query:
                    search_query = normalized_question
                else:
                    search_query = await rag_service.condense_query(
                        normalized_question, chat_history, model=model
                    )

                selected_chunks = await run_retrieval(
                    rag_service=rag_service,
                    session=session,
                    profile=profile,
                    language=language,
                    search_query=search_query,
                    original_query=normalized_question,
                    allowed_doc_ids=allowed_doc_ids,
                    retrieval_top_k=retrieval_top_k,
                    final_top_k=top_k,
                    notebook_id=notebook_id,
                )

                if not selected_chunks:
                    preparation = StreamChatPreparation(
                        **base_preparation,
                        chat_history=chat_history,
                        context=[],
                        context_metadata=[],
                        sources=[],
                        immediate_answer=no_data_answer,
                    )
                else:
                    doc_id_set = {
                        item["metadata"].get("doc_id")
                        for item in selected_chunks
                        if item["metadata"].get("doc_id") is not None
                    }
                    doc_name_map: dict[int, str] = {}
                    if doc_id_set:
                        docs_result = await session.exec(
                            select(Document).where(Document.id.in_(doc_id_set))
                        )
                        for doc in docs_result.all():
                            doc_name_map[doc.id] = doc.name
                    chunk_texts = await load_chunk_texts(session, selected_chunks)

                    sources: list[SourceItem] = []
                    expanded_context = await expand_with_neighbors(
                        selected_chunks, session
                    )
                    text_to_meta: dict[str, dict] = {}
                    for item in selected_chunks:
                        _meta = item["metadata"]
                        _doc_id = _meta.get("doc_id")
                        text_to_meta[item["text"]] = {
                            "doc_name": _meta.get("doc_name")
                            or doc_name_map.get(_doc_id),
                            "page": _meta.get("page"),
                        }

                    filtered_context: list[str] = list(expanded_context)
                    context_metadata: list[dict[str, Any]] = [
                        text_to_meta.get(t, {}) for t in filtered_context
                    ]
                    for item in selected_chunks:
                        chunk_text = item["text"]
                        meta = item["metadata"]
                        doc_id = meta.get("doc_id")
                        doc_name = meta.get("doc_name") or doc_name_map.get(doc_id)
                        page = meta.get("page")
                        chunk_id = item["chunk_id"]
                        if chunk_text not in filtered_context:
                            filtered_context.append(chunk_text)
                            context_metadata.append(
                                {"doc_name": doc_name, "page": page}
                            )
                        sources.append(
                            SourceItem(
                                source_type="source",
                                doc_id=doc_id,
                                doc_name=doc_name,
                                page=page,
                                chunk_id=chunk_id,
                                quote=build_quote(chunk_id, chunk_texts),
                            )
                        )

                    preparation = StreamChatPreparation(
                        **base_preparation,
                        chat_history=chat_history,
                        context=filtered_context,
                        context_metadata=context_metadata,
                        sources=sources,
                    )
    except Exception as exc:
        logger.exception("Streaming chat preparation failed")
        # У HTTPException берём detail: str(exc) добавил бы к нему код статуса,
        # который и так уходит отдельным полем.
        detail = getattr(exc, "detail", None) or str(exc)
        payload: dict[str, Any] = {"detail": detail or "Chat preparation failed"}
        status_code = getattr(exc, "status_code", None)
        if status_code:
            payload["status"] = status_code
        yield stream_event("error", payload)
        return

    if preparation.immediate_answer is not None:
        answer = preparation.immediate_answer
        sources = preparation.sources
        if is_no_data_answer(answer):
            sources = []
        log_id, warning = await persist_chat_log_short_lived(
            question=chat_request.question,
            answer=answer,
            sources=sources,
            started=started,
            user_id=user_id,
            notebook_id=preparation.notebook_id,
            domain_profile=preparation.domain_profile,
        )
        yield stream_event("token", {"token": answer})
        done_payload = {
            "answer": answer,
            "sources": [item.model_dump() for item in sources],
            "log_id": log_id,
        }
        if warning:
            done_payload["warning"] = warning
        yield stream_event("done", done_payload)
        return

    yield stream_event("status", {"stage": "generation"})

    answer_parts: list[str] = []
    try:
        async for token in rag_service.stream_answer(
            query=preparation.normalized_question,
            context=preparation.context,
            chat_history=preparation.chat_history,
            language=preparation.language,
            model=preparation.model,
            assistant_name=preparation.assistant_name,
            answer_rules=preparation.answer_rules,
            no_data_answer=preparation.no_data_answer,
            context_metadata=preparation.context_metadata,
        ):
            answer_parts.append(token)
            yield stream_event("token", {"token": token})
    except Exception as exc:
        logger.exception("Streaming chat generation failed")
        payload = {"detail": str(exc) or "Chat generation failed"}
        status_code = getattr(exc, "status_code", None)
        if status_code:
            payload["status"] = status_code
        yield stream_event("error", payload)
        return

    raw_answer = "".join(answer_parts)
    answer = rag_service._sanitize_answer_text(raw_answer)
    if not answer:
        answer = rag_service._generation._fallback_from_context(
            preparation.context, preparation.no_data_answer
        )
        yield stream_event("token", {"token": answer})

    sources = preparation.sources
    if is_no_data_answer(answer):
        sources = []

    log_id, warning = await persist_chat_log_short_lived(
        question=chat_request.question,
        answer=answer,
        sources=sources,
        started=started,
        user_id=user_id,
        notebook_id=preparation.notebook_id,
        domain_profile=preparation.domain_profile,
    )
    done_payload = {
        "answer": answer,
        "sources": [item.model_dump() for item in sources],
        "log_id": log_id,
    }
    if warning:
        done_payload["warning"] = warning
    yield stream_event("done", done_payload)
