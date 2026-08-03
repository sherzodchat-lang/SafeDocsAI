import json
from time import perf_counter

from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.models.models import Chunk, Document, Log, User
from app.modules.ask.schemas import AskRequest, AskResponse, CitationItem
from app.modules.chat.service import (
    build_quote,
    expand_with_neighbors,
    is_greeting,
    is_no_data_answer,
    load_chunk_texts,
    require_log_author,
    resolve_notebook_scope,
    resolve_retrieval_limits,
    run_retrieval,
)
from app.modules.rag.constants import DEFAULT_CHAT_MODEL
from app.services.profile_resolver import resolve_profile
from app.services.rag_service import RAGService
from app.services.runtime_settings_service import RuntimeSettingsService


async def _resolve_neighbor_metadata(
    selected_chunks: list[dict],
    doc_name_map: dict[int, str],
    session: AsyncSession,
) -> dict[str, dict]:
    """Метаданные соседних чанков, разложенные по их тексту.

    expand_with_neighbors отдаёт соседей одним плоским списком строк, без doc_id
    и страницы, поэтому связать сосед → документ можно только повторным запросом
    по тем же парам (doc_id, chunk_index), которые она использует.
    """
    from sqlalchemy import and_, or_

    conditions = []
    for item in selected_chunks:
        meta = item.get("metadata") or {}
        doc_id = meta.get("doc_id")
        chunk_index = meta.get("chunk_index")
        if doc_id is None or chunk_index is None:
            continue
        for offset in (-1, 1):
            conditions.append(
                and_(Chunk.doc_id == doc_id, Chunk.chunk_index == chunk_index + offset)
            )
    if not conditions:
        return {}

    result = await session.exec(select(Chunk).where(or_(*conditions)))
    metadata_by_text: dict[str, dict] = {}
    for chunk in result.all():
        if not chunk.text or chunk.text in metadata_by_text:
            continue
        metadata_by_text[chunk.text] = {
            "doc_name": doc_name_map.get(chunk.doc_id),
            "page": chunk.page,
        }
    return metadata_by_text


async def handle_ask_request(
    ask_request: AskRequest,
    current_user: User,
    session: AsyncSession,
) -> AskResponse:
    started = perf_counter()
    author_id = require_log_author(current_user)
    rag_service = RAGService()
    normalized_question = rag_service.normalize_query(ask_request.question)
    language = rag_service.detect_language(normalized_question)
    runtime_settings = RuntimeSettingsService.get_settings()
    retrieval_top_k, top_k = resolve_retrieval_limits(
        runtime_settings,
        requested_top_k=ask_request.top_k,
    )
    model = runtime_settings.get("chat_model") or runtime_settings.get(
        "model", DEFAULT_CHAT_MODEL
    )

    notebook, allowed_doc_ids = await resolve_notebook_scope(
        notebook_id=ask_request.notebook_id,
        session=session,
        current_user=current_user,
    )

    profile = resolve_profile(notebook=notebook, requested=ask_request.domain_profile)
    no_data_answer = profile.no_data_answer(language)

    if is_greeting(ask_request.question):
        answer = profile.greeting(language)
        log_entry = Log(
            question=ask_request.question,
            answer=answer,
            sources="[]",
            time_ms=int((perf_counter() - started) * 1000),
            user_id=author_id,
            notebook_id=notebook.id if notebook else None,
            domain_profile=profile.name,
        )
        session.add(log_entry)
        await session.commit()
        await session.refresh(log_entry)
        return AskResponse(answer=answer, citations=[], log_id=log_entry.id)

    if rag_service.is_prompt_injection_attempt(normalized_question):
        answer = profile.prompt_injection_message(language)
        log_entry = Log(
            question=ask_request.question,
            answer=answer,
            sources="[]",
            time_ms=int((perf_counter() - started) * 1000),
            user_id=author_id,
            notebook_id=notebook.id if notebook else None,
            domain_profile=profile.name,
        )
        session.add(log_entry)
        await session.commit()
        await session.refresh(log_entry)
        return AskResponse(answer=answer, citations=[], log_id=log_entry.id)

    # Конденсации здесь нет намеренно, в отличие от чата и /chat/retrieve.
    # /ask — разовый запрос: история диалога не читается, в generate_answer
    # ниже уходит chat_history=[]. Конденсация переписывает follow-up с
    # местоимениями («а какая у него ставка?») по истории, и на пустой истории
    # condense_query возвращает запрос как есть — то есть добавить её сюда
    # значило бы тратить вызов модели на каждый запрос ради того же самого
    # текста. По той же причине не читается и enable_condense_query:
    # выключателю нечего выключать.
    search_query = normalized_question

    selected_chunks: list[dict] = []
    selected_chunks = await run_retrieval(
        rag_service=rag_service,
        session=session,
        profile=profile,
        language=language,
        search_query=search_query,
        allowed_doc_ids=allowed_doc_ids,
        retrieval_top_k=retrieval_top_k,
        final_top_k=top_k,
        notebook_id=notebook.id if notebook else None,
    )

    if not selected_chunks:
        answer = no_data_answer
        log_entry = Log(
            question=ask_request.question,
            answer=answer,
            sources="[]",
            time_ms=int((perf_counter() - started) * 1000),
            user_id=author_id,
            notebook_id=notebook.id if notebook else None,
            domain_profile=profile.name,
        )
        session.add(log_entry)
        await session.commit()
        await session.refresh(log_entry)
        return AskResponse(answer=answer, citations=[], log_id=log_entry.id)

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

    expanded_context = await expand_with_neighbors(selected_chunks, session)
    neighbor_metadata = await _resolve_neighbor_metadata(
        selected_chunks, doc_name_map, session
    )
    selected_metadata: list[dict] = [
        {
            "doc_name": item["metadata"].get("doc_name")
            or doc_name_map.get(item["metadata"].get("doc_id")),
            "page": item["metadata"].get("page"),
        }
        for item in selected_chunks
    ]
    filtered_context = list(expanded_context)
    # Контракт expand_with_neighbors: сначала тексты selected_chunks в исходном
    # порядке, затем тексты соседей без метаданных. Голову выравниваем по индексу,
    # хвост разрешаем по тексту — иначе имя файла в промпте остаётся пустым.
    context_metadata: list[dict] = []
    for index, text in enumerate(filtered_context):
        if index < len(selected_metadata) and text == selected_chunks[index]["text"]:
            context_metadata.append(selected_metadata[index])
        else:
            context_metadata.append(neighbor_metadata.get(text, {}))

    # Цитата — из chunk.text, а не из текста кандидата: в индексе он обогащён
    # служебным префиксом (см. load_chunk_texts).
    chunk_texts = await load_chunk_texts(session, selected_chunks)
    citations: list[CitationItem] = []
    for item in selected_chunks:
        meta = item["metadata"]
        source_id = meta.get("doc_id")
        source_name = meta.get("doc_name") or doc_name_map.get(source_id)
        page = meta.get("page")
        chunk_id = item["chunk_id"]
        citations.append(
            CitationItem(
                source_id=source_id,
                source_name=source_name,
                page=page,
                chunk_id=chunk_id,
                quote=build_quote(chunk_id, chunk_texts),
            )
        )

    answer = await rag_service.generate_answer(
        query=normalized_question,
        context=filtered_context,
        chat_history=[],
        language=language,
        model=model,
        assistant_name=profile.assistant_name,
        answer_rules=profile.answer_rules(language),
        no_data_answer=no_data_answer,
        context_metadata=context_metadata,
    )
    if is_no_data_answer(answer):
        citations = []

    log_entry = Log(
        question=ask_request.question,
        answer=answer,
        sources=json.dumps(
            [item.model_dump() for item in citations], ensure_ascii=False
        ),
        time_ms=int((perf_counter() - started) * 1000),
        user_id=author_id,
        notebook_id=notebook.id if notebook else None,
        domain_profile=profile.name,
    )
    session.add(log_entry)
    await session.commit()
    await session.refresh(log_entry)
    return AskResponse(answer=answer, citations=citations, log_id=log_entry.id)
