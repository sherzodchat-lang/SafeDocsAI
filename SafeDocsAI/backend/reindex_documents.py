"""
Reindex all documents using the new HybridChunker pipeline.

Usage: python reindex_documents.py
"""

import asyncio
import json
import os
import re
import sys
from sqlmodel import select
from sqlalchemy.orm import sessionmaker
from sqlmodel.ext.asyncio.session import AsyncSession

# Add backend to path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from app.core.database import engine
from app.models.models import Document, Chunk
from app.services.document_service import DocumentService
from app.services.hybrid_chunker import HybridChunker
from app.services.rag_service import RAGService
from app.shared.settings.runtime_settings import RuntimeSettingsService


def _build_embedding_text(chunk_text, doc_name, page=None, section_json=None):
    parts = []
    clean_name = re.sub(r'\.[a-zA-Z0-9]+$', '', doc_name or '').strip()
    if clean_name:
        parts.append(clean_name)
    if section_json:
        try:
            sections = json.loads(section_json)
            if isinstance(sections, list) and sections:
                header = str(sections[-1]).strip()
                if len(header) > 80:
                    header = header[:80].rstrip() + '…'
                if header:
                    parts.append(header)
        except (json.JSONDecodeError, TypeError):
            pass
    if page is not None:
        parts.append(f'стр. {page}')
    if parts:
        return f'[{" | ".join(parts)}] {chunk_text}'
    return chunk_text


async def _generate_llm_context(chunk_text, doc_name, doc_language, model,
                               doc_intro="", section_path=None):
    """Call LLM to generate contextual description for a chunk."""
    from app.modules.rag.model_manager import ModelManager
    lang_instruction = {
        "ru": "Отвечай на русском языке.",
        "tj": "Ба забони тоҷикӣ ҷавоб деҳ.",
        "en": "Answer in English.",
    }.get(doc_language, "Answer in the same language as the text.")
    doc_intro_block = f"Document beginning:\n{doc_intro[:1500]}\n\n" if doc_intro else ""
    section_block = ""
    if section_path:
        section_str = " > ".join(section_path)
        section_block = f"Section: {section_str}\n\n"
    prompt = (
        f"You are a document analyst. Write 1-2 concise sentences describing "
        f"what this chunk is about and where it fits in the document. "
        f"Use the document beginning and section path as context. "
        f"Be specific. {lang_instruction}\n\n"
        f"Document: {doc_name}\n"
        f"{doc_intro_block}"
        f"{section_block}"
        f"Chunk:\n{chunk_text[:600]}\n\nContext:"
    )
    try:
        manager = ModelManager()
        result = await manager.chat(
            messages=[{"role": "user", "content": prompt}],
            model=model,
        )
        return result.strip()
    except Exception as exc:
        print(f"  WARNING: LLM context call failed: {exc}")
        return ""


async def reindex_all_documents():
    print("Starting re-indexing with HybridChunker pipeline...")

    async_session = sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )

    rag_service = RAGService()
    from app.modules.rag.chunker_config import (
        CHUNKER_TARGET_TOKENS,
        CHUNKER_MAX_TOKENS,
        CHUNKER_MIN_TOKENS,
        CHUNKER_OVERLAP_TOKENS,
        CHUNKER_MAX_CHARS,
    )
    chunker = HybridChunker(
        target_tokens=CHUNKER_TARGET_TOKENS,
        max_tokens=CHUNKER_MAX_TOKENS,
        min_tokens=CHUNKER_MIN_TOKENS,
        overlap_tokens=CHUNKER_OVERLAP_TOKENS,
        max_chars=CHUNKER_MAX_CHARS,
    )

    rt = RuntimeSettingsService.get_settings()
    ctx_enabled = rt.get("contextual_embedding_enabled", False)
    ctx_model = rt.get("contextual_embedding_model", "")
    if ctx_enabled:
        print(f"Contextual embedding enabled (model: {ctx_model})")
    else:
        print("Contextual embedding disabled")

    async with async_session() as session:
        result = await session.exec(select(Document))
        documents = result.all()

        print(f"Found {len(documents)} documents to re-index.")

        for doc in documents:
            print(f"\nProcessing Document ID: {doc.id} ({doc.name})...")

            file_path = doc.path
            if not file_path or not os.path.exists(file_path):
                print(f"  WARNING: File not found at {file_path}. Skipping.")
                continue

            # 1. Delete existing chunks
            chunks_result = await session.exec(
                select(Chunk).where(Chunk.doc_id == doc.id)
            )
            existing_chunks = chunks_result.all()
            existing_chunk_ids = [str(c.id) for c in existing_chunks if c.id is not None]

            print(f"  Deleting {len(existing_chunks)} existing chunks...")

            try:
                rag_service.delete_documents(existing_chunk_ids)
            except Exception as e:
                print(f"  Error deleting from Chroma: {e}")

            for chunk in existing_chunks:
                await session.delete(chunk)
            await session.commit()

            # 2. Extract & chunk with new pipeline
            print("  Extracting blocks & chunking...")
            file_ext = DocumentService.get_extension(doc.name)

            try:
                chunk_results = DocumentService.extract_and_chunk(
                    file_path, file_ext, chunker
                )
            except Exception as e:
                print(f"  Error extracting text: {e}")
                continue

            if not chunk_results:
                print("  WARNING: No chunks extracted.")
                continue

            print(f"  Generated {len(chunk_results)} new chunks.")

            # 3. Save to DB & index in Chroma
            docs_text = []
            ids = []
            metadatas = []

            doc_intro = " ".join(cr.text for cr in chunk_results[:3])[:1500] if ctx_enabled else ""
            for cr in chunk_results:
                chunk = Chunk(
                    text=cr.text,
                    page=cr.page_start,
                    chunk_index=cr.chunk_index,
                    section=cr.section_path_json if cr.section_path else None,
                    doc_id=doc.id,
                )
                session.add(chunk)
                await session.flush()
                await session.refresh(chunk)

                base_text = _build_embedding_text(chunk.text, doc.name, chunk.page, chunk.section)
                if ctx_enabled and ctx_model:
                    llm_ctx = await _generate_llm_context(
                        chunk.text, doc.name, doc.language or "ru", ctx_model,
                        doc_intro=doc_intro, section_path=cr.section_path or [],
                    )
                    embedding_text = f"{llm_ctx} {base_text}" if llm_ctx else base_text
                else:
                    embedding_text = base_text
                docs_text.append(embedding_text)
                ids.append(str(chunk.id))
                metadata = {
                    "doc_id": doc.id,
                    "doc_name": doc.name,
                    "page": chunk.page,
                    "chunk_index": cr.chunk_index,
                }
                if chunk.section:
                    metadata["section"] = chunk.section
                metadatas.append(metadata)

            await session.commit()

            print(f"  Indexing {len(ids)} chunks in ChromaDB...")
            try:
                rag_service.add_documents(docs_text, metadatas, ids)
            except Exception as e:
                print(f"  Error indexing in Chroma: {e}")

            print("  Done.")

    print("\nRe-indexing complete.")


if __name__ == "__main__":
    try:
        asyncio.run(reindex_all_documents())
    except Exception as e:
        print(f"Critical error: {e}")
