import asyncio
import json
import logging
import os
import re
from typing import Any, Awaitable, Callable, Optional

from fastapi import UploadFile
from fastapi.concurrency import run_in_threadpool
from sqlalchemy import delete, text
from sqlmodel import func, select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.core.exceptions import ApiError, SourceErrors
from app.models.models import Chunk, Document, Job, Notebook
from app.modules.jobs.service import JOB_INDEX_DOCUMENT, JobsService
from app.services.document_service import UploadValidationError
from app.services.hybrid_chunker import HybridChunker
from app.services.source_service import SourceService
from app.modules.rag.service import RAGService

logger = logging.getLogger(__name__)

# Ограничивает фонового воркера, а не HTTP-обработчики: одновременно
# индексируется один документ на процесс.
_INDEXING_SEMAPHORE = asyncio.Semaphore(1)


class DocumentIndexingError(Exception):
    """Ожидаемый отказ индексации: текст показываем пользователю как есть.

    error_code воркер кладёт в document.error_code рядом с текстом — по нему
    интерфейс подбирает перевод.
    """

    def __init__(
        self, message: str, error_code: str = SourceErrors.INDEXING_FAILED
    ) -> None:
        super().__init__(message)
        self.error_code = error_code


_YEAR_RE = re.compile(r'((?:19|20)\d{2})')

# Путь к файлу на сервере внутри текста ошибки. Библиотеки разбора подставляют
# его охотно — PyMuPDF отвечает «FileDataError: Failed to open file
# 'data/uploads/<uuid>_QA_broken.pdf' as type pdf», — а document.error_text
# уходит клиенту в GET /sources/ и показывается подсказкой при наведении.
# Наружу нужна причина отказа, а не устройство диска: имя документа клиент и
# так знает, полный текст остаётся в логе.
#
# Две ветки: путь от корня или с буквой диска — по ведущему разделителю;
# относительный (`data/uploads/файл.pdf`) — по двум и более разделителям.
# Ведущий просмотр назад обязателен для обеих: без него «и/или» и «20 MB/s»
# тоже считались бы путём, потому что начать матч можно прямо с косой черты.
# Пробелы в элемент пути не пускаем сознательно: с ними выражение слизывало бы
# соседние слова предложения, а цель — убрать каталоги, и они здесь всегда без
# пробелов. Имя файла с пробелом теряет только начало — путь всё равно уходит.
_FS_PATH_RE = re.compile(
    r"(?<![\w.+-])(?:"
    r"(?:[A-Za-z]:[\\/]|[\\/])[\w.+-]+(?:[\\/][\w.+-]+)*"
    r"|(?:[\w.+-]+[\\/]){2,}[\w.+-]*"
    r")"
)
_REDACTED_PATH = "<файл>"


def redact_server_paths(message: str) -> str:
    """Убрать серверные пути из текста, который увидит клиент."""
    return _FS_PATH_RE.sub(_REDACTED_PATH, message or "")


def _build_embedding_text(
    chunk_text: str,
    doc_name: str,
    page: int | None = None,
    section_json: str | None = None,
) -> str:
    """Prepend contextual metadata to chunk text before embedding.

    Result looks like:
      [Обложка.pdf | Введение в финансовую науку | стр. 12] <chunk text>
    """
    parts: list[str] = []
    # Clean doc name (remove extension for readability)
    clean_name = re.sub(r'\.[a-zA-Z0-9]+$', '', doc_name or '').strip()
    if clean_name:
        parts.append(clean_name)

    # Extract most specific section header
    if section_json:
        try:
            sections = json.loads(section_json)
            if isinstance(sections, list) and sections:
                # Use last (most specific) section
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
        prefix = ' | '.join(parts)
        return f'[{prefix}] {chunk_text}'
    return chunk_text



async def _generate_llm_context(
    chunk_text: str,
    doc_name: str,
    doc_language: str,
    model: str,
    doc_intro: str = "",
    section_path: list | None = None,
) -> str:
    """Call LLM to generate a 1-2 sentence contextual description for the chunk."""
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
        f"You are a search-index assistant. Your output will be prepended to a document chunk "
        f"to improve retrieval. Follow the format exactly.\n\n"
        f"Output format (do not add any other text):\n"
        f"DESCRIPTION: <2 sentences — what this chunk is about and where it fits in the document. "
        f"Do NOT copy sentences from the chunk verbatim.>\n"
        f"KEYWORDS: <8-12 comma-separated keywords and synonyms, including: key terms from the chunk, "
        f"alternative phrasings, ALL numbers/dates/amounts mentioned, and relevant words from the filename \"{doc_name}\".>\n\n"
        f"Rules:\n"
        f"- Always include every number, date, percentage, and monetary amount from the chunk in KEYWORDS.\n"
        f"- Include synonyms from both the document language and Russian if they differ.\n"
        f"- DESCRIPTION must explain placement (e.g. 'This chunk is from the section on...'), not just repeat content.\n"
        f"- {lang_instruction}\n\n"
        f"Document: {doc_name}\n"
        f"{doc_intro_block}"
        f"{section_block}"
        f"Chunk:\n{chunk_text[:600]}\n\nOutput:"
    )
    try:
        from app.services.runtime_settings_service import RuntimeSettingsService
        ctx_num_ctx = RuntimeSettingsService.get_settings().get("contextual_embedding_num_ctx", 8192)
        manager = ModelManager()
        result = await manager.chat(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            max_tokens=220,
            num_ctx=ctx_num_ctx,
        )
        return result.strip()
    except Exception as exc:
        logger.warning("Contextual embedding LLM call failed: %s", exc)
        return ""


class DocumentModuleService:
    @staticmethod
    def _build_ingestion_chunker() -> HybridChunker:
        from app.modules.rag.chunker_config import (
            CHUNKER_TARGET_TOKENS,
            CHUNKER_MAX_TOKENS,
            CHUNKER_MIN_TOKENS,
            CHUNKER_OVERLAP_TOKENS,
            CHUNKER_MAX_CHARS,
        )
        return HybridChunker(
            target_tokens=CHUNKER_TARGET_TOKENS,
            max_tokens=CHUNKER_MAX_TOKENS,
            min_tokens=CHUNKER_MIN_TOKENS,
            overlap_tokens=CHUNKER_OVERLAP_TOKENS,
            max_chars=CHUNKER_MAX_CHARS,
        )

    @staticmethod
    def _upload_rejected(exc: ValueError) -> ApiError:
        """Отказ приёма файла → HTTP-ошибка с машинным кодом."""
        error_code = getattr(exc, "error_code", SourceErrors.INVALID_UPLOAD)
        # 413, а не 400: превышение лимита клиент обрабатывает иначе —
        # предлагает разбить файл, а не сменить формат.
        status_code = 413 if error_code == SourceErrors.TOO_LARGE else 400
        return ApiError(status_code, error_code, str(exc))

    @staticmethod
    async def upload_document(
        session: AsyncSession,
        file: UploadFile,
        notebook_id: Optional[int] = None,
        owner_id: Optional[int] = None,
    ) -> Document:
        try:
            file_ext = SourceService.validate_upload_file(file)
        except ValueError as exc:
            raise DocumentModuleService._upload_rejected(exc) from exc

        notebook: Notebook | None = None
        if notebook_id is not None:
            notebook = await session.get(Notebook, notebook_id)
            if not notebook:
                raise ApiError(
                    404, SourceErrors.NOTEBOOK_NOT_FOUND, "Notebook not found"
                )

        try:
            file_path = await SourceService.save_upload_file(file)
        except ValueError as exc:
            raise DocumentModuleService._upload_rejected(exc) from exc
        try:
            actual_size = os.path.getsize(file_path)
        except OSError:
            actual_size = 0

        doc = Document(
            # Не file.filename как есть: имя уходит в список источников и в
            # Content-Disposition превью, а клиент волен прислать туда
            # `../../../../etc/passwd.txt` или строку в килобайт.
            name=SourceService.sanitize_display_name(file.filename),
            path=file_path,
            size=actual_size,
            status="pending",
            notebook_id=notebook.id if notebook else None,
            owner_id=owner_id,
        )
        session.add(doc)
        # flush, а не commit: документ и задача на его индексацию должны стать
        # видимы одной транзакцией. Иначе падение между двумя коммитами
        # оставит документ в 'pending' без задачи, то есть навсегда.
        await session.flush()
        await JobsService.enqueue(
            session,
            job_type=JOB_INDEX_DOCUMENT,
            payload={"file_ext": file_ext},
            source_id=doc.id,
            notebook_id=doc.notebook_id,
            created_by=owner_id,
        )
        await session.refresh(doc)
        return doc

    @staticmethod
    async def index_document_job(
        session: AsyncSession,
        doc_id: int,
        progress_cb: Callable[[int], Awaitable[None]] | None = None,
    ) -> dict[str, Any]:
        """Полный цикл индексации одного документа. Вызывается воркером.

        Извлечение текста и чанкинг живут здесь, а не в HTTP-обработчике:
        на сканированном PDF с OCR это минуты, и клиент столько не ждёт.
        Статусы документа выставляет вызывающий воркер — он один отвечает
        за терминальное состояние, в том числе при отмене.
        """

        async def _progress(value: int) -> None:
            if progress_cb is not None:
                await progress_cb(value)

        doc = await session.get(Document, doc_id)
        if doc is None:
            raise DocumentIndexingError(
                "Документ удалён до начала индексации",
                SourceErrors.DELETED_BEFORE_INDEXING,
            )
        if not doc.path or not os.path.exists(doc.path):
            raise DocumentIndexingError(
                "Файл документа не найден на диске", SourceErrors.FILE_MISSING
            )

        # Повторная попытка не должна удвоить чанки: прерванный проход мог
        # успеть записать часть в БД и в ChromaDB.
        await DocumentModuleService._purge_chunks(session, doc_id)

        ext = SourceService.get_extension(doc.name or doc.path)
        chunker = DocumentModuleService._build_ingestion_chunker()
        try:
            chunk_results = await run_in_threadpool(
                SourceService.extract_and_chunk, doc.path, ext, chunker
            )
        except UploadValidationError as exc:
            # Разбор файла тоже отказывает осмысленно (например, не-UTF-8 TXT);
            # без этой ветки код терялся, а пользователь видел «ValueError: ...».
            raise DocumentIndexingError(str(exc), exc.error_code) from exc
        if not chunk_results:
            raise DocumentIndexingError(
                "Не удалось извлечь текст из файла",
                SourceErrors.TEXT_EXTRACTION_FAILED,
            )
        await _progress(30)

        sample_text = " ".join(cr.text for cr in chunk_results[:5])
        doc.language = SourceService.detect_language(sample_text)
        session.add(doc)
        await session.commit()

        await DocumentModuleService._index_document(
            session=session, doc=doc, chunk_results=chunk_results,
        )
        await _progress(100)
        return {"chunks": len(chunk_results)}

    @staticmethod
    async def _purge_chunks(session: AsyncSession, doc_id: int) -> None:
        result = await session.exec(select(Chunk.id).where(Chunk.doc_id == doc_id))
        chunk_ids = [str(chunk_id) for chunk_id in result.all() if chunk_id is not None]
        if not chunk_ids:
            return
        try:
            await run_in_threadpool(RAGService().delete_documents, chunk_ids)
        except Exception as exc:
            # Недоступный ChromaDB здесь не повод отменять попытку: те же id
            # всё равно будут перезаписаны на шаге add_documents.
            logger.warning("Could not drop old vectors for doc %s: %s", doc_id, exc)
        await session.execute(delete(Chunk).where(Chunk.doc_id == doc_id))
        await session.commit()

    @staticmethod
    async def reconcile_stuck_documents(session: AsyncSession) -> dict[str, int]:
        """Согласовать статусы документов с очередью.

        Вызывается на старте и периодически. Документ в 'indexing' без живой
        задачи остался от убитого процесса — ровно тот случай, ради которого
        в DEPLOY.md жил ручной UPDATE document SET status='error'.
        """
        params = {"job_type": JOB_INDEX_DOCUMENT}
        # Задача вернулась в очередь — документ снова ждёт своей очереди.
        back_to_queue = await session.execute(
            text(
                """
                UPDATE document d
                SET status = 'pending'
                WHERE d.status = 'indexing'
                  AND EXISTS (
                      SELECT 1 FROM job j
                      WHERE j.source_id = d.id
                        AND j.job_type = :job_type
                        AND j.status = 'queued'
                  )
                RETURNING d.id
                """
            ),
            params,
        )
        requeued = len(back_to_queue.all())

        # Живой задачи нет вовсе — дальше ждать нечего.
        orphaned = await session.execute(
            text(
                """
                UPDATE document d
                SET status = 'error',
                    error_text = COALESCE(d.error_text, :message),
                    error_code = COALESCE(d.error_code, :error_code)
                WHERE d.status IN ('pending', 'indexing')
                  AND NOT EXISTS (
                      SELECT 1 FROM job j
                      WHERE j.source_id = d.id
                        AND j.job_type = :job_type
                        AND j.status IN ('queued', 'running')
                  )
                RETURNING d.id
                """
            ),
            {
                **params,
                "message": "Индексация не завершилась: задача потеряна",
                "error_code": SourceErrors.INDEXING_INTERRUPTED,
            },
        )
        failed = len(orphaned.all())
        await session.commit()
        return {"requeued": requeued, "failed": failed}

    @staticmethod
    async def _index_document(session, doc, chunk_results):
        rag_service = RAGService()
        from app.shared.settings.runtime_settings import RuntimeSettingsService as _RSS
        _rt = _RSS.get_settings()
        _ctx_enabled = _rt.get("contextual_embedding_enabled", False)
        _ctx_model = _rt.get("contextual_embedding_model", "")
        detected_language = doc.language or "ru"
        doc_intro = " ".join(cr.text for cr in chunk_results[:3])[:1500] if _ctx_enabled else ""

        # Сохраняем все чанки в БД сразу, получаем их id
        chunks: list[Chunk] = []
        for cr in chunk_results:
            chunk = Chunk(
                text=cr.text,
                page=cr.page_start,
                chunk_index=cr.chunk_index,
                section=cr.section_path_json if cr.section_path else None,
                doc_id=doc.id,
            )
            session.add(chunk)
            chunks.append(chunk)
        await session.flush()

        # Извлекаем все нужные данные из ORM-объектов ДО commit,
        # чтобы не держать соединение открытым во время LLM-фазы
        doc_id = doc.id
        doc_name = doc.name
        doc_notebook_id = doc.notebook_id
        chunk_data = [
            {
                "id": str(chunk.id),
                "text": chunk.text,
                "page": chunk.page,
                "section": chunk.section,
                "section_path": cr.section_path or [],
                "chunk_index": cr.chunk_index,
            }
            for chunk, cr in zip(chunks, chunk_results)
        ]

        # Коммит освобождает DB-соединение обратно в пул
        await session.commit()

        # Параллельная генерация LLM-контекста (до 5 одновременно)
        # DB-соединение НЕ удерживается во время LLM-вызовов
        _llm_sem = asyncio.Semaphore(5)

        async def _process_chunk(cd: dict) -> str:
            base_text = _build_embedding_text(cd["text"], doc_name, cd["page"], cd["section"])
            if _ctx_enabled and _ctx_model:
                async with _llm_sem:
                    llm_ctx = await _generate_llm_context(
                        cd["text"], doc_name, detected_language, _ctx_model,
                        doc_intro=doc_intro, section_path=cd["section_path"],
                    )
                return f"{llm_ctx} {base_text}" if llm_ctx else base_text
            return base_text

        embedding_texts = await asyncio.gather(
            *[_process_chunk(cd) for cd in chunk_data]
        )

        ids: list[str] = [cd["id"] for cd in chunk_data]
        metadatas: list[dict[str, Any]] = []
        for cd, emb_text in zip(chunk_data, embedding_texts):
            metadata = {
                "doc_id": doc_id,
                "doc_name": doc_name,
                "page": cd["page"],
                "chunk_index": cd["chunk_index"],
            }
            if cd["section"]:
                metadata["section"] = cd["section"]
            if doc_notebook_id is not None:
                metadata["notebook_id"] = doc_notebook_id
            metadatas.append(metadata)

        docs_text = list(embedding_texts)
        # Синхронный клиент Ollama внутри: без threadpool весь воркер стоит
        # на всё время эмбеддинга (батчи по 20, таймаут 120 с на батч).
        # Исключение отсюда не гасим: терминальный статус документа ставит
        # воркер, у которого для этого есть отдельная живая сессия.
        await run_in_threadpool(rag_service.add_documents, docs_text, metadatas, ids)
        return doc

    @staticmethod
    async def read_documents(
        session: AsyncSession,
        skip: int = 0,
        limit: int = 100,
        notebook_id: int | None = None,
        owner_id: int | None = None,
    ) -> tuple[list[Document], int]:
        """Страница документов и общее число записей под теми же фильтрами.

        total считается отдельным запросом, а не по длине страницы: иначе
        клиент не может отличить «это всё» от «есть ещё» и не строит пагинацию.
        """

        def _filtered(statement):
            if notebook_id is not None:
                statement = statement.where(Document.notebook_id == notebook_id)
            # owner_id=None — вызов от админа: отдаём всё, включая документы без владельца
            if owner_id is not None:
                statement = statement.where(Document.owner_id == owner_id)
            return statement

        # ORDER BY обязателен: с OFFSET/LIMIT без него порядок строк не
        # определён, и соседние страницы одни записи дублируют, другие теряют.
        # id вторым ключом — тай-брейк для документов, загруженных в одну
        # миллисекунду (массовая загрузка), иначе они прыгают между страницами.
        page = _filtered(select(Document)).order_by(
            Document.created_at.desc(), Document.id.desc()
        )
        result = await session.exec(page.offset(skip).limit(limit))
        documents = list(result.all())

        total_result = await session.exec(
            _filtered(select(func.count()).select_from(Document))
        )
        return documents, int(total_result.first() or 0)

    @staticmethod
    async def attach_documents_to_notebook(
        session: AsyncSession,
        notebook_id: int,
        source_ids: list[int],
        owner_id: int | None = None,
    ) -> list[Document]:
        notebook = await session.get(Notebook, notebook_id)
        if not notebook:
            raise ApiError(404, SourceErrors.NOTEBOOK_NOT_FOUND, "Notebook not found")

        normalized_source_ids = list(dict.fromkeys(source_ids))
        if not normalized_source_ids:
            raise ApiError(
                400, SourceErrors.NO_IDS_PROVIDED, "No source ids provided"
            )

        statement = select(Document).where(Document.id.in_(normalized_source_ids))
        # Чужой документ не находится запросом и попадает в missing_ids —
        # снаружи он неотличим от несуществующего.
        if owner_id is not None:
            statement = statement.where(Document.owner_id == owner_id)
        result = await session.exec(statement)
        documents = result.all()

        if len(documents) != len(normalized_source_ids):
            found_ids = {
                document.id for document in documents if document.id is not None
            }
            missing_ids = [
                source_id
                for source_id in normalized_source_ids
                if source_id not in found_ids
            ]
            raise ApiError(
                404,
                SourceErrors.NOT_FOUND,
                f"Documents not found: {', '.join(str(source_id) for source_id in missing_ids)}",
            )

        for document in documents:
            document.notebook_id = notebook_id
            session.add(document)

        await session.commit()

        for document in documents:
            await session.refresh(document)

        return documents

    @staticmethod
    async def get_document_chunks(session: AsyncSession, document_id: int):
        doc = await session.get(Document, document_id)
        if not doc:
            raise ApiError(404, SourceErrors.NOT_FOUND, "Document not found")
        result = await session.exec(
            select(Chunk).where(Chunk.doc_id == document_id).order_by(Chunk.chunk_index)
        )
        return result.all()

    @staticmethod
    async def delete_document(session: AsyncSession, document_id: int) -> Document:
        doc = await session.get(Document, document_id)
        if not doc:
            raise ApiError(404, SourceErrors.NOT_FOUND, "Document not found")
        chunks_result = await session.exec(
            select(Chunk).where(Chunk.doc_id == document_id)
        )
        chunks = chunks_result.all()
        chunk_ids = [str(chunk.id) for chunk in chunks if chunk.id is not None]
        rag_service = RAGService()
        try:
            # Чистит все коллекции andozai_docs_*, а не только активную:
            # документ мог быть проиндексирован ДО смены embedding-модели, и
            # его векторы лежат в коллекции той, прежней модели
            # (ChromaGateway.delete_documents). Отказ по-прежнему отменяет
            # удаление целиком — документ должен исчезнуть вместе с векторами.
            rag_service.delete_documents(chunk_ids)
        except Exception as exc:
            raise ApiError(
                503,
                SourceErrors.VECTOR_STORE_UNAVAILABLE,
                "Failed to delete embeddings from ChromaDB.",
            ) from exc
        if doc.path and os.path.exists(doc.path):
            try:
                os.remove(doc.path)
            except OSError as exc:
                raise ApiError(
                    500,
                    SourceErrors.DELETE_FILE_FAILED,
                    "Failed to delete document file from disk.",
                ) from exc
        for chunk in chunks:
            await session.delete(chunk)
        # Job.source_id ссылается на document.id без ON DELETE CASCADE, поэтому
        # задачи удаляем сами и отдельным flush: иначе порядок DELETE внутри
        # одного flush не гарантирован и упирается во внешний ключ.
        # Работающая задача обнаружит пропажу документа и закроется сама.
        jobs_result = await session.exec(
            select(Job).where(Job.source_id == document_id)
        )
        for job in jobs_result.all():
            await session.delete(job)
        await session.flush()
        await session.delete(doc)
        await session.commit()
        return doc

    @staticmethod
    async def reindex_all_documents(session: AsyncSession) -> dict[str, Any]:
        import logging

        logger = logging.getLogger(__name__)
        result = await session.exec(select(Document))
        documents = result.all()
        if not documents:
            return {
                "status": "ok",
                "message": "No documents to reindex",
                "total_chunks": 0,
            }
        # Тот же семафор, что и у воркера: массовая переиндексация сносит все
        # чанки разом, и одновременная индексация свежей загрузки потеряла бы
        # свои векторы.
        async with _INDEXING_SEMAPHORE:
            return await DocumentModuleService._reindex_all(session, documents, logger)

    @staticmethod
    async def _reindex_all(session: AsyncSession, documents, logger) -> dict[str, Any]:
        rag_service = RAGService()
        chunker = DocumentModuleService._build_ingestion_chunker()
        from app.shared.settings.runtime_settings import RuntimeSettingsService as _RSS
        _rt = _RSS.get_settings()
        _ctx_enabled = _rt.get("contextual_embedding_enabled", False)
        _ctx_model = _rt.get("contextual_embedding_model", "")
        all_chunks_result = await session.exec(select(Chunk))
        all_chunks = all_chunks_result.all()
        old_chunk_ids = [str(c.id) for c in all_chunks if c.id is not None]
        if old_chunk_ids:
            try:
                # Переиндексация чаще всего и запускается после смены
                # embedding-модели, то есть старые векторы лежат в коллекции
                # ПРЕЖНЕЙ модели, а не в активной. delete_documents проходит по
                # всем коллекциям andozai_docs_* — иначе прежняя оставалась бы
                # полной копией всей базы и оживала при откате настройки.
                rag_service.delete_documents(old_chunk_ids)
            except Exception as exc:
                # Не отменяет переиндексацию: id всё равно будут перезаписаны в
                # активной коллекции. Причину пишем — по «Could not delete old
                # chunks» без неё нельзя было отличить лежащую ChromaDB от
                # недоступной коллекции прежней модели.
                logger.warning(
                    "Could not delete old chunks from ChromaDB (%d ids): %s",
                    len(old_chunk_ids),
                    exc,
                )
        for chunk in all_chunks:
            await session.delete(chunk)
        await session.flush()
        total_chunks = 0
        errors: list[str] = []
        for doc in documents:
            try:
                if not doc.path or not os.path.exists(doc.path):
                    errors.append(f"File missing for document {doc.id}: {doc.name}")
                    doc.status = "error"
                    doc.error_text = "Файл документа не найден на диске"
                    doc.error_code = SourceErrors.FILE_MISSING
                    session.add(doc)
                    continue
                ext = SourceService.get_extension(doc.name or doc.path)
                chunk_results = await run_in_threadpool(
                    SourceService.extract_and_chunk, doc.path, ext, chunker
                )
                if not chunk_results:
                    errors.append(
                        f"No text extracted from document {doc.id}: {doc.name}"
                    )
                    doc.status = "error"
                    doc.error_text = "Не удалось извлечь текст из файла"
                    doc.error_code = SourceErrors.TEXT_EXTRACTION_FAILED
                    session.add(doc)
                    continue
                doc_intro = " ".join(cr.text for cr in chunk_results[:3])[:1500] if _ctx_enabled else ""
                docs_text: list[str] = []
                ids: list[str] = []
                metadatas: list[dict[str, Any]] = []
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
                    base_text = _build_embedding_text(
                        chunk.text, doc.name, chunk.page, chunk.section,
                    )
                    if _ctx_enabled and _ctx_model:
                        llm_ctx = await _generate_llm_context(
                            chunk.text, doc.name, doc.language or "ru", _ctx_model,
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
                    if doc.notebook_id is not None:
                        metadata["notebook_id"] = doc.notebook_id
                    metadatas.append(metadata)
                await run_in_threadpool(
                    rag_service.add_documents, docs_text, metadatas, ids
                )
                doc.status = "indexed"
                doc.error_text = None
                doc.error_code = None
                session.add(doc)
                total_chunks += len(chunk_results)
                logger.info(
                    f"Reindexed doc {doc.id} ({doc.name}): {len(chunk_results)} chunks"
                )
            except Exception as exc:
                # Оба текста уходят по API (error_text — всем, errors — админу
                # в ответе POST /reindex), поэтому путь к файлу из них убираем;
                # в логе ниже он остаётся полностью.
                errors.append(
                    redact_server_paths(
                        f"Error reindexing doc {doc.id} ({doc.name}): {exc}"
                    )
                )
                logger.warning(
                    "Reindex of doc %s (%s) failed: %s", doc.id, doc.name, exc,
                    exc_info=True,
                )
                doc.status = "error"
                doc.error_text = redact_server_paths(
                    f"{type(exc).__name__}: {exc}"
                )[:500]
                doc.error_code = getattr(
                    exc, "error_code", SourceErrors.INDEXING_FAILED
                )
                session.add(doc)
        await session.commit()
        return {
            "status": "ok" if not errors else "partial",
            "total_documents": len(documents),
            "total_chunks": total_chunks,
            "errors": errors,
        }
