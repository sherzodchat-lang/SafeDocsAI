import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useSearchParams } from 'react-router-dom';
import {
    AlertTriangle,
    CheckCircle2,
    ChevronDown,
    ChevronLeft,
    ChevronRight,
    Clock,
    CloudUpload,
    Eye,
    FileText,
    Loader2,
    RefreshCw,
    Tag,
    Trash2,
    X,
} from 'lucide-react';
import { sourcesService } from '../services/sourcesService';
import { Button } from '../components/ui/Button';
import NotebookScopeBadge from '../components/notebook/NotebookScopeBadge';
import { cn } from '../lib/utils';
import { useActiveNotebookScope } from '../hooks/useActiveNotebookScope';
import { useModalDialog } from '../hooks/useModalDialog';
import { useSources, useSourcesActions } from '../contexts/SourcesContext';
import { useLocale } from '../i18n';
import { resolveApiErrorMessage } from '../lib/apiError';
import { formatDocumentLanguage, formatLocaleDate, parseTimestamp } from '../lib/locale';
import { formatSize, resolveSourceErrorMessage, resolveStatus } from '../lib/sources';
import { TOPIC_LABEL_PARAM, TOPIC_PARAM, matchesTopicFilter, readTopicFilter, resolveTopicLabel } from '../lib/topics';

const ALLOWED_EXTENSIONS = ['pdf', 'docx', 'txt'];
const ALLOWED_MIME_TYPES = new Set([
    'application/pdf',
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    'text/plain',
    'application/octet-stream',
    'binary/octet-stream',
    'application/zip',
]);

const normalizeMimeType = (mimeType) => String(mimeType || '').split(';', 1)[0].trim().toLowerCase();

const STATUS_ORDER = ['all', 'ready', 'pending', 'indexing', 'error'];

const PAGE_SIZE_OPTIONS = [10, 25, 50];
const DEFAULT_PAGE_SIZE = 25;

// Раскрытие плотного слоя запоминаем: тому, кому фильтры и страницы нужны
// каждый день, иначе пришлось бы открывать их при каждом заходе. Значение по
// умолчанию — свёрнуто, поэтому отсутствие ключа означает «закрыто».
const DETAILS_STORAGE_KEY = 'knowledgeai.sources.detailsOpen';

const readDetailsPreference = () => {
    if (typeof window === 'undefined') return false;
    return localStorage.getItem(DETAILS_STORAGE_KEY) === 'true';
};

const AdminDocumentsPage = ({ notebookId }) => {
    const { locale, t } = useLocale();
    const [isUploading, setIsUploading] = useState(false);
    const [uploadError, setUploadError] = useState('');
    const [statusFilter, setStatusFilter] = useState('all');
    const [isDragActive, setIsDragActive] = useState(false);
    const [sortField, setSortField] = useState('created_at');
    const [sortOrder, setSortOrder] = useState('desc');
    const [page, setPage] = useState(1);
    const [pageSize, setPageSize] = useState(DEFAULT_PAGE_SIZE);
    const [detailsOpen, setDetailsOpen] = useState(readDetailsPreference);
    const [deleteTarget, setDeleteTarget] = useState(null);
    const [isDeleting, setIsDeleting] = useState(false);
    const [deleteError, setDeleteError] = useState('');

    // Поиск читаем из URL param ?q= (устанавливается хедером)
    const [searchParams, setSearchParams] = useSearchParams();
    const searchTerm = searchParams.get('q') || '';
    // Фильтр по теме приходит с экрана «Темы» ссылкой. В адресе он и живёт:
    // так его видно, им можно поделиться, и «Назад» возвращает к полному списку.
    const topicFilter = readTopicFilter(searchParams);
    const [chunksModal, setChunksModal] = useState({
        isOpen: false,
        docId: null,
        docName: '',
        chunks: [],
        isLoading: false,
        error: null,
    });

    const fileInputRef = useRef(null);
    const chunksDialogRef = useRef(null);
    const chunksCloseRef = useRef(null);
    const deleteDialogRef = useRef(null);
    const deleteCancelRef = useRef(null);

    // Внутри блокнота страница работает панелью: область задана маршрутом, её имя
    // уже стоит в шапке блокнота, поэтому ни бейджа, ни запроса за именем там нет.
    const isNotebookPanel = notebookId != null;
    // Самостоятельная страница берёт область из того же активного блокнота, что и
    // глобальный чат. Раньше список молча сужался по нему; теперь область видна в
    // бейдже и снимается одной кнопкой, а сам фильтр сохранён.
    const {
        notebookId: effectiveNotebookId,
        notebookName,
        canResetScope,
        resetScope,
    } = useActiveNotebookScope(notebookId, { withNotebookName: !isNotebookPanel });

    // Внутри блокнота плотный слой (фильтры, сортировка, страницы, колонки
    // «Язык»/«Размер»/«Дата», просмотр фрагментов) уходит под «Подробности» —
    // одинаково для всех ролей. Отдельная страница «Все источники» остаётся
    // рабочим местом администратора и открыта целиком.
    const showDetails = !isNotebookPanel || detailsOpen;

    const toggleDetails = useCallback(() => {
        setDetailsOpen((prev) => {
            const next = !prev;
            localStorage.setItem(DETAILS_STORAGE_KEY, String(next));
            return next;
        });
    }, []);

    const statusMeta = useMemo(() => ({
        ready: {
            label: t('documents.status.ready'),
            badgeClass: 'bg-emerald-100 text-emerald-700',
        },
        pending: {
            label: t('documents.status.pending'),
            badgeClass: 'bg-slate-100 text-slate-600',
        },
        indexing: {
            label: t('documents.status.indexing'),
            badgeClass: 'bg-amber-100 text-amber-700',
        },
        error: {
            label: t('documents.status.error'),
            badgeClass: 'bg-red-100 text-red-700',
        },
    }), [t]);

    // Единый слой данных: список, постраничное чтение, опрос индексации и обработка
    // ошибок живут в SourcesProvider — тот же кэш читают панель блокнота и его шапка.
    const {
        items: documents,
        isTruncated: isListTruncated,
        isLoading,
        error: loadError,
        reload: retryFetchDocuments,
    } = useSources(effectiveNotebookId);

    const { uploadSource, deleteSource, updateSource, invalidate } = useSourcesActions();

    const validateFile = useCallback((file) => {
        const ext = file.name.split('.').pop()?.toLowerCase() || '';
        const hasAllowedExt = ALLOWED_EXTENSIONS.includes(ext);
        if (!hasAllowedExt) return false;
        if (ext === 'txt') return true;

        const hasAllowedMime = ALLOWED_MIME_TYPES.has(normalizeMimeType(file.type));
        return hasAllowedExt && hasAllowedMime;
    }, []);

    const resetFileInput = useCallback(() => {
        // Без сброса value повторный выбор того же файла не вызывает onChange.
        if (fileInputRef.current) {
            fileInputRef.current.value = '';
        }
    }, []);

    const handleDragEnter = useCallback((event) => {
        event.preventDefault();
        event.stopPropagation();
        setIsDragActive(true);
    }, []);

    const handleDragLeave = useCallback((event) => {
        event.preventDefault();
        event.stopPropagation();
        setIsDragActive(false);
    }, []);

    const handleDragOver = useCallback((event) => {
        event.preventDefault();
        event.stopPropagation();
    }, []);

    const uploadFiles = useCallback(async (files) => {
        if (!files || files.length === 0) return;

        setIsUploading(true);

        const uploadPromises = files.map(async (file) => {
            try {
                await uploadSource(file, effectiveNotebookId);
                return { success: true, name: file.name };
            } catch (error) {
                return {
                    success: false,
                    name: file.name,
                    error: resolveApiErrorMessage(error, t, 'documents.uploadError'),
                };
            }
        });

        const results = await Promise.all(uploadPromises);
        const failed = results.filter((r) => !r.success);

        if (failed.length > 0) {
            // Дописываем, а не заменяем: в смешанном наборе рядом уже стоит
            // предупреждение об отброшенных файлах, и оно тоже нужно.
            const failedMessage = t('notebook.uploadFailed', {
                files: failed.map((f) => `${f.name}${f.error ? ` (${f.error})` : ''}`).join(', '),
            });
            setUploadError((prev) => (prev ? `${prev} ${failedMessage}` : failedMessage));
        }

        resetFileInput();
        // Общая инвалидация: список обновляется и здесь, и в панели блокнота, и в его шапке.
        await invalidate();
        setIsUploading(false);
    }, [effectiveNotebookId, invalidate, resetFileInput, t, uploadSource]);

    // Выбор файла и есть команда «загрузить»: отдельная кнопка «Загрузить файл»
    // была вторым шагом там, где в «Обзоре» его нет, и одно и то же действие
    // стоило разного числа кликов на соседних экранах.
    const applyFileSelection = useCallback((files) => {
        const valid = files.filter(validateFile);
        const invalid = files.filter((file) => !validateFile(file));

        // Предупреждение о непринятых файлах не затираем: при смешанном наборе (PDF + ZIP)
        // это единственный сигнал о том, что часть файлов отброшена.
        if (invalid.length > 0) {
            setUploadError(t('documents.unsupportedFiles', { files: invalid.map((file) => file.name).join(', ') }));
        } else if (valid.length > 0) {
            setUploadError('');
        } else if (files.length > 0) {
            setUploadError(t('documents.invalidFormats'));
        }

        if (valid.length > 0) {
            uploadFiles(valid);
        } else {
            // Ни один файл не принят: без сброса value повторный выбор того же
            // файла не вызовет onChange, и экран будет выглядеть неотзывчивым.
            resetFileInput();
        }
    }, [resetFileInput, t, uploadFiles, validateFile]);

    const handleDrop = useCallback((event) => {
        event.preventDefault();
        event.stopPropagation();
        setIsDragActive(false);

        applyFileSelection(Array.from(event.dataTransfer.files));
    }, [applyFileSelection]);

    const fetchChunks = useCallback(async (docId, docName) => {
        setChunksModal({
            isOpen: true,
            docId,
            docName,
            chunks: [],
            isLoading: true,
            error: null,
        });

        try {
            const response = await sourcesService.getChunks(docId);
            const fetchedChunks = response.data || [];

            if (fetchedChunks.length === 0) {
                updateSource(docId, { status: 'indexing' });
            }

            setChunksModal((prev) => ({
                ...prev,
                chunks: fetchedChunks,
                isLoading: false,
            }));
        } catch (error) {
            console.error('Failed to fetch chunks:', error);
            setChunksModal((prev) => ({
                ...prev,
                isLoading: false,
                error: resolveApiErrorMessage(error, t, 'documents.chunksLoadFailed'),
            }));
        }
    }, [t, updateSource]);

    const closeChunksModal = useCallback(() => {
        setChunksModal({
            isOpen: false,
            docId: null,
            docName: '',
            chunks: [],
            isLoading: false,
            error: null,
        });
    }, []);

    const closeDeleteDialog = useCallback(() => {
        setDeleteTarget(null);
        setDeleteError('');
        setIsDeleting(false);
    }, []);

    const confirmDelete = useCallback(async () => {
        if (!deleteTarget) return;

        setIsDeleting(true);
        setDeleteError('');

        try {
            await deleteSource(deleteTarget.id);
            setDeleteTarget(null);
        } catch (error) {
            console.error('Delete failed:', error);
            setDeleteError(resolveApiErrorMessage(error, t, 'documents.deleteFailed'));
        } finally {
            setIsDeleting(false);
        }
    }, [deleteSource, deleteTarget, t]);

    useModalDialog(chunksModal.isOpen, closeChunksModal, chunksDialogRef, chunksCloseRef);
    useModalDialog(Boolean(deleteTarget), closeDeleteDialog, deleteDialogRef, deleteCancelRef);

    const stats = useMemo(() => {
        const counts = {
            all: documents.length,
            ready: 0,
            pending: 0,
            indexing: 0,
            error: 0,
        };

        documents.forEach((doc) => {
            const status = resolveStatus(doc.status);
            counts[status] += 1;
        });

        return counts;
    }, [documents]);

    const topicClusterIndex = topicFilter.clusterIndex;

    const filteredDocuments = useMemo(() => {
        const query = searchTerm.trim().toLowerCase();

        return documents.filter((doc) => {
            const status = resolveStatus(doc.status);
            const statusMatch = statusFilter === 'all' || statusFilter === status;
            const queryMatch = !query || String(doc.name || '').toLowerCase().includes(query);
            return statusMatch && queryMatch && matchesTopicFilter(doc, topicClusterIndex);
        });
    }, [documents, searchTerm, statusFilter, topicClusterIndex]);

    // Подпись темы для бейджа: сначала та, что пришла ссылкой, затем — взятая у
    // самих документов. Обе могут отсутствовать (адрес набрали руками, под тему
    // не попал ни один источник), и тогда бейдж всё равно обязан сказать, что
    // фильтр стоит, — иначе список опять неполон без объяснения.
    const topicFilterLabel = useMemo(() => {
        if (topicClusterIndex == null) return '';
        if (topicFilter.label) return topicFilter.label;

        const labelled = documents.find((doc) => matchesTopicFilter(doc, topicClusterIndex) && resolveTopicLabel(doc));
        return labelled ? resolveTopicLabel(labelled) : t('documents.topicFilterUnknown');
    }, [documents, t, topicClusterIndex, topicFilter.label]);

    const resetTopicFilter = useCallback(() => {
        const params = new URLSearchParams(searchParams);
        params.delete(TOPIC_PARAM);
        params.delete(TOPIC_LABEL_PARAM);
        setSearchParams(params, { replace: true });
    }, [searchParams, setSearchParams]);

    const sortedDocuments = useMemo(() => {
        const sorted = [...filteredDocuments];
        sorted.sort((a, b) => {
            let valA = a[sortField];
            let valB = b[sortField];

            if (sortField === 'name') {
                valA = String(valA || '').toLowerCase();
                valB = String(valB || '').toLowerCase();
                if (valA < valB) return sortOrder === 'asc' ? -1 : 1;
                if (valA > valB) return sortOrder === 'asc' ? 1 : -1;
                return 0;
            } else if (sortField === 'created_at') {
                // Наивные метки без зоны разбираем как UTC, иначе порядок «поплывёт» на границе суток.
                const timeA = parseTimestamp(valA)?.getTime() || 0;
                const timeB = parseTimestamp(valB)?.getTime() || 0;
                return sortOrder === 'asc' ? timeA - timeB : timeB - timeA;
            } else if (sortField === 'size') {
                const sizeA = Number(valA || 0);
                const sizeB = Number(valB || 0);
                return sortOrder === 'asc' ? sizeA - sizeB : sizeB - sizeA;
            }
            return 0;
        });
        return sorted;
    }, [filteredDocuments, sortField, sortOrder]);

    // Смена фильтра, поиска или размера страницы возвращает пользователя на первую страницу.
    useEffect(() => {
        setPage(1);
    }, [searchTerm, statusFilter, sortField, sortOrder, pageSize, topicClusterIndex]);

    const pageCount = Math.max(1, Math.ceil(sortedDocuments.length / pageSize));
    const currentPage = Math.min(page, pageCount);
    const rangeStart = sortedDocuments.length === 0 ? 0 : (currentPage - 1) * pageSize + 1;
    const rangeEnd = Math.min(currentPage * pageSize, sortedDocuments.length);
    const visibleDocuments = useMemo(
        () => sortedDocuments.slice((currentPage - 1) * pageSize, currentPage * pageSize),
        [sortedDocuments, currentPage, pageSize],
    );

    const listStatusMessage = isLoading
        ? t('documents.loading')
        : loadError || t('documents.listStatus', { count: sortedDocuments.length });

    // Свёрнутое состояние оставляет имя, статус и действия; язык, дата и размер
    // уходят под «Подробности» вместе с остальным плотным слоем.
    const columnCount = showDetails ? 6 : 3;

    const openDeleteDialog = (doc) => {
        setDeleteError('');
        setDeleteTarget({ id: doc.id, name: doc.name });
    };

    return (
        <div className="space-y-6 px-4">
            {/* Бейдж стоит над загрузкой, а не только над таблицей: активный блокнот
                сужает список и одновременно задаёт, куда попадёт новый источник.
                Рядом — тема, если пришли по ней: обе области сужают один список,
                и показывать их порознь значило бы прятать одну из них. */}
            {!isNotebookPanel || topicClusterIndex != null ? (
                <div className="flex flex-wrap items-center gap-2">
                    {!isNotebookPanel ? (
                        <NotebookScopeBadge
                            notebookId={effectiveNotebookId}
                            notebookName={notebookName}
                            canReset={canResetScope}
                            onReset={resetScope}
                            resetTitle={t('documents.scopeResetTitle')}
                        />
                    ) : null}

                    {topicClusterIndex != null ? (
                        <span className="inline-flex items-center gap-1.5 rounded-full bg-[#1f3a60]/10 px-3 py-1 text-xs font-semibold text-[#1f3a60]">
                            <Tag className="h-3.5 w-3.5" />
                            {t('documents.topic', { value: topicFilterLabel })}
                            <button
                                type="button"
                                onClick={resetTopicFilter}
                                title={t('documents.topicFilterReset')}
                                aria-label={t('documents.topicFilterReset')}
                                className="rounded-full p-0.5 transition hover:bg-[#1f3a60]/20 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#1f3a60]"
                            >
                                <X className="h-3 w-3" />
                            </button>
                        </span>
                    ) : null}
                </div>
            ) : null}

            <section className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">

                <div
                    onDragEnter={handleDragEnter}
                    onDragLeave={handleDragLeave}
                    onDragOver={handleDragOver}
                    onDrop={handleDrop}
                    className={cn(
                        'rounded-2xl border-2 border-dashed p-8 text-center transition-all duration-200',
                        isDragActive
                            ? 'border-[#1f3a60] bg-[#1f3a60]/5'
                            : 'border-slate-300 bg-slate-50',
                    )}
                >
                    <div className={cn(
                        'mx-auto mb-4 flex h-14 w-14 items-center justify-center rounded-full transition-all duration-200',
                        isDragActive
                            ? 'bg-[#1f3a60] text-white'
                            : 'bg-[#1f3a60]/10 text-[#1f3a60]',
                    )}>
                        <CloudUpload className={cn('h-7 w-7', isDragActive && 'animate-bounce')} />
                    </div>
                    <h3 className="text-2xl font-bold text-slate-800">
                        {isDragActive ? t('documents.dropActive') : t('documents.dropIdle')}
                    </h3>
                    <p className="mt-1 text-sm text-slate-500">{t('documents.supportedFormats')}</p>

                    {/* Одна кнопка на всё действие: выбор файла сразу запускает
                        загрузку, поэтому она же и показывает её ход. */}
                    <div className="mt-5 flex flex-wrap items-center justify-center gap-3" aria-live="polite">
                        {/* Кнопка, а не label: скрытый input недоступен с клавиатуры, label не фокусируется. */}
                        <button
                            type="button"
                            onClick={() => fileInputRef.current?.click()}
                            disabled={isUploading}
                            className="inline-flex items-center gap-2 rounded-lg bg-[#c5a059] px-5 py-2 text-sm font-semibold text-[#1f3a60] transition hover:bg-[#d7b878] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#1f3a60] focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-60"
                        >
                            {isUploading ? (
                                <>
                                    <Loader2 className="h-4 w-4 animate-spin" />
                                    {t('documents.uploadLoading')}
                                </>
                            ) : (
                                <>
                                    <CloudUpload className="h-4 w-4" />
                                    {t('documents.chooseFile')}
                                </>
                            )}
                        </button>
                        <input
                            ref={fileInputRef}
                            type="file"
                            multiple
                            className="hidden"
                            accept=".pdf,.docx,.txt,text/plain,application/pdf,application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                            onChange={(event) => applyFileSelection(Array.from(event.target.files || []))}
                        />
                    </div>

                    {uploadError && (
                        <p role="alert" className="mt-3 text-sm font-semibold text-red-600">{uploadError}</p>
                    )}
                </div>
            </section>

            <section className="overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-sm">
                <div className="border-b border-slate-200 px-5 py-4">
                    {isNotebookPanel ? (
                        <div className="flex flex-wrap items-center justify-between gap-3">
                            <button
                                type="button"
                                onClick={toggleDetails}
                                aria-expanded={detailsOpen}
                                className="inline-flex items-center gap-2 rounded-lg border border-slate-200 bg-white px-3 py-1.5 text-xs font-semibold text-slate-600 transition hover:border-slate-300 hover:text-[#1f3a60] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#1f3a60] focus-visible:ring-offset-1"
                            >
                                <ChevronDown className={cn('h-4 w-4 transition-transform duration-150', detailsOpen && 'rotate-180')} />
                                {t('documents.details')}
                            </button>

                            {/* Свёрнутый блок не должен читаться как «сломалось»: рядом
                                сказано словами, что внутри. Применённый фильтр и неполная
                                страница вынесены наружу — иначе список молча неполон, а
                                причины на экране нет. */}
                            {!detailsOpen ? (
                                <div className="flex flex-wrap items-center gap-2 text-xs font-semibold text-slate-500">
                                    {statusFilter !== 'all' ? (
                                        <span className="inline-flex items-center gap-1.5 rounded-full bg-[#1f3a60]/10 px-2.5 py-1 text-[#1f3a60]">
                                            {t('documents.activeFilter', { value: statusMeta[statusFilter].label })}
                                            <button
                                                type="button"
                                                onClick={() => setStatusFilter('all')}
                                                title={t('documents.activeFilterReset')}
                                                aria-label={t('documents.activeFilterReset')}
                                                className="rounded-full p-0.5 transition hover:bg-[#1f3a60]/20 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#1f3a60]"
                                            >
                                                <X className="h-3 w-3" />
                                            </button>
                                        </span>
                                    ) : null}
                                    {pageCount > 1 ? (
                                        <span>{t('documents.pagination.range', { from: rangeStart, to: rangeEnd, total: sortedDocuments.length })}</span>
                                    ) : null}
                                    <span className="font-normal text-slate-400">{t('documents.detailsHint')}</span>
                                </div>
                            ) : null}
                        </div>
                    ) : null}

                    {showDetails ? (
                    <div className={cn('flex flex-wrap items-center justify-between gap-3', isNotebookPanel && 'mt-4')}>
                        <div className="flex flex-wrap gap-2" role="group" aria-label={t('documents.statusFilterLabel')}>
                            {STATUS_ORDER.map((status) => (
                                <button
                                    key={status}
                                    type="button"
                                    onClick={() => setStatusFilter(status)}
                                    aria-pressed={statusFilter === status}
                                    className={cn(
                                        'rounded-lg px-3 py-1.5 text-xs font-semibold transition focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#1f3a60] focus-visible:ring-offset-1',
                                        statusFilter === status
                                            ? 'bg-[#1f3a60] text-white'
                                            : 'bg-slate-100 text-slate-600 hover:bg-slate-200',
                                    )}
                                >
                                    {status === 'all' ? t('documents.status.all') : statusMeta[status].label}
                                    <span className="ml-1">{stats[status]}</span>
                                </button>
                            ))}
                        </div>

                        <div className="flex items-center gap-2">
                            <label className="text-sm font-semibold text-slate-500" htmlFor="documents-sort">{t('documents.sort')}</label>
                            <select
                                id="documents-sort"
                                value={`${sortField}-${sortOrder}`}
                                onChange={(e) => {
                                    const [field, order] = e.target.value.split('-');
                                    setSortField(field);
                                    setSortOrder(order);
                                }}
                                className="rounded-lg border border-slate-200 bg-slate-50 px-3 py-1.5 text-xs font-semibold text-slate-700 outline-none focus:border-[#1f3a60] focus:ring-1 focus:ring-[#1f3a60]"
                            >
                                <option value="created_at-desc">{t('documents.sortOptions.dateDesc')}</option>
                                <option value="created_at-asc">{t('documents.sortOptions.dateAsc')}</option>
                                <option value="name-asc">{t('documents.sortOptions.nameAsc')}</option>
                                <option value="name-desc">{t('documents.sortOptions.nameDesc')}</option>
                                <option value="size-desc">{t('documents.sortOptions.sizeDesc')}</option>
                                <option value="size-asc">{t('documents.sortOptions.sizeAsc')}</option>
                            </select>
                        </div>
                    </div>
                    ) : null}
                </div>

                {/* Единственный источник голосового статуса списка для скринридера. */}
                <p className="sr-only" role="status" aria-live="polite">{listStatusMessage}</p>

                {/* Сбой фонового обновления не должен прятать уже загруженный список. */}
                {loadError && documents.length > 0 && (
                    <div role="alert" className="flex flex-wrap items-center justify-between gap-3 border-b border-red-200 bg-red-50 px-5 py-3">
                        <span className="flex items-center gap-2 text-sm font-semibold text-red-700">
                            <AlertTriangle className="h-4 w-4" />
                            {loadError}
                        </span>
                        <Button type="button" variant="outline" size="sm" onClick={retryFetchDocuments}>
                            <RefreshCw className="h-4 w-4" />
                            {t('documents.retry')}
                        </Button>
                    </div>
                )}

                {isListTruncated && !loadError && (
                    <p className="border-b border-amber-200 bg-amber-50 px-5 py-2 text-xs font-semibold text-amber-800">
                        {t('documents.pagination.truncated', { count: documents.length })}
                    </p>
                )}

                <div className="overflow-x-auto">
                    {/* Свёрнутый вид — три колонки, и таблице больше не нужна ширина
                        под шесть: горизонтальная прокрутка появлялась даже там, где
                        показывать нечего. */}
                    <table className={cn('w-full text-left', showDetails ? 'min-w-[820px]' : 'min-w-[420px]')}>
                        <thead className="bg-slate-50 text-xs uppercase tracking-[0.08em] text-slate-500">
                            <tr>
                                <th className="px-5 py-3 font-semibold">{t('documents.table.fileName')}</th>
                                {showDetails ? (
                                    <>
                                        <th className="px-5 py-3 font-semibold">{t('documents.table.language')}</th>
                                        <th className="px-5 py-3 font-semibold">{t('documents.table.uploadedAt')}</th>
                                        <th className="px-5 py-3 font-semibold">{t('documents.table.size')}</th>
                                    </>
                                ) : null}
                                <th className="px-5 py-3 font-semibold">{t('documents.table.status')}</th>
                                <th className="px-5 py-3 text-right font-semibold">{t('documents.table.actions')}</th>
                            </tr>
                        </thead>

                        <tbody>
                            {isLoading ? (
                                <tr>
                                    <td colSpan={columnCount} className="px-5 py-12 text-center text-slate-500">
                                        <div className="inline-flex items-center gap-2">
                                            <Loader2 className="h-5 w-5 animate-spin" />
                                            {t('documents.loading')}
                                        </div>
                                    </td>
                                </tr>
                            ) : loadError && documents.length === 0 ? (
                                <tr>
                                    <td colSpan={columnCount} className="px-5 py-12">
                                        {/* Сбой загрузки: явно отличаем от пустой базы и даём повторить. */}
                                        <div role="alert" className="mx-auto flex max-w-md flex-col items-center gap-3 rounded-xl bg-red-50 p-6 text-center">
                                            <AlertTriangle className="h-6 w-6 text-red-600" />
                                            <p className="text-sm font-semibold text-red-700">{loadError}</p>
                                            <Button type="button" variant="outline" onClick={retryFetchDocuments}>
                                                <RefreshCw className="h-4 w-4" />
                                                {t('documents.retry')}
                                            </Button>
                                        </div>
                                    </td>
                                </tr>
                            ) : visibleDocuments.length === 0 ? (
                                <tr>
                                    <td colSpan={columnCount} className="px-5 py-12 text-center text-slate-500">
                                        {t('documents.empty')}
                                    </td>
                                </tr>
                            ) : (
                                visibleDocuments.map((doc) => {
                                    const statusKey = resolveStatus(doc.status);
                                    const documentStatusMeta = statusMeta[statusKey];
                                    const languageTag = showDetails ? formatDocumentLanguage(doc.language) : null;
                                    // Причина провала индексации: error_code документа переводится
                                    // той же таблицей, что и коды HTTP-ошибок.
                                    const documentErrorMessage = resolveSourceErrorMessage(doc, t);
                                    // Тема — подпись, а не статус: строкой ниже имени и
                                    // серым. Поля может не быть вовсе (модель не обучена,
                                    // источник не размечен) — тогда не показываем ничего:
                                    // прочерк на месте темы сообщал бы о ней больше, чем
                                    // о самом источнике.
                                    const topicLabel = resolveTopicLabel(doc);

                                    return (
                                        <tr key={doc.id} className="border-t border-slate-100 text-sm hover:bg-slate-50/70">
                                            <td className="px-5 py-3">
                                                <div className="flex items-center gap-3">
                                                    <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-red-100 text-red-500">
                                                        <FileText className="h-4 w-4" />
                                                    </div>
                                                    <div className="min-w-0">
                                                        <p className="max-w-[320px] break-words font-semibold text-slate-800">{doc.name}</p>
                                                        {topicLabel ? (
                                                            <p
                                                                className="mt-0.5 flex max-w-[320px] items-center gap-1 text-xs text-slate-400"
                                                                title={t('documents.topic', { value: topicLabel })}
                                                            >
                                                                <Tag className="h-3 w-3 shrink-0" />
                                                                <span className="truncate">{topicLabel}</span>
                                                            </p>
                                                        ) : null}
                                                    </div>
                                                </div>
                                            </td>

                                            {showDetails ? (
                                                <>
                                                    <td className="px-5 py-3">
                                                        <span className={cn('rounded px-2 py-0.5 text-xs font-semibold', languageTag.className)}>
                                                            {languageTag.text}
                                                        </span>
                                                    </td>

                                                    <td className="px-5 py-3 text-slate-500">
                                                        {formatLocaleDate(doc.created_at, locale, {
                                                            month: 'short',
                                                            day: 'numeric',
                                                            year: 'numeric',
                                                        })}
                                                    </td>

                                                    <td className="px-5 py-3 text-slate-500">
                                                        {formatSize(doc.size, t)}
                                                    </td>
                                                </>
                                            ) : null}

                                            <td className="px-5 py-3">
                                                <span className={cn('inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-semibold', documentStatusMeta.badgeClass)}>
                                                    {statusKey === 'indexing' ? (
                                                        <Loader2 className="h-3.5 w-3.5 animate-spin" />
                                                    ) : statusKey === 'pending' ? (
                                                        <Clock className="h-3.5 w-3.5" />
                                                    ) : statusKey === 'error' ? (
                                                        <AlertTriangle className="h-3.5 w-3.5" />
                                                    ) : (
                                                        <CheckCircle2 className="h-3.5 w-3.5" />
                                                    )}
                                                    {documentStatusMeta.label}
                                                </span>
                                                {documentErrorMessage ? (
                                                    /* error_text — техническая строка на одном языке:
                                                       она нужна администратору, но в ячейке таблицы
                                                       ей не место, поэтому уходит в подсказку. */
                                                    <p
                                                        className="mt-1 max-w-[240px] text-xs leading-5 text-red-600"
                                                        title={doc.error_text || undefined}
                                                    >
                                                        {documentErrorMessage}
                                                    </p>
                                                ) : null}
                                            </td>

                                            <td className="px-5 py-3 text-right">
                                                <div className="inline-flex items-center gap-1">
                                                    {/* Разбор источника на фрагменты — работа отладочная:
                                                        она осталась целиком, но под тем же раскрытием. */}
                                                    {showDetails ? (
                                                        <button
                                                            type="button"
                                                            onClick={() => fetchChunks(doc.id, doc.name)}
                                                            className="rounded-md p-1.5 text-slate-500 transition hover:bg-blue-50 hover:text-blue-600 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#1f3a60]"
                                                            title={t('documents.viewChunks')}
                                                            aria-label={t('documents.viewChunksFor', { name: doc.name })}
                                                        >
                                                            <Eye className="h-4 w-4" />
                                                        </button>
                                                    ) : null}
                                                    <button
                                                        type="button"
                                                        onClick={() => openDeleteDialog(doc)}
                                                        className="rounded-md p-1.5 text-slate-500 transition hover:bg-red-50 hover:text-red-600 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#1f3a60]"
                                                        title={t('documents.delete')}
                                                        aria-label={t('documents.deleteSource', { name: doc.name })}
                                                    >
                                                        <Trash2 className="h-4 w-4" />
                                                    </button>
                                                </div>
                                            </td>
                                        </tr>
                                    );
                                })
                            )}
                        </tbody>
                    </table>
                </div>

                {showDetails && !isLoading && sortedDocuments.length > 0 && (
                    <nav
                        className="flex flex-wrap items-center justify-between gap-3 border-t border-slate-200 px-5 py-3"
                        aria-label={t('documents.pagination.label')}
                    >
                        <div className="flex flex-wrap items-center gap-3 text-xs font-semibold text-slate-500">
                            <span>{t('documents.pagination.range', { from: rangeStart, to: rangeEnd, total: sortedDocuments.length })}</span>
                            <span className="flex items-center gap-2">
                                <label htmlFor="documents-page-size">{t('documents.pagination.pageSize')}</label>
                                <select
                                    id="documents-page-size"
                                    value={pageSize}
                                    onChange={(event) => setPageSize(Number(event.target.value))}
                                    className="rounded-lg border border-slate-200 bg-slate-50 px-2 py-1 text-xs font-semibold text-slate-700 outline-none focus:border-[#1f3a60] focus:ring-1 focus:ring-[#1f3a60]"
                                >
                                    {PAGE_SIZE_OPTIONS.map((option) => (
                                        <option key={option} value={option}>{option}</option>
                                    ))}
                                </select>
                            </span>
                        </div>

                        <div className="flex items-center gap-2">
                            <Button
                                type="button"
                                variant="outline"
                                size="sm"
                                disabled={currentPage <= 1}
                                onClick={() => setPage((prev) => Math.max(1, prev - 1))}
                                aria-label={t('documents.pagination.previous')}
                            >
                                <ChevronLeft className="h-4 w-4" />
                            </Button>
                            <Button
                                type="button"
                                variant="outline"
                                size="sm"
                                disabled={currentPage >= pageCount}
                                onClick={() => setPage((prev) => Math.min(pageCount, prev + 1))}
                                aria-label={t('documents.pagination.next')}
                            >
                                <ChevronRight className="h-4 w-4" />
                            </Button>
                        </div>
                    </nav>
                )}
            </section>

            {/* Chunks Modal */}
            {chunksModal.isOpen && (
                <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
                    <div
                        className="absolute inset-0 bg-black/50"
                        onClick={closeChunksModal}
                        aria-hidden="true"
                    />
                    <div
                        ref={chunksDialogRef}
                        role="dialog"
                        aria-modal="true"
                        aria-labelledby="chunks-dialog-title"
                        className="relative max-h-[90vh] w-full max-w-4xl overflow-hidden rounded-2xl bg-white shadow-xl"
                    >
                        <div className="flex items-center justify-between border-b border-slate-200 px-6 py-4">
                            <div className="min-w-0">
                                <h3 id="chunks-dialog-title" className="text-lg font-bold text-slate-800">{t('documents.chunksTitle')}</h3>
                                <p className="break-words text-sm text-slate-500">{chunksModal.docName}</p>
                            </div>
                            <button
                                ref={chunksCloseRef}
                                type="button"
                                onClick={closeChunksModal}
                                className="rounded-lg p-2 text-slate-400 transition hover:bg-slate-100 hover:text-slate-600 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#1f3a60]"
                                aria-label={t('documents.close')}
                            >
                                <X className="h-5 w-5" />
                            </button>
                        </div>

                        <div className="max-h-[calc(90vh-80px)] overflow-y-auto p-6" aria-live="polite">
                            {chunksModal.isLoading ? (
                                <div className="flex items-center justify-center py-12 text-slate-500">
                                    <Loader2 className="h-6 w-6 animate-spin" />
                                    <span className="ml-2">{t('documents.loadingChunks')}</span>
                                </div>
                            ) : chunksModal.error ? (
                                <div role="alert" className="rounded-lg bg-red-50 p-4 text-center text-red-600">
                                    {chunksModal.error}
                                </div>
                            ) : chunksModal.chunks.length === 0 ? (
                                <div className="rounded-lg bg-slate-50 p-8 text-center text-slate-500">
                                    {t('documents.emptyChunks')}
                                </div>
                            ) : (
                                <div className="space-y-4">
                                    <div className="mb-4 text-sm text-slate-500">
                                        {t('documents.totalChunks', { count: chunksModal.chunks.length })}
                                    </div>
                                    {chunksModal.chunks.map((chunk, index) => (
                                        <div
                                            key={chunk.id || index}
                                            className="rounded-xl border border-slate-200 bg-slate-50 p-4"
                                        >
                                            <div className="mb-2 flex items-center gap-2">
                                                <span className="rounded-full bg-[#1f3a60]/10 px-2 py-0.5 text-xs font-semibold text-[#1f3a60]">
                                                    {t('documents.page', { page: chunk.page || '?' })}
                                                </span>
                                            </div>
                                            <p className="text-sm leading-relaxed text-slate-700 whitespace-pre-wrap">
                                                {chunk.text}
                                            </p>
                                        </div>
                                    ))}
                                </div>
                            )}
                        </div>
                    </div>
                </div>
            )}

            {/* Delete confirmation modal — вместо window.confirm/alert, чтобы кнопки были на языке приложения */}
            {deleteTarget && (
                <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
                    <div
                        className="absolute inset-0 bg-black/50"
                        onClick={closeDeleteDialog}
                        aria-hidden="true"
                    />
                    <div
                        ref={deleteDialogRef}
                        role="dialog"
                        aria-modal="true"
                        aria-labelledby="delete-dialog-title"
                        aria-describedby="delete-dialog-description"
                        className="relative w-full max-w-md overflow-hidden rounded-2xl bg-white shadow-xl"
                    >
                        <div className="flex items-start gap-3 px-6 py-5">
                            <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full bg-red-100 text-red-600">
                                <AlertTriangle className="h-5 w-5" />
                            </div>
                            <div className="min-w-0">
                                <h3 id="delete-dialog-title" className="text-base font-semibold text-slate-900">
                                    {t('documents.deleteConfirm')}
                                </h3>
                                <p id="delete-dialog-description" className="mt-1 break-words text-sm text-slate-500">
                                    {t('documents.deleteDescription', { name: deleteTarget.name })}
                                </p>
                            </div>
                        </div>

                        {deleteError && (
                            <p role="alert" className="mx-6 mb-2 rounded-lg bg-red-50 px-3 py-2 text-sm font-semibold text-red-700">
                                {deleteError}
                            </p>
                        )}

                        <div className="flex justify-end gap-2 border-t border-slate-200 px-6 py-4">
                            <Button
                                ref={deleteCancelRef}
                                type="button"
                                variant="outline"
                                onClick={closeDeleteDialog}
                                disabled={isDeleting}
                            >
                                {t('documents.cancel')}
                            </Button>
                            <Button
                                type="button"
                                variant="destructive"
                                onClick={confirmDelete}
                                isLoading={isDeleting}
                            >
                                {isDeleting ? t('documents.deleting') : t('documents.delete')}
                            </Button>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
};

export default AdminDocumentsPage;
