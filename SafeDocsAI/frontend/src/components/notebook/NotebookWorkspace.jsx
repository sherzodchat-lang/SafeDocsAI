import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  AlertTriangle,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  FilePlus2,
  FileText,
  Link2,
  Loader2,
  MessageSquareText,
  NotebookPen,
  Plus,
  RefreshCw,
  Search,
  X,
} from 'lucide-react';
import { Link } from 'react-router-dom';

import { Button } from '../ui/Button';
import Input from '../ui/Input';
import { cn } from '../../lib/utils';
import { useModalDialog } from '../../hooks/useModalDialog';
import { useSources, useSourcesActions } from '../../contexts/SourcesContext';
import { useLocale } from '../../i18n';
import { resolveApiErrorMessage } from '../../lib/apiError';
import { formatLocaleDate } from '../../lib/locale';
import { formatSize, resolveStatus } from '../../lib/sources';
import { notesService } from '../../services/notesService';
import { notebooksService } from '../../services/notebooksService';
import ChatPage from '../../pages/ChatPage';

const getNotebookLabel = (source, notebookNameById, t) => {
  if (source.notebook_id == null) return t('notebook.notebookUnlinked');

  return notebookNameById[source.notebook_id] || t('notebook.notebookFallback', { id: source.notebook_id });
};

/* ── Split-dropdown button (like Google NotebookLM) ─────────────────────── */
const AddSourceSplitButton = ({ onUpload, onExisting, isLoading, uploadProgress, labels }) => {
  const [open, setOpen] = useState(false);
  const containerRef = useRef(null);

  // Close dropdown when clicking outside
  useEffect(() => {
    if (!open) return undefined;
    const handleClick = (e) => {
      if (containerRef.current && !containerRef.current.contains(e.target)) {
        setOpen(false);
      }
    };
    document.addEventListener('mousedown', handleClick);
    return () => document.removeEventListener('mousedown', handleClick);
  }, [open]);

  return (
    <div ref={containerRef} className="relative flex w-full">
      {/* Primary action */}
      <button
        type="button"
        onClick={onUpload}
        disabled={isLoading}
        className="flex flex-1 items-center justify-center gap-2 rounded-l-lg bg-[#1f3a60] px-4 py-2 text-sm font-semibold text-white shadow-[0_8px_18px_rgba(31,58,96,0.22)] transition hover:bg-[#162945] disabled:pointer-events-none disabled:opacity-60"
      >
        {isLoading ? (
          <Loader2 className="h-4 w-4 animate-spin" />
        ) : (
          <Plus className="h-4 w-4" />
        )}
        {isLoading && uploadProgress ? `${labels.loading} ${uploadProgress}` : labels.addSource}
      </button>

      {/* Divider */}
      <div className="w-px bg-white/20" />

      {/* Chevron toggle */}
      <button
        type="button"
        onClick={() => setOpen((prev) => !prev)}
        disabled={isLoading}
        className="flex items-center justify-center rounded-r-lg bg-[#1f3a60] px-2.5 py-2 text-white shadow-[0_8px_18px_rgba(31,58,96,0.22)] transition hover:bg-[#162945] disabled:pointer-events-none disabled:opacity-60"
        aria-label={labels.openMenu}
        aria-expanded={open}
      >
        <ChevronDown className={cn('h-4 w-4 transition-transform duration-150', open && 'rotate-180')} />
      </button>

      {/* Dropdown */}
      {open && (
        <div className="absolute left-0 top-full z-50 mt-1.5 w-full overflow-hidden rounded-xl border border-slate-200 bg-white py-1 shadow-lg">
          <button
            type="button"
            onClick={() => { setOpen(false); onUpload(); }}
            className="flex w-full items-center gap-2.5 px-3.5 py-2.5 text-sm text-slate-700 transition hover:bg-slate-50"
          >
            <Plus className="h-4 w-4 text-slate-400" />
            {labels.addSource}
          </button>
          <button
            type="button"
            onClick={() => { setOpen(false); onExisting(); }}
            className="flex w-full items-center gap-2.5 px-3.5 py-2.5 text-sm text-slate-700 transition hover:bg-slate-50"
          >
            <Link2 className="h-4 w-4 text-slate-400" />
            {labels.addExistingSources}
          </button>
        </div>
      )}
    </div>
  );
};

const NotebookSidePanel = ({
  icon,
  title,
  collapsed,
  onToggle,
  action,
  actionLabel,
  actionLoading,
  actionDisabled,
  renderAction,
  children,
  footerLink,
  expandLabel,
  collapseLabel,
}) => {
  return (
    <section
      className={cn(
        'relative flex h-full shrink-0 overflow-hidden rounded-3xl border border-slate-200 bg-white shadow-sm transition-[width] duration-300 ease-out',
        collapsed ? 'w-11' : 'w-[19rem] xl:w-[21rem]',
      )}
    >
      {collapsed ? (
        <button
          type="button"
          onClick={onToggle}
          className="flex h-full w-full flex-col items-center justify-between bg-slate-50 py-4 text-slate-500 transition hover:bg-slate-100 hover:text-[#1f3a60]"
          aria-label={expandLabel}
        >
          <ChevronRight className="h-4 w-4" />
          <div className="flex items-center gap-2" style={{ writingMode: 'vertical-rl', transform: 'rotate(180deg)' }}>
            {React.createElement(icon, { className: 'h-4 w-4' })}
            <span className="text-xs font-semibold tracking-[0.24em] uppercase">{title}</span>
          </div>
          <span className="h-4 w-4" />
        </button>
      ) : (
        <div className="flex h-full min-h-0 w-full flex-col">
          <div className="flex items-start justify-between gap-3 border-b border-slate-200 px-4 py-4">
            <div>
              <div className="inline-flex items-center gap-2 rounded-full bg-[#1f3a60]/10 px-2.5 py-1 text-[11px] font-semibold uppercase tracking-[0.14em] text-[#1f3a60]">
                {React.createElement(icon, { className: 'h-3.5 w-3.5' })}
                {title}
              </div>
            </div>

            <button
              type="button"
              onClick={onToggle}
              className="inline-flex h-9 w-9 items-center justify-center rounded-xl border border-slate-200 bg-white text-slate-500 transition hover:border-slate-300 hover:text-slate-900"
              aria-label={collapseLabel}
            >
              <ChevronLeft className="h-4 w-4" />
            </button>
          </div>

          <div className="border-b border-slate-200 px-4 py-3">
            {renderAction ? renderAction() : (
              <Button
                type="button"
                onClick={action}
                isLoading={actionLoading}
                disabled={actionDisabled}
                className="w-full justify-center"
              >
                <Plus className="h-4 w-4" />
                {actionLabel}
              </Button>
            )}
          </div>

          <div className="scrollbar-soft min-h-0 flex-1 overflow-y-auto px-4 py-4">{children}</div>

          {footerLink ? (
            <div className="border-t border-slate-200 px-4 py-3">
              <Link to={footerLink.to} className="text-sm font-semibold text-[#1f3a60] transition hover:text-[#162945] hover:underline">
                {footerLink.label}
              </Link>
            </div>
          ) : null}
        </div>
      )}
    </section>
  );
};

const EmptyPanelState = ({ icon, title, description }) => (
  <div className="flex h-full min-h-[280px] flex-col items-center justify-center rounded-3xl border border-dashed border-slate-200 bg-slate-50/70 px-6 text-center">
    <div className="rounded-2xl bg-white p-4 text-[#1f3a60] shadow-sm">
      {React.createElement(icon, { className: 'h-6 w-6' })}
    </div>
    <p className="mt-4 text-sm font-semibold text-slate-800">{title}</p>
    <p className="mt-2 text-sm leading-6 text-slate-500">{description}</p>
  </div>
);

const NotebookWorkspace = ({ notebookId }) => {
  const { locale, t } = useLocale();
  const currentNotebookId = Number(notebookId);
  const sourceInputRef = useRef(null);
  const existingDialogRef = useRef(null);
  const existingCloseRef = useRef(null);
  const [sourcesCollapsed, setSourcesCollapsed] = useState(false);
  const [notesCollapsed, setNotesCollapsed] = useState(false);

  const [uploadingSource, setUploadingSource] = useState(false);
  const [uploadProgress, setUploadProgress] = useState('');  // e.g. "2 / 5"
  const [uploadError, setUploadError] = useState('');
  const [sourceSheetOpen, setSourceSheetOpen] = useState(false);
  const [sourceSheetMode, setSourceSheetMode] = useState('actions');
  const [sourceSearch, setSourceSearch] = useState('');
  const [attachError, setAttachError] = useState('');
  const [selectedExistingSourceIds, setSelectedExistingSourceIds] = useState([]);
  const [attachingExistingSources, setAttachingExistingSources] = useState(false);
  const [notebooks, setNotebooks] = useState([]);
  const [notebooksError, setNotebooksError] = useState('');

  // Общий слой данных: тот же кэш читают шапка блокнота и таблица «Все источники».
  const {
    items: sources,
    isLoading: sourcesLoading,
    error: sourcesError,
    reload: reloadSources,
  } = useSources(currentNotebookId);

  // Список для привязки берём из области «все источники» — той же, что у таблицы.
  const isExistingSheetOpen = sourceSheetOpen && sourceSheetMode === 'existing';
  const {
    items: existingSources,
    isLoading: existingSourcesLoading,
    error: existingSourcesError,
    reload: reloadExistingSources,
  } = useSources(null, { enabled: isExistingSheetOpen });

  const { uploadSource, attachSources, invalidate } = useSourcesActions();

  const [notes, setNotes] = useState([]);
  const [notesLoading, setNotesLoading] = useState(true);
  const [notesError, setNotesError] = useState('');
  const [selectedNote, setSelectedNote] = useState(null);
  const [noteComposerOpen, setNoteComposerOpen] = useState(false);
  const [noteTitle, setNoteTitle] = useState('');
  const [noteBody, setNoteBody] = useState('');
  const [savingNote, setSavingNote] = useState(false);

  useEffect(() => {
    let active = true;

    const fetchNotes = async () => {
      try {
        setNotesLoading(true);
        setNotesError('');
        const response = await notesService.getAll(currentNotebookId);
        if (!active) return;
        setNotes(response.data || []);
      } catch (error) {
        if (!active) return;
        console.error('Failed to fetch notebook notes', error);
        setNotes([]);
        setNotesError(error.response?.data?.detail || t('notebook.loadingNotes'));
      } finally {
        if (active) {
          setNotesLoading(false);
        }
      }
    };

    fetchNotes();

    return () => {
      active = false;
    };
  }, [currentNotebookId, t]);

  // Названия блокнотов нужны только в модалке привязки — сами источники берёт общий слой данных.
  useEffect(() => {
    if (!isExistingSheetOpen) return undefined;

    let active = true;

    const fetchNotebooks = async () => {
      try {
        setNotebooksError('');
        const response = await notebooksService.getAll();
        if (!active) return;
        setNotebooks(response.data || []);
      } catch (error) {
        if (!active) return;
        console.error('Failed to fetch notebooks', error);
        setNotebooks([]);
        setNotebooksError(resolveApiErrorMessage(error, t, 'notebook.existingLoadFailed'));
      }
    };

    fetchNotebooks();

    return () => {
      active = false;
    };
  }, [isExistingSheetOpen, t]);

  const sourceStatusLabels = useMemo(() => ({
    ready: t('documents.status.ready'),
    pending: t('documents.status.pending'),
    indexing: t('documents.status.indexing'),
    error: t('documents.status.error'),
  }), [t]);

  const noteCountLabel = useMemo(() => t('notebook.noteCount', { count: notes.length }), [notes.length, t]);

  const notebookNameById = useMemo(
    () => Object.fromEntries((notebooks || []).map((notebook) => [notebook.id, notebook.name])),
    [notebooks],
  );

  const attachableSources = useMemo(
    () => existingSources.filter((source) => source.notebook_id !== currentNotebookId),
    [currentNotebookId, existingSources],
  );

  const closeSourceSheet = useCallback(() => {
    setSourceSheetOpen(false);
    setSourceSheetMode('actions');
    setAttachError('');
    setSelectedExistingSourceIds([]);
    setSourceSearch('');
  }, []);

  useModalDialog(isExistingSheetOpen, closeSourceSheet, existingDialogRef, existingCloseRef);

  const handleUploadSourceClick = () => {
    closeSourceSheet();
    sourceInputRef.current?.click();
  };

  const handleOpenExistingSources = () => {
    setAttachError('');
    setSelectedExistingSourceIds([]);
    setSourceSheetMode('existing');
    setSourceSheetOpen(true);
  };

  const handleExistingSourceSelection = (sourceId) => {
    setSelectedExistingSourceIds((prev) => (
      prev.includes(sourceId)
        ? prev.filter((id) => id !== sourceId)
        : [...prev, sourceId]
    ));
  };

  const handleSourceUpload = async (event) => {
    const files = Array.from(event.target.files || []);
    if (files.length === 0) return;

    setUploadingSource(true);
    setUploadError('');
    const errors = [];

    for (let i = 0; i < files.length; i++) {
      setUploadProgress(`${i + 1} / ${files.length}`);
      try {
        await uploadSource(files[i], currentNotebookId);
        setSourcesCollapsed(false);
      } catch (error) {
        console.error(`Failed to upload ${files[i].name}`, error);
        errors.push(`${files[i].name} (${resolveApiErrorMessage(error, t, 'documents.uploadError')})`);
      }
    }

    if (errors.length > 0) {
      setUploadError(t('notebook.uploadFailed', { files: errors.join(', ') }));
    }

    // Общая инвалидация: список и дата «Обновлён» в шапке подтягивают серверное состояние.
    await invalidate();
    setUploadingSource(false);
    setUploadProgress('');
    event.target.value = '';
  };

  const handleAttachExistingSources = async () => {
    if (selectedExistingSourceIds.length === 0) return;

    try {
      setAttachingExistingSources(true);
      setAttachError('');
      await attachSources(currentNotebookId, selectedExistingSourceIds);
      setSourcesCollapsed(false);
      closeSourceSheet();
    } catch (error) {
      console.error('Failed to attach existing sources', error);
      setAttachError(resolveApiErrorMessage(error, t, 'notebook.attachFailed'));
    } finally {
      setAttachingExistingSources(false);
    }
  };

  const handleCreateNote = async (event) => {
    event.preventDefault();
    if (!noteTitle.trim()) return;

    try {
      setSavingNote(true);
      setNotesError('');
      const response = await notesService.create({
        notebook_id: currentNotebookId,
        title: noteTitle.trim(),
        body: noteBody.trim(),
      });

      setNotes((prev) => [response.data, ...prev]);
      setNoteTitle('');
      setNoteBody('');
      setNoteComposerOpen(false);
      setNotesCollapsed(false);
    } catch (error) {
      console.error('Failed to create note', error);
      setNotesError(error.response?.data?.detail || t('notebook.createNoteFailed'));
    } finally {
      setSavingNote(false);
    }
  };

  return (
    <div className="flex h-full min-h-0 flex-col gap-4">
      <input ref={sourceInputRef} type="file" className="hidden" multiple accept=".pdf,.docx,.txt" onChange={handleSourceUpload} />

      <div className="min-h-0 flex-1 overflow-x-auto pb-1">
        <div className="flex h-full min-w-[940px] gap-4">
          <NotebookSidePanel
            icon={FileText}
            title={t('notebook.sources')}
            collapsed={sourcesCollapsed}
            onToggle={() => setSourcesCollapsed((prev) => !prev)}
            expandLabel={t('notebook.expandPanel', { title: t('notebook.sources') })}
            collapseLabel={t('notebook.collapsePanel', { title: t('notebook.sources') })}
            renderAction={() => (
              <AddSourceSplitButton
                onUpload={handleUploadSourceClick}
                onExisting={handleOpenExistingSources}
                isLoading={uploadingSource}
                uploadProgress={uploadProgress}
                labels={{
                  loading: t('documents.uploadLoading'),
                  addSource: t('notebook.addSource'),
                  addExistingSources: t('notebook.addExistingSources'),
                  openMenu: t('notebook.openAddSourceMenu'),
                }}
              />
            )}
            footerLink={{ to: `/notebooks/${notebookId}/sources`, label: t('notebook.openAllSources') }}
          >
            <div className="space-y-3" aria-live="polite">
              {uploadError ? (
                <p role="alert" className="rounded-2xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-600">
                  {uploadError}
                </p>
              ) : null}

              {/* Сбой загрузки списка показываем баннером: панель с уже полученными источниками остаётся на месте. */}
              {sourcesError ? (
                <div role="alert" className="flex flex-col gap-2 rounded-2xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-600">
                  <span className="flex items-center gap-2 font-semibold">
                    <AlertTriangle className="h-4 w-4" />
                    {sourcesError}
                  </span>
                  <Button type="button" variant="outline" size="sm" className="self-start" onClick={reloadSources}>
                    <RefreshCw className="h-4 w-4" />
                    {t('documents.retry')}
                  </Button>
                </div>
              ) : null}

              {sourcesLoading && sources.length === 0 ? (
                <div className="flex h-full items-center justify-center text-sm text-slate-500">
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  {t('notebook.loadingSources')}
                </div>
              ) : sources.length === 0 ? (
                sourcesError ? null : (
                  <EmptyPanelState
                    icon={FilePlus2}
                    title={t('notebook.noSourcesTitle')}
                    description={t('notebook.noSourcesDescription')}
                  />
                )
              ) : (
                sources.map((source) => (
                  <article key={source.id} className="rounded-2xl border border-slate-200 p-4 transition hover:border-slate-300 hover:bg-slate-50">
                    <div className="flex items-start justify-between gap-3">
                      <div className="min-w-0">
                        <p className="truncate text-sm font-semibold text-slate-900">{source.name}</p>
                        <p className="mt-1 text-xs text-slate-400">ID #{source.id}</p>
                      </div>
                      <span className="rounded-full bg-slate-100 px-2.5 py-1 text-[11px] font-semibold text-slate-600">
                        {sourceStatusLabels[resolveStatus(source.status)]}
                      </span>
                    </div>
                    <div className="mt-3 flex flex-wrap gap-2 text-xs text-slate-500">
                      <span>{t('notebook.createdAt', { date: formatLocaleDate(source.created_at, locale, { day: 'numeric', month: 'short', year: 'numeric' }, '—') })}</span>
                      <span>{formatSize(source.size, t)}</span>
                    </div>
                  </article>
                ))
              )}
            </div>
          </NotebookSidePanel>

          <NotebookSidePanel
            icon={NotebookPen}
            title={t('notebook.notes')}
            collapsed={notesCollapsed}
            onToggle={() => setNotesCollapsed((prev) => !prev)}
            expandLabel={t('notebook.expandPanel', { title: t('notebook.notes') })}
            collapseLabel={t('notebook.collapsePanel', { title: t('notebook.notes') })}
            action={() => setNoteComposerOpen(true)}
            actionLabel={t('notebook.writeNote')}
            footerLink={{ to: `/notebooks/${notebookId}/notes`, label: t('notebook.openAllNotes') }}
          >

            {notesLoading ? (
              <div className="flex h-full items-center justify-center text-sm text-slate-500">
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                {t('notebook.loadingNotes')}
              </div>
            ) : notesError ? (
              <div className="rounded-2xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-600">
                {notesError}
              </div>
            ) : notes.length === 0 ? (
              <EmptyPanelState
                icon={MessageSquareText}
                title={t('notebook.noNotesTitle')}
                description={t('notebook.noNotesDescription')}
              />
            ) : (
              <div className="space-y-3">
                <div className="rounded-2xl bg-slate-50 px-4 py-3 text-xs font-semibold uppercase tracking-[0.14em] text-slate-500">
                  {noteCountLabel}
                </div>
                {notes.map((note) => (
                  <article
                    key={note.id}
                    role="button"
                    tabIndex={0}
                    onClick={() => setSelectedNote(note)}
                    onKeyDown={(e) => e.key === 'Enter' && setSelectedNote(note)}
                    className="cursor-pointer rounded-2xl border border-slate-200 p-4 transition hover:border-[#1f3a60]/40 hover:bg-slate-50"
                  >
                    <div className="flex items-start justify-between gap-3">
                      <h3 className="text-sm font-semibold text-slate-900">{note.title}</h3>
                      <span className="rounded-full bg-[#1f3a60]/10 px-2.5 py-1 text-[11px] font-semibold text-[#1f3a60]">
                        {note.kind || t('notebook.noteKindManual')}
                      </span>
                    </div>
                    <p className="mt-2 line-clamp-4 whitespace-pre-wrap text-sm leading-6 text-slate-500">
                      {note.body || t('notebook.noteTextMissing')}
                    </p>
                    <div className="mt-3 text-xs text-slate-400">{t('notebook.updatedAt', { date: formatLocaleDate(note.updated_at || note.created_at, locale, { day: 'numeric', month: 'short', year: 'numeric' }, '—') })}</div>
                  </article>
                ))}
              </div>
            )}
          </NotebookSidePanel>

          <section className="flex h-full min-w-0 flex-1 overflow-hidden rounded-3xl border border-slate-200 bg-white shadow-sm">
            <ChatPage notebookId={currentNotebookId} mode="notebookPanel" />
          </section>
        </div>
      </div>

      {/* Existing-sources MODAL — opened via dropdown “Добавить существующие источники” */}
      {isExistingSheetOpen ? (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
          {/* Backdrop */}
          <div
            className="absolute inset-0 bg-black/40 backdrop-blur-[2px]"
            onClick={closeSourceSheet}
            aria-hidden="true"
          />

          {/* Dialog */}
          <div
            ref={existingDialogRef}
            role="dialog"
            aria-modal="true"
            aria-label={t('notebook.addExistingTitle')}
            className="relative flex w-full max-w-lg flex-col overflow-hidden rounded-2xl bg-white shadow-2xl max-h-[80vh]"
          >
            {/* Header */}
            <div className="flex items-start justify-between gap-4 px-6 py-5">
              <div>
                <div className="flex items-center gap-2">
                  <Link2 className="h-4 w-4 text-[#1f3a60]" />
                  <p className="text-[15px] font-semibold text-slate-900">{t('notebook.addExistingTitle')}</p>
                </div>
                <p className="mt-1 text-sm text-slate-500">
                  {t('notebook.addExistingDescription')}
                </p>
              </div>
              <button
                ref={existingCloseRef}
                type="button"
                onClick={closeSourceSheet}
                className="inline-flex h-8 w-8 shrink-0 items-center justify-center rounded-lg text-slate-400 transition hover:bg-slate-100 hover:text-slate-700"
                aria-label={t('notebook.close')}
              >
                <X className="h-4 w-4" />
              </button>
            </div>

            {/* Search */}
            <div className="px-6 pb-3">
              {(() => {
                const filtered = attachableSources.filter((s) =>
                  !sourceSearch.trim() ||
                  s.name?.toLowerCase().includes(sourceSearch.toLowerCase())
                );
                const allIds = filtered.map(s => s.id);
                const areAllSelected = allIds.length > 0 && allIds.every(id => selectedExistingSourceIds.includes(id));
                
                return (
                  <div className="flex items-center gap-3">
                    <div className="relative flex-1">
                      <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-400" />
                      <input
                        type="text"
                        placeholder={t('notebook.sourceSearchPlaceholder')}
                        value={sourceSearch}
                        onChange={(e) => setSourceSearch(e.target.value)}
                        className="h-10 w-full rounded-lg border border-slate-300 bg-white pl-9 pr-3 text-sm text-slate-800 placeholder-slate-400 transition focus:border-[#1f3a60] focus:outline-none focus:ring-2 focus:ring-[#1f3a60]/20"
                      />
                    </div>
                    {filtered.length > 0 && (
                      <Button
                        type="button"
                        variant="outline"
                        onClick={() => {
                          if (areAllSelected) {
                            setSelectedExistingSourceIds(prev => prev.filter(id => !allIds.includes(id)));
                          } else {
                            setSelectedExistingSourceIds(prev => Array.from(new Set([...prev, ...allIds])));
                          }
                        }}
                        className="h-10 shrink-0 px-3"
                      >
                        {areAllSelected ? t('notebook.resetAll') : t('notebook.selectAll')}
                      </Button>
                    )}
                  </div>
                );
              })()}
            </div>

            {/* Body */}
            <div className="min-h-[240px] flex-1 overflow-y-auto px-6 pb-2" aria-live="polite">
              {existingSourcesError || notebooksError ? (
                <div role="alert" className="mb-2 flex flex-col gap-2 rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-600">
                  <span>{existingSourcesError || notebooksError}</span>
                  {existingSourcesError ? (
                    <Button type="button" variant="outline" size="sm" className="self-start" onClick={reloadExistingSources}>
                      <RefreshCw className="h-4 w-4" />
                      {t('documents.retry')}
                    </Button>
                  ) : null}
                </div>
              ) : null}

              {existingSourcesLoading && existingSources.length === 0 ? (
                <div className="flex h-48 items-center justify-center text-sm text-slate-500">
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  {t('notebook.loadingAvailableSources')}
                </div>
              ) : existingSourcesError && existingSources.length === 0 ? null : (() => {
                const filtered = attachableSources.filter((s) =>
                  !sourceSearch.trim() ||
                  s.name?.toLowerCase().includes(sourceSearch.toLowerCase())
                );
                if (filtered.length === 0) {
                  return (
                    <div className="flex h-48 flex-col items-center justify-center gap-2 text-sm text-slate-500">
                      <FileText className="h-8 w-8 text-slate-300" />
                      {attachableSources.length === 0 ? t('notebook.noAttachableSources') : t('notebook.nothingFound')}
                    </div>
                  );
                }
                return (
                  <div className="space-y-2 py-2">
                    {filtered.map((source) => {
                      const isSelected = selectedExistingSourceIds.includes(source.id);
                      return (
                        <label
                          key={source.id}
                          className={cn(
                            'flex cursor-pointer items-center gap-3 rounded-xl border px-4 py-3 transition',
                            isSelected
                              ? 'border-[#1f3a60] bg-[#1f3a60]/5'
                              : 'border-slate-200 hover:border-slate-300 hover:bg-slate-50',
                          )}
                        >
                          <input
                            type="checkbox"
                            className="h-4 w-4 shrink-0 rounded border-slate-300 text-[#1f3a60] focus:ring-[#1f3a60]"
                            checked={isSelected}
                            onChange={() => handleExistingSourceSelection(source.id)}
                          />
                          <div className="min-w-0 flex-1">
                            <p className="truncate text-sm font-semibold text-slate-900">{source.name}</p>
                            <p className="mt-0.5 text-xs text-slate-400">
                              {getNotebookLabel(source, notebookNameById, t)} · {formatSize(source.size, t)}
                            </p>
                          </div>
                          <span className="shrink-0 rounded-full bg-slate-100 px-2 py-0.5 text-[11px] font-semibold text-slate-600">
                            {sourceStatusLabels[resolveStatus(source.status)]}
                          </span>
                        </label>
                      );
                    })}
                  </div>
                );
              })()}
            </div>

            {attachError ? (
              <p role="alert" className="mx-6 mb-2 rounded-lg bg-red-50 px-3 py-2 text-sm font-semibold text-red-700">
                {attachError}
              </p>
            ) : null}

            {/* Footer */}
            <div className="flex items-center justify-end gap-3 border-t border-slate-200 px-6 py-4">
              <Button type="button" variant="ghost" onClick={closeSourceSheet}>
                {t('notebook.cancel')}
              </Button>
              <Button
                type="button"
                onClick={handleAttachExistingSources}
                isLoading={attachingExistingSources}
                disabled={
                  selectedExistingSourceIds.length === 0 ||
                  existingSourcesLoading ||
                  attachableSources.length === 0
                }
              >
                {t('notebook.addSelected')}
              </Button>
            </div>
          </div>
        </div>
      ) : null}

      {/* Note VIEW modal */}
      {selectedNote ? (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center p-4"
          role="dialog"
          aria-modal="true"
          aria-label={t('notebook.viewNote')}
        >
          {/* Backdrop */}
          <div
            className="absolute inset-0 bg-black/40 backdrop-blur-[2px]"
            onClick={() => setSelectedNote(null)}
          />

          {/* Dialog */}
          <div className="relative flex w-full max-w-lg flex-col overflow-hidden rounded-2xl bg-white shadow-2xl">
            {/* Header */}
            <div className="flex items-start justify-between gap-4 px-6 py-5">
              <div className="min-w-0">
                <div className="flex items-center gap-2">
                  <NotebookPen className="h-4 w-4 shrink-0 text-[#1f3a60]" />
                  <p className="truncate text-[15px] font-semibold text-slate-900">{selectedNote.title}</p>
                </div>
                <p className="mt-1 text-xs text-slate-400">
                  {t('notebook.noteCreatedAt', { date: formatLocaleDate(selectedNote.created_at, locale, { day: 'numeric', month: 'short', year: 'numeric' }, '—') })}
                  {selectedNote.updated_at && selectedNote.updated_at !== selectedNote.created_at
                    ? ` · ${t('notebook.noteUpdatedAt', { date: formatLocaleDate(selectedNote.updated_at, locale, { day: 'numeric', month: 'short', year: 'numeric' }, '—') })}`
                    : ''}
                </p>
              </div>
              <button
                type="button"
                onClick={() => setSelectedNote(null)}
                className="inline-flex h-8 w-8 shrink-0 items-center justify-center rounded-lg text-slate-400 transition hover:bg-slate-100 hover:text-slate-700"
                aria-label={t('notebook.close')}
              >
                <X className="h-4 w-4" />
              </button>
            </div>

            {/* Body */}
            <div className="max-h-[60vh] overflow-y-auto px-6 pb-6">
              {selectedNote.body ? (
                <p className="whitespace-pre-wrap text-sm leading-7 text-slate-700">{selectedNote.body}</p>
              ) : (
                <p className="text-sm italic text-slate-400">{t('notebook.noteBodyMissing')}</p>
              )}
            </div>

            {/* Footer */}
            <div className="flex items-center justify-between border-t border-slate-200 px-6 py-4">
              <span className="rounded-full bg-[#1f3a60]/10 px-3 py-1 text-[11px] font-semibold text-[#1f3a60]">
                {selectedNote.kind || t('notebook.noteKindManual')}
              </span>
              <Button type="button" variant="ghost" onClick={() => setSelectedNote(null)}>
                {t('notebook.close')}
              </Button>
            </div>
          </div>
        </div>
      ) : null}

      {/* Note composer MODAL */}
      {noteComposerOpen ? (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center p-4"
          role="dialog"
          aria-modal="true"
          aria-label={t('notebook.writeNoteTitle')}
        >
          {/* Backdrop */}
          <div
            className="absolute inset-0 bg-black/40 backdrop-blur-[2px]"
            onClick={() => { setNoteComposerOpen(false); setNoteTitle(''); setNoteBody(''); }}
          />

          {/* Dialog */}
          <div className="relative flex w-full max-w-lg flex-col overflow-hidden rounded-2xl bg-white shadow-2xl">
            {/* Header */}
            <div className="flex items-start justify-between gap-4 px-6 py-5">
              <div>
                <div className="flex items-center gap-2">
                  <NotebookPen className="h-4 w-4 text-[#1f3a60]" />
                  <p className="text-[15px] font-semibold text-slate-900">{t('notebook.writeNoteTitle')}</p>
                </div>
                <p className="mt-1 text-sm text-slate-500">
                  {t('notebook.writeNoteDescription')}
                </p>
              </div>
              <button
                type="button"
                onClick={() => { setNoteComposerOpen(false); setNoteTitle(''); setNoteBody(''); }}
                className="inline-flex h-8 w-8 shrink-0 items-center justify-center rounded-lg text-slate-400 transition hover:bg-slate-100 hover:text-slate-700"
                aria-label={t('notebook.close')}
              >
                <X className="h-4 w-4" />
              </button>
            </div>

            {/* Form body */}
            <form onSubmit={handleCreateNote} className="flex flex-col gap-4 px-6 pb-6">
              <input
                type="text"
                value={noteTitle}
                onChange={(e) => setNoteTitle(e.target.value)}
                placeholder={t('notebook.noteTitlePlaceholder')}
                className="h-10 w-full rounded-lg border border-slate-300 bg-white px-3 text-sm text-slate-800 placeholder-slate-400 transition focus:border-[#1f3a60] focus:outline-none focus:ring-2 focus:ring-[#1f3a60]/20"
                autoFocus
              />
              <textarea
                value={noteBody}
                onChange={(e) => setNoteBody(e.target.value)}
                placeholder={t('notebook.noteBodyPlaceholder')}
                rows={6}
                className="w-full resize-none rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm text-slate-800 placeholder-slate-400 transition focus:border-[#1f3a60] focus:outline-none focus:ring-2 focus:ring-[#1f3a60]/20"
              />
              {notesError ? (
                <div className="rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-600">
                  {notesError}
                </div>
              ) : null}
              <div className="flex items-center justify-end gap-3 border-t border-slate-200 pt-4">
                <Button
                  type="button"
                  variant="ghost"
                  onClick={() => { setNoteComposerOpen(false); setNoteTitle(''); setNoteBody(''); }}
                >
                  {t('notebook.cancel')}
                </Button>
                <Button type="submit" isLoading={savingNote} disabled={!noteTitle.trim()}>
                  {t('notebook.save')}
                </Button>
              </div>
            </form>
          </div>
        </div>
      ) : null}
    </div>
  );
};

export default NotebookWorkspace;
