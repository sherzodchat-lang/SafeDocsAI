import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  AlertTriangle,
  Archive,
  ArchiveRestore,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  FilePlus2,
  FileText,
  Link2,
  Loader2,
  MessageSquareText,
  NotebookPen,
  Pencil,
  Plus,
  RefreshCw,
  Search,
  Tag,
  Trash2,
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
import { formatSize, resolveSourceErrorMessage, resolveStatus } from '../../lib/sources';
import { resolveTopicLabel } from '../../lib/topics';
import { notesService } from '../../services/notesService';
import { notebooksService } from '../../services/notebooksService';
import ChatPage from '../../pages/ChatPage';

// Значения Note.status с бэкенда (backend/app/api/endpoints/notes.py):
// архивация — это ровно смена статуса, отдельной ручки под неё нет.
const NOTE_STATUS_ACTIVE = 'active';
const NOTE_STATUS_ARCHIVED = 'archived';

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
        // Полосой панель становится только там, где рядом помещаются три колонки:
        // до этого она занимает всю ширину и стоит над чатом, а не за краем экрана.
        'relative flex w-full shrink-0 overflow-hidden rounded-3xl border border-slate-200 bg-white shadow-sm transition-[width] duration-300 ease-out xl:h-full',
        collapsed ? 'xl:w-11' : 'xl:w-[19rem] 2xl:w-[21rem]',
      )}
    >
      {collapsed ? (
        <button
          type="button"
          onClick={onToggle}
          className="flex w-full items-center justify-center gap-2 bg-slate-50 px-4 py-3 text-slate-500 transition hover:bg-slate-100 hover:text-[#1f3a60] xl:h-full xl:flex-col xl:justify-between xl:px-0 xl:py-4"
          aria-label={expandLabel}
        >
          <ChevronDown className="h-4 w-4 xl:hidden" />
          <ChevronRight className="hidden h-4 w-4 xl:block" />
          {/* Вертикальная надпись — только у узкой полосы: в развёрнутой на всю
              ширину строке её пришлось бы читать боком. */}
          <span className="flex items-center gap-2 xl:rotate-180 xl:[writing-mode:vertical-rl]">
            {React.createElement(icon, { className: 'h-4 w-4' })}
            <span className="text-xs font-semibold tracking-[0.24em] uppercase">{title}</span>
          </span>
          <span className="hidden h-4 w-4 xl:block" />
        </button>
      ) : (
        <div className="flex min-h-0 w-full flex-col xl:h-full">
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

          {/* На узком экране высота панели не задана, поэтому список ограничиваем
              сами: иначе страница превращается в один длинный столбец. */}
          <div className="scrollbar-soft max-h-[60vh] min-h-0 flex-1 overflow-y-auto px-4 py-4 xl:max-h-none">{children}</div>

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

/**
 * Пустая колонка объясняет себя одной строкой: раньше это была рамка в треть
 * панели с иконкой в отдельной плашке и абзацем текста — и так в каждой из трёх
 * колонок сразу. Смысл тот же, места втрое меньше.
 */
const EmptyPanelState = ({ icon, title, description }) => (
  <div className="flex flex-col items-center gap-1.5 rounded-2xl border border-dashed border-slate-200 bg-slate-50/70 px-4 py-5 text-center">
    {React.createElement(icon, { className: 'h-5 w-5 text-slate-400' })}
    <p className="text-sm font-semibold text-slate-800">{title}</p>
    <p className="text-xs leading-5 text-slate-500">{description}</p>
  </div>
);

const NotebookWorkspace = ({
  notebookId,
  notes = [],
  notesLoading = false,
  notesError = '',
  onReloadNotes,
  onNoteCreated,
  onNoteUpdated,
  onNoteDeleted,
}) => {
  const { locale, t } = useLocale();
  const currentNotebookId = Number(notebookId);
  const sourceInputRef = useRef(null);
  const existingDialogRef = useRef(null);
  const existingCloseRef = useRef(null);
  const noteViewDialogRef = useRef(null);
  const noteViewCloseRef = useRef(null);
  const noteComposerDialogRef = useRef(null);
  const noteTitleInputRef = useRef(null);
  const noteDeleteDialogRef = useRef(null);
  const noteDeleteCancelRef = useRef(null);
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

  // Список заметок — общий с шапкой блокнота, поэтому приходит сверху: своего запроса панель не шлёт.
  const [selectedNote, setSelectedNote] = useState(null);
  const [noteComposerOpen, setNoteComposerOpen] = useState(false);
  // Заметка, которую правим. null — режим создания: форма одна на оба случая,
  // поля и проверки у создания и правки совпадают полностью.
  const [editingNote, setEditingNote] = useState(null);
  const [noteTitle, setNoteTitle] = useState('');
  const [noteBody, setNoteBody] = useState('');
  const [savingNote, setSavingNote] = useState(false);
  // Ошибку создания держим отдельно от ошибки загрузки: она не должна прятать уже показанный список.
  const [noteCreateError, setNoteCreateError] = useState('');
  // Архивация идёт прямо из списка: помним id заметки, по которой ждём ответ,
  // чтобы крутилка стояла на нужной карточке, а не на всей панели.
  const [noteStatusPendingId, setNoteStatusPendingId] = useState(null);
  const [noteStatusError, setNoteStatusError] = useState('');
  const [noteDeleteTarget, setNoteDeleteTarget] = useState(null);
  const [deletingNote, setDeletingNote] = useState(false);
  const [noteDeleteError, setNoteDeleteError] = useState('');

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

  const closeNoteViewer = useCallback(() => {
    setSelectedNote(null);
  }, []);

  const openNoteComposer = useCallback(() => {
    setEditingNote(null);
    setNoteTitle('');
    setNoteBody('');
    setNoteCreateError('');
    setNoteComposerOpen(true);
  }, []);

  const openNoteEditor = useCallback((note) => {
    if (!note) return;
    setEditingNote(note);
    setNoteTitle(note.title || '');
    setNoteBody(note.body || '');
    setNoteCreateError('');
    setNoteComposerOpen(true);
  }, []);

  const closeNoteComposer = useCallback(() => {
    // Пока идёт запрос, закрывать нечего: ответ всё равно придёт, а сообщение
    // об ошибке пользователь бы уже не увидел.
    if (savingNote) return;
    setNoteComposerOpen(false);
    setEditingNote(null);
    setNoteTitle('');
    setNoteBody('');
    setNoteCreateError('');
  }, [savingNote]);

  const closeNoteDeleteDialog = useCallback(() => {
    if (deletingNote) return;
    setNoteDeleteTarget(null);
    setNoteDeleteError('');
  }, [deletingNote]);

  useModalDialog(Boolean(selectedNote), closeNoteViewer, noteViewDialogRef, noteViewCloseRef);
  useModalDialog(noteComposerOpen, closeNoteComposer, noteComposerDialogRef, noteTitleInputRef);
  useModalDialog(Boolean(noteDeleteTarget), closeNoteDeleteDialog, noteDeleteDialogRef, noteDeleteCancelRef);

  const handleSubmitNote = async (event) => {
    event.preventDefault();
    // Второй клик до ответа сервера создавал бы вторую заметку с тем же текстом.
    if (savingNote || !noteTitle.trim()) return;

    const nextTitle = noteTitle.trim();
    const nextBody = noteBody.trim();

    if (editingNote) {
      // PATCH частичный: шлём только изменённое. Пустое тело бэкенд отвергает
      // кодом source.nothing_to_update, поэтому «правку без правок» закрываем сами.
      const payload = {};
      if (nextTitle !== (editingNote.title || '')) payload.title = nextTitle;
      if (nextBody !== (editingNote.body || '')) payload.body = nextBody;

      if (Object.keys(payload).length === 0) {
        closeNoteComposer();
        return;
      }

      try {
        setSavingNote(true);
        setNoteCreateError('');
        const response = await notesService.update(editingNote.id, payload);
        onNoteUpdated?.(response.data);
        // Открытый просмотр той же заметки иначе показывал бы старый текст.
        setSelectedNote((prev) => (prev && prev.id === response.data.id ? response.data : prev));
        setNoteComposerOpen(false);
        setEditingNote(null);
        setNoteTitle('');
        setNoteBody('');
      } catch (error) {
        console.error('Failed to update note', error);
        setNoteCreateError(resolveApiErrorMessage(error, t, 'notebook.updateNoteFailed'));
      } finally {
        setSavingNote(false);
      }
      return;
    }

    try {
      setSavingNote(true);
      setNoteCreateError('');
      const response = await notesService.create({
        notebook_id: currentNotebookId,
        title: nextTitle,
        body: nextBody,
      });

      onNoteCreated?.(response.data);
      setNoteComposerOpen(false);
      setNoteTitle('');
      setNoteBody('');
      setNotesCollapsed(false);
    } catch (error) {
      console.error('Failed to create note', error);
      // Ошибку показываем в модалке — там, где пользователь её ждёт, список заметок остаётся на месте.
      setNoteCreateError(resolveApiErrorMessage(error, t, 'notebook.createNoteFailed'));
    } finally {
      setSavingNote(false);
    }
  };

  const handleToggleNoteStatus = async (note) => {
    if (!note || noteStatusPendingId != null) return;

    const nextStatus = note.status === NOTE_STATUS_ARCHIVED ? NOTE_STATUS_ACTIVE : NOTE_STATUS_ARCHIVED;

    try {
      setNoteStatusPendingId(note.id);
      setNoteStatusError('');
      const response = await notesService.update(note.id, { status: nextStatus });
      onNoteUpdated?.(response.data);
      setSelectedNote((prev) => (prev && prev.id === response.data.id ? response.data : prev));
    } catch (error) {
      console.error('Failed to change note status', error);
      // Список остаётся на месте: ошибка архивации не повод прятать заметки.
      setNoteStatusError(resolveApiErrorMessage(error, t, 'notebook.archiveNoteFailed'));
    } finally {
      setNoteStatusPendingId(null);
    }
  };

  const openNoteDeleteDialog = (note) => {
    if (!note) return;
    setNoteDeleteError('');
    setNoteDeleteTarget(note);
  };

  const handleDeleteNote = async () => {
    if (!noteDeleteTarget || deletingNote) return;

    try {
      setDeletingNote(true);
      setNoteDeleteError('');
      await notesService.delete(noteDeleteTarget.id);
      onNoteDeleted?.(noteDeleteTarget.id);
      // Удалённую заметку нельзя оставить открытой в просмотре.
      setSelectedNote((prev) => (prev && prev.id === noteDeleteTarget.id ? null : prev));
      setNoteDeleteTarget(null);
    } catch (error) {
      console.error('Failed to delete note', error);
      setNoteDeleteError(resolveApiErrorMessage(error, t, 'notebook.deleteNoteFailed'));
    } finally {
      setDeletingNote(false);
    }
  };

  return (
    <div className="flex min-h-0 flex-col gap-4 xl:h-full">
      <input ref={sourceInputRef} type="file" className="hidden" multiple accept=".pdf,.docx,.txt" onChange={handleSourceUpload} />

      {/* Три колонки требуют места, которого на ноутбуке нет. Раньше строка
          держала min-w-[940px] и уезжала за край с горизонтальной прокруткой —
          вместо этого колонки складываются в столбец, пока ширина не позволит. */}
      <div className="flex flex-col gap-4 pb-1 xl:h-full xl:min-h-0 xl:flex-1 xl:flex-row">
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
              sources.map((source) => {
                const sourceStatus = resolveStatus(source.status);
                // Статус error без объяснения не подсказывает, что делать: причину берём
                // из error_code документа общей таблицей переводов.
                const sourceErrorMessage = resolveSourceErrorMessage(source, t);
                // Тема источника — подпись рядом с датой и размером. Нет поля или
                // источник не размечен — строки просто нет: «не определено» на
                // карточке рассказывало бы о состоянии модели, а не о документе.
                const sourceTopicLabel = resolveTopicLabel(source, locale);

                return (
                  <article key={source.id} className="rounded-2xl border border-slate-200 p-4 transition hover:border-slate-300 hover:bg-slate-50">
                    <div className="flex items-start justify-between gap-3">
                      <div className="min-w-0">
                        <p className="truncate text-sm font-semibold text-slate-900" title={source.name || undefined}>{source.name}</p>
                      </div>
                      <span
                        className={cn(
                          'shrink-0 rounded-full px-2.5 py-1 text-[11px] font-semibold',
                          sourceStatus === 'error' ? 'bg-red-100 text-red-700' : 'bg-slate-100 text-slate-600',
                        )}
                      >
                        {sourceStatusLabels[sourceStatus]}
                      </span>
                    </div>
                    <div className="mt-3 flex flex-wrap gap-2 text-xs text-slate-500">
                      <span>{t('notebook.createdAt', { date: formatLocaleDate(source.created_at, locale, { day: 'numeric', month: 'short', year: 'numeric' }, '—') })}</span>
                      <span>{formatSize(source.size, t)}</span>
                      {sourceTopicLabel ? (
                        <span className="inline-flex min-w-0 items-center gap-1" title={t('documents.topic', { value: sourceTopicLabel })}>
                          <Tag className="h-3.5 w-3.5 shrink-0" />
                          <span className="truncate">{sourceTopicLabel}</span>
                        </span>
                      ) : null}
                    </div>
                    {sourceErrorMessage ? (
                      /* error_text — техническая строка на одном языке (путь, имя библиотеки):
                         показываем её подсказкой при наведении, а в тексте оставляем перевод. */
                      <p
                        className="mt-3 flex items-start gap-1.5 rounded-xl bg-red-50 px-3 py-2 text-xs leading-5 text-red-600"
                        title={source.error_text || undefined}
                      >
                        <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0" />
                        <span className="min-w-0 break-words">{sourceErrorMessage}</span>
                      </p>
                    ) : null}
                  </article>
                );
              })
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
          action={openNoteComposer}
          actionLabel={t('notebook.writeNote')}
        >

          {notesLoading ? (
            <div className="flex h-full items-center justify-center text-sm text-slate-500">
              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              {t('notebook.loadingNotes')}
            </div>
          ) : notesError ? (
            /* Сбой загрузки списка: показывать вместо него нечего, поэтому ошибка занимает панель. */
            <div role="alert" className="flex flex-col gap-2 rounded-2xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-600">
              <span className="flex items-center gap-2 font-semibold">
                <AlertTriangle className="h-4 w-4" />
                {notesError}
              </span>
              {onReloadNotes ? (
                <Button type="button" variant="outline" size="sm" className="self-start" onClick={onReloadNotes}>
                  <RefreshCw className="h-4 w-4" />
                  {t('documents.retry')}
                </Button>
              ) : null}
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

              {/* Сбой архивации показываем баннером над списком: сами заметки на месте. */}
              {noteStatusError ? (
                <p role="alert" className="rounded-2xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-600">
                  {noteStatusError}
                </p>
              ) : null}

              {notes.map((note) => {
                const isArchived = note.status === NOTE_STATUS_ARCHIVED;
                const isStatusPending = noteStatusPendingId === note.id;

                return (
                  <article
                    key={note.id}
                    role="button"
                    tabIndex={0}
                    onClick={() => setSelectedNote(note)}
                    onKeyDown={(e) => e.key === 'Enter' && setSelectedNote(note)}
                    className={cn(
                      'cursor-pointer rounded-2xl border border-slate-200 p-4 transition hover:border-[#1f3a60]/40 hover:bg-slate-50',
                      isArchived && 'bg-slate-50/70',
                    )}
                  >
                    <div className="flex items-start justify-between gap-3">
                      <h3 className="min-w-0 break-words text-sm font-semibold text-slate-900">{note.title}</h3>
                      {isArchived ? (
                        <span className="shrink-0 rounded-full bg-slate-200 px-2.5 py-1 text-[11px] font-semibold text-slate-600">
                          {t('notebook.noteStatusArchived')}
                        </span>
                      ) : null}
                    </div>
                    <p className="mt-2 line-clamp-4 whitespace-pre-wrap break-words text-sm leading-6 text-slate-500">
                      {note.body || t('notebook.noteTextMissing')}
                    </p>
                    <div className="mt-3 flex items-center justify-between gap-2">
                      <span className="text-xs text-slate-400">{t('notebook.updatedAt', { date: formatLocaleDate(note.updated_at || note.created_at, locale, { day: 'numeric', month: 'short', year: 'numeric' }, '—') })}</span>
                      {/* Действия внутри карточки-кнопки: без stopPropagation каждый клик
                          заодно открывал бы просмотр заметки. */}
                      <div className="flex shrink-0 items-center gap-1">
                        <button
                          type="button"
                          onClick={(event) => { event.stopPropagation(); openNoteEditor(note); }}
                          title={t('notebook.editNote')}
                          aria-label={t('notebook.editNote')}
                          className="inline-flex h-8 w-8 items-center justify-center rounded-lg text-slate-400 transition hover:bg-slate-100 hover:text-[#1f3a60]"
                        >
                          <Pencil className="h-4 w-4" />
                        </button>
                        <button
                          type="button"
                          onClick={(event) => { event.stopPropagation(); handleToggleNoteStatus(note); }}
                          disabled={isStatusPending}
                          title={isArchived ? t('notebook.unarchiveNote') : t('notebook.archiveNote')}
                          aria-label={isArchived ? t('notebook.unarchiveNote') : t('notebook.archiveNote')}
                          className="inline-flex h-8 w-8 items-center justify-center rounded-lg text-slate-400 transition hover:bg-slate-100 hover:text-[#1f3a60] disabled:pointer-events-none disabled:opacity-60"
                        >
                          {isStatusPending ? (
                            <Loader2 className="h-4 w-4 animate-spin" />
                          ) : isArchived ? (
                            <ArchiveRestore className="h-4 w-4" />
                          ) : (
                            <Archive className="h-4 w-4" />
                          )}
                        </button>
                        <button
                          type="button"
                          onClick={(event) => { event.stopPropagation(); openNoteDeleteDialog(note); }}
                          title={t('notebook.deleteNote')}
                          aria-label={t('notebook.deleteNote')}
                          className="inline-flex h-8 w-8 items-center justify-center rounded-lg text-slate-400 transition hover:bg-red-50 hover:text-red-600"
                        >
                          <Trash2 className="h-4 w-4" />
                        </button>
                      </div>
                    </div>
                  </article>
                );
              })}
            </div>
          )}
        </NotebookSidePanel>

        {/* Пока колонки сложены, чату нужна собственная высота: полосу ввода
            и последний ответ видно без прокрутки страницы. */}
        <section className="flex h-[28rem] w-full min-w-0 overflow-hidden rounded-3xl border border-slate-200 bg-white shadow-sm xl:h-full xl:flex-1">
          <ChatPage notebookId={currentNotebookId} mode="notebookPanel" />
        </section>
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
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
          {/* Backdrop */}
          <div
            className="absolute inset-0 bg-black/40 backdrop-blur-[2px]"
            onClick={closeNoteViewer}
            aria-hidden="true"
          />

          {/* Dialog */}
          <div
            ref={noteViewDialogRef}
            role="dialog"
            aria-modal="true"
            aria-label={t('notebook.viewNote')}
            className="relative flex w-full max-w-lg flex-col overflow-hidden rounded-2xl bg-white shadow-2xl"
          >
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
                ref={noteViewCloseRef}
                type="button"
                onClick={closeNoteViewer}
                className="inline-flex h-8 w-8 shrink-0 items-center justify-center rounded-lg text-slate-400 transition hover:bg-slate-100 hover:text-slate-700"
                aria-label={t('notebook.close')}
              >
                <X className="h-4 w-4" />
              </button>
            </div>

            {/* Body */}
            <div className="max-h-[60vh] overflow-y-auto px-6 pb-6">
              {selectedNote.body ? (
                <p className="whitespace-pre-wrap break-words text-sm leading-7 text-slate-700">{selectedNote.body}</p>
              ) : (
                <p className="text-sm italic text-slate-400">{t('notebook.noteBodyMissing')}</p>
              )}
            </div>

            {/* Footer */}
            <div className="flex items-center justify-end gap-3 border-t border-slate-200 px-6 py-4">
              <div className="flex items-center gap-2">
                {/* Правку добавляем и сюда: увидев текст целиком, пользователь
                    чаще всего и хочет его поправить, а не искать кнопку в списке. */}
                <Button
                  type="button"
                  variant="outline"
                  onClick={() => {
                    const note = selectedNote;
                    closeNoteViewer();
                    openNoteEditor(note);
                  }}
                >
                  <Pencil className="h-4 w-4" />
                  {t('notebook.editNote')}
                </Button>
                <Button type="button" variant="ghost" onClick={closeNoteViewer}>
                  {t('notebook.close')}
                </Button>
              </div>
            </div>
          </div>
        </div>
      ) : null}

      {/* Note composer MODAL */}
      {noteComposerOpen ? (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
          {/* Backdrop */}
          <div
            className="absolute inset-0 bg-black/40 backdrop-blur-[2px]"
            onClick={closeNoteComposer}
            aria-hidden="true"
          />

          {/* Dialog */}
          <div
            ref={noteComposerDialogRef}
            role="dialog"
            aria-modal="true"
            aria-label={editingNote ? t('notebook.editNoteTitle') : t('notebook.writeNoteTitle')}
            className="relative flex w-full max-w-lg flex-col overflow-hidden rounded-2xl bg-white shadow-2xl"
          >
            {/* Header */}
            <div className="flex items-start justify-between gap-4 px-6 py-5">
              <div>
                <div className="flex items-center gap-2">
                  <NotebookPen className="h-4 w-4 text-[#1f3a60]" />
                  <p className="text-[15px] font-semibold text-slate-900">
                    {editingNote ? t('notebook.editNoteTitle') : t('notebook.writeNoteTitle')}
                  </p>
                </div>
                <p className="mt-1 text-sm text-slate-500">
                  {editingNote ? t('notebook.editNoteDescription') : t('notebook.writeNoteDescription')}
                </p>
              </div>
              <button
                type="button"
                onClick={closeNoteComposer}
                disabled={savingNote}
                className="inline-flex h-8 w-8 shrink-0 items-center justify-center rounded-lg text-slate-400 transition hover:bg-slate-100 hover:text-slate-700 disabled:pointer-events-none disabled:opacity-60"
                aria-label={t('notebook.close')}
              >
                <X className="h-4 w-4" />
              </button>
            </div>

            {/* Form body */}
            <form onSubmit={handleSubmitNote} className="flex flex-col gap-4 px-6 pb-6">
              <input
                ref={noteTitleInputRef}
                type="text"
                value={noteTitle}
                onChange={(e) => setNoteTitle(e.target.value)}
                placeholder={t('notebook.noteTitlePlaceholder')}
                disabled={savingNote}
                className="h-10 w-full rounded-lg border border-slate-300 bg-white px-3 text-sm text-slate-800 placeholder-slate-400 transition focus:border-[#1f3a60] focus:outline-none focus:ring-2 focus:ring-[#1f3a60]/20 disabled:cursor-not-allowed disabled:opacity-50"
              />
              <textarea
                value={noteBody}
                onChange={(e) => setNoteBody(e.target.value)}
                placeholder={t('notebook.noteBodyPlaceholder')}
                rows={6}
                disabled={savingNote}
                className="w-full resize-none rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm text-slate-800 placeholder-slate-400 transition focus:border-[#1f3a60] focus:outline-none focus:ring-2 focus:ring-[#1f3a60]/20 disabled:cursor-not-allowed disabled:opacity-50"
              />
              {noteCreateError ? (
                <div role="alert" className="rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-600">
                  {noteCreateError}
                </div>
              ) : null}
              <div className="flex items-center justify-end gap-3 border-t border-slate-200 pt-4">
                <Button
                  type="button"
                  variant="ghost"
                  onClick={closeNoteComposer}
                  disabled={savingNote}
                >
                  {t('notebook.cancel')}
                </Button>
                <Button type="submit" isLoading={savingNote} disabled={!noteTitle.trim()}>
                  {savingNote ? t('notebook.saving') : t('notebook.save')}
                </Button>
              </div>
            </form>
          </div>
        </div>
      ) : null}

      {/* Note DELETE confirmation — свой диалог вместо window.confirm, чтобы кнопки
          были на языке приложения, а ошибка удаления оставалась рядом с вопросом */}
      {noteDeleteTarget ? (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
          <div className="absolute inset-0 bg-black/50" onClick={closeNoteDeleteDialog} aria-hidden="true" />

          <div
            ref={noteDeleteDialogRef}
            role="dialog"
            aria-modal="true"
            aria-labelledby="note-delete-dialog-title"
            aria-describedby="note-delete-dialog-description"
            className="relative w-full max-w-md overflow-hidden rounded-2xl bg-white shadow-xl"
          >
            <div className="flex items-start gap-3 px-6 py-5">
              <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full bg-red-100 text-red-600">
                <AlertTriangle className="h-5 w-5" />
              </div>
              <div className="min-w-0">
                <h3 id="note-delete-dialog-title" className="text-base font-semibold text-slate-900">
                  {t('notebook.deleteNoteConfirm')}
                </h3>
                <p id="note-delete-dialog-description" className="mt-1 break-words text-sm text-slate-500">
                  {t('notebook.deleteNoteDescription', { title: noteDeleteTarget.title })}
                </p>
              </div>
            </div>

            {noteDeleteError ? (
              <p role="alert" className="mx-6 mb-2 rounded-lg bg-red-50 px-3 py-2 text-sm font-semibold text-red-700">
                {noteDeleteError}
              </p>
            ) : null}

            <div className="flex justify-end gap-2 border-t border-slate-200 px-6 py-4">
              <Button
                ref={noteDeleteCancelRef}
                type="button"
                variant="outline"
                onClick={closeNoteDeleteDialog}
                disabled={deletingNote}
              >
                {t('notebook.cancel')}
              </Button>
              <Button type="button" variant="destructive" onClick={handleDeleteNote} isLoading={deletingNote}>
                {deletingNote ? t('notebook.deleting') : t('notebook.delete')}
              </Button>
            </div>
          </div>
        </div>
      ) : null}
    </div>
  );
};

export default NotebookWorkspace;
