import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  AlertTriangle,
  Archive,
  ArchiveRestore,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  ChevronUp,
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
import FolderGlyph from '../ui/FolderGlyph';
import Input from '../ui/Input';
import { cn } from '../../lib/utils';
import { useFolderDisclosure } from '../../hooks/useFolderDisclosure';
import { useModalDialog } from '../../hooks/useModalDialog';
import { useSources, useSourcesActions } from '../../contexts/SourcesContext';
import { useLocale } from '../../i18n';
import { resolveApiErrorMessage } from '../../lib/apiError';
import { formatLocaleDate } from '../../lib/locale';
import { SOURCE_STATUS_BADGE_CLASS, formatSize, resolveSourceErrorMessage, resolveStatus } from '../../lib/sources';
import {
  TOPIC_FOLDER_AUTO_OPEN_LIMIT,
  TOPIC_FOLDER_HINT_KEY,
  buildTopicFolders,
  filterTopicFolders,
  resolveTopicFolderName,
  resolveTopicLabel,
} from '../../lib/topics';
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

/**
 * Папка в выборе источников для блокнота.
 *
 * Галочка у папки берёт её целиком — ради этого папки сюда и добавлены: «взять
 * всё законодательство» одним нажатием вместо десяти галочек подряд. Раскрытие
 * и выбор разведены по разным элементам: иначе взять папку целиком можно было бы
 * только раскрыв её, то есть заглянув внутрь того, что и так берёшь полностью.
 */
const AttachableFolder = ({
  folder,
  isOpen,
  onToggle,
  selectedIds,
  onSelectFolder,
  onToggleSource,
  notebookNameById,
  statusLabels,
  t,
}) => {
  const name = resolveTopicFolderName(folder, t);
  const hintKey = TOPIC_FOLDER_HINT_KEY[folder.kind];
  const panelId = `attachable-folder-${folder.key}`;
  const selectedCount = folder.sources.reduce((sum, source) => sum + (selectedIds.has(source.id) ? 1 : 0), 0);
  const allSelected = selectedCount > 0 && selectedCount === folder.sources.length;

  return (
    <div className="overflow-hidden rounded-xl border border-slate-200">
      <div className="flex items-stretch bg-slate-50/70">
        <input
          type="checkbox"
          // indeterminate живёт только в свойстве узла, атрибута для него нет.
          // Без него «выбрана часть папки» выглядело бы как «не выбрано ничего»,
          // и человек нажал бы галочку второй раз, сняв свой же выбор.
          ref={(element) => { if (element) element.indeterminate = selectedCount > 0 && !allSelected; }}
          checked={allSelected}
          onChange={() => onSelectFolder(folder, !allSelected)}
          aria-label={allSelected ? t('notebook.folderClear', { name }) : t('notebook.folderSelectAll', { name })}
          className="my-auto ml-4 h-4 w-4 shrink-0 rounded border-slate-300 text-[#1f3a60] focus:ring-[#1f3a60]"
        />
        <button
          type="button"
          onClick={() => onToggle(!isOpen)}
          aria-expanded={isOpen}
          aria-controls={panelId}
          className="flex min-w-0 flex-1 items-center gap-2.5 py-2.5 pl-2.5 pr-4 text-left transition hover:bg-slate-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-[#1f3a60]/40"
        >
          <ChevronRight className={cn('h-4 w-4 shrink-0 text-slate-400 transition-transform duration-150', isOpen && 'rotate-90')} />
          <FolderGlyph folder={folder} isOpen={isOpen} />
          {/* break-words, а не truncate: название темы приходит от модели и
              бывает длинным, а обрезанное имя папки — папка без имени. */}
          <span className="min-w-0 flex-1 break-words text-sm font-semibold text-slate-900">{name}</span>
          <span
            className="shrink-0 tabular-nums text-xs text-slate-500"
            title={t('notebook.folderSelected', { selected: selectedCount, count: folder.count })}
          >
            {selectedCount > 0 ? `${selectedCount} / ${folder.count}` : folder.count}
          </span>
        </button>
      </div>

      {isOpen ? (
        /* Отступ слева ставит документы под названием папки: вложенность
           должно быть видно, не вчитываясь в подписи. */
        <div id={panelId} className="space-y-1.5 border-t border-slate-200 py-1.5 pl-7 pr-1.5">
          {hintKey ? (
            <p className="px-2.5 pt-1 text-xs leading-5 text-slate-500">{t(hintKey)}</p>
          ) : null}

          {folder.sources.map((source) => {
            const isSelected = selectedIds.has(source.id);

            return (
              <label
                key={source.id}
                className={cn(
                  'flex cursor-pointer items-center gap-3 rounded-lg border px-3 py-2.5 transition',
                  isSelected
                    ? 'border-[#1f3a60] bg-[#1f3a60]/5'
                    : 'border-transparent bg-white hover:border-slate-200 hover:bg-slate-50',
                )}
              >
                <input
                  type="checkbox"
                  className="h-4 w-4 shrink-0 rounded border-slate-300 text-[#1f3a60] focus:ring-[#1f3a60]"
                  checked={isSelected}
                  onChange={() => onToggleSource(source.id)}
                />
                <div className="min-w-0 flex-1">
                  <p className="truncate text-sm font-semibold text-slate-900" title={source.name || undefined}>{source.name}</p>
                  <p className="mt-0.5 truncate text-xs text-slate-400">
                    {getNotebookLabel(source, notebookNameById, t)} · {formatSize(source.size, t)}
                  </p>
                </div>
                <span className="shrink-0 rounded-full bg-slate-100 px-2 py-0.5 text-[11px] font-semibold text-slate-600">
                  {statusLabels[resolveStatus(source.status)]}
                </span>
              </label>
            );
          })}
        </div>
      ) : null}
    </div>
  );
};

const RAIL_SOURCES = 'sources';
const RAIL_NOTES = 'notes';

/**
 * Вспомогательная полоса блокнота: источники и заметки.
 *
 * Раньше это были две одинаковые панели, стоявшие слева от чата. Каждая несла
 * свою шапку, свою кнопку сворачивания и свою кнопку действия, и вдвоём они
 * занимали больше места, чем сам чат, ради которого блокнот и открывают. Теперь
 * полоса одна, а источники и заметки — вкладки внутри неё: шапка, сворачивание и
 * кнопка действия стали общими, а счётчик у каждой вкладки виден всегда, поэтому
 * невыбранная половина не пропадает из виду.
 */
const NotebookRail = ({
  tabs,
  activeTab,
  onTabChange,
  collapsed,
  onToggle,
  expandLabel,
  collapseLabel,
  panelLabel,
  children,
}) => {
  const active = tabs.find((tab) => tab.id === activeTab) || tabs[0];

  return (
    <aside
      aria-label={panelLabel}
      className={cn(
        // Полосой панель становится только там, где рядом помещается вторая
        // колонка: до этого она занимает всю ширину и стоит под чатом, а не за
        // краем экрана.
        'flex w-full shrink-0 overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-sm transition-[width] duration-200 ease-out xl:h-full',
        collapsed ? 'xl:w-12' : 'xl:w-[20rem] 2xl:w-[22rem]',
      )}
    >
      {collapsed ? (
        <button
          type="button"
          onClick={onToggle}
          className="flex w-full items-center justify-center gap-2 bg-slate-50 px-4 py-2.5 text-slate-500 transition hover:bg-slate-100 hover:text-[#1f3a60] xl:h-full xl:flex-col xl:justify-between xl:px-0 xl:py-4"
          aria-label={expandLabel}
        >
          <ChevronDown className="h-4 w-4 xl:hidden" />
          <ChevronLeft className="hidden h-4 w-4 xl:block" />
          {/* Вертикальная надпись — только у узкой полосы: в развёрнутой на всю
              ширину строке её пришлось бы читать боком. */}
          <span className="flex items-center gap-2 xl:rotate-180 xl:[writing-mode:vertical-rl]">
            {React.createElement(active.icon, { className: 'h-4 w-4' })}
            <span className="text-xs font-semibold uppercase tracking-[0.24em]">{panelLabel}</span>
          </span>
          <span className="hidden h-4 w-4 xl:block" />
        </button>
      ) : (
        <div className="flex min-h-0 w-full flex-col xl:h-full">
          <div className="flex items-center gap-2 border-b border-slate-200 px-3 py-2.5">
            <div role="tablist" aria-label={panelLabel} className="flex min-w-0 flex-1 gap-1">
              {tabs.map((tab) => {
                const isActive = tab.id === active.id;

                return (
                  <button
                    key={tab.id}
                    type="button"
                    role="tab"
                    id={`notebook-rail-tab-${tab.id}`}
                    aria-selected={isActive}
                    aria-controls={`notebook-rail-panel-${tab.id}`}
                    onClick={() => onTabChange(tab.id)}
                    className={cn(
                      'inline-flex min-w-0 flex-1 items-center justify-center gap-1.5 rounded-lg px-2 py-1.5 text-xs font-semibold transition focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#1f3a60] focus-visible:ring-offset-1',
                      isActive ? 'bg-[#1f3a60] text-white' : 'text-slate-600 hover:bg-slate-100 hover:text-[#1f3a60]',
                    )}
                  >
                    {React.createElement(tab.icon, { className: 'h-3.5 w-3.5 shrink-0' })}
                    <span className="truncate">{tab.title}</span>
                    {/* Счётчик у невыбранной вкладки — единственный признак, что
                        за ней что-то есть: без него переключение стало бы
                        проверкой наугад. */}
                    <span className={cn('shrink-0 tabular-nums', isActive ? 'text-white/75' : 'text-slate-400')}>
                      {tab.count}
                    </span>
                  </button>
                );
              })}
            </div>

            <button
              type="button"
              onClick={onToggle}
              className="inline-flex h-8 w-8 shrink-0 items-center justify-center rounded-lg text-slate-400 transition hover:bg-slate-100 hover:text-slate-700"
              aria-label={collapseLabel}
            >
              <ChevronUp className="h-4 w-4 xl:hidden" />
              <ChevronRight className="hidden h-4 w-4 xl:block" />
            </button>
          </div>

          <div className="border-b border-slate-200 px-3 py-3">{active.action}</div>

          {/* На узком экране высота полосы не задана, поэтому список ограничиваем
              сами: иначе страница превращается в один длинный столбец. */}
          <div
            role="tabpanel"
            id={`notebook-rail-panel-${active.id}`}
            aria-labelledby={`notebook-rail-tab-${active.id}`}
            className="scrollbar-soft max-h-[55vh] min-h-0 flex-1 overflow-y-auto px-3 py-3 xl:max-h-none"
          >
            {children}
          </div>

          {active.footerLink ? (
            <div className="border-t border-slate-200 px-3 py-2.5">
              <Link to={active.footerLink.to} className="text-sm font-semibold text-[#1f3a60] transition hover:text-[#162945] hover:underline">
                {active.footerLink.label}
              </Link>
            </div>
          ) : null}
        </div>
      )}
    </aside>
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
  // Одна полоса на источники и заметки: вместо двух состояний сворачивания
  // осталось одно, а вкладка помнит, что человек смотрел последним.
  const [railCollapsed, setRailCollapsed] = useState(false);
  const [railTab, setRailTab] = useState(RAIL_SOURCES);

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

  // Загрузка и привязка источника, как и создание заметки, приводят полосу к тому
  // списку, который только что изменился: иначе результат действия оказывается за
  // невыбранной вкладкой и выглядит как «ничего не произошло».
  const revealRail = useCallback((tab) => {
    setRailTab(tab);
    setRailCollapsed(false);
  }, []);

  const notebookNameById = useMemo(
    () => Object.fromEntries((notebooks || []).map((notebook) => [notebook.id, notebook.name])),
    [notebooks],
  );

  const attachableSources = useMemo(
    () => existingSources.filter((source) => source.notebook_id !== currentNotebookId),
    [currentNotebookId, existingSources],
  );

  // Те же папки, что в разделе «Темы»: раскладку делает lib/topics, поэтому
  // документ лежит в одном и том же месте, откуда на него ни смотри.
  // Распределения тут нет — папки строятся по самим документам, и счётчик
  // папки равен числу документов в ней.
  const attachableFolders = useMemo(
    () => buildTopicFolders(attachableSources, locale),
    [attachableSources, locale],
  );

  // Поиск идёт по всем папкам сразу — и по названиям тем, и по именам файлов.
  const visibleFolders = useMemo(
    () => filterTopicFolders(attachableFolders, sourceSearch, t),
    [attachableFolders, sourceSearch, t],
  );

  // «Выбрать все» берёт ровно то, что видно: под поиском это найденное, без
  // поиска — весь список. Брать спрятанное фильтром — то же самое, что
  // подсунуть человеку документы, которых он не видел.
  const visibleSourceIds = useMemo(
    () => visibleFolders.flatMap((folder) => folder.sources.map((source) => source.id)),
    [visibleFolders],
  );

  const selectedIds = useMemo(() => new Set(selectedExistingSourceIds), [selectedExistingSourceIds]);
  const areAllVisibleSelected = visibleSourceIds.length > 0
    && visibleSourceIds.every((id) => selectedIds.has(id));

  const folderDisclosure = useFolderDisclosure(sourceSearch.trim());
  // Папки открыты сразу, пока список помещается на экран или пока идёт поиск:
  // складывать в папки десяток документов — значит прятать то, что и так видно,
  // а прятать найденное по запросу нельзя вовсе.
  const foldersOpenByDefault = Boolean(sourceSearch.trim())
    || attachableSources.length <= TOPIC_FOLDER_AUTO_OPEN_LIMIT;

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

  const handleExistingSourceSelection = useCallback((sourceId) => {
    setSelectedExistingSourceIds((prev) => (
      prev.includes(sourceId)
        ? prev.filter((id) => id !== sourceId)
        : [...prev, sourceId]
    ));
  }, []);

  // Одно правило на выбор группой: и «выбрать все», и галочка папки. Разные
  // ветки набора и снятия расходились бы ровно в том, что выбранное вне группы
  // трогать нельзя.
  const setSourcesSelection = useCallback((sourceIds, selected) => {
    setSelectedExistingSourceIds((prev) => (selected
      ? Array.from(new Set([...prev, ...sourceIds]))
      : prev.filter((id) => !sourceIds.includes(id))));
  }, []);

  const handleSelectFolder = useCallback((folder, selected) => {
    setSourcesSelection(folder.sources.map((source) => source.id), selected);
  }, [setSourcesSelection]);

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
        revealRail(RAIL_SOURCES);
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
      revealRail(RAIL_SOURCES);
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
      revealRail(RAIL_NOTES);
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

  const sourcesContent = (
    <div className="space-y-3" aria-live="polite">
      {uploadError ? (
        <p role="alert" className="rounded-xl border border-red-200 bg-red-50 px-3 py-2.5 text-sm text-red-600">
          {uploadError}
        </p>
      ) : null}

      {/* Сбой загрузки списка показываем баннером: полоса с уже полученными источниками остаётся на месте. */}
      {sourcesError ? (
        <div role="alert" className="flex flex-col gap-2 rounded-xl border border-red-200 bg-red-50 px-3 py-2.5 text-sm text-red-600">
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
        <div className="flex items-center justify-center py-6 text-sm text-slate-500">
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
            <article key={source.id} className="rounded-xl border border-slate-200 p-3 transition hover:border-slate-300 hover:bg-slate-50">
              <div className="flex items-start justify-between gap-2">
                <p className="min-w-0 truncate text-sm font-semibold text-slate-900" title={source.name || undefined}>{source.name}</p>
                <span
                  className={cn(
                    'shrink-0 rounded-full px-2 py-0.5 text-[11px] font-semibold',
                    SOURCE_STATUS_BADGE_CLASS[sourceStatus],
                  )}
                >
                  {sourceStatusLabels[sourceStatus]}
                </span>
              </div>
              <div className="mt-2 flex flex-wrap gap-x-2 gap-y-1 text-xs text-slate-500">
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
                  className="mt-2 flex items-start gap-1.5 rounded-lg bg-red-50 px-2.5 py-1.5 text-xs leading-5 text-red-600"
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
  );

  const notesContent = notesLoading ? (
    <div className="flex items-center justify-center py-6 text-sm text-slate-500">
      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
      {t('notebook.loadingNotes')}
    </div>
  ) : notesError ? (
    /* Сбой загрузки списка: показывать вместо него нечего, поэтому ошибка занимает полосу. */
    <div role="alert" className="flex flex-col gap-2 rounded-xl border border-red-200 bg-red-50 px-3 py-2.5 text-sm text-red-600">
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
      {/* Сбой архивации показываем баннером над списком: сами заметки на месте. */}
      {noteStatusError ? (
        <p role="alert" className="rounded-xl border border-red-200 bg-red-50 px-3 py-2.5 text-sm text-red-600">
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
              'cursor-pointer rounded-xl border border-slate-200 p-3 transition hover:border-[#1f3a60]/40 hover:bg-slate-50',
              isArchived && 'bg-slate-50/70',
            )}
          >
            <div className="flex items-start justify-between gap-2">
              <h3 className="min-w-0 break-words text-sm font-semibold text-slate-900">{note.title}</h3>
              {isArchived ? (
                <span className="shrink-0 rounded-full bg-slate-200 px-2 py-0.5 text-[11px] font-semibold text-slate-600">
                  {t('notebook.noteStatusArchived')}
                </span>
              ) : null}
            </div>
            {/* Две строки вместо четырёх: в узкой полосе превью на четверть
                экрана вытесняло соседние заметки, а текст целиком открывается
                одним кликом по карточке. */}
            <p className="mt-1.5 line-clamp-2 whitespace-pre-wrap break-words text-sm leading-6 text-slate-500">
              {note.body || t('notebook.noteTextMissing')}
            </p>
            <div className="mt-2 flex items-center justify-between gap-2">
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
                  className="inline-flex h-8 w-8 items-center justify-center rounded-lg text-slate-400 transition hover:bg-slate-100 hover:text-red-600"
                >
                  <Trash2 className="h-4 w-4" />
                </button>
              </div>
            </div>
          </article>
        );
      })}
    </div>
  );

  const railTabs = [
    {
      id: RAIL_SOURCES,
      title: t('notebook.sources'),
      icon: FileText,
      count: sources.length,
      action: (
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
      ),
      footerLink: { to: `/notebooks/${notebookId}/sources`, label: t('notebook.openAllSources') },
    },
    {
      id: RAIL_NOTES,
      title: t('notebook.notes'),
      icon: NotebookPen,
      count: notes.length,
      action: (
        <Button type="button" onClick={openNoteComposer} className="w-full justify-center">
          <Plus className="h-4 w-4" />
          {t('notebook.writeNote')}
        </Button>
      ),
    },
  ];

  const activeRailTitle = railTab === RAIL_NOTES ? t('notebook.notes') : t('notebook.sources');

  return (
    <div className="flex min-h-0 flex-col gap-4 xl:h-full xl:flex-row">
      <input ref={sourceInputRef} type="file" className="hidden" multiple accept=".pdf,.docx,.txt" onChange={handleSourceUpload} />

      {/* Чат стоит первым и забирает всё свободное место: блокнот открывают,
          чтобы спросить по его документам, а не чтобы разглядывать списки.
          Источники и заметки ушли в полосу справа, а на узком экране — под чат,
          где раньше стояли над ним и отодвигали поле вопроса за нижний край. */}
      <section
        aria-label={t('chat.notebookChatTitle')}
        className="flex h-[68vh] min-h-[22rem] w-full min-w-0 overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-sm xl:h-full xl:min-h-0 xl:flex-1"
      >
        <ChatPage notebookId={currentNotebookId} mode="notebookPanel" />
      </section>

      <NotebookRail
        tabs={railTabs}
        activeTab={railTab}
        onTabChange={setRailTab}
        collapsed={railCollapsed}
        onToggle={() => setRailCollapsed((prev) => !prev)}
        expandLabel={t('notebook.expandPanel', { title: activeRailTitle })}
        collapseLabel={t('notebook.collapsePanel', { title: activeRailTitle })}
        panelLabel={t('notebook.panelLabel')}
      >
        {railTab === RAIL_NOTES ? notesContent : sourcesContent}
      </NotebookRail>

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
                {visibleSourceIds.length > 0 ? (
                  <Button
                    type="button"
                    variant="outline"
                    onClick={() => setSourcesSelection(visibleSourceIds, !areAllVisibleSelected)}
                    className="h-10 shrink-0 px-3"
                  >
                    {areAllVisibleSelected ? t('notebook.resetAll') : t('notebook.selectAll')}
                  </Button>
                ) : null}
              </div>
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
              ) : existingSourcesError && existingSources.length === 0 ? null : visibleFolders.length === 0 ? (
                <div className="flex h-48 flex-col items-center justify-center gap-2 text-sm text-slate-500">
                  <FileText className="h-8 w-8 text-slate-300" />
                  {attachableSources.length === 0 ? t('notebook.noAttachableSources') : t('notebook.nothingFound')}
                </div>
              ) : (
                <div className="space-y-2 py-2">
                  {visibleFolders.map((folder) => (
                    <AttachableFolder
                      key={folder.key}
                      folder={folder}
                      isOpen={folderDisclosure.isOpen(folder.key, foldersOpenByDefault)}
                      onToggle={(open) => folderDisclosure.setOpen(folder.key, open)}
                      selectedIds={selectedIds}
                      onSelectFolder={handleSelectFolder}
                      onToggleSource={handleExistingSourceSelection}
                      notebookNameById={notebookNameById}
                      statusLabels={sourceStatusLabels}
                      t={t}
                    />
                  ))}
                </div>
              )}
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
