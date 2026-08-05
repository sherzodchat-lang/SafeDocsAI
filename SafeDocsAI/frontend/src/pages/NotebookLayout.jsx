import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Link, Outlet, useLocation, useNavigate, useParams } from 'react-router-dom';

import { useNotebookHeader } from '../components/layout/NotebookHeaderContext';
import NotebookEditDialog from '../components/notebook/NotebookEditDialog';
import { useSources, useSourcesActions } from '../contexts/SourcesContext';
import { useSessionRole } from '../hooks/useSessionRole';
import { useLocale } from '../i18n';
import { cn } from '../lib/utils';
import { resolveApiErrorMessage } from '../lib/apiError';
import { formatLocaleDate } from '../lib/locale';
import { resolveDomainProfileLabel } from '../lib/notebooks';
import { notebooksService } from '../services/notebooksService';
import { notesService } from '../services/notesService';

const ACTIVE_NOTEBOOK_STORAGE_KEY = 'knowledgeai.activeNotebookId';
const CHAT_HISTORY_STORAGE_PREFIX = 'knowledgeai.chat.history.';

/**
 * Локальное состояние удалённого блокнота: активный id читают Ask, Insights, Notes и Chat,
 * а история чата лежит по ключу вида knowledgeai.chat.history.<user>.notebook.<id>,
 * поэтому её ищем по суффиксу — имя пользователя в ключе может быть любым.
 */
const clearNotebookLocalState = (notebookId) => {
  if (localStorage.getItem(ACTIVE_NOTEBOOK_STORAGE_KEY) === String(notebookId)) {
    localStorage.removeItem(ACTIVE_NOTEBOOK_STORAGE_KEY);
  }

  const chatHistorySuffix = `.notebook.${notebookId}`;
  Object.keys(localStorage)
    .filter((key) => key.startsWith(CHAT_HISTORY_STORAGE_PREFIX) && key.endsWith(chatHistorySuffix))
    .forEach((key) => localStorage.removeItem(key));
};

/**
 * Разделы блокнота. Маршруты у них были и раньше, а перехода — нет: в источники
 * и заметки вели ссылки со дна панелей рабочего пространства, и найти их можно
 * было, только зная, что они там есть. Презентации так не открыть вовсе,
 * поэтому разделы собраны в одну полосу вкладок над содержимым.
 *
 * contentOnly — вкладка контент-менеджера: пользовательской роли она не
 * рисуется. Это не защита (её держит сервер и гард маршрута), а отсутствие
 * тупика: ссылка, ведущая на «недостаточно прав», — не пункт меню.
 *
 * Вкладки «Заметки» здесь нет: страница блокнота была беднее панели заметок в
 * «Обзоре» — ни правки, ни архивации, ни удаления, ни статуса и дат. Две точки
 * работы с одними и теми же заметками, из которых одна умеет меньше, — повод
 * гадать, где работать; работа с заметками осталась в «Обзоре».
 */
const NOTEBOOK_TABS = [
  { path: '', labelKey: 'notebookTabs.overview' },
  { path: 'sources', labelKey: 'notebookTabs.sources' },
  { path: 'presentations', labelKey: 'notebookTabs.presentations', contentOnly: true },
];

const readActiveNotebookId = () => {
  if (typeof window === 'undefined') return null;
  return localStorage.getItem(ACTIVE_NOTEBOOK_STORAGE_KEY) || null;
};

const resolveLatestNotebookActivity = (notebook, sources, notes) => {
  const candidates = [notebook?.created_at];

  (sources || []).forEach((source) => {
    candidates.push(source?.created_at);
  });

  (notes || []).forEach((note) => {
    candidates.push(note?.updated_at || note?.created_at);
  });

  const timestamps = candidates
    .map((value) => new Date(value).getTime())
    .filter((value) => Number.isFinite(value));

  if (timestamps.length === 0) return null;

  return new Date(Math.max(...timestamps)).toISOString();
};

const NotebookLayout = () => {
  const { notebookId } = useParams();
  const navigate = useNavigate();
  const location = useLocation();
  const { locale, t } = useLocale();
  const { canManageContent } = useSessionRole();
  const [notebook, setNotebook] = useState(null);
  const [notes, setNotes] = useState([]);
  const [notesLoading, setNotesLoading] = useState(true);
  const [notesError, setNotesError] = useState('');
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState('');
  const [deleting, setDeleting] = useState(false);
  const [isEditOpen, setIsEditOpen] = useState(false);
  const { setNotebookHeader, setNotebookActions } = useNotebookHeader();

  // Тот же кэш, что и у панели источников: своего запроса за списком шапка не шлёт,
  // а дата «Обновлён» пересчитывается сразу после загрузки или удаления источника.
  const { items: sources } = useSources(notebookId);
  const { invalidate } = useSourcesActions();

  // Ответ на запрос по прежнему блокноту не должен попасть в состояние нового.
  const notebookIdRef = useRef(notebookId);

  useEffect(() => {
    notebookIdRef.current = notebookId;
  }, [notebookId]);

  // Активный блокнот задаётся только кнопкой в шапке. Раньше он записывался на
  // любом заходе на /notebooks/:id, и просмотр заметки молча сужал область
  // глобального чата: пользователь выходил из блокнота, а чат продолжал
  // отвечать только по нему. Со слов такое не отлаживается.
  const [activeNotebookId, setActiveNotebookId] = useState(readActiveNotebookId);

  // Тот же ключ правят другая вкладка и сама страница чата: без слушателя
  // кнопка в шапке показывала бы состояние, которого уже нет.
  useEffect(() => {
    // event.key === null — localStorage.clear() в другой вкладке.
    const handleStorage = (event) => {
      if (event.key === ACTIVE_NOTEBOOK_STORAGE_KEY || event.key === null) {
        setActiveNotebookId(readActiveNotebookId());
      }
    };

    window.addEventListener('storage', handleStorage);
    return () => window.removeEventListener('storage', handleStorage);
  }, []);

  const isActiveForChat = Boolean(notebookId) && String(activeNotebookId) === String(notebookId);

  useEffect(() => {
    let isMounted = true;

    const fetchNotebook = async () => {
      if (!notebookId) return;

      try {
        // Смена блокнота: данные прошлого убираем сразу, иначе рядом с новым id
        // в шапке ещё несколько секунд висят имя, описание, даты и профиль предыдущего.
        setNotebook(null);
        setNotes([]);
        setNotesError('');
        setNotesLoading(true);
        setIsLoading(true);
        setError('');
        const [notebookResponse, notesResponse] = await Promise.allSettled([
          notebooksService.getById(notebookId),
          notesService.getAll(notebookId),
        ]);

        if (!isMounted) return;

        if (notebookResponse.status !== 'fulfilled') {
          throw notebookResponse.reason;
        }

        setNotebook(notebookResponse.value.data || null);

        if (notesResponse.status === 'fulfilled') {
          setNotes(notesResponse.value.data || []);
        } else {
          // Молча подставлять пустой список нельзя: по заметкам считается дата «Обновлён»,
          // а панель заметок покажет ошибку вместо ложного «заметок пока нет».
          console.error('Failed to fetch notebook notes', notesResponse.reason);
          setNotes([]);
          setNotesError(resolveApiErrorMessage(notesResponse.reason, t, 'notebook.notesLoadFailed'));
        }
      } catch (fetchError) {
        if (!isMounted) return;
        console.error('Failed to fetch notebook', fetchError);
        setNotebook(null);
        setNotes([]);
        setError(resolveApiErrorMessage(fetchError, t, 'notebookLayout.loadFailed'));
      } finally {
        if (isMounted) {
          setIsLoading(false);
          setNotesLoading(false);
        }
      }
    };

    fetchNotebook();

    return () => {
      isMounted = false;
    };
  }, [notebookId, t]);

  // Заметки блокнота живут здесь: рабочее пространство читает тот же список,
  // поэтому запрос уходит один, а дата «Обновлён» меняется сразу после создания заметки.
  const reloadNotes = useCallback(async () => {
    if (!notebookId) return;

    try {
      setNotesLoading(true);
      setNotesError('');
      const response = await notesService.getAll(notebookId);
      if (notebookIdRef.current !== notebookId) return;
      setNotes(response.data || []);
    } catch (notesFetchError) {
      if (notebookIdRef.current !== notebookId) return;
      console.error('Failed to fetch notebook notes', notesFetchError);
      setNotes([]);
      setNotesError(resolveApiErrorMessage(notesFetchError, t, 'notebook.notesLoadFailed'));
    } finally {
      if (notebookIdRef.current === notebookId) {
        setNotesLoading(false);
      }
    }
  }, [notebookId, t]);

  const addNote = useCallback((note) => {
    if (!note) return;
    setNotes((prev) => (prev.some((item) => item.id === note.id) ? prev : [note, ...prev]));
  }, []);

  // Правка и архивация возвращают заметку целиком: подменяем её на месте, чтобы
  // порядок в панели не прыгал, а дата «Обновлён» в шапке пересчиталась сразу.
  const updateNote = useCallback((note) => {
    if (!note) return;
    setNotes((prev) => prev.map((item) => (item.id === note.id ? note : item)));
  }, []);

  const removeNote = useCallback((noteId) => {
    if (noteId == null) return;
    setNotes((prev) => prev.filter((item) => item.id !== noteId));
  }, []);

  const lastUpdatedAt = useMemo(
    () => resolveLatestNotebookActivity(notebook, sources, notes),
    [notebook, notes, sources],
  );

  const contextValue = useMemo(
    () => ({
      notebook,
      isLoading,
      error,
      lastUpdatedAt,
      notes,
      notesLoading,
      notesError,
      reloadNotes,
      addNote,
      updateNote,
      removeNote,
    }),
    [
      notebook,
      isLoading,
      error,
      lastUpdatedAt,
      notes,
      notesLoading,
      notesError,
      reloadNotes,
      addNote,
      updateNote,
      removeNote,
    ],
  );

  useEffect(() => {
    if (!notebook || !notebookId) {
      setNotebookHeader(null);
      return undefined;
    }

    setNotebookHeader({
      name: notebook.name || t('notebookLayout.defaultName'),
      description: notebook.description || '',
      createdAtText: formatLocaleDate(notebook.created_at, locale, {
        day: 'numeric',
        month: 'long',
        year: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
      }, '—'),
      updatedAtText: formatLocaleDate(lastUpdatedAt || notebook.created_at, locale, {
        day: 'numeric',
        month: 'long',
        year: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
      }, '—'),
      domainProfile: resolveDomainProfileLabel(notebook.domain_profile, t) || t('notebookLayout.domainProfileFallback'),
    });

    return () => {
      setNotebookHeader(null);
    };
  }, [locale, notebook, notebookId, lastUpdatedAt, setNotebookHeader, t]);

  const handleDeleteNotebook = useCallback(async () => {
    if (!notebookId) return;
    if (!window.confirm(t('notebookLayout.deleteConfirm'))) return;
    try {
      setDeleting(true);
      await notebooksService.delete(notebookId);
      // Мёртвый активный id и история чата удалённого блокнота иначе остаются
      // в localStorage и приводят другие экраны к 404.
      clearNotebookLocalState(notebookId);
      navigate('/notebooks', { replace: true });
      // Кэш источников общий: без инвалидации «Все источники» ещё до 15 секунд
      // показывают документы удалённого блокнота.
      await invalidate();
    } catch (deleteError) {
      console.error('Failed to delete notebook', deleteError);
      alert(resolveApiErrorMessage(deleteError, t, 'notebookLayout.deleteFailed'));
    } finally {
      setDeleting(false);
    }
  }, [notebookId, navigate, invalidate, t]);

  // Одна кнопка на оба хода: повторное назначение уже активного блокнота
  // ничего не меняет, поэтому в этом состоянии она предлагает сброс.
  const handleToggleActiveForChat = useCallback(() => {
    if (!notebookId) return;

    if (isActiveForChat) {
      localStorage.removeItem(ACTIVE_NOTEBOOK_STORAGE_KEY);
      setActiveNotebookId(null);
      return;
    }

    localStorage.setItem(ACTIVE_NOTEBOOK_STORAGE_KEY, String(notebookId));
    setActiveNotebookId(String(notebookId));
  }, [isActiveForChat, notebookId]);

  const handleOpenEdit = useCallback(() => {
    setIsEditOpen(true);
  }, []);

  const handleCloseEdit = useCallback(() => {
    setIsEditOpen(false);
  }, []);

  // Ответ PATCH — тот же объект, что у GET: кладём его в состояние, и шапка
  // берёт новое имя из того же эффекта, что и при первой загрузке.
  const handleNotebookSaved = useCallback((updated) => {
    if (!updated) return;
    setNotebook(updated);
  }, []);

  useEffect(() => {
    setNotebookActions({
      onToggleActiveForChat: notebook ? handleToggleActiveForChat : undefined,
      isActiveForChat,
      toggleActiveForChatDisabled: !notebook || deleting,
      toggleActiveForChatTitle: isActiveForChat
        ? t('notebookLayout.resetActiveForChat')
        : t('notebookLayout.makeActiveForChat'),
      onEdit: notebook ? handleOpenEdit : undefined,
      onDelete: handleDeleteNotebook,
      editDisabled: !notebook || deleting,
      deleteDisabled: deleting,
      editTitle: t('notebookLayout.editNotebook'),
      deleteTitle: deleting ? t('notebookLayout.deleting') : t('notebookLayout.deleteNotebook'),
    });

    return () => {
      setNotebookActions(null);
    };
  }, [
    setNotebookActions,
    handleDeleteNotebook,
    handleOpenEdit,
    handleToggleActiveForChat,
    isActiveForChat,
    notebook,
    deleting,
    t,
  ]);

  const notebookBasePath = notebookId ? `/notebooks/${notebookId}` : '';
  const visibleTabs = NOTEBOOK_TABS.filter((tab) => !tab.contentOnly || canManageContent);

  return (
    <div className="flex h-full min-h-0 flex-col gap-4">
      {notebookBasePath ? (
        <nav aria-label={t('notebookTabs.label')} className="flex flex-wrap gap-2">
          {visibleTabs.map((tab) => {
            const href = tab.path ? `${notebookBasePath}/${tab.path}` : notebookBasePath;
            // Обзор — индексный маршрут, поэтому сравнение точное: без него он
            // подсвечивался бы вместе с любой вложенной вкладкой.
            const isActive = tab.path
              ? location.pathname === href || location.pathname.startsWith(`${href}/`)
              : location.pathname === notebookBasePath || location.pathname === `${notebookBasePath}/`;

            return (
              <Link
                key={tab.labelKey}
                to={href}
                aria-current={isActive ? 'page' : undefined}
                className={cn(
                  'rounded-lg border px-3.5 py-1.5 text-sm font-semibold transition focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#1f3a60]/40 focus-visible:ring-offset-2',
                  isActive
                    ? 'border-[#1f3a60] bg-[#1f3a60] text-white'
                    : 'border-slate-200 bg-white text-slate-600 hover:border-slate-300 hover:text-[#1f3a60]',
                )}
              >
                {t(tab.labelKey)}
              </Link>
            );
          })}
        </nav>
      ) : null}

      <div className="min-h-0 flex-1">
        <Outlet context={contextValue} />
      </div>

      <NotebookEditDialog
        notebook={notebook}
        isOpen={isEditOpen}
        onClose={handleCloseEdit}
        onSaved={handleNotebookSaved}
      />
    </div>
  );
};

export default NotebookLayout;
