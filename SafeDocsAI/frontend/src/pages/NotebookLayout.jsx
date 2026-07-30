import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { Outlet, useNavigate, useParams } from 'react-router-dom';

import { useNotebookHeader } from '../components/layout/NotebookHeaderContext';
import { useSources } from '../contexts/SourcesContext';
import { useLocale } from '../i18n';
import { formatLocaleDate } from '../lib/locale';
import { notebooksService } from '../services/notebooksService';
import { notesService } from '../services/notesService';

const ACTIVE_NOTEBOOK_STORAGE_KEY = 'knowledgeai.activeNotebookId';

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
  const { locale, t } = useLocale();
  const [notebook, setNotebook] = useState(null);
  const [notes, setNotes] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState('');
  const [deleting, setDeleting] = useState(false);
  const { setNotebookHeader, setNotebookActions } = useNotebookHeader();

  // Тот же кэш, что и у панели источников: своего запроса за списком шапка не шлёт,
  // а дата «Обновлён» пересчитывается сразу после загрузки или удаления источника.
  const { items: sources } = useSources(notebookId);

  useEffect(() => {
    if (!notebookId) return;
    localStorage.setItem(ACTIVE_NOTEBOOK_STORAGE_KEY, notebookId);
  }, [notebookId]);

  useEffect(() => {
    let isMounted = true;

    const fetchNotebook = async () => {
      if (!notebookId) return;

      try {
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
        setNotes(notesResponse.status === 'fulfilled' ? (notesResponse.value.data || []) : []);
      } catch (fetchError) {
        if (!isMounted) return;
        console.error('Failed to fetch notebook', fetchError);
        setNotebook(null);
        setNotes([]);
        setError(fetchError.response?.data?.detail || t('notebookLayout.loadFailed'));
      } finally {
        if (isMounted) {
          setIsLoading(false);
        }
      }
    };

    fetchNotebook();

    return () => {
      isMounted = false;
    };
  }, [notebookId, t]);

  const lastUpdatedAt = useMemo(
    () => resolveLatestNotebookActivity(notebook, sources, notes),
    [notebook, notes, sources],
  );

  const contextValue = useMemo(
    () => ({ notebook, isLoading, error, lastUpdatedAt }),
    [notebook, isLoading, error, lastUpdatedAt],
  );

  useEffect(() => {
    if (!notebook || !notebookId) {
      setNotebookHeader(null);
      return undefined;
    }

    setNotebookHeader({
      id: notebookId,
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
      domainProfile: notebook.domain_profile || t('notebookLayout.domainProfileFallback'),
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
      navigate('/notebooks', { replace: true });
    } catch (deleteError) {
      console.error('Failed to delete notebook', deleteError);
      alert(deleteError.response?.data?.detail || t('notebookLayout.deleteFailed'));
    } finally {
      setDeleting(false);
    }
  }, [notebookId, navigate, t]);

  useEffect(() => {
    setNotebookActions({
      onArchive: undefined,
      onDelete: handleDeleteNotebook,
      archiveDisabled: true,
      deleteDisabled: deleting,
      archiveTitle: t('notebookLayout.archiveUnsupported'),
      deleteTitle: deleting ? t('notebookLayout.deleting') : t('notebookLayout.deleteNotebook'),
    });

    return () => {
      setNotebookActions(null);
    };
  }, [setNotebookActions, handleDeleteNotebook, deleting, t]);

  return (
    <div className="flex h-full min-h-0 flex-col gap-6">
      <div className="min-h-0 flex-1">
        <Outlet context={contextValue} />
      </div>
    </div>
  );
};

export default NotebookLayout;
