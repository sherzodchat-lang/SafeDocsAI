import React, { useEffect, useState } from 'react';
import { NotebookPen } from 'lucide-react';

import { notesService } from '../services/notesService';
import { Button } from '../components/ui/Button';
import Input from '../components/ui/Input';
import { useActiveNotebookScope } from '../hooks/useActiveNotebookScope';
import { useLocale } from '../i18n';
import { resolveApiErrorMessage } from '../lib/apiError';


const NotesPage = ({ notebookId }) => {
  const { t } = useLocale();
  // Тот же хук, что у чата и списка источников: он же приносит имя блокнота —
  // по номеру из localStorage пользователь всё равно не узнаёт, чей это блокнот.
  const { notebookId: activeNotebookId, notebookName } = useActiveNotebookScope(notebookId);
  const [notes, setNotes] = useState([]);
  const [title, setTitle] = useState('');
  const [body, setBody] = useState('');
  const [error, setError] = useState('');
  const [isSaving, setIsSaving] = useState(false);

  useEffect(() => {
    const fetchNotes = async () => {
      try {
        const response = await notesService.getAll(activeNotebookId);
        setNotes(response.data || []);
      } catch (fetchError) {
        console.error('Failed to fetch notes', fetchError);
        setError(resolveApiErrorMessage(fetchError, t, 'notesPage.loadFailed'));
      }
    };

    fetchNotes();
  }, [activeNotebookId, t]);

  const handleCreateNote = async (event) => {
    event.preventDefault();
    if (!activeNotebookId || !title.trim()) return;
    try {
      setIsSaving(true);
      setError('');
      const response = await notesService.create({
        notebook_id: activeNotebookId,
        title: title.trim(),
        body,
      });
      setNotes((prev) => [response.data, ...prev]);
      setTitle('');
      setBody('');
    } catch (saveError) {
      console.error('Failed to create note', saveError);
      setError(resolveApiErrorMessage(saveError, t, 'notesPage.createFailed'));
    } finally {
      setIsSaving(false);
    }
  };

  return (
    <div className="space-y-6 px-4">
      <section className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
        <div className="mb-4 flex items-center gap-3">
          <div className="rounded-xl bg-[#1f3a60]/10 p-2 text-[#1f3a60]">
            <NotebookPen className="h-5 w-5" />
          </div>
          <div>
            <h2 className="text-lg font-semibold text-slate-900">{t('notesPage.title')}</h2>
            <p className="text-sm text-slate-500">{t('notesPage.description')}</p>
          </div>
        </div>

        {/* Строку собирает словарь целиком: порядок слов и знак после метки в ru и tg разный. */}
        <div className="rounded-xl bg-slate-50 p-4 text-sm font-semibold text-slate-700">
          {/* Пока имя не пришло, показываем «#id» — тем же способом, что бейдж области
              чата: это заметно временное значение, а не выдача номера за название. */}
          {t('notesPage.activeNotebook', {
            value: activeNotebookId == null
              ? t('notesPage.notebookNotSelected')
              : notebookName || `#${activeNotebookId}`,
          })}
          {/* Активный блокнот больше не проставляется сам при заходе в блокнот, поэтому
              «не выбран» — обычное состояние. Без пояснения пользователь видит только
              выключенную кнопку создания и не понимает, чего не хватает. */}
          {!activeNotebookId ? (
            <p className="mt-1 font-normal text-slate-500">{t('notesPage.notebookHint')}</p>
          ) : null}
        </div>
        <form onSubmit={handleCreateNote} className="mt-4 space-y-3">
          <Input value={title} onChange={(e) => setTitle(e.target.value)} placeholder={t('notesPage.titlePlaceholder')} />
          <textarea
            value={body}
            onChange={(e) => setBody(e.target.value)}
            placeholder={t('notesPage.bodyPlaceholder')}
            className="min-h-28 w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm text-slate-800 shadow-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#1f3a60]/30"
          />
          <div className="flex justify-end">
            <Button type="submit" isLoading={isSaving} disabled={!activeNotebookId}>{t('notesPage.create')}</Button>
          </div>
        </form>
        {error ? <p role="alert" className="mt-3 text-sm text-red-600">{error}</p> : null}
      </section>

      <section className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
        <h3 className="mb-4 text-base font-semibold text-slate-900">{t('notesPage.savedTitle')}</h3>
        {notes.length === 0 ? (
          <p className="text-sm text-slate-500">{t('notesPage.empty')}</p>
        ) : (
          <div className="space-y-3">
            {notes.map((note) => (
              <div key={note.id} className="rounded-xl border border-slate-200 p-4">
                <div className="break-words text-sm font-semibold text-slate-900">{note.title}</div>
                <div className="mt-1 whitespace-pre-wrap break-words text-sm text-slate-600">{note.body || t('notesPage.noContent')}</div>
              </div>
            ))}
          </div>
        )}
      </section>
    </div>
  );
};


export default NotesPage;
