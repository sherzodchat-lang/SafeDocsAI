import React, { useEffect, useState } from 'react';
import { NotebookPen } from 'lucide-react';

import { notesService } from '../services/notesService';
import { Button } from '../components/ui/Button';
import Input from '../components/ui/Input';


const ACTIVE_NOTEBOOK_STORAGE_KEY = 'knowledgeai.activeNotebookId';


const resolveNotebookId = (notebookId) => {
  if (notebookId !== undefined) {
    return notebookId == null ? null : Number(notebookId);
  }

  const storedValue = localStorage.getItem(ACTIVE_NOTEBOOK_STORAGE_KEY);
  return storedValue ? Number(storedValue) : null;
};


const NotesPage = ({ notebookId }) => {
  const activeNotebookId = resolveNotebookId(notebookId);
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
        setError('Не удалось загрузить notes.');
      }
    };

    fetchNotes();
  }, [activeNotebookId]);

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
      setError(saveError.response?.data?.detail || 'Не удалось создать note.');
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
            <h2 className="text-lg font-semibold text-slate-900">Notes</h2>
            <p className="text-sm text-slate-500">Здесь будут рабочие заметки по notebook и сохраненные результаты Ask.</p>
          </div>
        </div>

        <div className="rounded-xl bg-slate-50 p-4 text-sm text-slate-600">
          Активный notebook: <span className="font-semibold text-slate-800">{activeNotebookId || 'default'}</span>
        </div>
        <form onSubmit={handleCreateNote} className="mt-4 space-y-3">
          <Input value={title} onChange={(e) => setTitle(e.target.value)} placeholder="Note title" />
          <textarea
            value={body}
            onChange={(e) => setBody(e.target.value)}
            placeholder="Write note body..."
            className="min-h-28 w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm text-slate-800 shadow-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#1f3a60]/30"
          />
          <div className="flex justify-end">
            <Button type="submit" isLoading={isSaving} disabled={!activeNotebookId}>Create note</Button>
          </div>
        </form>
        {error ? <p className="mt-3 text-sm text-red-600">{error}</p> : null}
      </section>

      <section className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
        <h3 className="mb-4 text-base font-semibold text-slate-900">Saved Notes</h3>
        {notes.length === 0 ? (
          <p className="text-sm text-slate-500">Заметок пока нет.</p>
        ) : (
          <div className="space-y-3">
            {notes.map((note) => (
              <div key={note.id} className="rounded-xl border border-slate-200 p-4">
                <div className="text-sm font-semibold text-slate-900">{note.title}</div>
                <div className="mt-1 whitespace-pre-wrap text-sm text-slate-600">{note.body || 'No content'}</div>
              </div>
            ))}
          </div>
        )}
      </section>
    </div>
  );
};


export default NotesPage;
