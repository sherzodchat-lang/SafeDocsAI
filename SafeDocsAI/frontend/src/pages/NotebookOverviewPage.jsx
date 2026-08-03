import React from 'react';
import { useOutletContext, useParams } from 'react-router-dom';

import NotebookWorkspace from '../components/notebook/NotebookWorkspace';
import { useLocale } from '../i18n';

const NotebookOverviewPage = () => {
  const { notebookId } = useParams();
  const {
    error,
    isLoading,
    notes,
    notesLoading,
    notesError,
    reloadNotes,
    addNote,
    updateNote,
    removeNote,
  } = useOutletContext();
  const { t } = useLocale();

  if (isLoading) {
    return (
      <section className="rounded-3xl border border-slate-200 bg-white p-6 text-sm text-slate-500 shadow-sm">
        {t('notebookOverview.loading')}
      </section>
    );
  }

  if (error) {
    return (
      <section className="rounded-3xl border border-red-200 bg-red-50 p-6 text-sm text-red-600 shadow-sm">
        {error}
      </section>
    );
  }

  // Заметки берём из общего состояния блокнота: тот же список считает дату «Обновлён» в шапке.
  return (
    <NotebookWorkspace
      notebookId={Number(notebookId)}
      notes={notes}
      notesLoading={notesLoading}
      notesError={notesError}
      onReloadNotes={reloadNotes}
      onNoteCreated={addNote}
      onNoteUpdated={updateNote}
      onNoteDeleted={removeNote}
    />
  );
};

export default NotebookOverviewPage;
