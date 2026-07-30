import React from 'react';
import { useOutletContext, useParams } from 'react-router-dom';

import NotebookWorkspace from '../components/notebook/NotebookWorkspace';
import { useLocale } from '../i18n';

const NotebookOverviewPage = () => {
  const { notebookId } = useParams();
  const { error, isLoading } = useOutletContext();
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

  return <NotebookWorkspace notebookId={Number(notebookId)} />;
};

export default NotebookOverviewPage;
