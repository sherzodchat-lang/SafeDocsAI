import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { Bookmark, CheckCircle2, Plus } from 'lucide-react';
import { useNavigate } from 'react-router-dom';

import { Button } from '../components/ui/Button';
import Input from '../components/ui/Input';
import { useLocale } from '../i18n';
import { notebooksService } from '../services/notebooksService';


const ACTIVE_NOTEBOOK_STORAGE_KEY = 'knowledgeai.activeNotebookId';


const NotebooksPage = () => {
  const navigate = useNavigate();
  const { t } = useLocale();
  const [notebooks, setNotebooks] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [domainProfile, setDomainProfile] = useState('general');
  const [error, setError] = useState('');
  const [activeNotebookId, setActiveNotebookId] = useState(() => localStorage.getItem(ACTIVE_NOTEBOOK_STORAGE_KEY) || '');

  const fetchNotebooks = useCallback(async () => {
    try {
      setError('');
      const response = await notebooksService.getAll();
      setNotebooks(response.data || []);
    } catch (fetchError) {
      console.error('Failed to fetch notebooks', fetchError);
      setError(fetchError.response?.data?.detail || t('notebooksPage.loadFailed'));
    } finally {
      setIsLoading(false);
    }
  }, [t]);

  useEffect(() => {
    fetchNotebooks();
  }, [fetchNotebooks]);

  const activeNotebook = useMemo(
    () => notebooks.find((item) => String(item.id) === String(activeNotebookId)) || null,
    [notebooks, activeNotebookId],
  );

  const handleSelectNotebook = (notebookId) => {
    const nextValue = String(notebookId);
    setActiveNotebookId(nextValue);
    localStorage.setItem(ACTIVE_NOTEBOOK_STORAGE_KEY, nextValue);
  };

  const handleOpenNotebook = (notebookId) => {
    handleSelectNotebook(notebookId);
    navigate(`/notebooks/${notebookId}`);
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    if (!name.trim()) return;

    try {
      setError('');
      const response = await notebooksService.create({
        name: name.trim(),
        description: description.trim() || null,
        domain_profile: domainProfile,
      });
      const created = response.data;
      setNotebooks((prev) => [created, ...prev]);
      handleOpenNotebook(created.id);
      setName('');
      setDescription('');
      setDomainProfile('general');
    } catch (submitError) {
      console.error('Failed to create notebook', submitError);
      setError(submitError.response?.data?.detail || t('notebooksPage.createFailed'));
    }
  };

  const getDomainProfileLabel = (value) => {
    const key = `notebooksPage.profiles.${value}`;
    const translated = t(key);
    return translated === key ? value : translated;
  };

  return (
    <div className="space-y-6">
      <div className="rounded-2xl bg-white p-6 shadow-sm">
        <div className="mb-4 flex items-center gap-3">
          <div className="rounded-xl bg-[#1f3a60]/10 p-2 text-[#1f3a60]">
            <Bookmark className="h-5 w-5" />
          </div>
          <div>
            <h2 className="text-lg font-semibold text-slate-900">{t('notebooksPage.title')}</h2>
            <p className="text-sm text-slate-500">{t('notebooksPage.description')}</p>
          </div>
        </div>

        <form className="grid gap-3 md:grid-cols-4" onSubmit={handleSubmit}>
          <Input value={name} onChange={(event) => setName(event.target.value)} placeholder={t('notebooksPage.namePlaceholder')} />
          <Input value={description} onChange={(event) => setDescription(event.target.value)} placeholder={t('notebooksPage.descriptionPlaceholder')} />
          <select
            value={domainProfile}
            onChange={(event) => setDomainProfile(event.target.value)}
            className="h-10 rounded-lg border border-slate-200 bg-white px-3 text-sm text-slate-700"
          >
            <option value="general">{t('notebooksPage.profiles.general')}</option>
            <option value="tax">{t('notebooksPage.profiles.tax')}</option>
            <option value="legal">{t('notebooksPage.profiles.legal')}</option>
          </select>
          <Button type="submit" className="inline-flex items-center justify-center gap-2">
            <Plus className="h-4 w-4" />
            {t('notebooksPage.create')}
          </Button>
        </form>
        {error ? <p className="mt-3 text-sm text-red-600">{error}</p> : null}
      </div>

      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
        {isLoading ? (
          <div className="rounded-2xl bg-white p-6 text-sm text-slate-500 shadow-sm">{t('notebooksPage.loading')}</div>
        ) : null}

        {!isLoading && notebooks.length === 0 ? (
          <div className="rounded-2xl bg-white p-6 text-sm text-slate-500 shadow-sm">{t('notebooksPage.empty')}</div>
        ) : null}

        {notebooks.map((notebook) => {
          const isActive = String(notebook.id) === String(activeNotebookId);
          return (
            <button
              key={notebook.id}
              type="button"
              onClick={() => handleOpenNotebook(notebook.id)}
              className={`rounded-2xl border bg-white p-5 text-left shadow-sm transition ${isActive ? 'border-[#1f3a60] ring-2 ring-[#1f3a60]/10' : 'border-slate-200 hover:border-slate-300'}`}
            >
              <div className="mb-3 flex items-start justify-between gap-3">
                <div>
                  <h3 className="text-base font-semibold text-slate-900">{notebook.name}</h3>
                  <p className="mt-1 text-sm text-slate-500">{notebook.description || t('notebooksPage.noDescription')}</p>
                </div>
                {isActive ? <CheckCircle2 className="h-5 w-5 text-emerald-600" /> : null}
              </div>
              <div className="inline-flex rounded-full bg-slate-100 px-2.5 py-1 text-xs font-semibold text-slate-600">
                {t('notebooksPage.profileLabel', { value: getDomainProfileLabel(notebook.domain_profile) })}
              </div>
            </button>
          );
        })}
      </div>

      {activeNotebook ? (
        <div className="rounded-2xl border border-emerald-200 bg-emerald-50 p-4 text-sm text-emerald-800">
          {t('notebooksPage.activeNotebook', { name: activeNotebook.name })}
        </div>
      ) : null}
    </div>
  );
};


export default NotebooksPage;
