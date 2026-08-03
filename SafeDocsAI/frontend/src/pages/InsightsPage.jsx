import React, { useEffect, useState } from 'react';
import { Lightbulb } from 'lucide-react';

import { insightsService } from '../services/insightsService';
import { Button } from '../components/ui/Button';
import Input from '../components/ui/Input';
import { useLocale } from '../i18n';
import { resolveApiErrorMessage } from '../lib/apiError';


const ACTIVE_NOTEBOOK_STORAGE_KEY = 'knowledgeai.activeNotebookId';


// В localStorage может лежать что угодно от прошлых версий: Number('abc') даёт
// NaN, в JSON он превращается в null, и сервер отвечает 422 вместо ответа.
const resolveNotebookId = (value) => {
  const parsed = Number(value);
  return Number.isInteger(parsed) && parsed > 0 ? parsed : null;
};


const InsightsPage = () => {
  const { t } = useLocale();
  const activeNotebookId = resolveNotebookId(localStorage.getItem(ACTIVE_NOTEBOOK_STORAGE_KEY));
  const [insights, setInsights] = useState([]);
  const [title, setTitle] = useState('');
  const [body, setBody] = useState('');
  const [error, setError] = useState('');
  const [isSaving, setIsSaving] = useState(false);

  useEffect(() => {
    const fetchInsights = async () => {
      try {
        const response = await insightsService.getAll(activeNotebookId);
        setInsights(response.data || []);
      } catch (fetchError) {
        console.error('Failed to fetch insights', fetchError);
        setError(resolveApiErrorMessage(fetchError, t, 'insightsPage.loadFailed'));
      }
    };

    fetchInsights();
  }, [activeNotebookId, t]);

  const handleCreateInsight = async (event) => {
    event.preventDefault();
    if (!activeNotebookId || !title.trim()) return;
    try {
      setIsSaving(true);
      setError('');
      const response = await insightsService.create({
        notebook_id: activeNotebookId,
        title: title.trim(),
        body,
      });
      setInsights((prev) => [response.data, ...prev]);
      setTitle('');
      setBody('');
    } catch (saveError) {
      console.error('Failed to create insight', saveError);
      setError(resolveApiErrorMessage(saveError, t, 'insightsPage.createFailed'));
    } finally {
      setIsSaving(false);
    }
  };

  return (
    <div className="space-y-6 px-4">
      <section className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
        <div className="mb-4 flex items-center gap-3">
          <div className="rounded-xl bg-[#c5a059]/15 p-2 text-[#a27e35]">
            <Lightbulb className="h-5 w-5" />
          </div>
          <div>
            <h2 className="text-lg font-semibold text-slate-900">{t('insightsPage.title')}</h2>
            <p className="text-sm text-slate-500">{t('insightsPage.description')}</p>
          </div>
        </div>

        {/* Строку собирает словарь целиком: порядок слов и знак после метки в ru и tg разный. */}
        <div className="rounded-xl bg-slate-50 p-4 text-sm font-semibold text-slate-700">
          {t('insightsPage.activeNotebook', { value: activeNotebookId || t('insightsPage.notebookNotSelected') })}
          {/* Активный блокнот больше не проставляется сам при заходе в блокнот, поэтому
              «не выбран» — обычное состояние. Без пояснения пользователь видит только
              выключенную кнопку создания и не понимает, чего не хватает. */}
          {!activeNotebookId ? (
            <p className="mt-1 font-normal text-slate-500">{t('insightsPage.notebookHint')}</p>
          ) : null}
        </div>
        <form onSubmit={handleCreateInsight} className="mt-4 space-y-3">
          <Input value={title} onChange={(e) => setTitle(e.target.value)} placeholder={t('insightsPage.titlePlaceholder')} />
          <textarea
            value={body}
            onChange={(e) => setBody(e.target.value)}
            placeholder={t('insightsPage.bodyPlaceholder')}
            className="min-h-28 w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm text-slate-800 shadow-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#1f3a60]/30"
          />
          <div className="flex justify-end">
            <Button type="submit" isLoading={isSaving} disabled={!activeNotebookId}>{t('insightsPage.create')}</Button>
          </div>
        </form>
        {error ? <p role="alert" className="mt-3 text-sm text-red-600">{error}</p> : null}
      </section>

      <section className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
        <h3 className="mb-4 text-base font-semibold text-slate-900">{t('insightsPage.savedTitle')}</h3>
        {insights.length === 0 ? (
          <p className="text-sm text-slate-500">{t('insightsPage.empty')}</p>
        ) : (
          <div className="space-y-3">
            {insights.map((insight) => (
              <div key={insight.id} className="rounded-xl border border-slate-200 p-4">
                <div className="break-words text-sm font-semibold text-slate-900">{insight.title}</div>
                <div className="mt-1 whitespace-pre-wrap break-words text-sm text-slate-600">{insight.body || t('insightsPage.noContent')}</div>
              </div>
            ))}
          </div>
        )}
      </section>
    </div>
  );
};


export default InsightsPage;
