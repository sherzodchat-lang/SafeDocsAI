import React, { useEffect, useState } from 'react';
import { Lightbulb } from 'lucide-react';

import { insightsService } from '../services/insightsService';
import { Button } from '../components/ui/Button';
import Input from '../components/ui/Input';


const ACTIVE_NOTEBOOK_STORAGE_KEY = 'knowledgeai.activeNotebookId';


const InsightsPage = () => {
  const activeNotebookId = localStorage.getItem(ACTIVE_NOTEBOOK_STORAGE_KEY);
  const [insights, setInsights] = useState([]);
  const [title, setTitle] = useState('');
  const [body, setBody] = useState('');
  const [error, setError] = useState('');
  const [isSaving, setIsSaving] = useState(false);

  useEffect(() => {
    const fetchInsights = async () => {
      try {
        const response = await insightsService.getAll(activeNotebookId ? Number(activeNotebookId) : null);
        setInsights(response.data || []);
      } catch (fetchError) {
        console.error('Failed to fetch insights', fetchError);
        setError('Не удалось загрузить insights.');
      }
    };

    fetchInsights();
  }, [activeNotebookId]);

  const handleCreateInsight = async (event) => {
    event.preventDefault();
    if (!activeNotebookId || !title.trim()) return;
    try {
      setIsSaving(true);
      setError('');
      const response = await insightsService.create({
        notebook_id: Number(activeNotebookId),
        title: title.trim(),
        body,
      });
      setInsights((prev) => [response.data, ...prev]);
      setTitle('');
      setBody('');
    } catch (saveError) {
      console.error('Failed to create insight', saveError);
      setError(saveError.response?.data?.detail || 'Не удалось создать insight.');
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
            <h2 className="text-lg font-semibold text-slate-900">Insights</h2>
            <p className="text-sm text-slate-500">Здесь появятся структурированные выводы, summary и extracted findings по notebook.</p>
          </div>
        </div>

        <div className="rounded-xl bg-slate-50 p-4 text-sm text-slate-600">
          Активный notebook: <span className="font-semibold text-slate-800">{activeNotebookId || 'default'}</span>
        </div>
        <form onSubmit={handleCreateInsight} className="mt-4 space-y-3">
          <Input value={title} onChange={(e) => setTitle(e.target.value)} placeholder="Insight title" />
          <textarea
            value={body}
            onChange={(e) => setBody(e.target.value)}
            placeholder="Write insight body..."
            className="min-h-28 w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm text-slate-800 shadow-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#1f3a60]/30"
          />
          <div className="flex justify-end">
            <Button type="submit" isLoading={isSaving} disabled={!activeNotebookId}>Create insight</Button>
          </div>
        </form>
        {error ? <p className="mt-3 text-sm text-red-600">{error}</p> : null}
      </section>

      <section className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
        <h3 className="mb-4 text-base font-semibold text-slate-900">Saved Insights</h3>
        {insights.length === 0 ? (
          <p className="text-sm text-slate-500">Insights пока нет.</p>
        ) : (
          <div className="space-y-3">
            {insights.map((insight) => (
              <div key={insight.id} className="rounded-xl border border-slate-200 p-4">
                <div className="text-sm font-semibold text-slate-900">{insight.title}</div>
                <div className="mt-1 whitespace-pre-wrap text-sm text-slate-600">{insight.body || 'No content'}</div>
              </div>
            ))}
          </div>
        )}
      </section>
    </div>
  );
};


export default InsightsPage;
