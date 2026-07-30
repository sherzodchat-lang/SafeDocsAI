import React, { useMemo, useState } from 'react';
import { FileSearch, Sparkles } from 'lucide-react';

import { askService } from '../services/askService';
import { Button } from '../components/ui/Button';
import Input from '../components/ui/Input';


const ACTIVE_NOTEBOOK_STORAGE_KEY = 'knowledgeai.activeNotebookId';


const AskPage = () => {
  const [question, setQuestion] = useState('');
  const [result, setResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');

  const activeNotebookId = useMemo(() => localStorage.getItem(ACTIVE_NOTEBOOK_STORAGE_KEY), []);

  const handleSubmit = async (event) => {
    event.preventDefault();
    const cleanQuestion = question.trim();
    if (!cleanQuestion) return;

    try {
      setIsLoading(true);
      setError('');
      const response = await askService.ask(cleanQuestion, activeNotebookId ? Number(activeNotebookId) : null);
      setResult(response.data);
    } catch (requestError) {
      console.error('Ask request failed', requestError);
      setError(requestError.response?.data?.detail || 'Не удалось выполнить Ask запрос.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="space-y-6 px-4">
      <section className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
        <div className="mb-4 flex items-center gap-3">
          <div className="rounded-xl bg-[#1f3a60]/10 p-2 text-[#1f3a60]">
            <Sparkles className="h-5 w-5" />
          </div>
          <div>
            <h2 className="text-lg font-semibold text-slate-900">Ask</h2>
            <p className="text-sm text-slate-500">One-shot анализ по текущему notebook с автоматическим retrieval.</p>
          </div>
        </div>

        <form onSubmit={handleSubmit} className="space-y-3">
          <Input
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            placeholder="Сформулируйте вопрос по выбранному notebook..."
            className="h-12"
          />
          <div className="flex items-center justify-between gap-3">
            <div className="text-sm text-slate-500">
              Активный notebook: <span className="font-semibold text-slate-700">{activeNotebookId || 'default'}</span>
            </div>
            <Button type="submit" isLoading={isLoading}>
              <FileSearch className="h-4 w-4" />
              Ask
            </Button>
          </div>
        </form>

        {error ? <p className="mt-3 text-sm text-red-600">{error}</p> : null}
      </section>

      <section className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
        <div className="mb-4 flex items-center justify-between gap-3">
          <h3 className="text-base font-semibold text-slate-900">Result</h3>
          {result?.log_id ? <span className="text-xs font-semibold text-slate-500">log #{result.log_id}</span> : null}
        </div>

        {!result ? (
          <p className="text-sm text-slate-500">Здесь появится структурированный результат Ask запроса.</p>
        ) : (
          <div className="space-y-5">
            <div className="rounded-xl bg-slate-50 p-4 text-sm leading-relaxed text-slate-700 whitespace-pre-wrap">
              {result.answer}
            </div>

            <div>
              <h4 className="mb-2 text-sm font-semibold text-slate-900">Citations</h4>
              {Array.isArray(result.citations) && result.citations.length > 0 ? (
                <div className="space-y-2">
                  {result.citations.map((citation, index) => (
                    <div key={citation.chunk_id || index} className="rounded-lg border border-slate-200 bg-white p-3 text-sm text-slate-600">
                      <div className="font-semibold text-slate-800">
                        {citation.source_name || `Source #${citation.source_id ?? 'N/A'}`}
                        {citation.page ? `, page ${citation.page}` : ''}
                      </div>
                      {citation.quote ? <div className="mt-1">{citation.quote}</div> : null}
                    </div>
                  ))}
                </div>
              ) : (
                <p className="text-sm text-slate-500">Нет citations для этого ответа.</p>
              )}
            </div>
          </div>
        )}
      </section>
    </div>
  );
};


export default AskPage;
