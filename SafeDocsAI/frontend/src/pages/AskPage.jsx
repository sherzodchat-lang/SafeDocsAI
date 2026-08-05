import React, { useState } from 'react';
import { FileSearch, Sparkles } from 'lucide-react';

import { askService } from '../services/askService';
import { Button } from '../components/ui/Button';
import Input from '../components/ui/Input';
import { useActiveNotebookScope } from '../hooks/useActiveNotebookScope';
import { useLocale } from '../i18n';
import { resolveApiErrorMessage } from '../lib/apiError';


// Тот же предел, что у чата и у бэкенда (QUESTION_MAX_LENGTH в
// backend/app/api/deps.py): вопрос уходит в то же окно контекста модели.
const MAX_QUESTION_LENGTH = 2000;


const AskPage = () => {
  const { t } = useLocale();
  // Тот же хук, что у чата и списка источников: он же приносит имя блокнота —
  // по номеру из localStorage пользователь всё равно не узнаёт, чей это блокнот.
  const { notebookId: activeNotebookId, notebookName } = useActiveNotebookScope(undefined);
  const [question, setQuestion] = useState('');
  const [result, setResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');

  const handleSubmit = async (event) => {
    event.preventDefault();
    const cleanQuestion = question.trim();
    if (!cleanQuestion) return;

    try {
      setIsLoading(true);
      setError('');
      const response = await askService.ask(cleanQuestion, activeNotebookId);
      setResult(response.data);
    } catch (requestError) {
      console.error('Ask request failed', requestError);
      setError(resolveApiErrorMessage(requestError, t, 'askPage.requestFailed'));
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
            <h2 className="text-lg font-semibold text-slate-900">{t('askPage.title')}</h2>
            <p className="text-sm text-slate-500">{t('askPage.description')}</p>
          </div>
        </div>

        <form onSubmit={handleSubmit} className="space-y-3">
          <Input
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            placeholder={t('askPage.questionPlaceholder')}
            maxLength={MAX_QUESTION_LENGTH}
            className="h-12"
          />
          {/* Ввод по maxLength обрывается молча — объясняем предел на месте.
              Ключ общий с чатом: правило и формулировка там те же. */}
          {question.length >= MAX_QUESTION_LENGTH ? (
            <p className="text-xs font-medium text-amber-600">
              {t('chat.questionLimitReached', { max: MAX_QUESTION_LENGTH })}
            </p>
          ) : null}
          <div className="flex items-center justify-between gap-3">
            {/* Строку собирает словарь целиком: порядок слов и знак после метки в ru и tg разный. */}
            <div className="text-sm font-semibold text-slate-600">
              {/* Пока имя не пришло, показываем «#id» — тем же способом, что бейдж области
                  чата: это заметно временное значение, а не выдача номера за название. */}
              {t('askPage.activeNotebook', {
                value: activeNotebookId == null
                  ? t('askPage.notebookNotSelected')
                  : notebookName || `#${activeNotebookId}`,
              })}
            </div>
            {/* disabled рядом с isLoading: их объединяет сам Button. Пустой
                вопрос отправлять нечего — он не дойдёт даже до поиска. */}
            <Button type="submit" isLoading={isLoading} disabled={!question.trim()}>
              <FileSearch className="h-4 w-4" />
              {t('askPage.submit')}
            </Button>
          </div>
          {/* Активный блокнот больше не проставляется сам при заходе в блокнот, поэтому
              «не выбран» — обычное состояние. Объясняем, что будет и как выбрать. */}
          {!activeNotebookId ? (
            <p className="text-sm text-slate-500">{t('askPage.notebookHint')}</p>
          ) : null}
        </form>

        {error ? <p role="alert" className="mt-3 text-sm text-red-600">{error}</p> : null}
      </section>

      <section className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
        <div className="mb-4 flex items-center justify-between gap-3">
          <h3 className="text-base font-semibold text-slate-900">{t('askPage.resultTitle')}</h3>
          {result?.log_id ? <span className="text-xs font-semibold text-slate-500">{t('askPage.logId', { id: result.log_id })}</span> : null}
        </div>

        {!result ? (
          <p className="text-sm text-slate-500">{t('askPage.emptyResult')}</p>
        ) : (
          <div className="space-y-5">
            <div className="rounded-xl bg-slate-50 p-4 text-sm leading-relaxed text-slate-700 whitespace-pre-wrap">
              {result.answer}
            </div>

            <div>
              <h4 className="mb-2 text-sm font-semibold text-slate-900">{t('askPage.citations')}</h4>
              {Array.isArray(result.citations) && result.citations.length > 0 ? (
                <div className="space-y-2">
                  {result.citations.map((citation, index) => (
                    <div key={citation.chunk_id || index} className="rounded-lg border border-slate-200 bg-white p-3 text-sm text-slate-600">
                      {/* Подпись источника и номер страницы берём из тех же ключей, что и чат:
                          формулировка и знаки препинания у них в ru и tg уже согласованы. */}
                      <div className="break-words font-semibold text-slate-800">
                        {citation.source_name || t('chat.sourceFallback', { id: citation.source_id ?? 'N/A' })}
                        {citation.page ? t('chat.sourcePage', { page: citation.page }) : ''}
                      </div>
                      {citation.quote ? <div className="mt-1 break-words">{citation.quote}</div> : null}
                    </div>
                  ))}
                </div>
              ) : (
                <p className="text-sm text-slate-500">{t('askPage.noCitations')}</p>
              )}
            </div>
          </div>
        )}
      </section>
    </div>
  );
};


export default AskPage;
