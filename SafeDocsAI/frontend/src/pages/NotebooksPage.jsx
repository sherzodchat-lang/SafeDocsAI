import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Bookmark, CheckCircle2, Pencil, Plus } from 'lucide-react';
import { useNavigate, useSearchParams } from 'react-router-dom';

import NotebookEditDialog from '../components/notebook/NotebookEditDialog';
import { Button } from '../components/ui/Button';
import Input from '../components/ui/Input';
import { useLocale } from '../i18n';
import { resolveApiErrorMessage } from '../lib/apiError';
import { notebooksService } from '../services/notebooksService';


const ACTIVE_NOTEBOOK_STORAGE_KEY = 'knowledgeai.activeNotebookId';


const NotebooksPage = () => {
  const navigate = useNavigate();
  const { t } = useLocale();
  // Поиск читаем из URL param ?q= (устанавливается хедером)
  const [searchParams] = useSearchParams();
  const searchTerm = searchParams.get('q') || '';
  const [notebooks, setNotebooks] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [domainProfile, setDomainProfile] = useState('general');
  const [error, setError] = useState('');
  const [isCreating, setIsCreating] = useState(false);
  const [editTarget, setEditTarget] = useState(null);
  const [activeNotebookId, setActiveNotebookId] = useState(() => localStorage.getItem(ACTIVE_NOTEBOOK_STORAGE_KEY) || '');
  const nameInputRef = useRef(null);

  const fetchNotebooks = useCallback(async () => {
    try {
      setError('');
      const response = await notebooksService.getAll();
      setNotebooks(response.data || []);
    } catch (fetchError) {
      console.error('Failed to fetch notebooks', fetchError);
      setError(resolveApiErrorMessage(fetchError, t, 'notebooksPage.loadFailed'));
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

  const visibleNotebooks = useMemo(() => {
    const query = searchTerm.trim().toLowerCase();
    if (!query) return notebooks;

    return notebooks.filter((notebook) => {
      const name = String(notebook.name || '').toLowerCase();
      const description = String(notebook.description || '').toLowerCase();
      return name.includes(query) || description.includes(query);
    });
  }, [notebooks, searchTerm]);

  // Выбор задаёт контекст глобального чата, поэтому нужен и обратный ход:
  // иначе вернуть чат ко всем источникам можно было бы только удалив блокнот.
  const handleResetActiveNotebook = () => {
    setActiveNotebookId('');
    localStorage.removeItem(ACTIVE_NOTEBOOK_STORAGE_KEY);
  };

  // Открыть — только открыть. Раньше клик по карточке заодно делал блокнот
  // активным для чата, то есть менял область ответов по факту просмотра.
  // Теперь активный блокнот назначается явно — кнопкой в шапке блокнота.
  const handleOpenNotebook = (notebookId) => {
    navigate(`/notebooks/${notebookId}`);
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    // Второй клик до ответа сервера создавал ещё один блокнот с тем же именем,
    // а редирект уводил только на первый — дубликат оставался незамеченным.
    if (isCreating) return;

    if (!name.trim()) {
      setError(t('notebooksPage.nameRequired'));
      nameInputRef.current?.focus();
      return;
    }

    try {
      setIsCreating(true);
      setError('');
      const response = await notebooksService.create({
        name: name.trim(),
        description: description.trim() || null,
        domain_profile: domainProfile,
      });
      const created = response.data;
      setNotebooks((prev) => [created, ...prev]);
      setName('');
      setDescription('');
      setDomainProfile('general');
      handleOpenNotebook(created.id);
    } catch (submitError) {
      console.error('Failed to create notebook', submitError);
      setError(resolveApiErrorMessage(submitError, t, 'notebooksPage.createFailed'));
    } finally {
      setIsCreating(false);
    }
  };

  const handleOpenEdit = (notebook) => {
    setEditTarget(notebook);
  };

  const handleCloseEdit = () => {
    setEditTarget(null);
  };

  // Ответ PATCH — тот же объект, что в списке: подменяем строку на месте,
  // чтобы новое имя появилось без повторного запроса за списком.
  const handleNotebookSaved = (updated) => {
    if (!updated) return;
    setNotebooks((prev) => prev.map((item) => (item.id === updated.id ? updated : item)));
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
          <Input ref={nameInputRef} value={name} onChange={(event) => setName(event.target.value)} placeholder={t('notebooksPage.namePlaceholder')} />
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
          <Button
            type="submit"
            className="inline-flex items-center justify-center gap-2"
            isLoading={isCreating}
            disabled={!name.trim()}
          >
            {isCreating ? null : <Plus className="h-4 w-4" />}
            {isCreating ? t('notebooksPage.creating') : t('notebooksPage.create')}
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

        {/* Пустой результат поиска — это не «блокнотов нет»: подсказка должна отличаться,
            иначе фильтр читается как потеря данных. */}
        {!isLoading && notebooks.length > 0 && visibleNotebooks.length === 0 ? (
          <div className="rounded-2xl bg-white p-6 text-sm text-slate-500 shadow-sm">{t('notebook.notebooksNotFound')}</div>
        ) : null}

        {visibleNotebooks.map((notebook) => {
          const isActive = String(notebook.id) === String(activeNotebookId);
          return (
            /* Карточка перестала быть <button>: кнопку правки нельзя вложить в кнопку.
               Роль и обработчик клавиатуры оставляют её доступной так же, как раньше. */
            <div
              key={notebook.id}
              role="button"
              tabIndex={0}
              onClick={() => handleOpenNotebook(notebook.id)}
              onKeyDown={(event) => {
                if (event.key !== 'Enter' && event.key !== ' ') return;
                event.preventDefault();
                handleOpenNotebook(notebook.id);
              }}
              className={`cursor-pointer rounded-2xl border bg-white p-5 text-left shadow-sm transition focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#1f3a60]/40 ${isActive ? 'border-[#1f3a60] ring-2 ring-[#1f3a60]/10' : 'border-slate-200 hover:border-slate-300'}`}
            >
              <div className="mb-3 flex items-start justify-between gap-3">
                {/* Имя и описание — до 255 символов и без пробелов: без обрезки одна такая
                    карточка растягивает всю сетку. break-words рвёт и сплошную строку. */}
                <div className="min-w-0">
                  <h3 className="line-clamp-2 break-words text-base font-semibold text-slate-900" title={notebook.name}>{notebook.name}</h3>
                  <p className="mt-1 line-clamp-2 break-words text-sm text-slate-500">{notebook.description || t('notebooksPage.noDescription')}</p>
                </div>
                <div className="flex shrink-0 items-center gap-2">
                  {isActive ? <CheckCircle2 className="h-5 w-5 text-emerald-600" /> : null}
                  <button
                    type="button"
                    onClick={(event) => {
                      // Иначе клик по правке заодно открывал бы блокнот.
                      event.stopPropagation();
                      handleOpenEdit(notebook);
                    }}
                    title={t('notebooksPage.editNotebookTitle', { name: notebook.name })}
                    aria-label={t('notebooksPage.editNotebookTitle', { name: notebook.name })}
                    className="inline-flex h-8 w-8 items-center justify-center rounded-lg border border-slate-200 bg-white text-slate-500 transition hover:border-slate-300 hover:text-[#1f3a60]"
                  >
                    <Pencil className="h-4 w-4" />
                  </button>
                </div>
              </div>
              <div className="inline-flex rounded-full bg-slate-100 px-2.5 py-1 text-xs font-semibold text-slate-600">
                {t('notebooksPage.profileLabel', { value: getDomainProfileLabel(notebook.domain_profile) })}
              </div>
            </div>
          );
        })}
      </div>

      {activeNotebook ? (
        <div className="flex flex-wrap items-center justify-between gap-3 rounded-2xl border border-emerald-200 bg-emerald-50 p-4 text-sm text-emerald-800">
          <span className="min-w-0 break-words">{t('notebooksPage.activeNotebook', { name: activeNotebook.name })}</span>
          <button
            type="button"
            onClick={handleResetActiveNotebook}
            title={t('notebooksPage.resetActiveNotebookTitle')}
            className="rounded-lg border border-emerald-300 bg-white px-3 py-1 text-xs font-semibold text-emerald-700 transition hover:bg-emerald-100"
          >
            {t('notebooksPage.resetActiveNotebook')}
          </button>
        </div>
      ) : null}

      <NotebookEditDialog
        notebook={editTarget}
        isOpen={Boolean(editTarget)}
        onClose={handleCloseEdit}
        onSaved={handleNotebookSaved}
      />
    </div>
  );
};


export default NotebooksPage;
