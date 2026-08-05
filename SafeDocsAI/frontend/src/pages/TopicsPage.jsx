import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { Link, useSearchParams } from 'react-router-dom';
import { AlertTriangle, ChevronDown, ChevronRight, Loader2, RefreshCw, Shapes } from 'lucide-react';

import { Button } from '../components/ui/Button';
import NotebookScopeBadge from '../components/notebook/NotebookScopeBadge';
import { cn } from '../lib/utils';
import { useActiveNotebookScope } from '../hooks/useActiveNotebookScope';
import { useSessionRole } from '../hooks/useSessionRole';
import { useLocale } from '../i18n';
import { resolveApiErrorMessage, resolveErrorCode } from '../lib/apiError';
import { formatLocaleDate } from '../lib/locale';
import { buildTopicFilterSearch, isEmptyTopic, resolveTopicName, resolveTopicShare, sortTopics } from '../lib/topics';
import { topicsService } from '../services/topicsService';

// Модель могли ещё не обучить — для пользователя это состояние системы, а не
// ошибка запроса, поэтому код разбирается отдельно от остальных.
const MODEL_MISSING_CODE = 'topic.model_missing';
const REASSIGN_IN_PROGRESS_CODE = 'topic.reassign_in_progress';

// Раскрытие карточки модели запоминаем — как «Подробности» у источников: тому,
// кто смотрит метрики каждый день, иначе открывать их пришлось бы каждый заход.
const MODEL_DETAILS_STORAGE_KEY = 'knowledgeai.topics.modelDetailsOpen';

// Пустые темы свёрнуты тем же приёмом и с той же памятью. Модель различает
// двадцать тем, а в небольшой базе документы попадают в три-четыре — остальные
// строки показывали бы нули и вытесняли с экрана то, ради чего сюда пришли.
// Список при этом не выброшен: он отвечает на вопрос «что модель вообще
// умеет», и администратору этот ответ нужен.
const EMPTY_TOPICS_STORAGE_KEY = 'knowledgeai.topics.emptyOpen';

const readStoredFlag = (key) => {
    if (typeof window === 'undefined') return false;
    return localStorage.getItem(key) === 'true';
};

const readModelDetailsPreference = () => readStoredFlag(MODEL_DETAILS_STORAGE_KEY);
const readEmptyTopicsPreference = () => readStoredFlag(EMPTY_TOPICS_STORAGE_KEY);

const formatMetric = (value, fallback) => {
    const numeric = Number(value);
    return Number.isFinite(numeric) ? numeric.toFixed(3) : fallback;
};

const TopicsPage = () => {
    const { locale, t } = useLocale();
    const { isAdmin } = useSessionRole();
    // Поиск в шапке иначе был бы мёртвым контролом: он есть на всех обычных
    // страницах, поэтому здесь он сужает список тем по подписи.
    const [searchParams] = useSearchParams();
    const searchTerm = searchParams.get('q') || '';

    // Та же область, что у списка источников и чата: экран показывает
    // распределение внутри активного блокнота, и область видна бейджем.
    const { notebookId, notebookName, canResetScope, resetScope } = useActiveNotebookScope(undefined);

    const [topics, setTopics] = useState([]);
    const [isLoading, setIsLoading] = useState(true);
    const [loadError, setLoadError] = useState('');
    const [isModelTrained, setIsModelTrained] = useState(true);
    const [model, setModel] = useState(null);
    const [modelError, setModelError] = useState('');
    const [detailsOpen, setDetailsOpen] = useState(readModelDetailsPreference);
    const [emptyOpen, setEmptyOpen] = useState(readEmptyTopicsPreference);
    const [reloadToken, setReloadToken] = useState(0);
    const [isReassigning, setIsReassigning] = useState(false);
    const [reassignNotice, setReassignNotice] = useState(null);

    useEffect(() => {
        let cancelled = false;

        setIsLoading(true);
        setLoadError('');
        setModelError('');

        // Оба запроса независимы: сведения о модели не должны решать, покажем ли
        // мы распределение, и наоборот.
        Promise.allSettled([topicsService.getDistribution(notebookId), topicsService.getModel()])
            .then(([distribution, modelInfo]) => {
                if (cancelled) return;

                // Тело ответа — массив тем; всё остальное считаем пустым списком,
                // иначе неожиданный формат уронил бы экран вместо того, чтобы
                // показать «тем пока нет».
                const payload = distribution.status === 'fulfilled' ? distribution.value.data : null;
                const items = Array.isArray(payload) ? payload : [];
                const distributionCode = distribution.status === 'rejected' ? resolveErrorCode(distribution.reason) : null;
                const modelCode = modelInfo.status === 'rejected' ? resolveErrorCode(modelInfo.reason) : null;

                setTopics(items);
                setModel(modelInfo.status === 'fulfilled' ? modelInfo.value.data : null);

                // «Модель не обучена» — состояние всего экрана, но только когда
                // показывать действительно нечего. Если распределение пришло, а
                // сведений о модели нет, экран остаётся рабочим: пустеет одна
                // карточка, а не вся страница.
                const nothingTrained = distributionCode === MODEL_MISSING_CODE
                    || (modelCode === MODEL_MISSING_CODE && items.length === 0);
                setIsModelTrained(!nothingTrained);

                if (distribution.status === 'rejected' && !nothingTrained) {
                    console.error('Failed to fetch topics', distribution.reason);
                    setLoadError(resolveApiErrorMessage(distribution.reason, t, 'topics.loadFailed'));
                }

                if (modelInfo.status === 'rejected' && !nothingTrained) {
                    console.error('Failed to fetch topic model', modelInfo.reason);
                    setModelError(resolveApiErrorMessage(modelInfo.reason, t, 'topics.model.loadFailed'));
                }

                setIsLoading(false);
            });

        return () => {
            cancelled = true;
        };
    }, [notebookId, reloadToken, t]);

    const toggleDetails = useCallback(() => {
        setDetailsOpen((prev) => {
            const next = !prev;
            localStorage.setItem(MODEL_DETAILS_STORAGE_KEY, String(next));
            return next;
        });
    }, []);

    const toggleEmpty = useCallback(() => {
        setEmptyOpen((prev) => {
            const next = !prev;
            localStorage.setItem(EMPTY_TOPICS_STORAGE_KEY, String(next));
            return next;
        });
    }, []);

    const handleReassign = useCallback(async () => {
        setIsReassigning(true);
        setReassignNotice(null);

        try {
            await topicsService.reassign();
            setReassignNotice({ tone: 'info', text: t('topics.reassign.started') });
        } catch (error) {
            console.error('Failed to start topic reassignment', error);
            // 409 — не отказ, а ответ «работа уже идёт»: тон у него спокойный,
            // текст берётся из общей таблицы кодов, как у остальных ответов.
            const inProgress = resolveErrorCode(error) === REASSIGN_IN_PROGRESS_CODE;
            setReassignNotice({
                tone: inProgress ? 'info' : 'error',
                text: resolveApiErrorMessage(error, t, 'topics.reassign.failed'),
            });
        } finally {
            setIsReassigning(false);
        }
    }, [t]);

    const totalDocuments = useMemo(
        () => topics.reduce((sum, topic) => sum + Number(topic?.document_count || 0), 0),
        [topics],
    );

    // Список делится надвое: темы с источниками и пустые. Пустых на небольшой
    // базе большинство (модель различает двадцать тем, документы попадают в
    // три-четыре), и показанные вперемешку они превращают экран в столбец
    // нулей.
    //
    // При поиске деления нет: пользователь назвал тему сам, и прятать
    // найденное под второе раскрытие значило бы отфильтровать его же запрос
    // ещё раз.
    const { visibleTopics, filledTopics, emptyTopics } = useMemo(() => {
        const query = searchTerm.trim().toLowerCase();
        const matching = query
            ? topics.filter((topic) => resolveTopicName(topic, locale).toLowerCase().includes(query))
            : topics;
        const sorted = sortTopics(matching);

        return {
            visibleTopics: sorted,
            filledTopics: query ? sorted : sorted.filter((topic) => !isEmptyTopic(topic)),
            emptyTopics: query ? [] : sorted.filter(isEmptyTopic),
        };
    }, [locale, searchTerm, topics]);

    const modelClusterCount = model?.cluster_count ?? model?.k;
    const metricFallback = t('topics.model.unknown');

    return (
        <div className="space-y-6 px-4">
            <div className="flex flex-wrap items-center gap-2">
                <NotebookScopeBadge
                    notebookId={notebookId}
                    notebookName={notebookName}
                    canReset={canResetScope}
                    onReset={resetScope}
                    resetTitle={t('documents.scopeResetTitle')}
                />
            </div>

            <section className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
                <div className="mb-4 flex items-center gap-3">
                    <div className="rounded-xl bg-[#1f3a60]/10 p-2 text-[#1f3a60]">
                        <Shapes className="h-5 w-5" />
                    </div>
                    <div>
                        <h2 className="text-lg font-semibold text-slate-900">{t('topics.title')}</h2>
                        <p className="text-sm text-slate-500">{t('topics.description')}</p>
                    </div>
                </div>

                {isLoading ? (
                    <div className="flex items-center justify-center gap-2 py-10 text-sm text-slate-500">
                        <Loader2 className="h-5 w-5 animate-spin" />
                        {t('topics.loading')}
                    </div>
                ) : loadError ? (
                    <div role="alert" className="mx-auto flex max-w-md flex-col items-center gap-3 rounded-xl bg-red-50 p-6 text-center">
                        <AlertTriangle className="h-6 w-6 text-red-600" />
                        <p className="text-sm font-semibold text-red-700">{loadError}</p>
                        <Button type="button" variant="outline" onClick={() => setReloadToken((prev) => prev + 1)}>
                            <RefreshCw className="h-4 w-4" />
                            {t('documents.retry')}
                        </Button>
                    </div>
                ) : !isModelTrained ? (
                    /* Модели ещё нет — это ожидаемое состояние системы, а не сбой:
                       техническая ошибка на экране обычного пользователя означала бы,
                       что он виноват в том, чего не делал. */
                    <div className="rounded-xl bg-slate-50 p-8 text-center">
                        <p className="text-sm font-semibold text-slate-700">{t('topics.notTrainedTitle')}</p>
                        <p className="mx-auto mt-1 max-w-md text-sm text-slate-500">{t('topics.notTrainedDescription')}</p>
                    </div>
                ) : visibleTopics.length === 0 ? (
                    <p className="rounded-xl bg-slate-50 p-8 text-center text-sm text-slate-500">
                        {searchTerm.trim() ? t('topics.noMatches') : t('topics.empty')}
                    </p>
                ) : (
                    <>
                        {filledTopics.length === 0 ? (
                            <p className="rounded-xl bg-slate-50 p-8 text-center text-sm text-slate-500">{t('topics.empty')}</p>
                        ) : (
                            <ul className="space-y-2">
                                {filledTopics.map((topic) => {
                                    const share = resolveTopicShare(topic, totalDocuments);
                                    const percent = Math.round(share * 100);
                                    // Подпись на языке интерфейса, с откатом к
                                    // устойчивому имени темы: английских экранов в
                                    // продукте нет.
                                    const label = resolveTopicName(topic, locale);

                                    return (
                                        <li key={topic.cluster_index}>
                                            {/* Клик по теме ведёт к источникам этой темы. Номер
                                                кластера уходит в адрес параметром запроса и на
                                                экране не показывается ни здесь, ни там. */}
                                            <Link
                                                to={`/sources${buildTopicFilterSearch(topic.cluster_index, label)}`}
                                                aria-label={t('topics.open', { name: label })}
                                                className="block rounded-xl border border-slate-200 px-4 py-3 transition hover:border-[#1f3a60]/40 hover:bg-slate-50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#1f3a60]/40"
                                            >
                                                <div className="flex flex-wrap items-baseline justify-between gap-x-3 gap-y-1">
                                                    <span className="min-w-0 break-words text-sm font-semibold text-slate-900">{label}</span>
                                                    <span className="flex shrink-0 items-center gap-3 text-xs text-slate-500">
                                                        <span>{t('topics.documentCount', { count: topic.document_count })}</span>
                                                        <span className="font-semibold text-slate-700">{t('topics.share', { value: percent })}</span>
                                                        <ChevronRight className="h-4 w-4 text-slate-400" />
                                                    </span>
                                                </div>
                                                {/* Полоса вместо графика: доля читается быстрее
                                                    в сравнении с соседями, а библиотек ради этого
                                                    в проекте нет и заводить их не из-за чего. */}
                                                <div className="mt-2 h-1.5 overflow-hidden rounded-full bg-slate-100" aria-hidden="true">
                                                    <div className="h-full rounded-full bg-[#1f3a60]" style={{ width: `${Math.max(percent, 1)}%` }} />
                                                </div>
                                            </Link>
                                        </li>
                                    );
                                })}
                            </ul>
                        )}

                        {/* Пустые темы — под раскрытием, тем же приёмом, что «О модели»
                            здесь и «Подробности» в списке источников. Ссылками они НЕ
                            делаются: переход в список источников по пустой теме ведёт на
                            пустой экран, и предлагать его незачем. */}
                        {emptyTopics.length > 0 ? (
                            <div className="mt-4 border-t border-slate-100 pt-4">
                                <div className="flex flex-wrap items-center gap-3">
                                    <button
                                        type="button"
                                        onClick={toggleEmpty}
                                        aria-expanded={emptyOpen}
                                        className="inline-flex items-center gap-2 rounded-lg border border-slate-200 bg-white px-3 py-1.5 text-xs font-semibold text-slate-600 transition hover:border-slate-300 hover:text-[#1f3a60] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#1f3a60] focus-visible:ring-offset-1"
                                    >
                                        <ChevronDown className={cn('h-4 w-4 transition-transform duration-150', emptyOpen && 'rotate-180')} />
                                        {t('topics.emptyClusters.action', { count: emptyTopics.length })}
                                    </button>
                                    {!emptyOpen ? (
                                        <span className="text-xs text-slate-400">{t('topics.emptyClusters.hint')}</span>
                                    ) : null}
                                </div>

                                {emptyOpen ? (
                                    <ul className="mt-3 flex flex-wrap gap-2">
                                        {emptyTopics.map((topic) => (
                                            <li
                                                key={topic.cluster_index}
                                                className="rounded-lg border border-dashed border-slate-200 px-3 py-1.5 text-xs text-slate-400"
                                            >
                                                {resolveTopicName(topic, locale)}
                                            </li>
                                        ))}
                                    </ul>
                                ) : null}
                            </div>
                        ) : null}
                    </>
                )}
            </section>

            {/* Карточка модели показывается только когда модель есть: без неё
                раскрытие обещало бы сведения, которых нет, а сказано о них уже выше. */}
            {!isLoading && isModelTrained ? (
                <section className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
                    <div className="flex flex-wrap items-center justify-between gap-3">
                        <div className="flex flex-wrap items-center gap-3">
                            <button
                                type="button"
                                onClick={toggleDetails}
                                aria-expanded={detailsOpen}
                                className="inline-flex items-center gap-2 rounded-lg border border-slate-200 bg-white px-3 py-1.5 text-xs font-semibold text-slate-600 transition hover:border-slate-300 hover:text-[#1f3a60] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#1f3a60] focus-visible:ring-offset-1"
                            >
                                <ChevronDown className={cn('h-4 w-4 transition-transform duration-150', detailsOpen && 'rotate-180')} />
                                {t('topics.model.details')}
                            </button>
                            {!detailsOpen ? (
                                <span className="text-xs text-slate-400">{t('topics.model.hint')}</span>
                            ) : null}
                        </div>

                        {/* Переразметку запускает только админ: остальным кнопка
                            всё равно вернула бы 403, а тупик — не защита. */}
                        {isAdmin ? (
                            <Button type="button" variant="outline" size="sm" isLoading={isReassigning} onClick={handleReassign}>
                                <RefreshCw className="h-4 w-4" />
                                {t('topics.reassign.action')}
                            </Button>
                        ) : null}
                    </div>

                    {reassignNotice ? (
                        <p
                            role={reassignNotice.tone === 'error' ? 'alert' : 'status'}
                            className={cn(
                                'mt-3 rounded-xl px-3 py-2 text-sm',
                                reassignNotice.tone === 'error' ? 'bg-red-50 text-red-700' : 'bg-[#1f3a60]/5 text-[#1f3a60]',
                            )}
                        >
                            {reassignNotice.text}
                        </p>
                    ) : null}

                    {detailsOpen ? (
                        modelError ? (
                            <p role="alert" className="mt-4 rounded-xl bg-red-50 px-3 py-2 text-sm text-red-700">{modelError}</p>
                        ) : (
                            <dl className="mt-4 grid gap-x-6 gap-y-3 text-sm sm:grid-cols-2">
                                <div>
                                    <dt className="text-xs font-semibold uppercase tracking-[0.08em] text-slate-400">{t('topics.model.trainedAt')}</dt>
                                    <dd className="mt-0.5 text-slate-700">
                                        {formatLocaleDate(model?.trained_at, locale, {
                                            day: 'numeric',
                                            month: 'short',
                                            year: 'numeric',
                                            hour: '2-digit',
                                            minute: '2-digit',
                                        }, metricFallback)}
                                    </dd>
                                </div>
                                <div>
                                    <dt className="text-xs font-semibold uppercase tracking-[0.08em] text-slate-400">{t('topics.model.clusterCount')}</dt>
                                    <dd className="mt-0.5 text-slate-700">{modelClusterCount ?? metricFallback}</dd>
                                </div>
                                <div>
                                    <dt className="text-xs font-semibold uppercase tracking-[0.08em] text-slate-400">{t('topics.model.embeddingModel')}</dt>
                                    <dd className="mt-0.5 break-words text-slate-700">{model?.embedding_model || metricFallback}</dd>
                                </div>
                                <div>
                                    <dt className="text-xs font-semibold uppercase tracking-[0.08em] text-slate-400">{t('topics.model.transform')}</dt>
                                    <dd className="mt-0.5 break-words text-slate-700">{model?.transform || metricFallback}</dd>
                                </div>
                                <div className="sm:col-span-2">
                                    <dt className="text-xs font-semibold uppercase tracking-[0.08em] text-slate-400">{t('topics.model.metrics')}</dt>
                                    <dd className="mt-1 flex flex-wrap gap-x-6 gap-y-1 text-slate-700">
                                        <span>{t('topics.model.ariTopic')}: {formatMetric(model?.metrics?.ari_topic, metricFallback)}</span>
                                        <span>{t('topics.model.purity')}: {formatMetric(model?.metrics?.purity, metricFallback)}</span>
                                        <span>{t('topics.model.silhouette')}: {formatMetric(model?.metrics?.silhouette, metricFallback)}</span>
                                    </dd>
                                </div>
                            </dl>
                        )
                    ) : null}
                </section>
            ) : null}
        </div>
    );
};

export default TopicsPage;
