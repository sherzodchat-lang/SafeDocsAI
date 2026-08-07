import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { Link, useSearchParams } from 'react-router-dom';
import { AlertTriangle, ArrowRight, ChevronDown, ChevronRight, FileText, Loader2, RefreshCw, Shapes } from 'lucide-react';

import { Button } from '../components/ui/Button';
import DocumentViewer from '../components/DocumentViewer';
import FolderGlyph from '../components/ui/FolderGlyph';
import NotebookScopeBadge from '../components/notebook/NotebookScopeBadge';
import { cn } from '../lib/utils';
import { useActiveNotebookScope } from '../hooks/useActiveNotebookScope';
import { useFolderDisclosure } from '../hooks/useFolderDisclosure';
import { useSessionRole } from '../hooks/useSessionRole';
import { useSources } from '../contexts/SourcesContext';
import { useLocale } from '../i18n';
import { resolveApiErrorMessage, resolveErrorCode } from '../lib/apiError';
import { formatLocaleDate } from '../lib/locale';
import { SOURCE_STATUS_BADGE_CLASS, formatSize, resolveStatus } from '../lib/sources';
import {
    TOPIC_FOLDER_AUTO_OPEN_LIMIT,
    TOPIC_FOLDER_HINT_KEY,
    TOPIC_FOLDER_TOPIC,
    buildTopicFilterSearch,
    buildTopicFolders,
    filterTopicFolders,
    readUnclearCount,
    resolveTopicFolderName,
    resolveTopicShare,
} from '../lib/topics';
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

/** Документ внутри папки: открывается просмотром, не уводя с экрана. */
const FolderDocument = ({ source, locale, t, statusLabels, onOpen }) => {
    const status = resolveStatus(source.status);

    return (
        <li>
            <button
                type="button"
                onClick={() => onOpen(source)}
                className="flex w-full items-center gap-2.5 rounded-lg px-2.5 py-2 text-left transition hover:bg-white focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#1f3a60]/40"
                aria-label={t('topics.openDocument', { name: source.name })}
            >
                <FileText className="h-4 w-4 shrink-0 text-slate-400" />
                <span className="min-w-0 flex-1">
                    <span className="block truncate text-sm text-slate-800" title={source.name || undefined}>{source.name}</span>
                    <span className="mt-0.5 block text-xs text-slate-400">
                        {formatLocaleDate(source.created_at, locale, { day: 'numeric', month: 'short', year: 'numeric' }, '—')}
                        {' · '}
                        {formatSize(source.size, t)}
                    </span>
                </span>
                {/* Бейдж только у тех, с кем что-то не так: «Готово» на каждой из
                    сорока строк — это сорок раз повторённая норма, за которой
                    перестаёт быть виден единственный сбойный документ. */}
                {status !== 'ready' ? (
                    <span className={cn('shrink-0 rounded-full px-2 py-0.5 text-[11px] font-semibold', SOURCE_STATUS_BADGE_CLASS[status])}>
                        {statusLabels[status]}
                    </span>
                ) : null}
            </button>
        </li>
    );
};

/**
 * Папка темы: заголовок раскрывает содержимое, стрелка справа уводит в список
 * источников, суженный до этой темы.
 *
 * Два действия у одной строки разведены по разным кнопкам нарочно: раскрытие —
 * то, ради чего сюда пришли, и оно не должно каждый раз менять экран, а переход
 * к фильтру остался на месте, потому что на него завязан раздел источников.
 */
const TopicFolderRow = ({
    folder,
    isOpen,
    onToggle,
    totalDocuments,
    locale,
    t,
    statusLabels,
    sourcesLoading,
    onOpenDocument,
}) => {
    const name = resolveTopicFolderName(folder, t);
    const hintKey = TOPIC_FOLDER_HINT_KEY[folder.kind];
    // Доля — только у целой папки. Поиск оставляет в папке одни совпадения, и
    // «1 документ · 43 %» рядом означало бы, что доля посчитана от найденного,
    // а она посчитана от всей темы.
    const isWhole = folder.topic && Number(folder.topic.document_count) === folder.count;
    const percent = folder.kind === TOPIC_FOLDER_TOPIC && isWhole
        ? Math.round(resolveTopicShare(folder.topic, totalDocuments) * 100)
        : null;
    const panelId = `topic-folder-${folder.key}`;
    // Число из распределения, а список источников догружается страницами: пока
    // он не дочитан, папка честнее скажет «показаны 3 из 12», чем промолчит.
    const isPartial = !sourcesLoading && folder.sources.length < folder.count;

    return (
        <li className="overflow-hidden rounded-xl border border-slate-200 transition hover:border-[#1f3a60]/40">
            <div className="flex items-stretch">
                <button
                    type="button"
                    onClick={() => onToggle(!isOpen)}
                    aria-expanded={isOpen}
                    aria-controls={panelId}
                    className="flex min-w-0 flex-1 items-center gap-2.5 px-3 py-2.5 text-left transition hover:bg-slate-50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-[#1f3a60]/40"
                >
                    <ChevronRight className={cn('h-4 w-4 shrink-0 text-slate-400 transition-transform duration-150', isOpen && 'rotate-90')} />
                    <FolderGlyph folder={folder} isOpen={isOpen} />
                    {/* break-words, а не truncate: название темы приходит от
                        модели и бывает длинным («Паём ва баромадҳои Президент»),
                        а обрезанное имя папки — это папка без имени. */}
                    <span className="min-w-0 flex-1 break-words text-sm font-semibold text-slate-900">{name}</span>
                    <span className="shrink-0 tabular-nums text-xs text-slate-500" title={t('topics.documentCount', { count: folder.count })}>
                        {folder.count}
                    </span>
                    {percent != null ? (
                        <span className="hidden shrink-0 text-xs font-semibold text-slate-700 sm:block">{t('topics.share', { value: percent })}</span>
                    ) : null}
                </button>

                {folder.clusterIndex != null ? (
                    <Link
                        to={`/sources${buildTopicFilterSearch(folder.clusterIndex, name)}`}
                        title={t('topics.open', { name })}
                        aria-label={t('topics.open', { name })}
                        className="flex shrink-0 items-center border-l border-slate-100 px-3 text-slate-400 transition hover:bg-slate-50 hover:text-[#1f3a60] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-[#1f3a60]/40"
                    >
                        <ArrowRight className="h-4 w-4" />
                    </Link>
                ) : null}
            </div>

            {/* Полоса вместо графика: доля читается быстрее в сравнении с
                соседями, а библиотек ради этого в проекте нет. */}
            {percent != null ? (
                <div className="mx-3 mb-2.5 h-1 overflow-hidden rounded-full bg-slate-100" aria-hidden="true">
                    <div className="h-full rounded-full bg-[#1f3a60]" style={{ width: `${Math.max(percent, 1)}%` }} />
                </div>
            ) : null}

            {isOpen ? (
                /* Отступ слева — не украшение: документы встают под названием
                   папки, и вложенность видно, не вчитываясь. */
                <div id={panelId} className="border-t border-slate-100 bg-slate-50/70 py-1.5 pl-7 pr-2">
                    {hintKey ? (
                        <p className="px-2.5 py-1.5 text-xs leading-5 text-slate-500">{t(hintKey)}</p>
                    ) : null}

                    {folder.sources.length > 0 ? (
                        <ul>
                            {folder.sources.map((source) => (
                                <FolderDocument
                                    key={source.id}
                                    source={source}
                                    locale={locale}
                                    t={t}
                                    statusLabels={statusLabels}
                                    onOpen={onOpenDocument}
                                />
                            ))}
                        </ul>
                    ) : sourcesLoading ? (
                        <p className="flex items-center gap-2 px-2.5 py-2 text-xs text-slate-500">
                            <Loader2 className="h-3.5 w-3.5 animate-spin" />
                            {t('documents.loading')}
                        </p>
                    ) : (
                        /* Причину (сбой загрузки списка) называет баннер над
                           папками — один раз. Повторять её в каждой раскрытой
                           папке значит трижды сказать одно и то же красным. */
                        <p className="px-2.5 py-2 text-xs text-slate-500">{t('topics.folders.empty')}</p>
                    )}

                    {isPartial && folder.sources.length > 0 ? (
                        <p className="px-2.5 py-1.5 text-xs text-slate-400">
                            {t('topics.folders.partial', { shown: folder.sources.length, count: folder.count })}
                        </p>
                    ) : null}
                </div>
            ) : null}
        </li>
    );
};

const TopicsPage = () => {
    const { locale, t } = useLocale();
    const { isAdmin } = useSessionRole();
    // Поиск в шапке иначе был бы мёртвым контролом: он есть на всех обычных
    // страницах, поэтому здесь он ищет и по названиям папок, и по именам
    // документов внутри них.
    const [searchParams] = useSearchParams();
    const searchTerm = searchParams.get('q') || '';

    // Та же область, что у списка источников и чата: экран показывает
    // распределение внутри активного блокнота, и область видна бейджем.
    const { notebookId, notebookName, canResetScope, resetScope } = useActiveNotebookScope(undefined);

    // Документы берём из общего кэша источников — того же, что у таблицы «Все
    // источники» и панели блокнота: экран раскладывает по папкам ровно те файлы,
    // которые пользователь видит в других местах, и своего запроса не шлёт.
    const {
        items: sources,
        isLoading: sourcesLoading,
        error: sourcesError,
        reload: reloadSources,
    } = useSources(notebookId);

    const folderDisclosure = useFolderDisclosure(searchTerm);
    // Просмотр документа тем же компонентом, что открывает ссылки в чате: файл
    // читается поверх экрана и папка под ним остаётся раскрытой.
    const [viewerSource, setViewerSource] = useState(null);

    const [topics, setTopics] = useState([]);
    // Документы, которые модель посмотрела и оставила без темы. null — сервер
    // об этом не сказал (старый бэкенд), и строку показывать нельзя: ноль в
    // этом месте соврал бы, что таких документов нет.
    const [unclearCount, setUnclearCount] = useState(null);
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
                setUnclearCount(
                    distribution.status === 'fulfilled' ? readUnclearCount(distribution.value) : null,
                );
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

    const statusLabels = useMemo(() => ({
        ready: t('documents.status.ready'),
        pending: t('documents.status.pending'),
        indexing: t('documents.status.indexing'),
        error: t('documents.status.error'),
    }), [t]);

    const folders = useMemo(
        () => buildTopicFolders(sources, locale, { topics, unclearCount }),
        [locale, sources, topics, unclearCount],
    );

    // Список делится надвое: папки с документами и пустые темы. Пустых на
    // небольшой базе большинство (модель различает двадцать тем, документы
    // попадают в три-четыре), и показанные вперемешку они превращают экран в
    // столбец нулей.
    //
    // При поиске деления нет: пользователь назвал тему сам, и прятать
    // найденное под второе раскрытие значило бы отфильтровать его же запрос
    // ещё раз.
    const { visibleFolders, filledFolders, emptyFolders, isSearching } = useMemo(() => {
        const query = searchTerm.trim();
        const matching = filterTopicFolders(folders, query, t);

        return {
            isSearching: Boolean(query),
            visibleFolders: matching,
            filledFolders: query ? matching : matching.filter((folder) => folder.count > 0),
            emptyFolders: query ? [] : matching.filter((folder) => folder.count === 0),
        };
    }, [folders, searchTerm, t]);

    // Папки открыты сразу, пока документов немного или пока идёт поиск: в первом
    // случае прятать нечего, во втором прятать нельзя — человек уже назвал, что
    // ищет, и лишний клик по каждой папке был бы платой за собственный запрос.
    const foldersOpenByDefault = isSearching || sources.length <= TOPIC_FOLDER_AUTO_OPEN_LIMIT;

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
                ) : visibleFolders.length === 0 ? (
                    <p className="rounded-xl bg-slate-50 p-8 text-center text-sm text-slate-500">
                        {isSearching ? t('topics.noMatches') : t('topics.empty')}
                    </p>
                ) : (
                    <>
                        {/* Папки строит распределение, а содержимое — список
                            источников: его сбой не повод прятать сами папки, но и
                            молчать о нём нельзя — иначе раскрытая папка выглядит
                            пустой без всякой причины. */}
                        {sourcesError ? (
                            <div role="alert" className="mb-3 flex flex-col gap-2 rounded-xl border border-red-200 bg-red-50 px-3 py-2.5 text-sm text-red-600">
                                <span className="flex items-center gap-2 font-semibold">
                                    <AlertTriangle className="h-4 w-4" />
                                    {sourcesError}
                                </span>
                                <Button type="button" variant="outline" size="sm" className="self-start" onClick={reloadSources}>
                                    <RefreshCw className="h-4 w-4" />
                                    {t('documents.retry')}
                                </Button>
                            </div>
                        ) : null}

                        {filledFolders.length === 0 ? (
                            <p className="rounded-xl bg-slate-50 p-8 text-center text-sm text-slate-500">{t('topics.empty')}</p>
                        ) : (
                            <ul className="space-y-2">
                                {filledFolders.map((folder) => (
                                    <TopicFolderRow
                                        key={folder.key}
                                        folder={folder}
                                        isOpen={folderDisclosure.isOpen(folder.key, foldersOpenByDefault)}
                                        onToggle={(open) => folderDisclosure.setOpen(folder.key, open)}
                                        totalDocuments={totalDocuments}
                                        locale={locale}
                                        t={t}
                                        statusLabels={statusLabels}
                                        sourcesLoading={sourcesLoading}
                                        onOpenDocument={setViewerSource}
                                    />
                                ))}
                            </ul>
                        )}

                        {/* Пустые темы — под раскрытием, тем же приёмом, что «О модели»
                            здесь и «Подробности» в списке источников. Папками они НЕ
                            делаются: раскрывать в них нечего, а переход в список
                            источников по пустой теме ведёт на пустой экран. */}
                        {emptyFolders.length > 0 ? (
                            <div className="mt-4 border-t border-slate-100 pt-4">
                                <div className="flex flex-wrap items-center gap-3">
                                    <button
                                        type="button"
                                        onClick={toggleEmpty}
                                        aria-expanded={emptyOpen}
                                        className="inline-flex items-center gap-2 rounded-lg border border-slate-200 bg-white px-3 py-1.5 text-xs font-semibold text-slate-600 transition hover:border-slate-300 hover:text-[#1f3a60] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#1f3a60] focus-visible:ring-offset-1"
                                    >
                                        <ChevronDown className={cn('h-4 w-4 transition-transform duration-150', emptyOpen && 'rotate-180')} />
                                        {t('topics.emptyClusters.action', { count: emptyFolders.length })}
                                    </button>
                                    {!emptyOpen ? (
                                        <span className="text-xs text-slate-400">{t('topics.emptyClusters.hint')}</span>
                                    ) : null}
                                </div>

                                {emptyOpen ? (
                                    <ul className="mt-3 flex flex-wrap gap-2">
                                        {emptyFolders.map((folder) => (
                                            <li
                                                key={folder.key}
                                                className="rounded-lg border border-dashed border-slate-200 px-3 py-1.5 text-xs text-slate-400"
                                            >
                                                {resolveTopicFolderName(folder, t)}
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

            {viewerSource ? (
                <DocumentViewer
                    docId={viewerSource.id}
                    docName={viewerSource.name}
                    onClose={() => setViewerSource(null)}
                />
            ) : null}
        </div>
    );
};

export default TopicsPage;
