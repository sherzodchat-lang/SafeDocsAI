/* eslint-disable react-refresh/only-export-components */
import React, { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from 'react';

import { useIndexingPoll } from '../hooks/useIndexingPoll';
import { useLocale } from '../i18n';
import { resolveApiErrorMessage } from '../lib/apiError';
import { hasSourcesInProgress, normalizeSourcesResponse } from '../lib/sources';
import { sourcesService } from '../services/sourcesService';

// Бэкенд отдаёт максимум 100 записей за запрос, поэтому список долистываем постранично.
const FETCH_PAGE_LIMIT = 100;
// Предохранитель: даже при сбое пагинации на сервере не уходим в бесконечный цикл запросов.
const MAX_FETCH_PAGES = 20;

// Пока данные свежие, повторное открытие экрана не шлёт новый запрос.
const STALE_AFTER_MS = 15000;

export const ALL_SOURCES_SCOPE = 'all';

/** Область данных: либо все источники, либо источники одного блокнота. */
export const getSourcesScopeKey = (notebookId) => {
    const numericId = Number(notebookId);
    if (notebookId === null || notebookId === undefined || notebookId === '' || !Number.isFinite(numericId)) {
        return ALL_SOURCES_SCOPE;
    }

    return `notebook:${numericId}`;
};

const getNotebookIdFromScope = (scope) => (
    scope === ALL_SOURCES_SCOPE ? null : Number(scope.slice('notebook:'.length))
);

const EMPTY_ENTRY = Object.freeze({
    items: [],
    total: 0,
    isTruncated: false,
    isLoading: true,
    error: '',
    loadedAt: 0,
});

const SourcesContext = createContext(null);

const loadAllPages = async (notebookId) => {
    const collected = [];
    let reportedTotal = null;
    let isTruncated = false;

    for (let pageIndex = 0; pageIndex < MAX_FETCH_PAGES; pageIndex += 1) {
        // Страницы читаем последовательно: следующий skip зависит от того, вернулась ли полная страница.
        const response = await sourcesService.getPage({
            notebookId,
            skip: pageIndex * FETCH_PAGE_LIMIT,
            limit: FETCH_PAGE_LIMIT,
        });
        const { items, total } = normalizeSourcesResponse(response);
        collected.push(...items);

        if (total != null) reportedTotal = total;
        if (items.length < FETCH_PAGE_LIMIT) break;
        if (reportedTotal != null && collected.length >= reportedTotal) break;
        if (pageIndex === MAX_FETCH_PAGES - 1) isTruncated = true;
    }

    return {
        items: collected,
        total: reportedTotal != null ? reportedTotal : collected.length,
        isTruncated,
    };
};

/**
 * Единый слой данных по источникам.
 *
 * Один кэш на область (все источники / конкретный блокнот) обслуживает таблицу
 * «Все источники», панель источников блокнота и шапку блокнота: одновременные
 * потребители делят один запрос, а загрузка, удаление и привязка источника
 * обновляют сразу все открытые экраны.
 */
export const SourcesProvider = ({ children }) => {
    const { t } = useLocale();
    const [entries, setEntries] = useState({});
    const [subscribedScopes, setSubscribedScopes] = useState([]);

    // Кэш держим и в ref: колбэки читают актуальные данные, не пересоздаваясь на каждый ответ.
    const entriesRef = useRef(entries);
    const inFlightRef = useRef(new Map());
    const lastRequestIdRef = useRef(new Map());
    const subscriberCountRef = useRef(new Map());
    const tRef = useRef(t);

    useEffect(() => {
        tRef.current = t;
    }, [t]);

    const writeEntry = useCallback((scope, patch) => {
        const previous = entriesRef.current[scope] || EMPTY_ENTRY;
        entriesRef.current = { ...entriesRef.current, [scope]: { ...previous, ...patch } };
        setEntries(entriesRef.current);
    }, []);

    const runFetch = useCallback((scope, { showLoading = false, background = false, force = false } = {}) => {
        // Дедупликация: пока запрос по области в полёте, новые потребители делят его результат.
        const pending = inFlightRef.current.get(scope);
        if (pending && !force) return pending;

        const requestId = (lastRequestIdRef.current.get(scope) || 0) + 1;
        lastRequestIdRef.current.set(scope, requestId);

        if (showLoading) {
            writeEntry(scope, { isLoading: true, error: '' });
        }

        let request;
        request = (async () => {
            try {
                const result = await loadAllPages(getNotebookIdFromScope(scope));

                // Побеждает последний ЗАПРОШЕННЫЙ, а не последний ответивший:
                // ответ обгоняющего запроса по той же области отбрасываем.
                if (lastRequestIdRef.current.get(scope) !== requestId) return;

                writeEntry(scope, {
                    items: result.items,
                    total: result.total,
                    isTruncated: result.isTruncated,
                    isLoading: false,
                    error: '',
                    loadedAt: Date.now(),
                });
            } catch (error) {
                console.error('Failed to fetch sources:', error);
                if (lastRequestIdRef.current.get(scope) !== requestId) return;

                // Сбой фонового опроса не должен затирать уже показанный список сообщением об ошибке.
                if (background) return;

                // Сбой загрузки списка не должен выглядеть как пустая база: элементы сохраняем.
                writeEntry(scope, {
                    isLoading: false,
                    error: resolveApiErrorMessage(error, tRef.current, 'documents.loadFailed'),
                });
            } finally {
                if (inFlightRef.current.get(scope) === request) {
                    inFlightRef.current.delete(scope);
                }
            }
        })();

        inFlightRef.current.set(scope, request);
        return request;
    }, [writeEntry]);

    const ensureLoaded = useCallback((scope) => {
        const entry = entriesRef.current[scope];
        const isFresh = entry && !entry.error && entry.loadedAt > 0 && Date.now() - entry.loadedAt < STALE_AFTER_MS;
        if (isFresh || inFlightRef.current.has(scope)) return;

        runFetch(scope, { showLoading: true });
    }, [runFetch]);

    const refresh = useCallback((scope) => runFetch(scope, { showLoading: true, force: true }), [runFetch]);

    const subscribe = useCallback((scope) => {
        const count = (subscriberCountRef.current.get(scope) || 0) + 1;
        subscriberCountRef.current.set(scope, count);
        if (count === 1) {
            setSubscribedScopes((prev) => (prev.includes(scope) ? prev : [...prev, scope]));
        }

        return () => {
            const rest = (subscriberCountRef.current.get(scope) || 1) - 1;
            if (rest > 0) {
                subscriberCountRef.current.set(scope, rest);
                return;
            }

            subscriberCountRef.current.delete(scope);
            setSubscribedScopes((prev) => prev.filter((item) => item !== scope));
        };
    }, []);

    /** Общая инвалидация: открытые экраны перечитывают список, закрытые — при следующем открытии. */
    const invalidate = useCallback(async () => {
        const scopes = Object.keys(entriesRef.current);
        const active = scopes.filter((scope) => (subscriberCountRef.current.get(scope) || 0) > 0);

        scopes
            .filter((scope) => !active.includes(scope))
            .forEach((scope) => writeEntry(scope, { loadedAt: 0 }));

        await Promise.all(active.map((scope) => runFetch(scope, { force: true })));
    }, [runFetch, writeEntry]);

    const insertSource = useCallback((source, notebookId) => {
        const targetNotebookId = source?.notebook_id ?? notebookId;
        const scopes = new Set([ALL_SOURCES_SCOPE, getSourcesScopeKey(targetNotebookId)]);

        scopes.forEach((scope) => {
            const entry = entriesRef.current[scope];
            // Незагруженную область не «подсаживаем»: она прочитается целиком при первом открытии.
            if (!entry) return;
            if (entry.items.some((item) => item.id === source.id)) return;

            writeEntry(scope, { items: [source, ...entry.items], total: entry.total + 1 });
        });
    }, [writeEntry]);

    const removeSource = useCallback((sourceId) => {
        Object.keys(entriesRef.current).forEach((scope) => {
            const entry = entriesRef.current[scope];
            if (!entry.items.some((item) => item.id === sourceId)) return;

            writeEntry(scope, {
                items: entry.items.filter((item) => item.id !== sourceId),
                total: Math.max(0, entry.total - 1),
            });
        });
    }, [writeEntry]);

    const updateSource = useCallback((sourceId, patch) => {
        Object.keys(entriesRef.current).forEach((scope) => {
            const entry = entriesRef.current[scope];
            if (!entry.items.some((item) => item.id === sourceId)) return;

            writeEntry(scope, {
                items: entry.items.map((item) => (item.id === sourceId ? { ...item, ...patch } : item)),
            });
        });
    }, [writeEntry]);

    const uploadSource = useCallback(async (file, notebookId) => {
        const formData = new FormData();
        formData.append('file', file);
        if (notebookId != null) {
            formData.append('notebook_id', String(notebookId));
        }

        const response = await sourcesService.upload(formData);
        // Сразу добавляем источник в список: реальный статус придёт от сервера, иначе показываем очередь.
        const created = response.data || {
            id: `temp-${Date.now()}-${file.name}`,
            name: file.name,
            size: file.size,
            language: 'ru',
            created_at: new Date().toISOString(),
        };

        insertSource({ status: 'pending', ...created }, notebookId);
        return created;
    }, [insertSource]);

    const deleteSource = useCallback(async (sourceId) => {
        await sourcesService.delete(sourceId);
        removeSource(sourceId);
    }, [removeSource]);

    const attachSources = useCallback(async (notebookId, sourceIds) => {
        await sourcesService.attachExisting({ notebook_id: notebookId, source_ids: sourceIds });
        await invalidate();
    }, [invalidate]);

    /**
     * Снять источники с блокнота, не удаляя их. Обратная операция к
     * attachSources и сделана так же: сам документ остаётся в области «все
     * источники», меняется только его блокнот, поэтому точечной правки кэша
     * мало — область блокнота и область «все источники» расходятся, и обе
     * перечитываются общей инвалидацией.
     *
     * Наружу отдаём updated_count ответа: ноль означает «в этом блокноте его
     * уже не было» (повторное нажатие, вторая вкладка) — это успех, а не сбой,
     * и разговор с пользователем у него другой.
     */
    const detachSources = useCallback(async (notebookId, sourceIds) => {
        const response = await sourcesService.detachExisting({ notebook_id: notebookId, source_ids: sourceIds });
        await invalidate();
        return Number(response?.data?.updated_count ?? 0);
    }, [invalidate]);

    // Индексация идёт в фоне: опрашиваем только те области, что открыты и содержат источники в очереди.
    const scopesToPoll = useMemo(
        () => subscribedScopes.filter((scope) => hasSourcesInProgress(entries[scope]?.items)),
        [entries, subscribedScopes],
    );

    const pollSources = useCallback(async () => {
        await Promise.all(scopesToPoll.map((scope) => runFetch(scope, { background: true, force: true })));
    }, [runFetch, scopesToPoll]);

    useIndexingPoll(scopesToPoll.length > 0, pollSources);

    const value = useMemo(() => ({
        entries,
        ensureLoaded,
        subscribe,
        refresh,
        invalidate,
        uploadSource,
        deleteSource,
        attachSources,
        detachSources,
        updateSource,
    }), [attachSources, deleteSource, detachSources, ensureLoaded, entries, invalidate, refresh, subscribe, updateSource, uploadSource]);

    return <SourcesContext.Provider value={value}>{children}</SourcesContext.Provider>;
};

export const useSourcesActions = () => {
    const context = useContext(SourcesContext);
    if (!context) {
        throw new Error('useSourcesActions must be used within SourcesProvider');
    }

    const { uploadSource, deleteSource, attachSources, detachSources, updateSource, invalidate } = context;
    return useMemo(
        () => ({ uploadSource, deleteSource, attachSources, detachSources, updateSource, invalidate }),
        [attachSources, deleteSource, detachSources, invalidate, updateSource, uploadSource],
    );
};

/**
 * Список источников выбранной области. Все потребители читают один кэш,
 * поэтому одновременное открытие шапки, панели и таблицы даёт один запрос.
 */
export const useSources = (notebookId, { enabled = true } = {}) => {
    const context = useContext(SourcesContext);
    if (!context) {
        throw new Error('useSources must be used within SourcesProvider');
    }

    const { entries, ensureLoaded, subscribe, refresh } = context;
    const scope = getSourcesScopeKey(notebookId);

    useEffect(() => {
        if (!enabled) return undefined;
        return subscribe(scope);
    }, [enabled, scope, subscribe]);

    useEffect(() => {
        if (!enabled) return;
        ensureLoaded(scope);
    }, [enabled, ensureLoaded, scope]);

    const entry = entries[scope] || EMPTY_ENTRY;
    const reload = useCallback(() => refresh(scope), [refresh, scope]);

    return {
        items: entry.items,
        total: entry.total,
        isTruncated: entry.isTruncated,
        // До первого запроса область считается загружающейся, иначе экран моргает пустым состоянием.
        isLoading: enabled ? entry.isLoading : false,
        error: entry.error,
        reload,
    };
};

export default SourcesContext;
