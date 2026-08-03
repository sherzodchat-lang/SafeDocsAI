import { useCallback, useEffect, useState } from 'react';

import { notebooksService } from '../services/notebooksService';

export const ACTIVE_NOTEBOOK_STORAGE_KEY = 'knowledgeai.activeNotebookId';

export const readActiveNotebookId = () => {
    if (typeof window === 'undefined') return null;
    const storedValue = localStorage.getItem(ACTIVE_NOTEBOOK_STORAGE_KEY);
    if (!storedValue) return null;
    const parsed = Number(storedValue);
    return Number.isFinite(parsed) ? parsed : null;
};

const resolveScopeNotebookId = (notebookId, activeNotebookId) => {
    if (notebookId !== undefined) {
        return notebookId == null ? null : Number(notebookId);
    }

    return activeNotebookId;
};

/**
 * Область данных экрана: либо блокнот, заданный маршрутом, либо активный блокнот
 * из localStorage — тот же, что выбирается кнопкой в шапке блокнота.
 *
 * Хук общий для чата и списка источников: оба сужаются одним и тем же ключом, и
 * расходиться в том, как они его читают, сбрасывают и подписываются на чужие
 * вкладки, им нельзя — иначе пользователь снова получит область, которую видит
 * один экран и не видит другой.
 *
 * @param {number|null|undefined} notebookId блокнот маршрута; undefined — область берётся из localStorage.
 * @param {{ withNotebookName?: boolean }} options имя блокнота запрашиваем только там, где его показывают.
 */
export const useActiveNotebookScope = (notebookId, { withNotebookName = true } = {}) => {
    const [activeNotebookId, setActiveNotebookId] = useState(readActiveNotebookId);
    // Имя храним вместе с id, для которого оно получено: при смене области старое
    // имя иначе повисло бы рядом с новым id, пока не придёт ответ.
    const [loadedNotebook, setLoadedNotebook] = useState(null);

    const scopeNotebookId = resolveScopeNotebookId(notebookId, activeNotebookId);
    // Область из localStorage принадлежит пользователю: только её он и может сбросить.
    const isStoredScope = notebookId === undefined;

    useEffect(() => {
        if (!isStoredScope) return undefined;

        // event.key === null — localStorage.clear() в другой вкладке.
        const handleStorage = (event) => {
            if (event.key === ACTIVE_NOTEBOOK_STORAGE_KEY || event.key === null) {
                setActiveNotebookId(readActiveNotebookId());
            }
        };

        window.addEventListener('storage', handleStorage);
        return () => window.removeEventListener('storage', handleStorage);
    }, [isStoredScope]);

    // Бейдж обещает контекст запроса, поэтому показываем имя блокнота, а не голый id.
    useEffect(() => {
        if (scopeNotebookId == null || !withNotebookName) return undefined;

        let isMounted = true;

        notebooksService.getById(scopeNotebookId)
            .then((response) => {
                if (isMounted) setLoadedNotebook({ id: scopeNotebookId, name: response.data?.name || '' });
            })
            .catch((error) => {
                console.error('Failed to fetch scope notebook', error);
                if (!isMounted) return;
                // Блокнот удалён в другой вкладке: остаться с мёртвым id нельзя —
                // запросы уйдут в 404, а бейдж будет обещать несуществующий контекст.
                if (error?.response?.status === 404 && isStoredScope) {
                    localStorage.removeItem(ACTIVE_NOTEBOOK_STORAGE_KEY);
                    setActiveNotebookId(null);
                }
            });

        return () => {
            isMounted = false;
        };
    }, [isStoredScope, scopeNotebookId, withNotebookName]);

    // Область должна управляться с того же экрана, где она видна: иначе вернуться
    // ко всем источникам можно было бы только уйдя на страницу блокнотов. Событие
    // storage в своей же вкладке не срабатывает, поэтому состояние правим здесь
    // же; соседние вкладки получат событие от браузера.
    const resetScope = useCallback(() => {
        localStorage.removeItem(ACTIVE_NOTEBOOK_STORAGE_KEY);
        setActiveNotebookId(null);
    }, []);

    return {
        notebookId: scopeNotebookId,
        notebookName: loadedNotebook && loadedNotebook.id === scopeNotebookId ? loadedNotebook.name : '',
        // Сбрасывать нечего там, где область задана маршрутом или уже равна «все источники».
        canResetScope: isStoredScope && scopeNotebookId != null,
        resetScope,
    };
};
