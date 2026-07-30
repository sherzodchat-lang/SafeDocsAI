// Единая нормализация источников: статусы, размер и формат ответа списка.
// Раньше эти правила были продублированы в таблице «Все источники» и в панели блокнота.

// Статусы, при которых индексация ещё не завершилась и список надо опрашивать.
export const IN_PROGRESS_STATUSES = ['pending', 'indexing'];

const READY_HINTS = ['ready', 'done', 'success', 'complete'];

/**
 * Приводит статус бэкенда к одному из четырёх ключей интерфейса.
 * Неизвестный статус считаем «в работе»: индексация идёт в фоне, значение ещё уточнится.
 */
export const resolveStatus = (status) => {
    const normalized = String(status || '').toLowerCase();

    if (normalized.includes('error') || normalized.includes('fail')) return 'error';
    if (normalized === 'indexed' || READY_HINTS.some((hint) => normalized.includes(hint))) return 'ready';
    if (normalized === 'pending' || normalized === 'queued') return 'pending';

    return 'indexing';
};

export const isSourceInProgress = (status) => IN_PROGRESS_STATUSES.includes(resolveStatus(status));

export const hasSourcesInProgress = (sources) => (sources || []).some((source) => isSourceInProgress(source?.status));

const SIZE_UNIT_KEYS = ['documents.size.bytes', 'documents.size.kilobytes', 'documents.size.megabytes'];

/**
 * Размер файла с локализованной единицей измерения.
 * Единицы переводятся: раньше таблица показывала латинские B/KB/MB, а панель блокнота —
 * захардкоженные русские Б/КБ/МБ, которые не менялись при переключении на таджикский.
 */
export const formatSize = (size, t) => {
    const bytes = Number(size);
    if (!Number.isFinite(bytes) || bytes < 0) return t('documents.size.unknown');

    if (bytes < 1024) return t(SIZE_UNIT_KEYS[0], { value: Math.round(bytes) });

    const kilobytes = bytes / 1024;
    if (kilobytes < 1024) return t(SIZE_UNIT_KEYS[1], { value: kilobytes.toFixed(1) });

    return t(SIZE_UNIT_KEYS[2], { value: (kilobytes / 1024).toFixed(1) });
};

const TOTAL_COUNT_HEADER = 'x-total-count';

const readTotalHeader = (headers) => {
    if (!headers) return null;

    const raw = typeof headers.get === 'function'
        ? headers.get(TOTAL_COUNT_HEADER)
        : headers[TOTAL_COUNT_HEADER];
    const total = Number(raw);

    return Number.isFinite(total) && raw != null && raw !== '' ? total : null;
};

/**
 * Контракт GET /sources/ переходный: тело — голый массив (общее число приходит
 * заголовком X-Total-Count) либо объект { items, total }.
 */
export const normalizeSourcesResponse = (response) => {
    const payload = response?.data;
    const headerTotal = readTotalHeader(response?.headers);

    if (Array.isArray(payload)) {
        return { items: payload, total: headerTotal };
    }

    if (payload && Array.isArray(payload.items)) {
        const bodyTotal = Number(payload.total);
        return {
            items: payload.items,
            total: Number.isFinite(bodyTotal) ? bodyTotal : headerTotal,
        };
    }

    return { items: [], total: headerTotal };
};
