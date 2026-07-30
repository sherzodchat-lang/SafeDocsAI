// Машинные коды ошибок бэкенда -> ключи i18n. Неизвестный код осознанно деградирует
// в общее локализованное сообщение, поэтому новые коды не ломают интерфейс.
const ERROR_CODE_KEYS = {
    unsupported_file_type: 'documents.errors.unsupportedFileType',
    file_too_large: 'documents.errors.fileTooLarge',
    text_extraction_failed: 'documents.errors.extractionFailed',
    invalid_encoding: 'documents.errors.invalidEncoding',
    notebook_not_found: 'documents.errors.notebookNotFound',
    document_not_found: 'documents.errors.documentNotFound',
    unauthorized: 'documents.errors.unauthorized',
    forbidden: 'documents.errors.forbidden',
    rate_limited: 'documents.errors.rateLimited',
    network_error: 'documents.errors.network',
    server_error: 'documents.errors.server',
    // Коды аутентификации: мёртвый токен приходит с 401, нехватка прав — с 403.
    'auth.invalid_token': 'auth.errors.sessionExpired',
    'auth.token_revoked': 'auth.errors.sessionRevoked',
    'auth.forbidden': 'auth.errors.forbidden',
    'auth.invalid_credentials': 'auth.errors.invalidCredentials',
    'auth.weak_password': 'auth.errors.weakPassword',
    // Запрос по куке без совпадающего CSRF-заголовка: чаще всего кука сессии
    // истекла или была снята в другой вкладке.
    'auth.csrf_failed': 'auth.errors.csrfFailed',
};

// Пока бэкенд не отдаёт error_code, тот же код выводим из английского текста detail.
const DETAIL_CODE_PATTERNS = [
    [/unsupported file type|allowed:\s*pdf/i, 'unsupported_file_type'],
    [/too large|file size|exceeds/i, 'file_too_large'],
    [/extract text|no text could be|empty text/i, 'text_extraction_failed'],
    [/utf-?8|кодировк/i, 'invalid_encoding'],
    [/notebook not found/i, 'notebook_not_found'],
    [/(document|documents|source|sources|file|chunk)s? not found/i, 'document_not_found'],
    [/could not validate credentials|not authenticated/i, 'unauthorized'],
    [/enough privileges|forbidden|not allowed/i, 'forbidden'],
    [/rate limit|too many requests/i, 'rate_limited'],
];

const STATUS_CODE_KEYS = {
    401: 'unauthorized',
    403: 'forbidden',
    404: 'document_not_found',
    413: 'file_too_large',
    429: 'rate_limited',
};

export const extractDetail = (data) => {
    const detail = data?.detail ?? data?.message;

    if (typeof detail === 'string') return detail.trim();
    if (Array.isArray(detail)) return detail.map((item) => item?.msg || '').filter(Boolean).join('; ');
    if (detail && typeof detail === 'object') return String(detail.msg || '').trim();
    return '';
};

export const resolveErrorCode = (error) => {
    const data = error?.response?.data;
    const explicitCode = data?.error_code || data?.code;
    if (explicitCode) return String(explicitCode).trim().toLowerCase();

    // Ответа нет вовсе: сеть, CORS или таймаут.
    if (!error?.response) return 'network_error';

    const detail = extractDetail(data);
    const matched = DETAIL_CODE_PATTERNS.find(([pattern]) => pattern.test(detail));
    if (matched) return matched[1];

    const status = Number(error.response.status);
    if (STATUS_CODE_KEYS[status]) return STATUS_CODE_KEYS[status];
    if (status >= 500) return 'server_error';

    return null;
};

/**
 * Приоритет: локализованное сообщение по коду -> локализованный фолбэк.
 * Сырой detail с бэкенда (английский) показываем только когда сопоставить не удалось
 * и общего сообщения недостаточно, чтобы понять причину.
 */
export const resolveApiErrorMessage = (error, t, fallbackKey) => {
    const code = resolveErrorCode(error);
    const messageKey = code ? ERROR_CODE_KEYS[code] : null;

    if (messageKey) {
        const message = t(messageKey);
        if (message && message !== messageKey) return message;
    }

    const hasExplicitCode = Boolean(error?.response?.data?.error_code || error?.response?.data?.code);
    if (!code && !hasExplicitCode) {
        const detail = extractDetail(error?.response?.data);
        if (detail) return detail;
    }

    return t(fallbackKey);
};
