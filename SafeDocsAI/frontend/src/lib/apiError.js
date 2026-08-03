// Машинные коды ошибок бэкенда -> ключи i18n. Неизвестный код осознанно деградирует
// в общее локализованное сообщение, поэтому новые коды не ломают интерфейс.
const ERROR_CODE_KEYS = {
    // Внутренние коды без префикса: их выводят DETAIL_CODE_PATTERNS и STATUS_CODE_KEYS,
    // когда бэкенд error_code не прислал (например, старый ответ или чужой прокси).
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
    // Коды раздела источников ровно в том виде, в каком их отдаёт бэкенд
    // (backend/app/core/exceptions.py, SourceErrors). Имена там свои, поэтому
    // сопоставление идёт по смыслу, а не приписыванием префикса.
    'source.filename_required': 'documents.errors.filenameRequired',
    'source.filename_too_long': 'documents.errors.filenameTooLong',
    'source.unsupported_type': 'documents.errors.unsupportedFileType',
    'source.invalid_content_type': 'documents.errors.invalidContentType',
    'source.too_large': 'documents.errors.fileTooLarge',
    'source.invalid_upload': 'documents.errors.invalidUpload',
    'source.not_found': 'documents.errors.documentNotFound',
    'source.notebook_not_found': 'documents.errors.notebookNotFound',
    'source.chunk_not_found': 'documents.errors.chunkNotFound',
    'source.note_not_found': 'documents.errors.noteNotFound',
    'source.insight_not_found': 'documents.errors.insightNotFound',
    'source.file_missing': 'documents.errors.fileMissing',
    'source.no_ids_provided': 'documents.errors.noIdsProvided',
    // Блокноты: профиль предметной области проверяется и на создании, и на PATCH.
    'source.unsupported_domain_profile': 'documents.errors.unsupportedDomainProfile',
    // Частичное обновление: пустое тело PATCH и недопустимый статус заметки.
    'source.nothing_to_update': 'documents.errors.nothingToUpdate',
    'source.invalid_note_status': 'documents.errors.invalidNoteStatus',
    'source.vector_store_unavailable': 'documents.errors.vectorStoreUnavailable',
    'source.delete_file_failed': 'documents.errors.deleteFileFailed',
    // Конфликты удаления блокнота (409): работу либо уже делает параллельный
    // запрос, либо мешает незавершённая индексация — оба раза достаточно повторить.
    'source.notebook_delete_conflict': 'documents.errors.notebookDeleteConflict',
    'source.notebook_busy_indexing': 'documents.errors.notebookBusyIndexing',
    // Коды индексации приходят не в HTTP-ответе, а в поле document.error_code,
    // но проходят через ту же таблицу, когда карточка источника показывает причину.
    'source.indexing_failed': 'documents.errors.indexingFailed',
    'source.indexing_interrupted': 'documents.errors.indexingInterrupted',
    'source.text_extraction_failed': 'documents.errors.extractionFailed',
    'source.encoding_not_utf8': 'documents.errors.invalidEncoding',
    'source.deleted_before_indexing': 'documents.errors.deletedBeforeIndexing',
    // Коды аутентификации: мёртвый токен приходит с 401, нехватка прав — с 403.
    'auth.invalid_token': 'auth.errors.sessionExpired',
    'auth.token_revoked': 'auth.errors.sessionRevoked',
    'auth.forbidden': 'auth.errors.forbidden',
    'auth.invalid_credentials': 'auth.errors.invalidCredentials',
    'auth.weak_password': 'auth.errors.weakPassword',
    'auth.registration_disabled': 'auth.errors.registrationDisabled',
    'auth.registration_rejected': 'auth.errors.registrationRejected',
    // Запрос по куке без совпадающего CSRF-заголовка: чаще всего кука сессии
    // истекла или была снята в другой вкладке.
    'auth.csrf_failed': 'auth.errors.csrfFailed',
    // Журнал запросов живёт отдельной сущностью и своим префиксом
    // (backend/app/core/exceptions.py, LogErrors), поэтому и ключ перевода свой:
    // «запись журнала не найдена» — не то же самое, что «источник не найден».
    'log.not_found': 'logs.errors.notFound',
    // Вопрос к ассистенту: отказ приходит одинаково из чата и из Ask, поэтому и
    // префикс, и ключ перевода общие (backend/app/core/exceptions.py, ChatErrors).
    'chat.question_required': 'chat.errors.questionRequired',
    'chat.question_too_long': 'chat.errors.questionTooLong',
    // Непойманное исключение. Префикс верхнеуровневый: такое прилетает из любого
    // эндпоинта, секцию по нему не восстановить. Пользователю показываем одно и то
    // же независимо от причины — разбирается она по request_id в логе сервера.
    'internal.error': 'errors.internal',
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
    // 429 приходит обычным HTTPException без error_code (backend/app/core/
    // rate_limit.py, check_rate_limit) — и от auth с chat, и от загрузки
    // источников, создания блокнотов, заметок и инсайтов. Ловим его дважды:
    // по detail здесь и по статусу ниже, чтобы сообщение не зависело от того,
    // дошло ли тело ответа (за прокси 429 бывает и без JSON).
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
 * Локализованное сообщение по одному машинному коду — без объекта HTTP-ошибки вокруг.
 *
 * Реестр кодов у бэкенда общий: те же значения, что приходят в теле ответа полем
 * error_code, приходят и полем объекта (document.error_code объясняет, почему источник
 * не проиндексировался). Таблица переводов от этого не зависит, поэтому разбор кода
 * вынесен сюда, а resolveApiErrorMessage лишь добавляет к нему извлечение кода из ответа.
 *
 * Пустая строка означает «перевода нет» (код пуст, неизвестен или ключ не заполнен) —
 * решение, чем это заменить, остаётся за вызывающим: у карточки источника и у формы
 * фолбэки разные.
 */
export const resolveErrorCodeMessage = (errorCode, t) => {
    const code = String(errorCode || '').trim().toLowerCase();
    if (!code) return '';

    const messageKey = ERROR_CODE_KEYS[code];
    if (!messageKey) return '';

    const message = t(messageKey);
    return message && message !== messageKey ? message : '';
};

/**
 * Приоритет: локализованное сообщение по коду -> локализованный фолбэк.
 * Сырой detail с бэкенда (английский) показываем только когда сопоставить не удалось
 * и общего сообщения недостаточно, чтобы понять причину.
 */
export const resolveApiErrorMessage = (error, t, fallbackKey) => {
    const code = resolveErrorCode(error);

    const message = resolveErrorCodeMessage(code, t);
    if (message) return message;

    // Пришёл машинный код, которого нет в таблице — значит, он появился на бэкенде
    // позже фронта. Сырой detail в этом случае не показываем: он на одном языке и
    // описывает внутренности (пути, имена сервисов), а полный объект ошибки и так
    // уходит в console.error на месте вызова. Без кода detail остаётся единственной
    // подсказкой — например, для 422, где его собирает FastAPI, а не наш обработчик.
    const hasExplicitCode = Boolean(error?.response?.data?.error_code || error?.response?.data?.code);
    if (!code && !hasExplicitCode) {
        const detail = extractDetail(error?.response?.data);
        if (detail) return detail;
    }

    return t(fallbackKey);
};
