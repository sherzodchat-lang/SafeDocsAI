import { normalizeLocale } from '../i18n';

const INTL_LOCALES = {
    ru: 'ru-RU',
    tg: 'tg-TJ',
};

export const getIntlLocale = (locale) => INTL_LOCALES[normalizeLocale(locale) || 'tg'];

// Метка времени без часового пояса: "2026-07-30T09:15:00", "2026-07-30 09:15:00.123".
const NAIVE_TIMESTAMP_PATTERN = /^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(:\d{2})?(\.\d+)?$/;

/**
 * Разбирает и старый формат бэкенда (наивная метка без зоны, фактически UTC),
 * и новый однозначный ISO с зоной. Без явного 'Z' браузер трактовал бы наивную
 * метку как локальное время и сдвигал дату на несколько часов.
 */
export const parseTimestamp = (value) => {
    if (value instanceof Date) return value;
    if (typeof value === 'number') return new Date(value);

    const raw = String(value || '').trim();
    if (!raw) return null;

    if (NAIVE_TIMESTAMP_PATTERN.test(raw)) {
        return new Date(`${raw.replace(' ', 'T')}Z`);
    }

    return new Date(raw);
};

export const formatLocaleDate = (value, locale, options, fallback = '-') => {
    if (!value) return fallback;

    const date = parseTimestamp(value);
    if (!date || Number.isNaN(date.getTime())) return fallback;

    return new Intl.DateTimeFormat(getIntlLocale(locale), options).format(date);
};

export const formatDocumentLanguage = (language) => {
    const normalized = normalizeLocale(language);

    if (normalized === 'tg') {
        return { code: 'tg', text: 'TG', className: 'bg-yellow-100 text-yellow-800' };
    }

    if (normalized === 'ru') {
        return { code: 'ru', text: 'RU', className: 'bg-blue-100 text-blue-700' };
    }

    return { code: null, text: String(language || '—').trim().toUpperCase() || '—', className: 'bg-slate-100 text-slate-700' };
};
