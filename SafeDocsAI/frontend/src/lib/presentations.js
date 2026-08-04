// Единый слой правил раздела презентаций: границы заказа, статусы, языки и
// разбор причины отказа. Форма, список и опрос читают их отсюда, а не выписывают
// по месту — ровно по той же причине, по какой это сделано у источников
// (lib/sources.js): одни и те же числа в двух местах расходятся молча.

import { normalizeLocale } from '../i18n';
import { resolveErrorCodeMessage } from './apiError';

/* ────────────────────────── Границы заказа ──────────────────────────────
 *
 * ЗЕРКАЛО серверных констант: backend/app/modules/presentations/constants.py.
 * Оттуда их берёт единственная серверная проверка заказа
 * (PresentationService.validate / generate_presentation, service.py: строка
 * `if not SLIDE_COUNT_MIN <= presentation.slide_count <= SLIDE_COUNT_MAX`), и
 * оттуда же считается предел описания (DESCRIPTION_MAX выведен из бюджета
 * контекста план-вызова — расчёт приведён в самом файле констант).
 *
 * Python отсюда прочитать нечем, а в теле ответов границ нет, поэтому числа
 * приходится повторять. Повторение сделано в том же духе, что и границы
 * настроек в SettingsPage.jsx: отдельными именованными константами вида
 * `const NAME = <число>;` — их разбирает регулярным выражением тест зеркала
 * (backend/tests/test_settings_limits_single_source.py, _CONST_RE), и такой же
 * тест для презентаций сможет сверить эти объявления с constants.py, не
 * переписывая парсер. Числами по месту (в min/max поля, в проверке ввода) они
 * не пишутся: разъехаться этим двум применениям нельзя.
 *
 * ВНИМАНИЕ при сверке: границы здесь ПРОДУКТОВЫЕ, а не схемные. Схема
 * собирает колоду и от трёх слайдов (MIN_SLIDE_COUNT в llm_schemas.py:
 * титул, один содержательный и «Источники»), а бюджет план-вызова допускает
 * примерно до двадцати — но заказывать разрешено 5..15. Сверять надо с
 * SLIDE_COUNT_MIN/MAX в constants.py, не со схемными пределами.
 */

// backend/app/modules/presentations/constants.py: SLIDE_COUNT_MIN
const SLIDE_COUNT_MIN = 5;
// backend/app/modules/presentations/constants.py: SLIDE_COUNT_MAX
const SLIDE_COUNT_MAX = 15;
// backend/app/modules/presentations/constants.py: DESCRIPTION_MAX
const DESCRIPTION_MAX = 1800;

/**
 * Умолчание числа слайдов. Своего значения на сервере у него нет — заказ
 * приходит с явным slide_count, — поэтому это решение интерфейса, а не зеркало:
 * 10 слайдов лежат внутри границ и совпадают с конфигурацией, на которой
 * измерялся PRESENTATION_JOB_TIMEOUT (замеры этапа 0 в constants.py).
 */
const DEFAULT_SLIDE_COUNT = 10;

export { DEFAULT_SLIDE_COUNT, DESCRIPTION_MAX, SLIDE_COUNT_MAX, SLIDE_COUNT_MIN };

/* ────────────────────────── Языки колоды ─────────────────────────────── */

/**
 * Коды языков контракта: backend/app/modules/presentations/constants.py,
 * SUPPORTED_LANGUAGES. Таджикский там обозначен `tj`, а НЕ ISO `tg` — это
 * осознанное решение проекта, оно же действует в document.language и в
 * названиях шаблонов (manifest.json, ключи name.ru / name.tj).
 *
 * Локаль интерфейса при этом хранится как `tg` (i18n/index.jsx), поэтому между
 * ними нужен явный перевод — см. localeToPresentationLanguage.
 */
export const PRESENTATION_LANGUAGES = ['ru', 'tj'];

const DEFAULT_PRESENTATION_LANGUAGE = 'ru';

/** Язык интерфейса -> код языка колоды. Умолчание селектора языка в форме. */
export const localeToPresentationLanguage = (locale) => (
    normalizeLocale(locale) === 'ru' ? 'ru' : 'tj'
);

/**
 * Любой код языка -> код контракта.
 *
 * Локаль интерфейса `tg` переводится в `tj` здесь, а не оставляется на совесть
 * вызывающего: разница в одной букве, а цена ошибки — молчаливый откат к
 * русскому (неизвестный код деградирует в умолчание), то есть таджикский
 * интерфейс с русскими названиями шаблонов.
 */
export const normalizePresentationLanguage = (value) => {
    const normalized = String(value || '').trim().toLowerCase();
    if (normalized === 'tg') return 'tj';

    return PRESENTATION_LANGUAGES.includes(normalized) ? normalized : DEFAULT_PRESENTATION_LANGUAGE;
};

/* ────────────────────────── Статусы ──────────────────────────────────── */

export const PRESENTATION_STATUSES = ['queued', 'generating', 'ready', 'error'];

// Статусы, при которых генерация ещё не закончилась и список надо опрашивать.
export const IN_PROGRESS_STATUSES = ['queued', 'generating'];

/**
 * Статус ответа -> один из четырёх статусов контракта.
 *
 * Неизвестное значение считаем «в очереди»: работа идёт в фоне, и следующий же
 * опрос уточнит состояние. Обратная трактовка («готово») предложила бы кнопку
 * скачивания файла, которого нет.
 */
export const resolvePresentationStatus = (status) => {
    const normalized = String(status || '').trim().toLowerCase();
    return PRESENTATION_STATUSES.includes(normalized) ? normalized : 'queued';
};

export const isPresentationInProgress = (presentation) => (
    IN_PROGRESS_STATUSES.includes(resolvePresentationStatus(presentation?.status))
);

export const hasPresentationsInProgress = (presentations) => (
    (presentations || []).some(isPresentationInProgress)
);

/** Прогресс в целых процентах, подрезанный в 0..100: значение рисует полоса. */
export const resolveProgress = (value) => {
    const progress = Number(value);
    if (!Number.isFinite(progress)) return 0;
    return Math.min(100, Math.max(0, Math.round(progress)));
};

/**
 * Место в очереди: число или null, когда презентация уже не в очереди.
 * Нумерация человеческая, с единицы, — так её и отдаёт сервер.
 */
export const resolveQueuePosition = (presentation) => {
    const position = Number(presentation?.queue_position);
    return Number.isFinite(position) && position > 0 ? position : null;
};

/* ────────────────────────── Причина отказа ───────────────────────────── */

/**
 * Почему презентация не собралась.
 *
 * Ровно та же задача, что у resolveSourceErrorMessage в lib/sources.js: код
 * лежит в поле объекта (presentation.error_code), а не в теле HTTP-ответа, но
 * реестр кодов у бэкенда общий (backend/app/core/exceptions.py,
 * PresentationErrors), поэтому перевод берётся общей таблицей apiError.js.
 *
 * Порядок деградации:
 *   1. перевод по коду — единственный вариант на языке интерфейса;
 *   2. error_text — техническая строка на одном языке, но конкретная: она
 *      лучше, чем «Ошибка» без единого слова о причине, если код новый и
 *      таблица переводов его ещё не знает;
 *   3. общее «не удалось собрать презентацию».
 *
 * Пункт 2 — отличие от источников: там error_text показывается только
 * подсказкой при наведении. Здесь он остаётся и подсказкой (см. карточку в
 * PresentationList), но при неизвестном коде выходит в текст: заказ повторяют
 * вручную, и решение «повторять или нет» без причины не принимается.
 */
export const resolvePresentationErrorMessage = (presentation, t) => {
    if (resolvePresentationStatus(presentation?.status) !== 'error') return '';

    const byCode = resolveErrorCodeMessage(presentation?.error_code, t);
    if (byCode) return byCode;

    const errorText = String(presentation?.error_text || '').trim();
    if (errorText) return errorText;

    return t('presentations.errors.generationFailed');
};

/* ────────────────────────── Шаблоны ──────────────────────────────────── */

/**
 * Имя шаблона на языке интерфейса.
 *
 * Сервер отдаёт name объектом {ru, tj} (backend/app/modules/presentations/
 * templates.py, NAME_LANGUAGES). Пустое имя деградирует в ключ шаблона: он
 * машинный, но узнаваемый, и это лучше пустой карточки в галерее.
 */
export const resolveTemplateName = (template, language) => {
    if (!template) return '';

    const names = template.name;
    if (typeof names === 'string') return names;

    const preferred = names?.[normalizePresentationLanguage(language)];
    return String(preferred || names?.ru || template.key || '').trim();
};
