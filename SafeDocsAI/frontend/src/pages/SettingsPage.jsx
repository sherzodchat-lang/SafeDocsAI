import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useSearchParams } from 'react-router-dom';
import { AlertTriangle, ChevronLeft, ChevronRight, RefreshCw } from 'lucide-react';
import { settingsService } from '../services/settingsService';
import { documentsService } from '../services/sourcesService';
import { clearSessionRole, getSessionUsername } from '../services/api';
import { Button } from '../components/ui/Button';
import { useModalDialog } from '../hooks/useModalDialog';
import { useLocale } from '../i18n';
import { resolveApiErrorMessage, resolveErrorCode } from '../lib/apiError';
import { formatLocaleDate } from '../lib/locale';
// Разбор постраничного ответа общий на весь проект: тело — голый массив, общее
// число — заголовком X-Total-Count. Имя функции историческое (первым таким
// списком были источники), но разбирает она форму ответа, а не источники.
import { normalizeSourcesResponse } from '../lib/sources';

const ROLE_OPTIONS = ['admin', 'content_manager', 'user'];

/**
 * Размер страницы таблицы пользователей.
 *
 * 25 строк — примерно один экран без внутренней прокрутки, и ровно столько же
 * показывает таблица источников (DEFAULT_PAGE_SIZE в AdminDocumentsPage.jsx):
 * листание в двух админских таблицах устроено одинаково. Потолок сервера — 500
 * (limit 1..500), и брать его целиком незачем: роли меняют поштучно, а не
 * читают таблицу подряд, зато первый экран раздела при большой базе перестаёт
 * тащить все регистрации разом.
 */
const USERS_PAGE_SIZE = 25;

/**
 * Вычитка всего списка — для поиска.
 *
 * Серверного поиска по пользователям нет, только пагинация. Фильтр по открытой
 * странице отвечал бы «не найдено» там, где пользователь есть на соседней
 * странице, — это ровно то враньё интерфейса, от которого раздел лечат. Поэтому
 * на время поиска список вычитывается целиком, страницами по максимуму, который
 * сервер отдаёт за один запрос (limit 1..500).
 *
 * Потолок на число запросов нужен, чтобы вычитка не превратилась в бесконечную
 * при неожиданном ответе сервера и не выгружала произвольно большую таблицу.
 * Упёршись в него, экран не молчит: он говорит, скольких пользователей проверил
 * и сколько их всего (settings.search.partial).
 */
const SEARCH_FETCH_LIMIT = 500;
const MAX_SEARCH_REQUESTS = 10;

// Сервер отвечает этим кодом на запрос, который меняет embedding-модель без
// подтверждения. Ответ не отказ по существу: тело валидно, повторить его нужно
// тем же составом плюс confirm_reindex.
const REINDEX_CONFIRMATION_CODE = 'settings.reindex_confirmation_required';

// Ошибки смены роли, после которых список пользователей у клиента заведомо
// устарел: обе просят обновить его и решить заново.
const STALE_USERS_CODES = ['settings.user_not_found', 'settings.role_change_conflict'];

// Редактируемые поля формы, разбитые по типу: по этим спискам строится и патч
// (что реально изменилось), и сравнение с сохранённым состоянием.
//
// Каждое имя ниже обязано существовать в схеме RuntimeSettingsUpdate
// (backend/app/api/endpoints/settings.py): схема объявлена с extra="forbid",
// и лишний ключ в теле — уже не молчаливое игнорирование, а 422. Единственный
// ключ тела вне этих списков — confirm_reindex, и он в схеме тоже объявлен.
const TEXT_FIELDS = [
    'chat_model',
    'embedding_model',
    'contextual_embedding_model',
    'reranker_model',
    // Профиль предметной области — тоже строка из фиксированного набора:
    // допустимые значения приходят в available_domain_profiles, и всё, чего в
    // этом наборе нет, сервер отвергает (settings.unsupported_domain_profile).
    'default_domain_profile',
];
const BOOLEAN_FIELDS = ['enable_condense_query', 'contextual_embedding_enabled', 'reranker_enabled'];
// Числовые поля держатся в состоянии СТРОКАМИ. Number(event.target.value) на
// каждый ввод превращал очищенное поле в видимый 0, а на сервер уходило совсем
// другое (Number(x) || 10), то есть показанное и отправленное расходились.
const NUMBER_FIELDS = ['retrieval_top_k', 'top_k', 'chat_model_num_ctx', 'contextual_embedding_num_ctx'];

// Границы числовых полей — ЗЕРКАЛО серверных.
//
// Источник один и он на сервере: SETTING_LIMITS в
// backend/app/shared/settings/runtime_settings.py. Оттуда их берут все три
// серверных пути — строгая проверка при записи (_require_int_in_range),
// снисходительный кламп при чтении (_clamp_on_read) и схема ответа
// (RuntimeSettingsResponse в backend/app/api/endpoints/settings.py). Читать
// python отсюда нечем, поэтому числа приходится повторять, — но повторение
// проверяемое: backend/tests/test_settings_limits_single_source.py разбирает
// объявления ниже и сверяет их с SETTING_LIMITS, то есть правка на сервере без
// правки здесь (и наоборот) роняет тест, а не приезжает к пользователю формой,
// которая предлагает значение, на котором сохранение упирается в отказ.
//
// Держим их именованными константами, а не числами по месту: одно и то же
// число нужно и проверке (NUMBER_FIELD_RANGES), и атрибутам min/max поля, и
// разъехаться этим двум нельзя.

// Окно контекста: ограничение общее для обеих настроек — это память под
// KV-кэш на том же железе.
const MIN_NUM_CTX = 2048;
const MAX_NUM_CTX = 32768;

// Пул кандидатов.
const MIN_RETRIEVAL_TOP_K = 1;
const MAX_RETRIEVAL_TOP_K = 50;

// Число фрагментов для модели: верхняя граница своя и меньшая.
const MIN_TOP_K = 1;
const MAX_TOP_K = 20;

/**
 * Диапазоны числовых полей ровно те, что проверяет сервер при ЗАПИСИ.
 *
 * Раньше здесь стоял только top_k, а num_ctx проверять было незачем: сервер
 * подгонял значение под диапазон и отвечал 200 OK. Теперь он отвечает отказом
 * (settings.value_out_of_range), поэтому граница обязана быть и на клиенте —
 * иначе про неё узнают только после неудачного сохранения.
 *
 * Границы берутся константами выше, а числами здесь не пишутся: и сами
 * константы, и состав этой таблицы сверяются с серверным SETTING_LIMITS в
 * backend/tests/test_settings_limits_single_source.py — числом по месту эта
 * сверка обходится молча.
 */
const NUMBER_FIELD_RANGES = {
    retrieval_top_k: {
        min: MIN_RETRIEVAL_TOP_K,
        max: MAX_RETRIEVAL_TOP_K,
        messageKey: 'settings.retrievalTopKRange',
    },
    top_k: { min: MIN_TOP_K, max: MAX_TOP_K, messageKey: 'settings.topKRange' },
    chat_model_num_ctx: { min: MIN_NUM_CTX, max: MAX_NUM_CTX, messageKey: 'settings.numCtxRange' },
    contextual_embedding_num_ctx: { min: MIN_NUM_CTX, max: MAX_NUM_CTX, messageKey: 'settings.numCtxRange' },
};

const DEFAULT_FORM_VALUES = {
    chat_model: '',
    embedding_model: '',
    contextual_embedding_model: '',
    reranker_model: 'gemma4:e4b',
    enable_condense_query: true,
    contextual_embedding_enabled: false,
    reranker_enabled: false,
    retrieval_top_k: '20',
    top_k: '10',
    default_domain_profile: '',
    chat_model_num_ctx: '20000',
    contextual_embedding_num_ctx: '8192',
};

// Всё, что приходит с сервера, но формой не редактируется: каталог моделей,
// состояние Ollama и долг по переиндексации.
const DEFAULT_CATALOG = {
    available_models: [],
    available_chat_models: [],
    available_embedding_models: [],
    // Набор профилей предметной области задан на сервере (реестр
    // app/domain_profiles): выбирать можно только из него, поэтому список
    // приходит вместе с настройками и живёт здесь же, рядом с каталогом моделей.
    available_domain_profiles: [],
    ollama_available: true,
    ollama_error: '',
    reindex_required: false,
};

const SELECT_CLASS = 'h-10 w-full rounded-lg border border-slate-300 bg-white px-3 text-sm text-slate-700 focus:outline-none focus:ring-2 focus:ring-[#1f3a60]/25 disabled:cursor-not-allowed disabled:opacity-60';
const NUMBER_INPUT_CLASS = 'h-10 rounded-lg border border-slate-300 bg-white px-3 text-sm text-slate-700 focus:outline-none focus:ring-2 focus:ring-[#1f3a60]/25';

const toNumberField = (value, fallback) => {
    if (value === null || value === undefined || value === '') return fallback;
    const number = Number(value);
    return Number.isFinite(number) ? String(number) : fallback;
};

const toFormValues = (data = {}) => ({
    chat_model: String(data.chat_model || data.model || ''),
    embedding_model: String(data.embedding_model || ''),
    contextual_embedding_model: String(data.contextual_embedding_model || ''),
    reranker_model: String(data.reranker_model || DEFAULT_FORM_VALUES.reranker_model),
    enable_condense_query: Boolean(data.enable_condense_query),
    contextual_embedding_enabled: Boolean(data.contextual_embedding_enabled),
    reranker_enabled: Boolean(data.reranker_enabled),
    retrieval_top_k: toNumberField(data.retrieval_top_k, DEFAULT_FORM_VALUES.retrieval_top_k),
    top_k: toNumberField(data.top_k, DEFAULT_FORM_VALUES.top_k),
    default_domain_profile: String(data.default_domain_profile || ''),
    chat_model_num_ctx: toNumberField(data.chat_model_num_ctx, DEFAULT_FORM_VALUES.chat_model_num_ctx),
    contextual_embedding_num_ctx: toNumberField(
        data.contextual_embedding_num_ctx,
        DEFAULT_FORM_VALUES.contextual_embedding_num_ctx,
    ),
});

const toCatalog = (data = {}) => ({
    available_models: Array.isArray(data.available_models) ? data.available_models : [],
    available_chat_models: Array.isArray(data.available_chat_models) ? data.available_chat_models : [],
    available_embedding_models: Array.isArray(data.available_embedding_models) ? data.available_embedding_models : [],
    available_domain_profiles: Array.isArray(data.available_domain_profiles) ? data.available_domain_profiles : [],
    ollama_available: data.ollama_available !== false,
    ollama_error: data.ollama_error || '',
    reindex_required: Boolean(data.reindex_required),
});

/**
 * Сохранённой модели нет в каталоге Ollama.
 *
 * Прежде такое значение молча заменялось на availableModels[0]: админ, зашедший
 * поправить top_k, сохранял вместе с ним чужую embedding-модель и уводил поиск
 * в пустую коллекцию, ничего об этом не узнав. Теперь значение показывается как
 * есть, а несоответствие каталогу — отдельным предупреждением.
 *
 * Пустой каталог не означает «модели нет»: он означает, что список не пришёл
 * (Ollama недоступна), и обвинять модель в этом случае не в чем.
 */
const isModelUnavailable = (value, models) => (
    Boolean(value) && Array.isArray(models) && models.length > 0 && !models.includes(value)
);

const validateNumberField = (field, rawValue) => {
    const value = String(rawValue ?? '').trim();
    if (!value) return 'settings.numberRequired';

    const number = Number(value);
    if (!Number.isInteger(number)) return 'settings.numberInvalid';

    // Тот же диапазон проверяет и сервер, но подсказка у поля появляется до
    // отправки: упереться в границу заранее полезнее, чем узнать о ней из
    // отказа, потеряв на этом запрос.
    const range = NUMBER_FIELD_RANGES[field];
    if (range && (number < range.min || number > range.max)) return range.messageKey;

    return '';
};

/**
 * Пул кандидатов уже итоговой выдачи.
 *
 * Не ошибка: сервер такое сочетание принимает, границы полей у него
 * независимы (retrieval_top_k 1..50, top_k 1..20). Но в момент запроса
 * resolve_retrieval_limits (backend/app/modules/chat/service.py) берёт
 * retrieval_top_k с min_value, равным итоговому top_k, то есть молча
 * расширяет пул до top_k — сохранённое число перестаёт что-либо значить.
 * Показать это заранее честнее, чем оставить настройку, которая не действует.
 */
const isRetrievalPoolNarrow = (values) => {
    if (validateNumberField('retrieval_top_k', values.retrieval_top_k)) return false;
    if (validateNumberField('top_k', values.top_k)) return false;

    return Number(values.retrieval_top_k) < Number(values.top_k);
};

const collectFieldErrors = (values) => {
    const errors = {};

    NUMBER_FIELDS.forEach((field) => {
        const messageKey = validateNumberField(field, values[field]);
        if (messageKey) errors[field] = messageKey;
    });

    return errors;
};

/**
 * Тело PUT — только то, что отличается от загруженного с сервера.
 *
 * Сервер держит частичное обновление: он патчит ровно присутствующие ключи.
 * Пока клиент слал полный набор, правка top_k физически тащила за собой
 * embedding-модель и contextual-модель, то есть безобидное изменение могло
 * потребовать переиндексации или упереться в чужую валидацию.
 */
const buildPatch = (values, savedValues) => {
    const patch = {};

    [...TEXT_FIELDS, ...BOOLEAN_FIELDS].forEach((field) => {
        if (values[field] !== savedValues[field]) patch[field] = values[field];
    });

    NUMBER_FIELDS.forEach((field) => {
        // Невалидное поле не уходит на сервер вовсе: иначе показанное и
        // отправленное снова разъедутся.
        if (validateNumberField(field, values[field])) return;

        const next = Number(String(values[field]).trim());
        if (next !== Number(savedValues[field])) patch[field] = next;
    });

    return patch;
};

/**
 * Список моделей с честным показом текущего значения.
 *
 * <select> со значением, которого нет среди <option>, браузер рисует пустым —
 * поле выглядело бы «ничего не выбрано» там, где на сервере лежит конкретная
 * модель. Поэтому недостающее значение добавляется отдельным пунктом, а пустое
 * значение — явным «модель не выбрана», а не молчаливым первым пунктом списка.
 *
 * Пустая строка при этом значит разное в разных полях, и от этого зависит,
 * можно ли к ней ВЕРНУТЬСЯ:
 *   * модель чата и embedding-модель обязательны: пустое значение — «ещё не
 *     выбрано», пункт показывается, пока выбора нет, и исчезает после него
 *     (сохранить пустое поле всё равно нельзя — settings.model_required);
 *   * contextual_embedding_model необязательна: с недавних пор её умолчание —
 *     пустая строка, и это законное сохраняемое состояние «не выбрана», а не
 *     ошибка. Такому полю пункт нужен ВСЕГДА: без него выбранную однажды
 *     модель нечем снять, хотя сервер такое значение принимает. Отсюда
 *     emptyLabel — он и включает постоянный пункт.
 *
 * id обязателен: имя списку даёт связанная подпись (<label htmlFor>), а не
 * aria-label. Разница не в озвучке — она в том, что подпись без htmlFor не
 * переводит фокус в поле по клику и не увеличивает область попадания.
 */
const ModelSelect = ({ id, value, models, onChange, disabled, placeholderLabel, unknownLabel, emptyLabel }) => {
    const options = Array.isArray(models) ? models : [];

    return (
        <select
            id={id}
            value={value}
            onChange={onChange}
            disabled={disabled}
            className={SELECT_CLASS}
        >
            {emptyLabel
                ? <option value="">{emptyLabel}</option>
                : (!value && <option value="">{placeholderLabel}</option>)}
            {Boolean(value) && !options.includes(value) && (
                <option value={value}>{unknownLabel}</option>
            )}
            {options.map((model) => (
                <option key={model} value={model}>{model}</option>
            ))}
        </select>
    );
};

const SettingsPage = () => {
    const { locale, t } = useLocale();
    const [isLoading, setIsLoading] = useState(true);
    const [isSettingsBusy, setIsSettingsBusy] = useState(false);
    const [isUsersBusy, setIsUsersBusy] = useState(false);
    const [isSaving, setIsSaving] = useState(false);
    const [isResetting, setIsResetting] = useState(false);
    const [isReindexing, setIsReindexing] = useState(false);

    const [catalog, setCatalog] = useState(DEFAULT_CATALOG);
    // Форма и снимок последнего ответа сервера. Снимок — база сравнения для
    // частичного PUT и единственный источник правды о том, что сохранено.
    const [formValues, setFormValues] = useState(DEFAULT_FORM_VALUES);
    const [savedValues, setSavedValues] = useState(DEFAULT_FORM_VALUES);
    const [isSettingsLoaded, setIsSettingsLoaded] = useState(false);

    // Страница пользователей и общее их число: список постраничный, а число
    // приходит заголовком и о skip/limit ничего не знает.
    const [users, setUsers] = useState([]);
    const [usersTotal, setUsersTotal] = useState(null);
    const [usersPage, setUsersPage] = useState(1);
    const [isUsersLoaded, setIsUsersLoaded] = useState(false);

    // Полный список — только для поиска, подробности у SEARCH_FETCH_LIMIT.
    // Хранится вместе с общим числом на момент вычитки: если проверить удалось
    // не всех, экран обязан сказать об этом, а не выдать неполный результат за
    // полный.
    const [searchIndex, setSearchIndex] = useState(null);
    const [isSearchLoading, setIsSearchLoading] = useState(false);
    const [searchError, setSearchError] = useState(null);
    const [searchPage, setSearchPage] = useState(1);

    // Ошибки хранятся объектами, а не готовыми строками: перевод берётся при
    // отрисовке, поэтому смена языка переводит и уже показанное сообщение.
    const [settingsError, setSettingsError] = useState(null);
    const [usersError, setUsersError] = useState(null);
    const [saveError, setSaveError] = useState(null);
    const [reindexError, setReindexError] = useState(null);
    const [confirmError, setConfirmError] = useState(null);
    const [roleError, setRoleError] = useState(null);
    // Успешные сообщения — ключом перевода с параметрами, по той же причине.
    const [notice, setNotice] = useState(null);
    // Итог смены роли живёт отдельно от настроек: сообщение обязано стоять у
    // своей таблицы, а не в чужой секции, и объявляться своей живой областью.
    const [usersNotice, setUsersNotice] = useState(null);

    const [reindexResult, setReindexResult] = useState(null);
    // Подтверждение смены роли: пока цель выбрана, запрос ещё не ушёл.
    const [roleTarget, setRoleTarget] = useState(null);
    const [isSavingRole, setIsSavingRole] = useState(false);
    // Подтверждение операции над настройками: сохранение со сменой
    // embedding-модели или сброс к умолчаниям.
    const [confirmTarget, setConfirmTarget] = useState(null);

    const roleDialogRef = useRef(null);
    const roleCancelRef = useRef(null);
    const confirmDialogRef = useRef(null);
    const confirmCancelRef = useRef(null);

    const fieldErrors = useMemo(() => collectFieldErrors(formValues), [formValues]);
    const patch = useMemo(() => buildPatch(formValues, savedValues), [formValues, savedValues]);
    const hasFieldErrors = Object.keys(fieldErrors).length > 0;
    const isDirty = Object.keys(patch).length > 0;
    /**
     * Пустая обязательная модель блокирует сохранение — но только если она
     * действительно уходит в этом запросе.
     *
     * Здесь стояло `Boolean(formValues.chat_model && formValues.embedding_model)`,
     * то есть требование ко ВСЕЙ форме на любом сохранении. Это ровно та
     * неисправность, которую только что убрали на сервере (проверка модели
     * контекстного обогащения стояла безусловно и роняла правку одного top_k):
     * поле, которого патч не касается, не должно решать судьбу чужой правки.
     * С частичным PUT сервер пустую модель и не увидит — в теле её не будет.
     */
    const canSave = !(
        (patch.chat_model !== undefined && !patch.chat_model)
        || (patch.embedding_model !== undefined && !patch.embedding_model)
    );
    const isBusy = isSaving || isResetting || isReindexing;

    const setField = useCallback((field, value) => {
        setFormValues((prev) => ({ ...prev, [field]: value }));
    }, []);

    // Ответ сервера кладётся на экран целиком и без наложения отправленного
    // сверху: сохранённое состояние знает только сервер, и админ обязан видеть
    // его, а не то, что набрал. Значения вне диапазона он теперь не подгоняет,
    // а отвергает, но подмена остаётся возможной и на успешном ответе — чтение
    // настроек чинит негодные значения из файла на лету.
    const applySettingsData = useCallback((data) => {
        const values = toFormValues(data);
        setCatalog(toCatalog(data));
        setFormValues(values);
        setSavedValues(values);
        setIsSettingsLoaded(true);
    }, []);

    // Загрузка разбита на два независимых запроса: раньше Promise.all ронял обе
    // секции, если падала любая из них, — недоступная Ollama прятала и таблицу
    // ролей. Зависимости от t здесь намеренно нет: иначе переключение языка
    // перезапрашивало настройки и стирало несохранённые правки формы.
    const loadSettings = useCallback(async () => {
        setIsSettingsBusy(true);
        setSettingsError(null);

        try {
            const response = await settingsService.get();
            applySettingsData(response.data || {});
        } catch (err) {
            console.error('Failed to load settings', err);
            setSettingsError(err);
        } finally {
            setIsSettingsBusy(false);
        }
    }, [applySettingsData]);

    const loadUsers = useCallback(async (targetPage = 1) => {
        setIsUsersBusy(true);
        setUsersError(null);
        setUsersPage(targetPage);

        try {
            const response = await settingsService.getUsers({
                skip: (targetPage - 1) * USERS_PAGE_SIZE,
                limit: USERS_PAGE_SIZE,
            });
            // Тот же разбор, что у источников: тело — голый массив, общее число
            // приходит заголовком X-Total-Count (он объявлен в expose_headers,
            // иначе браузер не отдал бы его коду страницы).
            const { items, total } = normalizeSourcesResponse(response);
            setUsers(items);
            setUsersTotal(total);
            setIsUsersLoaded(true);
        } catch (err) {
            console.error('Failed to load users', err);
            // Что лежит на запрошенной странице — неизвестно, и оставлять под её
            // номером строки предыдущей значило бы показать чужую страницу как
            // эту. Ошибка с кнопкой «Повторить» стоит рядом.
            setUsers([]);
            setUsersError(err);
        } finally {
            setIsUsersBusy(false);
        }
    }, []);

    /**
     * Полная вычитка списка для поиска.
     *
     * Идёт по тому же контракту, что и страница таблицы: skip растёт на число
     * уже полученных записей, limit — потолок сервера. Порядок (created_at DESC,
     * id DESC) устойчив между страницами, поэтому склейка не теряет и не
     * дублирует пользователей.
     *
     * Признак полноты считается здесь же: неполный набор в поиске молчать не
     * имеет права.
     */
    const loadSearchIndex = useCallback(async () => {
        setIsSearchLoading(true);
        setSearchError(null);

        try {
            const collected = [];
            let total = null;

            for (let request = 0; request < MAX_SEARCH_REQUESTS; request += 1) {
                const response = await settingsService.getUsers({
                    skip: collected.length,
                    limit: SEARCH_FETCH_LIMIT,
                });
                const page = normalizeSourcesResponse(response);
                collected.push(...page.items);

                if (page.total != null) total = page.total;
                // Короткая страница — последняя: продолжать нечем и незачем.
                if (page.items.length < SEARCH_FETCH_LIMIT) break;
                if (total != null && collected.length >= total) break;
            }

            setSearchIndex({
                items: collected,
                total: total ?? collected.length,
                isComplete: total == null || collected.length >= total,
            });
        } catch (err) {
            console.error('Failed to load users for search', err);
            setSearchError(err);
        } finally {
            setIsSearchLoading(false);
        }
    }, []);

    const loadData = useCallback(async () => {
        setIsLoading(true);
        await Promise.allSettled([loadSettings(), loadUsers(1)]);
        setIsLoading(false);
    }, [loadSettings, loadUsers]);

    useEffect(() => {
        loadData();
    }, [loadData]);

    /**
     * Поиск из шапки: значение приходит в ?q=, как на странице источников.
     *
     * Поле в шапке существует давно, а страница его не читала вовсе — оно
     * выглядело работающим и не делало ничего. Читать его недостаточно:
     * серверного поиска по пользователям нет, и фильтр по открытой странице
     * отвечал бы «не найдено» про пользователя, который просто лежит на другой
     * странице. Поэтому поиск работает по полному списку, вычитанному отдельно
     * (loadSearchIndex), и всегда говорит, по скольким пользователям он прошёл.
     */
    const [searchParams] = useSearchParams();
    // В сообщениях показывается набранное, а сравнивается приведённое к нижнему
    // регистру: пользователю возвращают его запрос, а не его же обработанную копию.
    const searchTerm = (searchParams.get('q') || '').trim();
    const searchQuery = searchTerm.toLowerCase();
    const isSearching = Boolean(searchQuery);

    useEffect(() => {
        // Первая же вычитка кэшируется: набор запроса по буквам не должен
        // тащить список заново на каждое нажатие.
        if (!isSearching || searchIndex || isSearchLoading || searchError) return;
        loadSearchIndex();
    }, [isSearching, isSearchLoading, loadSearchIndex, searchError, searchIndex]);

    useEffect(() => {
        setSearchPage(1);
    }, [searchQuery]);

    const searchMatches = useMemo(() => {
        if (!isSearching || !searchIndex) return [];

        return searchIndex.items.filter(
            (user) => String(user.username || '').toLowerCase().includes(searchQuery),
        );
    }, [isSearching, searchIndex, searchQuery]);

    // Страницы в двух режимах считаются по-разному: список сервер отдаёт уже
    // нарезанным, найденное режется здесь. Наружу оба режима выглядят одинаково.
    const knownUsersTotal = usersTotal ?? ((usersPage - 1) * USERS_PAGE_SIZE + users.length);
    const listTotal = isSearching ? searchMatches.length : knownUsersTotal;
    const pageCount = Math.max(1, Math.ceil(listTotal / USERS_PAGE_SIZE));
    const currentPage = isSearching ? Math.min(searchPage, pageCount) : usersPage;
    const visibleUsers = useMemo(() => (
        isSearching
            ? searchMatches.slice((currentPage - 1) * USERS_PAGE_SIZE, currentPage * USERS_PAGE_SIZE)
            : users
    ), [currentPage, isSearching, searchMatches, users]);
    const rangeStart = visibleUsers.length === 0 ? 0 : (currentPage - 1) * USERS_PAGE_SIZE + 1;
    const rangeEnd = visibleUsers.length === 0 ? 0 : rangeStart + visibleUsers.length - 1;
    // Вычитанного для поиска списка достаточно, чтобы показать найденное, даже
    // если страница таблицы в этот момент не загрузилась.
    const showUsersTable = isUsersLoaded || Boolean(searchIndex);

    // Сброс вычитанного списка запускает её заново тем же условием, что и первый
    // поиск: отдельного пути загрузки для повтора нет.
    const retrySearchIndex = useCallback(() => {
        setSearchError(null);
        setSearchIndex(null);
    }, []);

    const goToPage = useCallback((nextPage) => {
        if (isSearching) {
            setSearchPage(nextPage);
            return;
        }

        loadUsers(nextPage);
    }, [isSearching, loadUsers]);

    /**
     * Название роли берётся из словаря, а не показывается служебным именем.
     *
     * В таблице и в диалоге подтверждения стояли «admin» и «content_manager» —
     * латиница в интерфейсе, у которого язык по умолчанию таджикский. Роль, о
     * которой словарь не знает (значение из старой схемы, ручная правка в БД),
     * показывается как есть: сервер такие значения допускает, и подменять их
     * ближайшим знакомым нельзя.
     */
    const getRoleLabel = useCallback((role) => {
        const key = `settings.roles.${role}`;
        const translated = t(key);
        return translated === key ? role : translated;
    }, [t]);

    const closeConfirmDialog = useCallback(() => {
        setConfirmTarget(null);
        setConfirmError(null);
    }, []);

    const submitSettings = useCallback(async (settingsPatch, { confirmReindex, fromDialog }) => {
        setIsSaving(true);
        setNotice(null);
        if (fromDialog) setConfirmError(null); else setSaveError(null);

        try {
            const body = confirmReindex ? { ...settingsPatch, confirm_reindex: true } : settingsPatch;
            const response = await settingsService.update(body);
            applySettingsData(response.data || {});
            setNotice({ key: 'settings.saved' });
            setConfirmTarget(null);
            setConfirmError(null);
        } catch (err) {
            console.error('Failed to update settings', err);

            // Сервер увидел смену embedding-модели там, где клиент её не ждал
            // (например, значение поменяли в другой вкладке). Обрабатываем сам
            // код, а не только собственную предварительную проверку: диалог с
            // последствиями показывается и в этом случае, и повторяется ТОТ ЖЕ
            // запрос с подтверждением.
            if (!confirmReindex && resolveErrorCode(err) === REINDEX_CONFIRMATION_CODE) {
                setConfirmError(null);
                setConfirmTarget({
                    kind: 'save',
                    patch: settingsPatch,
                    from: savedValues.embedding_model,
                    to: settingsPatch.embedding_model || '',
                });
                return;
            }

            if (fromDialog) setConfirmError(err); else setSaveError(err);
        } finally {
            setIsSaving(false);
        }
    }, [applySettingsData, savedValues.embedding_model]);

    const handleSave = useCallback(() => {
        setNotice(null);
        setSaveError(null);

        if (hasFieldErrors) return;
        if (!isDirty) {
            setNotice({ key: 'settings.noChanges' });
            return;
        }

        // Смена embedding-модели — не настройка, а операция над всем поиском:
        // спрашиваем до запроса, чтобы объяснить последствия, а не после отказа.
        if (patch.embedding_model !== undefined) {
            setConfirmError(null);
            setConfirmTarget({
                kind: 'save',
                patch,
                from: savedValues.embedding_model,
                to: patch.embedding_model,
            });
            return;
        }

        submitSettings(patch, { confirmReindex: false, fromDialog: false });
    }, [hasFieldErrors, isDirty, patch, savedValues.embedding_model, submitSettings]);

    const submitReset = useCallback(async (confirmReindex) => {
        setIsResetting(true);
        setNotice(null);
        setConfirmError(null);

        try {
            const response = await settingsService.reset(confirmReindex ? { confirm_reindex: true } : {});
            applySettingsData(response.data || {});
            setNotice({ key: 'settings.resetDone' });
            setConfirmTarget(null);
        } catch (err) {
            console.error('Failed to reset settings', err);

            // Сброс вернул бы embedding-модель к умолчанию — те же последствия,
            // что и у ручной смены. Диалог остаётся открытым, но теперь
            // спрашивает про переиндексацию, и повтор идёт с подтверждением.
            if (!confirmReindex && resolveErrorCode(err) === REINDEX_CONFIRMATION_CODE) {
                setConfirmTarget({ kind: 'reset', requiresReindex: true });
                return;
            }

            setConfirmError(err);
        } finally {
            setIsResetting(false);
        }
    }, [applySettingsData]);

    const handleReset = useCallback(() => {
        setNotice(null);
        setSaveError(null);
        setConfirmError(null);
        setConfirmTarget({ kind: 'reset', requiresReindex: false });
    }, []);

    const confirmDialogAction = useCallback(() => {
        if (!confirmTarget) return;

        if (confirmTarget.kind === 'reset') {
            submitReset(Boolean(confirmTarget.requiresReindex));
            return;
        }

        submitSettings(confirmTarget.patch, { confirmReindex: true, fromDialog: true });
    }, [confirmTarget, submitReset, submitSettings]);

    // Переиндексация долгая: кнопка блокируется на всё время запроса, а вместе
    // с ней сохранение и сброс — менять настройки под работающей перестройкой
    // индекса нечестно по отношению к её результату.
    const runReindex = useCallback(async () => {
        setIsReindexing(true);
        setReindexError(null);
        setNotice(null);

        try {
            const response = await documentsService.reindexAll();
            const data = response.data || {};
            setReindexResult(data);
            // Флаг приходит в ответе переиндексации: при "partial" он остаётся
            // поднятым, и баннер не исчезает — часть документов в индекс не
            // попала, напомнить об этом больше нечем.
            setCatalog((prev) => ({ ...prev, reindex_required: Boolean(data.reindex_required) }));

            if (data.status === 'ok') {
                setNotice({
                    key: 'settings.reindexDone',
                    params: {
                        documents: data.total_documents ?? 0,
                        chunks: data.total_chunks ?? 0,
                    },
                });
            }
        } catch (err) {
            console.error('Failed to reindex documents', err);
            setReindexError(err);
        } finally {
            setIsReindexing(false);
        }
    }, []);

    /**
     * Выбор в списке ничего не меняет — он только предлагает изменение.
     *
     * Колесо мыши над сфокусированным <select> переключает значение и в Chrome,
     * и в Firefox: с запросом прямо на onChange роль понижалась бы прокруткой
     * страницы, и отменить это было бы нечем. Значение самого списка остаётся
     * привязанным к user.role, поэтому после открытия диалога он возвращается
     * к текущей роли, а не показывает несостоявшуюся.
     */
    const requestRoleChange = useCallback((user, nextRole) => {
        if (!nextRole || nextRole === user.role) return;

        setRoleError(null);
        setUsersNotice(null);
        setRoleTarget({
            id: user.id,
            username: user.username,
            currentRole: user.role,
            nextRole,
        });
    }, []);

    const closeRoleDialog = useCallback(() => {
        setRoleTarget(null);
        setRoleError(null);
        setIsSavingRole(false);
    }, []);

    const confirmRoleChange = useCallback(async () => {
        if (!roleTarget || isSavingRole) return;

        setIsSavingRole(true);
        setRoleError(null);

        try {
            const response = await settingsService.updateUserRole(roleTarget.id, roleTarget.nextRole);
            // Роль в таблице обновляем только по ответу сервера: при отказе
            // строка обязана показывать то, что на сервере, а не то, что
            // выбрали в списке.
            const savedRole = response?.data?.role || roleTarget.nextRole;
            setUsers((prev) => prev.map((user) => (
                user.id === roleTarget.id ? { ...user, role: savedRole } : user
            )));
            // Вычитанный для поиска список — та же таблица, только целиком:
            // не обнови его здесь, и найденная строка показывала бы прежнюю
            // роль до следующей вычитки.
            setSearchIndex((prev) => (prev ? {
                ...prev,
                items: prev.items.map((user) => (
                    user.id === roleTarget.id ? { ...user, role: savedRole } : user
                )),
            } : prev));
            // Диалог просто закрывается, и об исходе операции сказать было
            // нечем: зрячий видел новое значение в списке, незрячий — ничего.
            setUsersNotice({ username: roleTarget.username, role: savedRole });

            // Роль сменили самому себе — подсказка интерфейса устарела в ту же
            // секунду. Сбрасываем её, чтобы админские разделы исчезли сразу, а
            // не после того, как сервер ответит 403 на первое же действие.
            if (roleTarget.username === getSessionUsername()) {
                clearSessionRole();
            }

            setRoleTarget(null);
        } catch (err) {
            console.error('Failed to update role', err);
            // Ошибка остаётся в диалоге, рядом с вопросом: так видно, к какому
            // именно пользователю она относится, и решение можно повторить или
            // отменить, не теряя контекст. Нативный alert для этого не годится.
            setRoleError(err);
            // «Пользователь не найден» и «конфликт смены роли» означают одно:
            // список у клиента устарел. Перечитываем его сразу, чтобы решение
            // принималось по актуальным данным, — ту же страницу, на которой
            // стоит админ, а не первую. Полный список для поиска устарел ровно
            // так же, поэтому сбрасывается: следующий поиск вычитает его заново.
            if (STALE_USERS_CODES.includes(resolveErrorCode(err))) {
                loadUsers(usersPage);
                setSearchIndex(null);
                setSearchError(null);
            }
        } finally {
            setIsSavingRole(false);
        }
    }, [isSavingRole, loadUsers, roleTarget, usersPage]);

    useModalDialog(Boolean(roleTarget), closeRoleDialog, roleDialogRef, roleCancelRef);
    useModalDialog(Boolean(confirmTarget), closeConfirmDialog, confirmDialogRef, confirmCancelRef);

    if (isLoading) {
        return (
                <div className="rounded-2xl border border-slate-200 bg-white p-8 text-center text-slate-500 shadow-sm">
                {t('settings.loading')}
            </div>
        );
    }

    const reindexErrors = Array.isArray(reindexResult?.errors) ? reindexResult.errors : [];
    const isReindexPartial = reindexResult?.status === 'partial';

    /**
     * Предупреждение под списком: сохранённое значение не годится для этого поля.
     *
     * Списки моделей разные (available_chat_models и available_embedding_models
     * содержат разное), поэтому «нет в списке» имеет два разных лечения, и
     * подсказка про ollama pull для второго случая вредна — модель уже стоит.
     */
    const renderModelWarning = (value, kindModels) => {
        if (!isModelUnavailable(value, kindModels)) return null;

        const isInstalled = catalog.available_models.includes(value);

        return (
            <p className="mt-2 text-xs font-semibold text-amber-600">
                {isInstalled
                    ? t('settings.modelWrongKindWarning', { model: value })
                    : t('settings.modelUnavailableWarning', { model: value })}
            </p>
        );
    };

    /**
     * Название профиля берём из общего словаря профилей.
     *
     * Тот же источник, что у диалога правки блокнота (notebooksPage.profiles):
     * профиль в настройках и профиль блокнота — одно и то же понятие, и второй
     * словарь на те же значения разошёлся бы с первым. Незнакомый профиль
     * показывается своим служебным именем, а не пустой строкой: реестр на
     * сервере пополняется без правки клиента.
     */
    const getDomainProfileLabel = (profile) => {
        const key = `notebooksPage.profiles.${profile}`;
        const translated = t(key);
        return translated === key ? profile : translated;
    };

    const isDomainProfileUnknown = Boolean(formValues.default_domain_profile)
        && catalog.available_domain_profiles.length > 0
        && !catalog.available_domain_profiles.includes(formValues.default_domain_profile);

    /**
     * Голосовой статус таблицы пользователей.
     *
     * Порядок веток — от отказа к фону: сначала то, что помешало показать
     * список, потом область поиска, потом обычное «показано N из M». Отдельная
     * ветка ошибки обязательна: молча показать пустую таблицу значит выдать сбой
     * загрузки за отсутствие пользователей.
     */
    const getUsersStatusMessage = () => {
        if (usersError) return resolveApiErrorMessage(usersError, t, 'settings.usersLoadFailed');

        if (isSearching) {
            if (searchError) return resolveApiErrorMessage(searchError, t, 'settings.search.failed');
            if (isSearchLoading || !searchIndex) return t('settings.search.loading');

            return searchMatches.length === 0
                ? t('settings.search.empty', { query: searchTerm, scanned: searchIndex.items.length })
                : t('settings.search.results', {
                    count: searchMatches.length,
                    scanned: searchIndex.items.length,
                    query: searchTerm,
                });
        }

        if (isUsersBusy) return t('settings.usersLoading');

        return t('settings.pagination.range', { from: rangeStart, to: rangeEnd, total: listTotal });
    };

    const getEmptyRowMessage = () => {
        if (!isSearching) return t('settings.usersEmpty');
        if (searchError) return t('settings.search.unavailable');
        if (isSearchLoading || !searchIndex) return t('settings.search.loading');

        return t('settings.search.noRows');
    };

    const renderConfirmDescription = () => {
        if (confirmTarget?.kind === 'reset') {
            return confirmTarget.requiresReindex
                ? t('settings.resetReindexDescription')
                : t('settings.resetConfirmDescription');
        }

        // Смена, которую заметил только сервер: показать «с X на Y» нечем.
        if (!confirmTarget?.to) return t('settings.reindexConfirmUnknownDescription');

        return t('settings.reindexConfirmDescription', {
            from: confirmTarget.from || '—',
            to: confirmTarget.to,
        });
    };

    return (
        <div className="space-y-6 px-4">
            {/* Долг за сменой embedding-модели: пока флаг стоит, поиск идёт по
                коллекции, которую никто не заполнял. */}
            {catalog.reindex_required && (
                <section className="rounded-2xl border border-amber-300 bg-amber-50 p-5 shadow-sm">
                    <div className="flex items-start gap-3">
                        <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full bg-amber-100 text-amber-700">
                            <AlertTriangle className="h-5 w-5" />
                        </div>
                        <div className="min-w-0 flex-1">
                            <h3 className="text-base font-bold text-amber-900">{t('settings.reindexBannerTitle')}</h3>
                            <p className="mt-1 text-sm text-amber-900/80">{t('settings.reindexBannerDescription')}</p>

                            {isReindexPartial && (
                                <div className="mt-3 rounded-xl border border-amber-300 bg-white/70 p-3">
                                    <p className="text-sm font-semibold text-amber-900">
                                        {t('settings.reindexPartial', {
                                            failed: reindexErrors.length,
                                            documents: reindexResult?.total_documents ?? 0,
                                        })}
                                    </p>
                                    {reindexErrors.length > 0 && (
                                        <>
                                            <p className="mt-2 text-xs font-semibold text-amber-900">
                                                {t('settings.reindexPartialErrors')}
                                            </p>
                                            <ul className="mt-1 list-disc space-y-1 pl-5 text-xs text-amber-900/80">
                                                {reindexErrors.slice(0, 5).map((item) => (
                                                    <li key={item} className="break-words">{item}</li>
                                                ))}
                                            </ul>
                                            {reindexErrors.length > 5 && (
                                                <p className="mt-1 text-xs text-amber-900/70">
                                                    {t('settings.reindexPartialMore', { count: reindexErrors.length - 5 })}
                                                </p>
                                            )}
                                        </>
                                    )}
                                </div>
                            )}

                            {reindexError && (
                                <p role="alert" className="mt-3 rounded-lg bg-red-50 px-3 py-2 text-sm font-semibold text-red-700">
                                    {resolveApiErrorMessage(reindexError, t, 'settings.reindexFailed')}
                                </p>
                            )}

                            <div className="mt-4 flex flex-wrap items-center gap-3">
                                <Button
                                    type="button"
                                    variant="secondary"
                                    onClick={runReindex}
                                    isLoading={isReindexing}
                                    disabled={isSaving || isResetting}
                                >
                                    <RefreshCw className="h-4 w-4" />
                                    {isReindexing
                                        ? t('settings.reindexRunning')
                                        : (isReindexPartial ? t('settings.reindexRetry') : t('settings.reindexStart'))}
                                </Button>
                                {isReindexing && (
                                    <span className="text-xs font-semibold text-amber-900/80">
                                        {t('settings.reindexProgressHint')}
                                    </span>
                                )}
                            </div>
                        </div>
                    </div>
                </section>
            )}

            <section className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
                <div className="mb-4 flex flex-wrap items-center gap-3">
                    <h2 className="text-3xl font-extrabold text-[#1f3a60]">{t('settings.runtimeTitle')}</h2>
                    <span className="rounded-full bg-[#1f3a60]/10 px-3 py-1 text-xs font-bold text-[#1f3a60]">{t('settings.runtimeBadge')}</span>
                    {isDirty && (
                        <span className="rounded-full bg-amber-100 px-3 py-1 text-xs font-bold text-amber-700">
                            {t('settings.unsavedChanges')}
                        </span>
                    )}
                </div>

                {settingsError && (
                    <div className="mb-4 flex flex-wrap items-center gap-3 rounded-xl border border-red-200 bg-red-50 px-4 py-3">
                        <span role="alert" className="text-sm font-semibold text-red-700">
                            {resolveApiErrorMessage(settingsError, t, 'settings.loadFailed')}
                        </span>
                        <Button type="button" variant="outline" size="sm" onClick={loadSettings} isLoading={isSettingsBusy}>
                            {t('settings.retry')}
                        </Button>
                    </div>
                )}

                {isSettingsLoaded && (
                    <>
                        <div className="grid gap-4 md:grid-cols-2">
                            <div>
                                <label className="mb-2 block text-sm font-semibold text-slate-700" htmlFor="settings-chat-model">
                                    {t('settings.chatModel')}
                                </label>
                                <ModelSelect
                                    id="settings-chat-model"
                                    value={formValues.chat_model}
                                    models={catalog.available_chat_models}
                                    onChange={(event) => setField('chat_model', event.target.value)}
                                    disabled={!catalog.available_chat_models.length && !formValues.chat_model}
                                    placeholderLabel={catalog.available_chat_models.length
                                        ? t('settings.modelPlaceholder')
                                        : t('settings.noChatModels')}
                                    unknownLabel={t('settings.modelUnavailableOption', { model: formValues.chat_model })}
                                />
                                {renderModelWarning(formValues.chat_model, catalog.available_chat_models)}
                            </div>

                            <div>
                                <label className="mb-2 block text-sm font-semibold text-slate-700" htmlFor="settings-embedding-model">
                                    {t('settings.embeddingModel')}
                                </label>
                                <ModelSelect
                                    id="settings-embedding-model"
                                    value={formValues.embedding_model}
                                    models={catalog.available_embedding_models}
                                    onChange={(event) => setField('embedding_model', event.target.value)}
                                    disabled={!catalog.available_embedding_models.length && !formValues.embedding_model}
                                    placeholderLabel={catalog.available_embedding_models.length
                                        ? t('settings.modelPlaceholder')
                                        : t('settings.noEmbeddingModels')}
                                    unknownLabel={t('settings.modelUnavailableOption', { model: formValues.embedding_model })}
                                />
                                {renderModelWarning(formValues.embedding_model, catalog.available_embedding_models)}
                                <p className="mt-2 text-xs text-slate-500">
                                    {t('settings.embeddingHint')}
                                </p>
                                <p className="mt-1 text-xs text-slate-500">
                                    {t('settings.embeddingReindexHint')}
                                </p>
                                {!catalog.ollama_available && (
                                    <p className="mt-2 text-xs font-semibold text-amber-600">
                                        {t('settings.ollamaFailed', { error: catalog.ollama_error || t('settings.ollamaFallback') })}
                                    </p>
                                )}
                            </div>
                        </div>

                        <div className="mt-4 rounded-xl border border-slate-200 bg-slate-50 p-4">
                            <label className="flex items-start gap-3">
                                <input
                                    type="checkbox"
                                    checked={Boolean(formValues.enable_condense_query)}
                                    onChange={(event) => setField('enable_condense_query', event.target.checked)}
                                    className="mt-1 h-4 w-4 rounded border-slate-300 text-[#1f3a60] focus:ring-[#1f3a60]/25"
                                />
                                <span>
                                    <span className="block text-sm font-semibold text-slate-800">{t('settings.condenseQuery')}</span>
                                    <span className="mt-1 block text-xs text-slate-500">
                                        {t('settings.condenseHint')}
                                    </span>
                                </span>
                            </label>
                        </div>

                        {/* Два числа одного конвейера: сначала из индекса
                            достаётся пул кандидатов (retrieval_top_k), потом
                            из него после слияния и переранжирования отбирается
                            top_k фрагментов для модели. Стоят рядом, потому что
                            зависят друг от друга — см. подсказку под полем. */}
                        <div className="mt-4 space-y-4 rounded-xl border border-slate-200 bg-slate-50 p-4">
                            <div>
                                <label className="mb-2 block text-sm font-semibold text-slate-700" htmlFor="settings-retrieval-top-k">
                                    {t('settings.retrievalTopK')}
                                </label>
                                <input
                                    id="settings-retrieval-top-k"
                                    type="number"
                                    min={MIN_RETRIEVAL_TOP_K}
                                    max={MAX_RETRIEVAL_TOP_K}
                                    value={formValues.retrieval_top_k}
                                    onChange={(event) => setField('retrieval_top_k', event.target.value)}
                                    aria-invalid={Boolean(fieldErrors.retrieval_top_k)}
                                    className={`${NUMBER_INPUT_CLASS} w-32`}
                                />
                                {fieldErrors.retrieval_top_k && (
                                    <p className="mt-1 text-xs font-semibold text-red-600">
                                        {t(fieldErrors.retrieval_top_k)}
                                    </p>
                                )}
                                {/* Не блокировка сохранения: сервер такое
                                    значение принимает, но при запросе поднимает
                                    пул до top_k, и настройка перестаёт
                                    действовать. Молчать об этом нельзя. */}
                                {isRetrievalPoolNarrow(formValues) && (
                                    <p className="mt-1 text-xs font-semibold text-amber-600">
                                        {t('settings.retrievalTopKBelowTopK', { topK: formValues.top_k })}
                                    </p>
                                )}
                                <p className="mt-2 text-xs text-slate-500">{t('settings.retrievalTopKHint')}</p>
                            </div>

                            <div>
                                <label className="mb-2 block text-sm font-semibold text-slate-700" htmlFor="settings-top-k">
                                    {t('settings.topK')}
                                </label>
                                <input
                                    id="settings-top-k"
                                    type="number"
                                    min={MIN_TOP_K}
                                    max={MAX_TOP_K}
                                    value={formValues.top_k}
                                    onChange={(event) => setField('top_k', event.target.value)}
                                    aria-invalid={Boolean(fieldErrors.top_k)}
                                    className={`${NUMBER_INPUT_CLASS} w-32`}
                                />
                                {fieldErrors.top_k && (
                                    <p className="mt-1 text-xs font-semibold text-red-600">{t(fieldErrors.top_k)}</p>
                                )}
                                <p className="mt-2 text-xs text-slate-500">{t('settings.topKHint')}</p>
                            </div>
                        </div>

                        <div className="mt-4 rounded-xl border border-slate-200 bg-slate-50 p-4">
                            <label className="mb-2 block text-sm font-semibold text-slate-700" htmlFor="settings-domain-profile">
                                {t('settings.domainProfile')}
                            </label>
                            {/* Список, а не свободный ввод: набор профилей
                                держит сервер, и всё, чего нет в
                                available_domain_profiles, он отвергает кодом
                                settings.unsupported_domain_profile. */}
                            <select
                                id="settings-domain-profile"
                                value={formValues.default_domain_profile}
                                onChange={(event) => setField('default_domain_profile', event.target.value)}
                                disabled={!catalog.available_domain_profiles.length && !formValues.default_domain_profile}
                                className={SELECT_CLASS}
                            >
                                {!formValues.default_domain_profile && (
                                    <option value="">
                                        {catalog.available_domain_profiles.length
                                            ? t('settings.domainProfilePlaceholder')
                                            : t('settings.noDomainProfiles')}
                                    </option>
                                )}
                                {/* Сохранённого профиля нет среди доступных —
                                    показываем как есть, а не подменяем первым
                                    пунктом: это ровно та тихая подмена, от
                                    которой сервер уже отказался. */}
                                {Boolean(formValues.default_domain_profile)
                                    && !catalog.available_domain_profiles.includes(formValues.default_domain_profile) && (
                                    <option value={formValues.default_domain_profile}>
                                        {t('settings.domainProfileUnknownOption', { profile: formValues.default_domain_profile })}
                                    </option>
                                )}
                                {catalog.available_domain_profiles.map((profile) => (
                                    <option key={profile} value={profile}>{getDomainProfileLabel(profile)}</option>
                                ))}
                            </select>
                            {isDomainProfileUnknown && (
                                <p className="mt-2 text-xs font-semibold text-amber-600">
                                    {t('settings.domainProfileUnknownWarning', { profile: formValues.default_domain_profile })}
                                </p>
                            )}
                            <p className="mt-2 text-xs text-slate-500">{t('settings.domainProfileHint')}</p>
                            <p className="mt-1 text-xs text-slate-500">{t('settings.domainProfileScopeHint')}</p>
                        </div>

                        <div className="mt-4 space-y-4 rounded-xl border border-slate-200 bg-slate-50 p-4">
                            <label className="flex items-start gap-3">
                                <input
                                    type="checkbox"
                                    checked={Boolean(formValues.contextual_embedding_enabled)}
                                    onChange={(event) => setField('contextual_embedding_enabled', event.target.checked)}
                                    className="mt-1 h-4 w-4 rounded border-slate-300 text-[#1f3a60] focus:ring-[#1f3a60]/25"
                                />
                                <span>
                                    <span className="block text-sm font-semibold text-slate-800">{t('settings.contextualEmbedding')}</span>
                                    <span className="mt-1 block text-xs text-slate-500">
                                        {t('settings.contextualEmbeddingHint')}
                                    </span>
                                </span>
                            </label>

                            <div>
                                <label className="mb-2 block text-sm font-semibold text-slate-700" htmlFor="settings-contextual-model">
                                    {t('settings.contextualModel')}
                                </label>
                                <ModelSelect
                                    id="settings-contextual-model"
                                    value={formValues.contextual_embedding_model}
                                    models={catalog.available_chat_models}
                                    onChange={(event) => setField('contextual_embedding_model', event.target.value)}
                                    disabled={!catalog.available_chat_models.length && !formValues.contextual_embedding_model}
                                    // Пункт «не выбрана» постоянный, а не только
                                    // при пустом значении: пустая строка здесь —
                                    // умолчание сервера и законное состояние, и
                                    // вернуться к нему надо чем-то.
                                    emptyLabel={t('settings.contextualModelNone')}
                                    unknownLabel={t('settings.modelUnavailableOption', { model: formValues.contextual_embedding_model })}
                                />
                                {renderModelWarning(formValues.contextual_embedding_model, catalog.available_chat_models)}
                                {/* Не блокировка сохранения, а предупреждение: сервер
                                    спрашивает модель только в момент включения
                                    обогащения, и мешать остальным правкам клиент
                                    не вправе. */}
                                {formValues.contextual_embedding_enabled && !formValues.contextual_embedding_model && (
                                    <p className="mt-2 text-xs font-semibold text-amber-600">
                                        {t('settings.contextualModelMissingWarning')}
                                    </p>
                                )}
                                {!catalog.available_chat_models.length && (
                                    <p className="mt-2 text-xs text-slate-500">{t('settings.noContextualModels')}</p>
                                )}
                                <p className="mt-2 text-xs text-slate-500">
                                    {t('settings.contextualModelHint')}
                                </p>
                            </div>

                            <div>
                                <label className="mb-2 block text-sm font-semibold text-slate-700" htmlFor="settings-contextual-num-ctx">
                                    {t('settings.contextualNumCtx')}
                                </label>
                                <input
                                    id="settings-contextual-num-ctx"
                                    type="number"
                                    min={MIN_NUM_CTX}
                                    max={MAX_NUM_CTX}
                                    step={1024}
                                    value={formValues.contextual_embedding_num_ctx}
                                    onChange={(event) => setField('contextual_embedding_num_ctx', event.target.value)}
                                    aria-invalid={Boolean(fieldErrors.contextual_embedding_num_ctx)}
                                    className={`${NUMBER_INPUT_CLASS} w-40`}
                                />
                                {fieldErrors.contextual_embedding_num_ctx && (
                                    <p className="mt-1 text-xs font-semibold text-red-600">
                                        {t(fieldErrors.contextual_embedding_num_ctx)}
                                    </p>
                                )}
                                <p className="mt-1 text-xs text-slate-500">{t('settings.contextualNumCtxHint')}</p>
                                <p className="mt-1 text-xs text-slate-500">{t('settings.numCtxStrictHint')}</p>
                            </div>
                        </div>

                        <div className="mt-4 space-y-4 rounded-xl border border-slate-200 bg-slate-50 p-4">
                            <label className="mb-2 block text-sm font-semibold text-slate-700" htmlFor="settings-chat-num-ctx">
                                {t('settings.chatNumCtx')}
                            </label>
                            <input
                                id="settings-chat-num-ctx"
                                type="number"
                                min={MIN_NUM_CTX}
                                max={MAX_NUM_CTX}
                                step={1024}
                                value={formValues.chat_model_num_ctx}
                                onChange={(event) => setField('chat_model_num_ctx', event.target.value)}
                                aria-invalid={Boolean(fieldErrors.chat_model_num_ctx)}
                                className={`${NUMBER_INPUT_CLASS} w-40`}
                            />
                            {fieldErrors.chat_model_num_ctx && (
                                <p className="mt-1 text-xs font-semibold text-red-600">
                                    {t(fieldErrors.chat_model_num_ctx)}
                                </p>
                            )}
                            <p className="mt-1 text-xs text-slate-500">{t('settings.chatNumCtxHint')}</p>
                            <p className="mt-1 text-xs text-slate-500">{t('settings.numCtxStrictHint')}</p>
                        </div>

                        <div className="mt-4 rounded-xl border border-slate-200 bg-slate-50 p-4">
                            <label className="flex items-start gap-3">
                                <input
                                    type="checkbox"
                                    checked={Boolean(formValues.reranker_enabled)}
                                    onChange={(event) => setField('reranker_enabled', event.target.checked)}
                                    className="mt-1 h-4 w-4 rounded border-slate-300 text-[#1f3a60] focus:ring-[#1f3a60]/25"
                                />
                                <span>
                                    <span className="block text-sm font-semibold text-slate-800">
                                        {t('settings.reranker')}
                                    </span>
                                    <span className="mt-1 block text-xs text-slate-500">
                                        {t('settings.rerankerHint')}
                                    </span>
                                </span>
                            </label>
                        </div>

                        <div className="mt-5 flex flex-wrap items-center gap-3">
                            <Button
                                type="button"
                                onClick={handleSave}
                                isLoading={isSaving}
                                disabled={!canSave || hasFieldErrors || isBusy}
                            >
                                {t('settings.save')}
                            </Button>
                            <Button
                                type="button"
                                variant="outline"
                                onClick={handleReset}
                                isLoading={isResetting}
                                disabled={isBusy}
                            >
                                {t('settings.reset')}
                            </Button>
                            {/* Живая область стоит в разметке всегда, а не
                                появляется вместе с текстом: область aria-live,
                                добавленная в DOM уже с содержимым, объявляется
                                не всеми скринридерами, и «Настройки сохранены»
                                прошло бы мимо незрячего админа. */}
                            <span role="status" aria-live="polite" className="text-sm font-semibold text-emerald-600">
                                {notice ? t(notice.key, notice.params) : ''}
                            </span>
                            {saveError && (
                                <span role="alert" className="text-sm font-semibold text-red-600">
                                    {resolveApiErrorMessage(saveError, t, 'settings.saveFailed')}
                                </span>
                            )}
                        </div>
                    </>
                )}
            </section>

            <section className="overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-sm">
                <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-200 px-5 py-4">
                    <div className="flex flex-wrap items-center gap-3">
                        <h3 className="text-lg font-bold text-[#1f3a60]">{t('settings.rolesTitle')}</h3>
                        {/* Общее число — из X-Total-Count, а не из длины
                            страницы: сколько всего пользователей в системе,
                            интерфейс раньше показать не мог вовсе. */}
                        {usersTotal != null && (
                            <span className="rounded-full bg-[#1f3a60]/10 px-3 py-1 text-xs font-bold text-[#1f3a60]">
                                {t('settings.usersTotal', { count: usersTotal })}
                            </span>
                        )}
                    </div>

                    <span role="status" aria-live="polite" className="text-sm font-semibold text-emerald-600">
                        {usersNotice
                            ? t('settings.roleChanged', {
                                username: usersNotice.username,
                                role: getRoleLabel(usersNotice.role),
                            })
                            : ''}
                    </span>
                </div>

                {/* Единственный голосовой статус таблицы: что показано, сколько
                    найдено и чем закончилась загрузка. */}
                <p className="sr-only" role="status" aria-live="polite">{getUsersStatusMessage()}</p>

                {usersError && (
                    <div className="flex flex-wrap items-center gap-3 border-b border-slate-200 bg-red-50 px-5 py-3">
                        <span role="alert" className="text-sm font-semibold text-red-700">
                            {resolveApiErrorMessage(usersError, t, 'settings.usersLoadFailed')}
                        </span>
                        {/* Повторяется та страница, которая не загрузилась, а не первая. */}
                        <Button
                            type="button"
                            variant="outline"
                            size="sm"
                            onClick={() => loadUsers(usersPage)}
                            isLoading={isUsersBusy}
                        >
                            {t('settings.retry')}
                        </Button>
                    </div>
                )}

                {/* Область поиска названа явно: сколько пользователей проверено.
                    Без этой строки «ничего не найдено» ничем не отличалось бы от
                    «нет на этой странице». */}
                {isSearching && (
                    <div className="border-b border-slate-200 bg-slate-50 px-5 py-3">
                        {searchError ? (
                            <div className="flex flex-wrap items-center gap-3">
                                <span role="alert" className="text-sm font-semibold text-red-700">
                                    {resolveApiErrorMessage(searchError, t, 'settings.search.failed')}
                                </span>
                                <Button
                                    type="button"
                                    variant="outline"
                                    size="sm"
                                    onClick={retrySearchIndex}
                                    isLoading={isSearchLoading}
                                >
                                    {t('settings.retry')}
                                </Button>
                            </div>
                        ) : (isSearchLoading || !searchIndex) ? (
                            <p className="text-xs font-semibold text-slate-500">{t('settings.search.loading')}</p>
                        ) : (
                            <>
                                <p className="text-xs font-semibold text-slate-600">
                                    {searchMatches.length === 0
                                        ? t('settings.search.empty', {
                                            query: searchTerm,
                                            scanned: searchIndex.items.length,
                                        })
                                        : t('settings.search.results', {
                                            count: searchMatches.length,
                                            scanned: searchIndex.items.length,
                                            query: searchTerm,
                                        })}
                                </p>
                                {/* Вычитка упёрлась в потолок запросов: часть
                                    пользователей поиск не видел, и выдавать
                                    неполный результат за полный нельзя. */}
                                {!searchIndex.isComplete && (
                                    <p className="mt-1 text-xs font-semibold text-amber-700">
                                        {t('settings.search.partial', {
                                            scanned: searchIndex.items.length,
                                            total: searchIndex.total,
                                        })}
                                    </p>
                                )}
                            </>
                        )}
                    </div>
                )}

                {showUsersTable && (
                    <div className="overflow-x-auto">
                        <table className="w-full min-w-[740px] text-left">
                            <thead className="bg-slate-50 text-xs uppercase tracking-[0.08em] text-slate-500">
                                <tr>
                                    {/* scope="col" — не украшение: без него скринридер
                                        не связывает ячейку с заголовком столбца и
                                        читает строку набором значений без имён. */}
                                    <th scope="col" className="px-5 py-3 font-semibold">{t('settings.table.user')}</th>
                                    <th scope="col" className="px-5 py-3 font-semibold">{t('settings.table.role')}</th>
                                    <th scope="col" className="px-5 py-3 font-semibold">{t('settings.table.createdAt')}</th>
                                </tr>
                            </thead>

                            <tbody>
                                {visibleUsers.length === 0 ? (
                                    <tr>
                                        <td colSpan="3" className="px-5 py-10 text-center text-sm text-slate-500">
                                            {getEmptyRowMessage()}
                                        </td>
                                    </tr>
                                ) : visibleUsers.map((user) => (
                                    <tr key={user.id} className="border-t border-slate-100 text-sm hover:bg-slate-50">
                                        <td className="px-5 py-3 font-semibold text-slate-800">{user.username}</td>
                                        <td className="px-5 py-3">
                                            <select
                                                value={user.role}
                                                onChange={(event) => requestRoleChange(user, event.target.value)}
                                                // Строка блокируется на время своего запроса: пока
                                                // ответ не пришёл, порядок параллельных PUT ничем
                                                // не гарантирован.
                                                disabled={isSavingRole && roleTarget?.id === user.id}
                                                // Имя списка включает пользователя: заголовка
                                                // столбца скринридеру не хватает — в перечне форм
                                                // он видит их вне таблицы, и N списков «Роль»
                                                // неразличимы между собой.
                                                aria-label={t('settings.roleSelectLabel', { username: user.username })}
                                                className="h-9 rounded-lg border border-slate-300 bg-white px-3 text-sm text-slate-700 focus:outline-none focus:ring-2 focus:ring-[#1f3a60]/25 disabled:cursor-not-allowed disabled:opacity-60"
                                            >
                                                {/* Роль вне известного набора (значение из старой
                                                    схемы, ручная правка в БД) показывается своим
                                                    пунктом: <select> со значением вне списка
                                                    браузер рисует пустым, то есть роль исчезала
                                                    бы с экрана вместе с правом её увидеть. */}
                                                {!ROLE_OPTIONS.includes(user.role) && (
                                                    <option value={user.role}>
                                                        {t('settings.roleUnknownOption', { role: user.role })}
                                                    </option>
                                                )}
                                                {ROLE_OPTIONS.map((role) => (
                                                    <option key={role} value={role}>{getRoleLabel(role)}</option>
                                                ))}
                                            </select>
                                        </td>
                                        <td className="px-5 py-3 text-slate-500">
                                            {formatLocaleDate(user.created_at, locale, {
                                                year: 'numeric',
                                                month: 'short',
                                                day: 'numeric',
                                                hour: '2-digit',
                                                minute: '2-digit',
                                            })}
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                )}

                {/* Листание появляется только там, где страница не вмещает
                    список: при десятке пользователей навигация была бы шумом. */}
                {showUsersTable && listTotal > USERS_PAGE_SIZE && (
                    <nav
                        className="flex flex-wrap items-center justify-between gap-3 border-t border-slate-200 px-5 py-3"
                        aria-label={t('settings.pagination.label')}
                    >
                        <span className="text-xs font-semibold text-slate-500">
                            {t('settings.pagination.range', { from: rangeStart, to: rangeEnd, total: listTotal })}
                        </span>

                        <div className="flex items-center gap-2">
                            <Button
                                type="button"
                                variant="outline"
                                size="sm"
                                disabled={currentPage <= 1 || isUsersBusy}
                                onClick={() => goToPage(currentPage - 1)}
                                aria-label={t('settings.pagination.previous')}
                            >
                                <ChevronLeft className="h-4 w-4" />
                            </Button>
                            <span className="text-xs font-semibold text-slate-600">
                                {t('settings.pagination.pageOf', { page: currentPage, pages: pageCount })}
                            </span>
                            <Button
                                type="button"
                                variant="outline"
                                size="sm"
                                disabled={currentPage >= pageCount || isUsersBusy}
                                onClick={() => goToPage(currentPage + 1)}
                                aria-label={t('settings.pagination.next')}
                            >
                                <ChevronRight className="h-4 w-4" />
                            </Button>
                        </div>
                    </nav>
                )}
            </section>

            {/* Подтверждение смены роли — свой диалог, как у удаления источника и
                заметки: запрос уходит только после явного согласия, а ошибка
                остаётся рядом с вопросом. */}
            {roleTarget && (
                <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
                    <div className="absolute inset-0 bg-black/50" onClick={closeRoleDialog} aria-hidden="true" />

                    <div
                        ref={roleDialogRef}
                        role="dialog"
                        aria-modal="true"
                        aria-labelledby="role-dialog-title"
                        aria-describedby="role-dialog-description"
                        className="relative w-full max-w-md overflow-hidden rounded-2xl bg-white shadow-xl"
                    >
                        <div className="flex items-start gap-3 px-6 py-5">
                            <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full bg-amber-100 text-amber-600">
                                <AlertTriangle className="h-5 w-5" />
                            </div>
                            <div className="min-w-0">
                                <h3 id="role-dialog-title" className="text-base font-semibold text-slate-900">
                                    {t('settings.roleChangeConfirm')}
                                </h3>
                                <p id="role-dialog-description" className="mt-1 break-words text-sm text-slate-500">
                                    {/* Роли называются так же, как в таблице:
                                        служебные admin/content_manager в вопросе
                                        «менять или нет» читались бы латиницей в
                                        интерфейсе, где язык по умолчанию —
                                        таджикский. */}
                                    {t('settings.roleChangeDescription', {
                                        username: roleTarget.username,
                                        from: getRoleLabel(roleTarget.currentRole),
                                        to: getRoleLabel(roleTarget.nextRole),
                                    })}
                                </p>
                            </div>
                        </div>

                        {roleError && (
                            <p role="alert" className="mx-6 mb-2 rounded-lg bg-red-50 px-3 py-2 text-sm font-semibold text-red-700">
                                {resolveApiErrorMessage(roleError, t, 'settings.roleUpdateFailed')}
                            </p>
                        )}

                        <div className="flex justify-end gap-2 border-t border-slate-200 px-6 py-4">
                            <Button
                                ref={roleCancelRef}
                                type="button"
                                variant="outline"
                                onClick={closeRoleDialog}
                                disabled={isSavingRole}
                            >
                                {t('settings.roleChangeCancel')}
                            </Button>
                            <Button type="button" onClick={confirmRoleChange} isLoading={isSavingRole}>
                                {isSavingRole ? t('settings.roleChangeSaving') : t('settings.roleChangeApply')}
                            </Button>
                        </div>
                    </div>
                </div>
            )}

            {/* Подтверждение операции над настройками: смена embedding-модели и
                сброс к умолчаниям одинаково уводят поиск в другую коллекцию,
                поэтому спрашиваются одинаково и одним диалогом. */}
            {confirmTarget && (
                <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
                    <div className="absolute inset-0 bg-black/50" onClick={closeConfirmDialog} aria-hidden="true" />

                    <div
                        ref={confirmDialogRef}
                        role="dialog"
                        aria-modal="true"
                        aria-labelledby="settings-confirm-title"
                        aria-describedby="settings-confirm-description"
                        className="relative w-full max-w-md overflow-hidden rounded-2xl bg-white shadow-xl"
                    >
                        <div className="flex items-start gap-3 px-6 py-5">
                            <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full bg-amber-100 text-amber-600">
                                <AlertTriangle className="h-5 w-5" />
                            </div>
                            <div className="min-w-0">
                                <h3 id="settings-confirm-title" className="text-base font-semibold text-slate-900">
                                    {confirmTarget.kind === 'reset'
                                        ? t('settings.resetConfirmTitle')
                                        : t('settings.reindexConfirmTitle')}
                                </h3>
                                <p id="settings-confirm-description" className="mt-1 break-words text-sm text-slate-500">
                                    {renderConfirmDescription()}
                                </p>
                            </div>
                        </div>

                        {confirmError && (
                            <p role="alert" className="mx-6 mb-2 rounded-lg bg-red-50 px-3 py-2 text-sm font-semibold text-red-700">
                                {resolveApiErrorMessage(
                                    confirmError,
                                    t,
                                    confirmTarget.kind === 'reset' ? 'settings.resetFailed' : 'settings.saveFailed',
                                )}
                            </p>
                        )}

                        <div className="flex justify-end gap-2 border-t border-slate-200 px-6 py-4">
                            <Button
                                ref={confirmCancelRef}
                                type="button"
                                variant="outline"
                                onClick={closeConfirmDialog}
                                disabled={isSaving || isResetting}
                            >
                                {t('settings.cancel')}
                            </Button>
                            <Button
                                type="button"
                                onClick={confirmDialogAction}
                                isLoading={isSaving || isResetting}
                            >
                                {confirmTarget.kind === 'reset'
                                    ? t('settings.resetApply')
                                    : t('settings.reindexConfirmApply')}
                            </Button>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
};

export default SettingsPage;
