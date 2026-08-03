import axios from 'axios';
import { resolveErrorCode } from '../lib/apiError';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '/api/v1';

// Токены живут в httpOnly-куках: JS их не видит, поэтому украсть их через XSS
// нечем. Читаемой остаётся только CSRF-кука — она не секрет, её значение всего
// лишь подтверждает, что запрос отправил наш же код, а не чужая страница.
const CSRF_COOKIE_NAME = 'sd_csrf_token';
export const CSRF_HEADER_NAME = 'X-CSRF-Token';

// Несекретные остатки сессии: имя нужно шапке и ключам истории чата, токенов
// здесь нет и быть не должно.
const USERNAME_STORAGE_KEY = 'knowledgeai.username';

// Роль владельца сессии — подсказка интерфейсу, а не право доступа: решение
// «можно или нельзя» принимает сервер, здесь она нужна лишь для того, чтобы не
// показывать разделы, которые всё равно ответят 403. Поэтому хранится вместе
// со сроком годности: бессрочная копия роли в localStorage хуже её отсутствия —
// разжалованный пользователь носил бы её до следующего входа.
const ROLE_STORAGE_KEY = 'knowledgeai.role';

// Событие, по которому подписчики перечитывают роль: 403 от сервера и новый
// вход должны убирать и возвращать пункты меню сразу, а не после перезагрузки.
export const SESSION_ROLE_EVENT = 'knowledgeai:session-role';

// Отказ по нехватке прав. Владение ресурсом бэкенд отдаёт как 404 (см.
// app/api/deps.py), поэтому 403 с этими кодами означает ровно одно: роль
// сессии не та, за которую её принимал клиент.
const FORBIDDEN_ROLE_CODES = ['auth.forbidden', 'forbidden'];

// Ключи прошлой схемы хранения. Их надо стереть у тех, кто уже вошёл: иначе
// мёртвые токены так и останутся лежать в браузере.
const LEGACY_TOKEN_STORAGE_KEYS = ['token', 'knowledgeai.refreshToken', 'knowledgeai.tokenExpiresAt'];

const LOGIN_PATH = '/login';
export const SESSION_EXPIRED_PARAM = 'session';
export const SESSION_EXPIRED_VALUE = 'expired';

// Изменяющие методы бэкенд пропускает только с CSRF-заголовком, если запрос
// аутентифицирован кукой. Вход и регистрация исключены: сессии ещё нет.
const CSRF_PROTECTED_METHODS = ['post', 'put', 'patch', 'delete'];
const CSRF_EXEMPT_PATHS = ['/auth/login', '/auth/register'];

// withCredentials обязателен: без него браузер не пошлёт куки на другой порт
// (5173 -> 8001) и не примет выданные бэкендом.
const api = axios.create({
    baseURL: API_BASE_URL,
    withCredentials: true,
    headers: {
        'Content-Type': 'application/json',
    },
});

// Отдельный клиент без перехватчиков обновления: обмен и гашение сессии не
// должны рекурсивно втягиваться в ту же логику, которую они обслуживают.
const authApi = axios.create({
    baseURL: API_BASE_URL,
    withCredentials: true,
    headers: {
        'Content-Type': 'application/json',
    },
});

let refreshPromise = null;
let redirectingToLogin = false;

const readCookie = (name) => {
    if (typeof document === 'undefined') return null;

    const prefix = `${name}=`;
    const match = document.cookie.split('; ').find((entry) => entry.startsWith(prefix));
    if (!match) return null;

    try {
        return decodeURIComponent(match.slice(prefix.length));
    } catch {
        return match.slice(prefix.length);
    }
};

/** Значение CSRF-куки: его же бэкенд ждёт в заголовке (double submit). */
export const getCsrfToken = () => readCookie(CSRF_COOKIE_NAME) || null;

/**
 * Признак живой сессии. Access и refresh лежат в httpOnly-куках, из JS их не
 * прочитать, поэтому «залогинен» определяется по CSRF-куке: бэкенд ставит и
 * гасит её вместе с остальными. Заводить в localStorage копию токена ради
 * такой проверки нельзя — это вернуло бы ровно ту проблему, от которой ушли.
 */
export const hasActiveSession = () => Boolean(getCsrfToken());

export const getSessionUsername = () => localStorage.getItem(USERNAME_STORAGE_KEY) || '';

const notifySessionRoleChange = () => {
    if (typeof window === 'undefined') return;
    window.dispatchEvent(new Event(SESSION_ROLE_EVENT));
};

/** Подписка на изменения роли; возвращает функцию отписки. */
export const subscribeSessionRole = (listener) => {
    if (typeof window === 'undefined') return () => {};

    window.addEventListener(SESSION_ROLE_EVENT, listener);
    return () => window.removeEventListener(SESSION_ROLE_EVENT, listener);
};

export const clearSessionRole = () => {
    localStorage.removeItem(ROLE_STORAGE_KEY);
    notifySessionRoleChange();
};

/**
 * Запомнить роль ровно на срок жизни access-токена: `expires_in` из того же
 * ответа. Дальше подсказка считается протухшей и добывается заново обменом
 * токенов — то есть у сервера, а не из памяти браузера.
 */
export const storeSessionRole = (role, expiresInSeconds) => {
    const normalized = typeof role === 'string' ? role.trim() : '';
    if (!normalized) {
        clearSessionRole();
        return '';
    }

    const ttlSeconds = Number(expiresInSeconds);
    const expiresAt = Date.now() + (Number.isFinite(ttlSeconds) && ttlSeconds > 0 ? ttlSeconds * 1000 : 0);

    localStorage.setItem(ROLE_STORAGE_KEY, JSON.stringify({ role: normalized, expiresAt }));
    notifySessionRoleChange();
    return normalized;
};

/** Роль сессии, пока она не протухла. Пустая строка — «неизвестно». */
export const getSessionRole = () => {
    const raw = localStorage.getItem(ROLE_STORAGE_KEY);
    if (!raw) return '';

    let stored = null;
    try {
        stored = JSON.parse(raw);
    } catch {
        localStorage.removeItem(ROLE_STORAGE_KEY);
        return '';
    }

    if (!stored?.role || !(Number(stored.expiresAt) > Date.now())) {
        localStorage.removeItem(ROLE_STORAGE_KEY);
        return '';
    }

    return String(stored.role);
};

const purgeLegacyTokenStorage = () => {
    LEGACY_TOKEN_STORAGE_KEYS.forEach((key) => localStorage.removeItem(key));
};

// Чистим сразу при загрузке модуля: у уже вошедшего пользователя старая пара
// токенов лежит в localStorage с прошлой версии клиента.
purgeLegacyTokenStorage();

/**
 * Начало сессии. Куки ставит бэкенд в ответе на вход, клиенту остаётся
 * запомнить только несекретное имя пользователя — его показывает шапка и по
 * нему разделяется локальная история чата.
 */
export const storeAuthSession = (username) => {
    const normalized = typeof username === 'string' ? username.trim() : '';
    if (!normalized) return '';

    localStorage.setItem(USERNAME_STORAGE_KEY, normalized);
    return normalized;
};

/**
 * Локальные следы сессии. httpOnly-куки гасит бэкенд, а читаемую CSRF-куку
 * сбрасываем сами: пока она жива, маршруты считают сессию действующей.
 * Best-effort — куку, выданную с атрибутом Domain, из JS так не удалить, но её
 * снимет ответ /auth/logout.
 */
export const clearAuthSession = () => {
    purgeLegacyTokenStorage();
    localStorage.removeItem(USERNAME_STORAGE_KEY);
    clearSessionRole();

    if (typeof document !== 'undefined') {
        document.cookie = `${CSRF_COOKIE_NAME}=; path=/; max-age=0`;
    }
};

// Ручки, которые нельзя обновлять и повторять: /auth/refresh при 401 повторить
// нечем, а вход, регистрация и выход в обновлении не нуждаются.
const NON_RETRYABLE_AUTH_PATHS = ['/auth/login', '/auth/register', '/auth/refresh', '/auth/logout'];

const isNonRetryableAuthRequest = (url = '') => NON_RETRYABLE_AUTH_PATHS.some((path) => url.includes(path));

const isCsrfExemptRequest = (url = '') => CSRF_EXEMPT_PATHS.some((path) => url.includes(path));

const applyCsrfHeader = (config) => {
    const method = (config.method || 'get').toLowerCase();
    if (!CSRF_PROTECTED_METHODS.includes(method) || isCsrfExemptRequest(config.url)) return config;

    const csrfToken = getCsrfToken();
    if (csrfToken) {
        config.headers[CSRF_HEADER_NAME] = csrfToken;
    }
    return config;
};

api.interceptors.request.use(applyCsrfHeader, (error) => Promise.reject(error));
authApi.interceptors.request.use(applyCsrfHeader, (error) => Promise.reject(error));

const requestNewSession = async () => {
    // Без куки обновлять нечего: пустой запрос всё равно вернёт 401.
    if (!hasActiveSession()) return false;

    // Тело пустое: refresh-токен бэкенд возьмёт из куки, а CSRF-заголовок
    // приложит перехватчик authApi.
    const { data } = await authApi.post('/auth/refresh', {});
    // Роль читаем при каждом обмене: если её сменили, пока пользователь сидел
    // на странице, обмен — первый момент, когда об этом можно узнать честно.
    storeSessionRole(data?.role, data?.expires_in);
    return true;
};

/**
 * Обновление строго в одном экземпляре: параллельные запросы, упёршиеся в
 * протухший access, ждут общий промис. Второй обмен того же refresh-токена
 * бэкенд трактует как кражу и гасит все сессии пользователя.
 */
export const refreshSession = () => {
    if (!refreshPromise) {
        refreshPromise = requestNewSession()
            .catch((error) => {
                // Отказ бэкенда (401 auth.invalid_token / auth.token_revoked,
                // 403 auth.csrf_failed) означает, что сессию не восстановить.
                // Сетевой сбой без ответа куки не обесценивает — оставляем их
                // до следующей попытки.
                if (error?.response) clearAuthSession();
                return false;
            })
            .finally(() => {
                refreshPromise = null;
            });
    }

    return refreshPromise;
};

/**
 * Роль текущей сессии: сохранённая, пока не истёк её срок, иначе — свежая, из
 * ответа на обмен токенов. Пустая строка означает «сервер не подтвердил роль»,
 * и трактовать её надо как отсутствие прав: интерфейс ничего не разрешает, он
 * только скрывает заведомо недоступное.
 */
export const ensureSessionRole = async () => {
    if (!hasActiveSession()) return '';

    const cached = getSessionRole();
    if (cached) return cached;

    await refreshSession();
    return getSessionRole();
};

/**
 * Сессия не восстановима: чистим локальные следы и уводим на вход. Флаг гасит
 * гонку, когда в 401 одновременно упёрлось несколько запросов.
 */
export const forceLogout = () => {
    clearAuthSession();

    if (typeof window === 'undefined' || redirectingToLogin) return;
    if (window.location.pathname === LOGIN_PATH) return;

    redirectingToLogin = true;
    window.location.replace(`${LOGIN_PATH}?${SESSION_EXPIRED_PARAM}=${SESSION_EXPIRED_VALUE}`);
};

/** Выход: гасим сессию на бэкенде, он же снимает куки, затем чистим локальное. */
export const logout = async () => {
    if (hasActiveSession()) {
        try {
            await authApi.post('/auth/logout', {});
        } catch {
            // Ручка всегда отвечает 200, сюда попадаем только на сетевом сбое.
            // Удерживать пользователя в системе из-за него неправильно.
        }
    }

    clearAuthSession();
};

/** Смена пароля обесценивает старые токены; новые куки бэкенд ставит сам. */
export const changePassword = (currentPassword, newPassword) => api.post('/auth/change-password', {
    current_password: currentPassword,
    new_password: newPassword,
});

// Негодный, отозванный или просроченный токен приходит как 401. Пробуем
// обновиться ровно один раз на запрос, иначе — корректный выход.
api.interceptors.response.use(
    (response) => response,
    async (error) => {
        const config = error?.config;

        // 403 по нехватке прав — это ответ сервера «роль не та». Он же
        // единственный достоверный признак того, что пользователя разжаловали
        // прямо сейчас: сбрасываем подсказку, и разделы, которых больше нет,
        // пропадают из интерфейса, не дожидаясь перезагрузки страницы.
        if (error?.response?.status === 403 && FORBIDDEN_ROLE_CODES.includes(resolveErrorCode(error))) {
            clearSessionRole();
            return Promise.reject(error);
        }

        if (error?.response?.status !== 401 || !config || isNonRetryableAuthRequest(config.url)) {
            return Promise.reject(error);
        }

        // Повтор с обновлённой сессией снова упёрся в 401 — обновляться дальше
        // бессмысленно, иначе получится бесконечный круг.
        if (config.__authRetried) {
            forceLogout();
            return Promise.reject(error);
        }

        config.__authRetried = true;

        const refreshed = await refreshSession();
        if (!refreshed) {
            forceLogout();
            return Promise.reject(error);
        }

        // Куки уже обновлены браузером, повтор уходит с ними; CSRF-заголовок
        // проставит перехватчик запроса.
        return api.request(config);
    }
);

export default api;
