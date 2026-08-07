// Общие правила темы документа: как читается доля, как темы упорядочены и как
// фильтр по теме переносится между экраном «Темы» и списком источников.
//
// Один файл на оба экрана намеренно: имена параметров и правило сравнения
// иначе разъехались бы, а фильтр, который ставит один экран и не понимает
// другой, — ровно та невидимая фильтрация, которую тут уже дважды чинили.

/**
 * Номер кластера в адресе. Пользователю он не показывается нигде: это
 * внутренний идентификатор, годный только как параметр запроса.
 */
export const TOPIC_PARAM = 'topic';

/**
 * Подпись темы рядом с номером. Дублирование выглядит лишним, но без него
 * ссылка на тему, под которую не попал ни один видимый источник, оставляла бы
 * бейдж фильтра без названия — а фильтр обязан называть себя раньше, чем
 * загрузится список.
 */
export const TOPIC_LABEL_PARAM = 'topicLabel';

const toClusterIndex = (value) => {
    if (value === null || value === undefined || value === '') return null;
    const parsed = Number(value);
    return Number.isInteger(parsed) ? parsed : null;
};

/** Адрес списка источников, суженного до одной темы. */
export const buildTopicFilterSearch = (clusterIndex, label) => {
    const params = new URLSearchParams();
    params.set(TOPIC_PARAM, String(clusterIndex));
    if (label) params.set(TOPIC_LABEL_PARAM, label);
    return `?${params.toString()}`;
};

/** Фильтр из адреса: clusterIndex === null означает «фильтра нет». */
export const readTopicFilter = (searchParams) => ({
    clusterIndex: toClusterIndex(searchParams?.get(TOPIC_PARAM)),
    label: String(searchParams?.get(TOPIC_LABEL_PARAM) || '').trim(),
});

export const matchesTopicFilter = (source, clusterIndex) => (
    clusterIndex == null || toClusterIndex(source?.topic_cluster_index) === clusterIndex
);

/**
 * Порядок предпочтения подписей по локали интерфейса.
 *
 * Английский в этом списке ПОСЛЕДНИЙ и без своего поля: английского интерфейса
 * в продукте нет, а topic_label / label — это устойчивый ключ темы, годный как
 * подпись только тогда, когда перевода не нашлось.
 *
 * Таджикский откатывается на русский раньше, чем на ключ: между английским
 * названием и русским таджикоязычному пользователю ближе русское.
 */
const LABEL_ORDER = {
    ru: ['ru'],
    tg: ['tg', 'ru'],
};

const pickLabel = (source, locale, prefix) => {
    const order = LABEL_ORDER[locale] || LABEL_ORDER.ru;
    for (const language of order) {
        const value = String(source?.[`${prefix}${language}`] || '').trim();
        if (value) return value;
    }
    return '';
};

/**
 * Подпись темы документа; пустая строка означает «темы у документа нет».
 *
 * Откат к topic_label обязателен: null в переводе — обычное состояние
 * (документ размечен моделью без переводов или до появления колонок), и
 * подпись при этом всё равно должна быть.
 */
export const resolveTopicLabel = (source, locale) => (
    pickLabel(source, locale, 'topic_label_') || String(source?.topic_label || '').trim()
);

const UNCLEAR_COUNT_HEADER = 'x-unclear-count';

/**
 * Сколько документов выборки модель посмотрела и оставила без темы.
 *
 * Приходит заголовком: тело ответа — список тем, и число, которое ни к одной
 * теме не относится, вложить в него некуда. Разбор тот же, что у X-Total-Count
 * в списке источников, включая обе формы заголовков (Headers и обычный объект):
 * axios отдаёт объект, fetch — Headers, и полагаться на одну из них значило бы
 * получить null на другом клиенте.
 *
 * null, а не ноль, когда заголовка нет: ноль означал бы «модель никого не
 * пропустила», а отсутствие — «сервер об этом не сказал», и строку на экране в
 * этом случае показывать нельзя.
 */
export const readUnclearCount = (response) => {
    const headers = response?.headers;
    if (!headers) return null;
    const raw = typeof headers.get === 'function'
        ? headers.get(UNCLEAR_COUNT_HEADER)
        : headers[UNCLEAR_COUNT_HEADER];
    const value = Number(raw);
    return Number.isFinite(value) && raw != null && raw !== '' ? value : null;
};

/**
 * Модель посмотрела документ и отказалась называть тему.
 *
 * Два отсутствия темы выглядят на экране одинаково, а означают разное:
 *
 *   версии модели нет                 — документ ещё не размечали;
 *   версия есть, номера кластера нет  — модель отказалась: документ стоит почти
 *                                       ровно между двумя темами, и любая из
 *                                       них была бы монеткой.
 *
 * Второе — законченный ответ, а не ожидание, и молчать о нём значит заставить
 * пользователя ждать разметки, которая уже прошла.
 */
export const isTopicUnclear = (source) => (
    source?.topic_model_version != null
    && source?.topic_cluster_index == null
    && !resolveTopicLabel(source, 'ru')
);

/**
 * Подпись темы в распределении — по тому же правилу, что у документа.
 *
 * Отдельная функция, потому что поля приходят из разных ответов API
 * (label_ru у темы против topic_label_ru у источника), а правило выбора обязано
 * остаться одним: разъехавшись, эти два экрана назвали бы одну тему по-разному.
 */
export const resolveTopicName = (topic, locale) => (
    pickLabel(topic, locale, 'label_') || String(topic?.label || '').trim()
);

/** Тема, в которую не попал ни один видимый источник. */
export const isEmptyTopic = (topic) => !(Number(topic?.document_count) > 0);

/**
 * Доля темы как число от 0 до 1.
 *
 * Бэкенд обещает долю единицы, но значение больше единицы долей быть не может —
 * такое приходит только процентами, и показать «4200 %» хуже, чем привести. Если
 * доли нет вовсе, считаем её от общего числа документов: количество на экране
 * уже есть, и оставлять полосу пустой не из-за чего.
 */
export const resolveTopicShare = (topic, totalDocuments) => {
    const raw = Number(topic?.share);
    if (Number.isFinite(raw) && raw > 0) return Math.min(raw > 1 ? raw / 100 : raw, 1);

    const count = Number(topic?.document_count);
    if (!Number.isFinite(count) || !totalDocuments) return 0;
    return Math.min(count / totalDocuments, 1);
};

/**
 * Темы по убыванию количества документов. Порядок задаём сами, а не полагаемся
 * на ответ: экран обещает «сначала самые крупные», и это обещание не должно
 * зависеть от сортировки на стороне сервера.
 */
export const sortTopics = (topics) => [...(topics || [])].sort((a, b) => (
    Number(b?.document_count || 0) - Number(a?.document_count || 0)
));

/* ── Папки ──────────────────────────────────────────────────────────────────
 *
 * Тема = папка с документами. Раскладку делает этот файл, а не экраны: раздел
 * «Темы» и выбор источников для блокнота показывают одни и те же папки, и
 * разъехавшись, они по-разному ответили бы на вопрос «где лежит документ».
 */

export const TOPIC_FOLDER_TOPIC = 'topic';

/** Модель посмотрела документ и не смогла назвать тему — см. isTopicUnclear. */
export const TOPIC_FOLDER_UNCLEAR = 'unclear';

/** Версии модели у документа нет вовсе: его ещё не размечали. */
export const TOPIC_FOLDER_UNLABELED = 'unlabeled';

/**
 * Две особые папки нарочно разные, хотя на экране обе выглядят как «темы нет».
 *
 * «Без темы» — законченный ответ модели, и делать с ним нечего; «Ещё не
 * размечены» пройдёт само после переразметки. Сваленные в одну кучу, они
 * заставили бы ждать того, что уже случилось, — или чинить то, что не ломалось.
 *
 * Подписи держим ключами перевода, а не строками: в этом файле нет локали, а
 * два экрана обязаны называть особую папку одинаково.
 */
export const TOPIC_FOLDER_NAME_KEY = {
    [TOPIC_FOLDER_UNCLEAR]: 'topics.folders.unclear',
    [TOPIC_FOLDER_UNLABELED]: 'topics.folders.unlabeled',
};

export const TOPIC_FOLDER_HINT_KEY = {
    [TOPIC_FOLDER_UNCLEAR]: 'topics.folders.unclearHint',
    [TOPIC_FOLDER_UNLABELED]: 'topics.folders.unlabeledHint',
};

/** Название папки: у темы — от модели, у особой — из переводов. */
export const resolveTopicFolderName = (folder, t) => (
    folder?.kind === TOPIC_FOLDER_TOPIC
        ? folder.name
        : t(TOPIC_FOLDER_NAME_KEY[folder?.kind] || '')
);

/**
 * До скольких документов папки открыты сразу.
 *
 * Папки нужны там, где список не окинуть взглядом; на десятке документов
 * закрытые папки — это лишний клик за тем, что и так помещалось на экран.
 * Порог общий для обоих экранов: одинаковый список не должен вести себя
 * по-разному в зависимости от того, откуда на него смотрят.
 */
export const TOPIC_FOLDER_AUTO_OPEN_LIMIT = 12;

// Тема документа полями отличается от темы в распределении (topic_label_ru
// против label_ru), а правило выбора подписи одно — переклеиваем поля и зовём
// ту же resolveTopicName вместо второй копии правила.
const topicFromSource = (source) => ({
    cluster_index: toClusterIndex(source?.topic_cluster_index),
    label: source?.topic_label,
    label_ru: source?.topic_label_ru,
    label_tg: source?.topic_label_tg,
});

/**
 * Документы, разложенные по папкам.
 *
 * topics — распределение с экрана «Темы». Оно задаёт и порядок папок, и их
 * счётчики: список источников догружается страницами, и считать по нему значило
 * бы показывать числа, которые меняются на глазах. Там, где распределения нет
 * (выбор источников для блокнота), папки строятся по самим документам.
 *
 * unclearCount — заголовок X-Unclear-Count, ответ сервера по всей выборке.
 * Берём большее из него и числа найденных документов: заголовок знает про
 * недогруженные страницы, а список — про то, что показываем прямо сейчас.
 */
export const buildTopicFolders = (sources, locale, { topics = null, unclearCount = null } = {}) => {
    const byCluster = new Map();
    const unclear = [];
    const unlabeled = [];

    (sources || []).forEach((source) => {
        const clusterIndex = toClusterIndex(source?.topic_cluster_index);
        if (clusterIndex != null) {
            const bucket = byCluster.get(clusterIndex);
            if (bucket) bucket.push(source);
            else byCluster.set(clusterIndex, [source]);
            return;
        }

        // Кластера нет — документ идёт в одну из двух особых папок, и признак
        // тут версия модели: она проставлена ровно у тех, кого модель уже
        // посмотрела. Документ с подписью темы, но без номера кластера сюда же:
        // разметку он прошёл, и обещать ему переразметку было бы неправдой.
        if (isTopicUnclear(source) || source?.topic_model_version != null) {
            unclear.push(source);
            return;
        }

        unlabeled.push(source);
    });

    const folders = [];
    const known = new Set();

    sortTopics(topics || []).forEach((topic) => {
        const clusterIndex = toClusterIndex(topic?.cluster_index);
        if (clusterIndex == null) return;

        known.add(clusterIndex);
        folders.push({
            key: `${TOPIC_FOLDER_TOPIC}:${clusterIndex}`,
            kind: TOPIC_FOLDER_TOPIC,
            clusterIndex,
            topic,
            name: resolveTopicName(topic, locale),
            count: Number(topic?.document_count) || 0,
            sources: byCluster.get(clusterIndex) || [],
        });
    });

    // Кластер, которого в распределении не оказалось. Молча выбросить такие
    // документы нельзя: пользователь считает папки полным ответом на вопрос
    // «где мои файлы», и пропавший файл он будет искать глазами, а не в API.
    [...byCluster.entries()]
        .filter(([clusterIndex]) => !known.has(clusterIndex))
        .sort((a, b) => b[1].length - a[1].length)
        .forEach(([clusterIndex, folderSources]) => {
            folders.push({
                key: `${TOPIC_FOLDER_TOPIC}:${clusterIndex}`,
                kind: TOPIC_FOLDER_TOPIC,
                clusterIndex,
                topic: null,
                name: resolveTopicName(topicFromSource(folderSources[0]), locale),
                count: folderSources.length,
                sources: folderSources,
            });
        });

    // Особые папки — последними: сначала темы, ради которых сюда пришли.
    if (unclear.length > 0 || Number(unclearCount) > 0) {
        folders.push({
            key: TOPIC_FOLDER_UNCLEAR,
            kind: TOPIC_FOLDER_UNCLEAR,
            clusterIndex: null,
            topic: null,
            name: '',
            count: Math.max(Number(unclearCount) || 0, unclear.length),
            sources: unclear,
        });
    }

    if (unlabeled.length > 0) {
        folders.push({
            key: TOPIC_FOLDER_UNLABELED,
            kind: TOPIC_FOLDER_UNLABELED,
            clusterIndex: null,
            topic: null,
            name: '',
            count: unlabeled.length,
            sources: unlabeled,
        });
    }

    return folders;
};

/**
 * Поиск идёт по всем папкам сразу — и по их названиям, и по именам документов.
 *
 * Папка, найденная по названию, остаётся целой: спросили тему — показываем
 * тему. Папка, найденная по документу, сжимается до совпавших: остальное её
 * содержимое к запросу отношения не имеет и только прячет ответ.
 */
export const filterTopicFolders = (folders, query, t) => {
    const needle = String(query || '').trim().toLowerCase();
    if (!needle) return folders || [];

    return (folders || []).reduce((visible, folder) => {
        const nameMatches = resolveTopicFolderName(folder, t).toLowerCase().includes(needle);
        const matched = folder.sources.filter(
            (source) => String(source?.name || '').toLowerCase().includes(needle),
        );

        if (nameMatches) visible.push(folder);
        else if (matched.length > 0) visible.push({ ...folder, count: matched.length, sources: matched });

        return visible;
    }, []);
};
