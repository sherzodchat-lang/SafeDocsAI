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

/** Подпись темы документа; пустая строка означает «темы у документа нет». */
export const resolveTopicLabel = (source) => String(source?.topic_label || '').trim();

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
