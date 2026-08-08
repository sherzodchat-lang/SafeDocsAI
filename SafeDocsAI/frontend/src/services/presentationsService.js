import api from './api';

// Список презентаций постраничный, как и список источников: тело — голый массив,
// общее число приходит заголовком X-Total-Count (разбирает
// normalizeSourcesResponse из lib/sources.js).
const buildListQuery = ({ skip, limit } = {}) => {
  const params = new URLSearchParams();
  if (skip != null) params.set('skip', String(skip));
  if (limit != null) params.set('limit', String(limit));

  const query = params.toString();
  return query ? `?${query}` : '';
};

/**
 * Превью запрашивается по адресу ИЗ ОТВЕТА (поле preview_url), а не по
 * склеенному здесь пути: как именно сервер раздаёт картинки — его дело, и
 * повторять эту схему на клиенте значит ломаться от любой её правки.
 *
 * Функция ОДНА на оба вида превью — шаблона оформления и самой колоды, — потому
 * что различаются они только адресом, который в обоих случаях приходит с
 * сервера. Вторая копия этих же трёх строк отличалась бы от первой ровно тем,
 * что однажды забыли бы поправить.
 *
 * Отсюда и разбор адреса: абсолютный (http…) или начинающийся со слэша
 * preview_url — это уже полный путь, и baseURL клиента к нему приписывать
 * нельзя, иначе получится /api/v1/api/v1/... Относительный адрес, наоборот,
 * дописывается базой, как любой другой вызов.
 *
 * Картинка тянется axios-клиентом в blob, а не подставляется прямо в src:
 * ручка закрыта сессией, и <img> отправил бы запрос без учётных данных, если
 * API живёт на другом origin (в разработке это 5173 -> 8001). Тот же приём уже
 * применяется к предпросмотру источника (sourcesService.getPreviewBlob).
 */
const isAbsoluteUrl = (url) => /^https?:\/\//i.test(url) || String(url).startsWith('/');

export const presentationsService = {
  getTemplates: () => api.get('/presentations/templates'),
  getPreviewBlob: (previewUrl) => api.get(previewUrl, {
    responseType: 'blob',
    ...(isAbsoluteUrl(previewUrl) ? { baseURL: '' } : {}),
  }),
  getPage: (notebookId, options = {}) => api.get(
    `/notebooks/${notebookId}/presentations${buildListQuery(options)}`,
  ),
  // 202: заказ принят в очередь, объект приходит со status = queued и
  // queue_position — готового файла в этом ответе нет и быть не может.
  create: (notebookId, payload) => api.post(`/notebooks/${notebookId}/presentations`, payload),
  getById: (id) => api.get(`/presentations/${id}`),
  downloadBlob: (id) => api.get(`/presentations/${id}/download`, { responseType: 'blob' }),
  delete: (id) => api.delete(`/presentations/${id}`),
};

export default presentationsService;
