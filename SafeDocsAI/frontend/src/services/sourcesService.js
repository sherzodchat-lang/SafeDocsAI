import api from './api';


const buildListQuery = ({ notebookId, skip, limit }) => {
  const params = new URLSearchParams();
  if (skip != null) params.set('skip', String(skip));
  if (limit != null) params.set('limit', String(limit));
  if (notebookId != null) params.set('notebook_id', String(notebookId));

  const query = params.toString();
  return query ? `?${query}` : '';
};

export const sourcesService = {
  // Список читают только через SourcesProvider: он дедуплицирует запросы и держит
  // общий кэш для таблицы источников, панели блокнота и шапки блокнота.
  getPage: (options = {}) => api.get(`/sources/${buildListQuery(options)}`),
  upload: (formData) => api.post('/sources/upload', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  }),
  attachExisting: (payload) => api.post('/sources/attach', payload),
  // Пара к attachExisting, и тело у неё намеренно то же самое
  // (backend/app/api/endpoints/documents.py, DetachSourcesPayload):
  // {notebook_id, source_ids}. notebook_id обязателен — он говорит, ИЗ ЧЕГО
  // убирают, поэтому вкладка со старым списком не снимет документ с блокнота,
  // куда его успели перенести. Ответ — {updated_count, documents}, где
  // updated_count это число реально изменённых строк: повторный вызов по уже
  // отвязанному документу отвечает 200 с нулём, а не ошибкой.
  detachExisting: (payload) => api.post('/sources/detach', payload),
  delete: (id) => api.delete(`/sources/${id}`),
  getChunks: (id) => api.get(`/sources/${id}/chunks`),
  getPreviewBlob: (id) => api.get(`/sources/${id}/preview`, { responseType: 'blob' }),
  getChunkContext: (docId, chunkId, neighbors = 2) =>
    api.get(`/sources/${docId}/chunk/${chunkId}/context?neighbors=${neighbors}`),
};


// Всё, что делается над источниками поштучно, живёт в /sources. Обслуживание
// индекса целиком — отдельный админский эндпоинт в /documents: переиндексация
// перестраивает векторы всех документов разом и гасит флаг reindex_required
// в настройках, поэтому её запускает раздел «Настройки», а не карточка файла.
export const documentsService = {
  ...sourcesService,
  reindexAll: () => api.post('/documents/reindex'),
};
