import api from './api';


export const insightsService = {
  getAll: (notebookId) => api.get(`/insights/${notebookId ? `?notebook_id=${notebookId}` : ''}`),
  create: (payload) => api.post('/insights/', payload),
  // Частичное обновление, как у заметок: применяются только присланные поля
  // (title, body, insight_type, evidence_json), null не принимается.
  update: (id, payload) => api.patch(`/insights/${id}`, payload),
  delete: (id) => api.delete(`/insights/${id}`),
};
