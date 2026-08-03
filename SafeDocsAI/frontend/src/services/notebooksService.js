import api from './api';


export const notebooksService = {
  getAll: () => api.get('/notebooks/'),
  getById: (id) => api.get(`/notebooks/${id}`),
  create: (payload) => api.post('/notebooks/', payload),
  // Частичное обновление: бэкенд применяет только присланные поля, поэтому
  // отправлять весь объект нельзя — не переданное описание он оставит как есть,
  // а description: null означает «очистить».
  update: (id, payload) => api.patch(`/notebooks/${id}`, payload),
  delete: (id) => api.delete(`/notebooks/${id}`),
};
