import api from './api';


export const settingsService = {
  get: () => api.get('/settings/'),
  // Тело — частичный патч: сервер меняет только присутствующие ключи. Поле
  // confirm_reindex настройкой не является и требуется, только когда запрос
  // меняет embedding_model (см. SettingsErrors.REINDEX_CONFIRMATION_REQUIRED).
  update: (payload) => api.put('/settings/', payload),
  // Сброс к умолчаниям. Возвращает тот же объект настроек, что GET и PUT,
  // поэтому экран обновляется одним ответом. Тело нужно только ради
  // confirm_reindex: сброс может увести embedding-модель к умолчанию.
  reset: (payload = {}) => api.post('/settings/reset', payload),
  getUsers: () => api.get('/settings/users'),
  updateUserRole: (userId, role) => api.put(`/settings/users/${userId}/role`, { role }),
};
