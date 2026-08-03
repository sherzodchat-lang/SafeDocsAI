import api from './api';


// Строка запроса списка пользователей. Параметры не подставляются, если их не
// задали: сервер сам знает свои умолчания (skip=0, limit=500), а значение вне
// диапазона он отвергает 422 БЕЗ машинного кода — то есть сообщением, которого
// интерфейсу нечем перевести. Поэтому числа сюда приходят из констант страницы,
// а не собираются по месту.
const buildUsersQuery = ({ skip, limit } = {}) => {
  const params = new URLSearchParams();
  if (skip != null) params.set('skip', String(skip));
  if (limit != null) params.set('limit', String(limit));

  const query = params.toString();
  return query ? `?${query}` : '';
};

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
  // Список постраничный: тело — прежний голый массив, а общее число приходит
  // заголовком X-Total-Count (backend/app/api/endpoints/settings.py). Разбирает
  // ответ normalizeSourcesResponse из lib/sources.js — тот же разбор, что у
  // списка источников, блокнотов и заметок.
  getUsers: (options) => api.get(`/settings/users${buildUsersQuery(options)}`),
  updateUserRole: (userId, role) => api.put(`/settings/users/${userId}/role`, { role }),
};
