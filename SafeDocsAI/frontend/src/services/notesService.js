import api from './api';


// Пустые параметры не отправляем: бэкенд отличает «фильтр не задан» от значения,
// и ?status= с пустой строкой он отверг бы как недопустимый статус. NaN
// отбрасываем отдельно: в localStorage от прошлых версий лежит что угодно, а
// ?notebook_id=NaN бэкенд встретил бы ошибкой валидации вместо списка.
const buildQuery = (params) => {
  const search = new URLSearchParams();
  Object.entries(params).forEach(([key, value]) => {
    if (value === undefined || value === null || value === '') return;
    if (typeof value === 'number' && !Number.isFinite(value)) return;
    search.append(key, String(value));
  });
  const query = search.toString();
  return query ? `?${query}` : '';
};


export const notesService = {
  // status — необязательный фильтр (active | archived). Без него приходят
  // заметки любого статуса, поэтому архивация не выкидывает их из панели.
  getAll: (notebookId, status) => api.get(`/notes/${buildQuery({ notebook_id: notebookId, status })}`),
  create: (payload) => api.post('/notes/', payload),
  // Частичное обновление: null бэкенд не принимает ни для одного поля, поэтому
  // в тело кладём только действительно изменённые значения.
  update: (id, payload) => api.patch(`/notes/${id}`, payload),
  delete: (id) => api.delete(`/notes/${id}`),
};
