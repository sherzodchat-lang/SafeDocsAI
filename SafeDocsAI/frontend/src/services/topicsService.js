import api from './api';


export const topicsService = {
  // Без notebook_id — распределение по всем доступным документам, с ним —
  // внутри блокнота: тот же ключ области, что у списка источников и чата.
  getDistribution: (notebookId) => api.get(`/topics${notebookId != null ? `?notebook_id=${notebookId}` : ''}`),
  // 404 с кодом topic.model_missing здесь — обычное состояние системы, а не
  // сбой: модель могли ещё не обучить. Разбирают его на экране.
  getModel: () => api.get('/topics/model'),
  // 202: переразметка идёт в фоне, ответ ничего о её результате не говорит.
  reassign: () => api.post('/topics/reassign'),
};
