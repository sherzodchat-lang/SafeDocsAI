import api from './api';


export const insightsService = {
  getAll: (notebookId) => api.get(`/insights/${notebookId ? `?notebook_id=${notebookId}` : ''}`),
  create: (payload) => api.post('/insights/', payload),
};
