import api from './api';


export const notebooksService = {
  getAll: () => api.get('/notebooks/'),
  getById: (id) => api.get(`/notebooks/${id}`),
  create: (payload) => api.post('/notebooks/', payload),
  delete: (id) => api.delete(`/notebooks/${id}`),
};
