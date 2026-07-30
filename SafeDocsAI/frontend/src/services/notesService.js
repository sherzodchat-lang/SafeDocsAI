import api from './api';


export const notesService = {
  getAll: (notebookId) => api.get(`/notes/${notebookId ? `?notebook_id=${notebookId}` : ''}`),
  create: (payload) => api.post('/notes/', payload),
};
