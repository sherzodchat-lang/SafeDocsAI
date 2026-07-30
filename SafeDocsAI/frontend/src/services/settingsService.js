import api from './api';


export const settingsService = {
  get: () => api.get('/settings/'),
  update: (payload) => api.put('/settings/', payload),
  getUsers: () => api.get('/settings/users'),
  updateUserRole: (userId, role) => api.put(`/settings/users/${userId}/role`, { role }),
};
