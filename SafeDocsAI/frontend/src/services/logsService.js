import api from './api';


export const logsService = {
  getAll: ({ skip = 0, limit = 100, startDate, endDate } = {}) => {
    const params = new URLSearchParams();
    params.set('skip', skip);
    params.set('limit', limit);
    if (startDate) params.set('start_date', startDate);
    if (endDate) params.set('end_date', endDate);
    return api.get(`/logs/?${params.toString()}`);
  },
  getAnalytics: () => api.get('/analytics/'),
  exportCsv: ({ startDate, endDate } = {}) => {
    const params = new URLSearchParams();
    if (startDate) params.set('start_date', startDate);
    if (endDate) params.set('end_date', endDate);
    const queryString = params.toString();
    return api.get(`/logs/export${queryString ? `?${queryString}` : ''}`, {
      responseType: 'blob',
    });
  },
};
