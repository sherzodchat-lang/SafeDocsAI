import api from './api';


export const askService = {
  ask: (question, notebookId, topK) => api.post('/ask/', {
    question,
    notebook_id: notebookId ?? null,
    ...(topK ? { top_k: topK } : {}),
  }),
};
