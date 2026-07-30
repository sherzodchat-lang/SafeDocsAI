import api, { CSRF_HEADER_NAME, forceLogout, getCsrfToken, refreshSession } from './api';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '/api/v1';

const parseSseEvent = (rawEvent) => {
  const lines = rawEvent.split('\n');
  let event = 'message';
  const dataLines = [];

  for (const line of lines) {
    if (line.startsWith('event:')) {
      event = line.slice(6).trim();
      continue;
    }
    if (line.startsWith('data:')) {
      dataLines.push(line.slice(5).trimStart());
    }
  }

  if (dataLines.length === 0) return null;

  try {
    return {
      event,
      data: JSON.parse(dataLines.join('\n')),
    };
  } catch {
    return null;
  }
};

const createHttpError = (response, payload) => {
  const detail = typeof payload?.detail === 'string' ? payload.detail : '';
  const error = new Error(detail || `Request failed with status ${response.status}`);
  // Payload отдаём целиком: в нём приходит error_code, по которому строится
  // локализованное сообщение.
  error.response = { status: response.status, data: payload || null };
  return error;
};

const createStreamError = (payload = {}) => {
  const detail = payload?.detail || 'Streaming request failed.';
  const status = payload?.status;
  const error = new Error(detail);
  error.response = {
    status,
    data: payload,
  };
  return error;
};

/**
 * Стрим ходит мимо axios, поэтому куки надо запросить явно (credentials), а
 * CSRF-заголовок приложить руками — стрим отправляется методом POST. Протухшая
 * сессия обрабатывается тем же порядком, что и в перехватчике: одно общее
 * обновление и один повтор. Тело запроса — строка, её можно отправить повторно
 * без пересборки.
 */
const authorizedFetch = async (url, { headers = {}, ...options }) => {
  const sendRequest = () => {
    const csrfToken = getCsrfToken();
    return fetch(url, {
      ...options,
      credentials: 'include',
      headers: {
        ...headers,
        ...(csrfToken ? { [CSRF_HEADER_NAME]: csrfToken } : {}),
      },
    });
  };

  const response = await sendRequest();
  if (response.status !== 401) return response;

  const refreshed = await refreshSession();
  if (!refreshed) {
    forceLogout();
    return response;
  }

  const retried = await sendRequest();
  if (retried.status === 401) forceLogout();
  return retried;
};

const streamMessage = async (question, notebookId, handlers = {}) => {
  const streamUrl = `${API_BASE_URL}/chat/stream`;
  const response = await authorizedFetch(streamUrl, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ question, notebook_id: notebookId ?? null }),
    signal: handlers.signal,
  });

  if (!response.ok) {
    let payload = null;
    try {
      payload = await response.json();
    } catch {
      payload = null;
    }
    throw createHttpError(response, payload);
  }

  if (!response.body) {
    throw new Error('Streaming is not supported by this browser.');
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let finalPayload = null;

  while (true) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done });

    const events = buffer.split('\n\n');
    buffer = events.pop() || '';

    for (const rawEvent of events) {
      const parsed = parseSseEvent(rawEvent.trim());
      if (!parsed) continue;

      if (parsed.event === 'token') {
        handlers.onToken?.(parsed.data?.token || '');
      } else if (parsed.event === 'done') {
        finalPayload = parsed.data;
        handlers.onDone?.(parsed.data);
      } else if (parsed.event === 'error') {
        throw createStreamError(parsed.data);
      }
    }

    if (done) break;
  }

  return finalPayload;
};

export const chatService = {
  sendMessage: (question, notebookId) => api.post('/chat/', { question, notebook_id: notebookId ?? null }),
  streamMessage,
  sendFeedback: (logId, rating) => api.post(`/logs/${logId}/rating`, { rating }),
};
