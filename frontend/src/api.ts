import type { ChatResponse, TraceEvent } from './types';

export const base = (import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000').replace(/\/$/, '');
const token = (import.meta.env.VITE_API_TOKEN || localStorage.getItem('analyst_api_token') || '').trim();
const headers: Record<string, string> = {
  'Content-Type': 'application/json',
  ...(token ? { 'X-API-Token': token } : {}),
};

export type HealthResponse = {
  status: string;
  database: string;
  models: Record<string, string>;
};

async function throwApiError(res: Response): Promise<never> {
  let msg = `Request failed (${res.status})`;
  let code = 'REQUEST_FAILED';
  try {
    const data = await res.json();
    msg = data.message || (typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail)) || msg;
    code = data.code || code;
  } catch {
    // non-json error response
  }
  throw Object.assign(new Error(msg), { code });
}

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${base}${path}`, { ...options, headers: { ...headers, ...options?.headers } });
  if (!res.ok) await throwApiError(res);
  return res.json();
}

export const api = {
  health: () => request<HealthResponse>('/health'),
  session: () => request<{ session_id: string }>('/api/v1/sessions', { method: 'POST' }),
  ask: (question: string, session_id: string | null) =>
    request<ChatResponse>('/api/v1/questions', { method: 'POST', body: JSON.stringify({ question, session_id }) }),
  askStream: async (question: string, session_id: string | null, onProgress: (evt: TraceEvent) => void): Promise<ChatResponse> => {
    const res = await fetch(`${base}/api/v1/questions/stream`, {
      method: 'POST',
      headers,
      body: JSON.stringify({ question, session_id }),
    });
    if (!res.ok) {
      if (res.status === 404 || res.status === 405) return api.ask(question, session_id);
      await throwApiError(res);
    }
    if (!res.body) return api.ask(question, session_id);

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let finalResponse: ChatResponse | null = null;

    const processLines = (text: string) => {
      for (const rawLine of text.split(/\r?\n/)) {
        const line = rawLine.trim();
        if (!line.startsWith('data:')) continue;
        const jsonStr = line.slice(5).trim();
        if (!jsonStr) continue;
        const payload = JSON.parse(jsonStr);
        if (payload.type === 'progress' && payload.event) {
          const e = payload.event;
          onProgress({
            stage: String(e.stage || e.agent || 'working'),
            message: String(e.message || e.action || 'Processing...'),
            status: e.status === 'error' ? 'error' : 'ok',
            timestamp: Number(e.timestamp) || Date.now() / 1000,
            task_id: e.task_id,
          });
        } else if (payload.type === 'complete' && payload.data) {
          finalResponse = payload.data as ChatResponse;
        } else if (payload.type === 'error') {
          throw Object.assign(new Error(payload.message || 'Streaming analysis failed'), { code: payload.code || 'STREAM_FAILED' });
        }
      }
    };

    while (true) {
      const { value, done } = await reader.read();
      if (done) {
        buffer += decoder.decode();
        if (buffer.trim()) processLines(buffer);
        break;
      }
      buffer += decoder.decode(value, { stream: true });
      const parts = buffer.split(/\r?\n\r?\n/);
      buffer = parts.pop() || '';
      parts.forEach(processLines);
    }

    if (!finalResponse) {
      throw Object.assign(new Error('Streaming connection closed before completion.'), { code: 'STREAM_INTERRUPTED' });
    }
    return finalResponse;
  },
  run: (id: string) => request<ChatResponse>(`/api/v1/runs/${encodeURIComponent(id)}`),
  history: (session: string) => request<(ChatResponse & { question?: string })[]>(`/api/v1/sessions/${encodeURIComponent(session)}/runs`),
  reportHtmlUrl: (runId: string) => `${base}/api/v1/runs/${encodeURIComponent(runId)}/report.html`,
  reportPdfUrl: (runId: string) => `${base}/api/v1/runs/${encodeURIComponent(runId)}/report.pdf`,
};
