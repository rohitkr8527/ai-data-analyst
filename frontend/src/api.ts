import type { ChatResponse, TraceEvent } from './types';

export const base = (import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000').replace(/\/$/, '');
const token = (import.meta.env.VITE_API_TOKEN || localStorage.getItem('analyst_api_token') || '').trim();

export type HealthResponse = {
  status: string;
  database: string;
  models: {
    router: string;
    sql_analyst: string;
    planner: string;
    rca: string;
    hypothesis?: string;
    evidence_reviewer?: string;
    insights: string;
    report_writer: string;
  };
};

export type ApiError = {
  code: string;
  message: string;
};

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const result = await fetch(`${base}${path}`, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...(token ? { 'X-API-Token': token } : {}),
      ...options?.headers,
    },
  });

  if (!result.ok) {
    let errorMsg = `Request failed (${result.status})`;
    let errorCode = 'REQUEST_FAILED';
    try {
      const data = await result.json();
      if (data.message) errorMsg = data.message;
      else if (data.detail) errorMsg = typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail);
      if (data.code) errorCode = data.code;
    } catch {
      // Fallback to text if not json
    }
    const err = new Error(errorMsg);
    (err as Error & { code?: string }).code = errorCode;
    throw err;
  }
  return result.json();
}

export const api = {
  health: () => request<HealthResponse>('/health'),
  session: () => request<{ session_id: string }>('/api/v1/sessions', { method: 'POST' }),
  ask: (question: string, session_id: string | null) =>
    request<ChatResponse>('/api/v1/questions', {
      method: 'POST',
      body: JSON.stringify({ question, session_id }),
    }),
  askStream: async (
    question: string,
    session_id: string | null,
    onProgress: (evt: TraceEvent) => void
  ): Promise<ChatResponse> => {
    const res = await fetch(`${base}/api/v1/questions/stream`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...(token ? { 'X-API-Token': token } : {}),
      },
      body: JSON.stringify({ question, session_id }),
    });
    if (!res.ok || !res.body) {
      return api.ask(question, session_id);
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let finalResponse: ChatResponse | null = null;

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const parts = buffer.split('\n\n');
      buffer = parts.pop() || '';
      for (const chunk of parts) {
        const line = chunk.trim();
        if (!line.startsWith('data:')) continue;
        const jsonStr = line.slice(5).trim();
        if (!jsonStr) continue;
        const payload = JSON.parse(jsonStr);
        if (payload.type === 'progress' && payload.event) {
          const raw = payload.event;
          onProgress({
            stage: String(raw.stage || raw.agent || raw.event || 'working'),
            message: String(raw.message || raw.action || raw.event || 'Processing step...'),
            status: raw.status === 'error' ? 'error' : 'ok',
            timestamp: Number(raw.timestamp) || Date.now() / 1000,
            task_id: raw.task_id,
          });
        } else if (payload.type === 'complete' && payload.data) {
          finalResponse = payload.data as ChatResponse;
        } else if (payload.type === 'error') {
          const err = new Error(payload.message || 'Streaming analysis failed');
          (err as Error & { code?: string }).code = payload.code || 'STREAM_FAILED';
          throw err;
        }
      }
    }

    if (!finalResponse) {
      return api.ask(question, session_id);
    }
    return finalResponse;
  },
  run: (id: string) => request<ChatResponse>(`/api/v1/runs/${encodeURIComponent(id)}`),
  history: (session: string) =>
    request<(ChatResponse & { question?: string })[]>(
      `/api/v1/sessions/${encodeURIComponent(session)}/runs`
    ),
  reportHtmlUrl: (runId: string) => `${base}/api/v1/runs/${encodeURIComponent(runId)}/report.html`,
  reportPdfUrl: (runId: string) => `${base}/api/v1/runs/${encodeURIComponent(runId)}/report.pdf`,
};
