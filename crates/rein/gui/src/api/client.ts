import type { ConceptState } from './types';

const BASE = '';

export async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const method = (init?.method ?? 'GET').toUpperCase();
  const needsMutationMarker = !['GET', 'HEAD'].includes(method);
  const res = await fetch(`${BASE}${path}`, {
    ...init,
    credentials: 'same-origin',
    headers: {
      'Content-Type': 'application/json',
      ...(needsMutationMarker ? { 'x-rein-action': '1' } : {}),
      ...init?.headers,
    },
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `HTTP ${res.status}`);
  }
  return res.json();
}

export const apiGet = <T>(path: string, init?: RequestInit) => api<T>(path, init);
export const apiPost = <T>(path: string, body: unknown) =>
  api<T>(path, { method: 'POST', body: JSON.stringify(body) });
export const apiPut = <T>(path: string, body: unknown) =>
  api<T>(path, { method: 'PUT', body: JSON.stringify(body) });
export const apiDelete = <T>(path: string) =>
  api<T>(path, { method: 'DELETE' });

/**
 * Fetch the current state of a concept — includes the auto-refreshed
 * `living_summary` (v0.24 ARS Capability A). Backed by
 * `GET /api/concepts/{concept_id}/state`.
 */
export const getConceptState = (conceptId: string): Promise<ConceptState> =>
  apiGet<ConceptState>(`/api/concepts/${encodeURIComponent(conceptId)}/state`);
