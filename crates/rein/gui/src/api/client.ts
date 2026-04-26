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
export const getConceptState = (
  conceptId: string,
  opts?: { queryType?: string; clusterId?: number },
): Promise<ConceptState> => {
  // v0.27 R6 P2 fix: thread optional bucket context so the Cap A adaptive
  // gate can suppress low-usefulness summaries on first-fetch. Backend
  // uses the computed `representative_cluster_id` as a fallback when
  // `clusterId` is omitted.
  const params = new URLSearchParams();
  if (opts?.queryType) params.set('query_type', opts.queryType);
  if (typeof opts?.clusterId === 'number') params.set('cluster_id', String(opts.clusterId));
  const qs = params.toString();
  const path = `/api/concepts/${encodeURIComponent(conceptId)}/state${qs ? `?${qs}` : ''}`;
  return apiGet<ConceptState>(path);
};
