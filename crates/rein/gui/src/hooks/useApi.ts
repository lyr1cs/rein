import { useQuery } from '@tanstack/react-query';
import { apiGet } from '../api/client';
import type {
  StoreStats,
  AdaptiveStatus,
  DoctorReport,
  Memory,
  RecallMemoryOutput,
  RecallPageResponse,
  Episode,
  Artifact,
  MemoryDetailResponse,
} from '../api/types';

const DEFAULT_INTERVAL = 5000;

/**
 * Read the user's polling cadence from localStorage.
 *
 * Passed as a function to React Query's `refetchInterval` so it is evaluated
 * after each fetch — letting a Settings-page change take effect on the next
 * poll without a full page reload. Before B6 #29 this was called once at
 * module init and the interval stayed frozen until reload.
 */
function getPollingInterval(): number {
  const saved = localStorage.getItem('rein_polling_interval');
  return saved ? parseInt(saved, 10) * 1000 : DEFAULT_INTERVAL;
}

export function useStats() {
  return useQuery({
    queryKey: ['stats'],
    queryFn: () => apiGet<StoreStats>('/api/stats'),
    refetchInterval: () => getPollingInterval(),
  });
}

export function useTopics() {
  return useQuery({
    queryKey: ['topics'],
    queryFn: () => apiGet<{ topics: string[] }>('/api/topics'),
  });
}

export function useRecent(limit = 20) {
  return useQuery({
    queryKey: ['recent', limit],
    queryFn: () => apiGet<{ memories: Memory[] }>(`/api/recent?limit=${limit}`),
    refetchInterval: () => getPollingInterval(),
  });
}

export function useRecall(
  query: string,
  options?: { topic?: string; limit?: number; synthesize?: boolean },
) {
  return useQuery({
    queryKey: ['recall', query, options],
    queryFn: () => {
      const params = new URLSearchParams({ q: query });
      if (options?.topic) params.set('topic', options.topic);
      if (options?.limit) params.set('limit', String(options.limit));
      // v0.25 ARS Cap B opt-in synthesis. Server defaults to false; we only
      // forward the flag when the user toggles it on so legacy URL inspection
      // stays clean.
      if (options?.synthesize) params.set('synthesize', 'true');
      return apiGet<RecallMemoryOutput>(`/api/memories?${params}`);
    },
    enabled: query.length > 0,
  });
}

export function useRecallStream(
  query: string,
  options?: { topic?: string; keyword?: string; limit?: number; offset?: number },
) {
  return useQuery({
    queryKey: ['recall-stream', query, options],
    queryFn: () => {
      const params = new URLSearchParams({ q: query });
      if (options?.topic) params.set('topic', options.topic);
      if (options?.keyword) params.set('keyword', options.keyword);
      if (options?.limit) params.set('limit', String(options.limit));
      if (options?.offset) params.set('offset', String(options.offset));
      return apiGet<RecallPageResponse>(`/api/recall_stream?${params}`);
    },
    enabled: query.length > 0,
  });
}

export function useMemoryDetail(id: string | null) {
  return useQuery({
    queryKey: ['memory-detail', id],
    queryFn: () => apiGet<MemoryDetailResponse>(`/api/memories/${id}`),
    enabled: !!id,
  });
}

export function useAdaptive() {
  return useQuery({
    queryKey: ['adaptive'],
    queryFn: () => apiGet<AdaptiveStatus>('/api/adaptive'),
    refetchInterval: () => getPollingInterval(),
  });
}

export function useDoctor(options?: { network?: boolean; fix?: boolean }) {
  return useQuery({
    queryKey: ['doctor', options],
    queryFn: () => {
      const params = new URLSearchParams();
      if (options?.network) params.set('network', 'true');
      if (options?.fix) params.set('fix', 'true');
      const qs = params.toString();
      return apiGet<DoctorReport>(`/api/doctor${qs ? `?${qs}` : ''}`);
    },
    refetchInterval: false,
    staleTime: 1000,
  });
}

export function useActivity(days = 14) {
  return useQuery({
    queryKey: ['activity', days],
    queryFn: () => apiGet<{ activity: Array<{ date: string; recalls: number; stores: number }> }>(`/api/activity?days=${days}`),
    refetchInterval: () => getPollingInterval(),
  });
}

export function useEpisodes(limit = 20) {
  return useQuery({
    queryKey: ['episodes', limit],
    queryFn: () => apiGet<{ episodes: Episode[] }>(`/api/episodes?limit=${limit}`),
  });
}

export function useArtifacts(limit = 20) {
  return useQuery({
    queryKey: ['artifacts', limit],
    queryFn: () => apiGet<{ artifacts: Artifact[] }>(`/api/artifacts?limit=${limit}`),
  });
}

export function useServerVersion() {
  return useQuery({
    queryKey: ['server-version'],
    queryFn: () => apiGet<{ version: string }>(`/api/version`),
    staleTime: 60 * 60 * 1000, // 1 hour — version is pinned per build
    refetchInterval: false,
  });
}
