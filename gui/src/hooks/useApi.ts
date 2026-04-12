import { useQuery } from '@tanstack/react-query';
import { apiGet } from '../api/client';
import type {
  StoreStats,
  AdaptiveStatus,
  DoctorReport,
  Memory,
  RecallResult,
  Episode,
  Artifact,
  MemoryDetailResponse,
} from '../api/types';

const DEFAULT_INTERVAL = 5000;

function getPollingInterval(): number {
  const saved = localStorage.getItem('rein_polling_interval');
  return saved ? parseInt(saved, 10) * 1000 : DEFAULT_INTERVAL;
}

export function useStats() {
  return useQuery({
    queryKey: ['stats'],
    queryFn: () => apiGet<StoreStats>('/api/stats'),
    refetchInterval: getPollingInterval(),
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
    refetchInterval: getPollingInterval(),
  });
}

export function useRecall(query: string, options?: { topic?: string; limit?: number }) {
  return useQuery({
    queryKey: ['recall', query, options],
    queryFn: () => {
      const params = new URLSearchParams({ q: query });
      if (options?.topic) params.set('topic', options.topic);
      if (options?.limit) params.set('limit', String(options.limit));
      return apiGet<{ results: RecallResult[]; count: number }>(`/api/memories?${params}`);
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
    refetchInterval: getPollingInterval(),
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
    refetchInterval: getPollingInterval(),
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
