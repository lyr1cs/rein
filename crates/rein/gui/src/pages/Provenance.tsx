import { useQuery } from '@tanstack/react-query';
import { useState } from 'react';
import { apiGet } from '../api/client';
import type { DedupResponse, MergeMetrics } from '../api/types';

function relationClasses(relation: string): string {
  switch (relation) {
    case 'duplicate':
      return 'bg-[var(--bg-secondary)] text-[var(--text-muted)] border-[var(--text-muted)]';
    case 'update':
      return 'bg-[var(--accent)]/15 text-[var(--accent)] border-[var(--accent)]/40';
    case 'related':
      return 'bg-[var(--warm)]/15 text-[var(--warm)] border-[var(--warm)]/40';
    case 'distinct':
      return 'bg-[var(--success)]/15 text-[var(--success)] border-[var(--success)]/40';
    default:
      return 'bg-[var(--bg-secondary)] text-[var(--text-muted)] border-[var(--text-muted)]';
  }
}

export default function Provenance() {
  const [operatorFilter, setOperatorFilter] = useState<'all' | 'llm_verdict' | 'auto'>('all');
  const [limit] = useState(100);

  const { data, isLoading, error, refetch } = useQuery<DedupResponse>({
    queryKey: ['dedup-decisions', operatorFilter, limit],
    queryFn: () => {
      const params = new URLSearchParams();
      if (operatorFilter !== 'all') params.set('operator', operatorFilter);
      params.set('limit', String(limit));
      return apiGet<DedupResponse>(`/api/dedup_decisions?${params}`);
    },
    refetchInterval: 15000,
  });

  const { data: metrics } = useQuery<MergeMetrics>({
    queryKey: ['intelligent-merge-metrics'],
    queryFn: () => apiGet<MergeMetrics>('/api/intelligent_merge_metrics'),
    refetchInterval: 10000,
  });

  const decisions = data?.decisions ?? [];

  return (
    <div className="flex flex-col h-full overflow-hidden">
      <div className="px-6 pt-5 pb-4 border-b border-[var(--border)]">
        <h1 className="text-2xl font-bold text-[var(--text-primary)]">Provenance</h1>
        <p className="text-sm text-[var(--text-muted)] mt-1">
          Dedup decisions & intelligent-merge verdicts explaining how canonicals were formed.
        </p>

        <div className="flex items-center gap-4 mt-4 flex-wrap">
          <div className="flex items-center gap-2 text-sm">
            <label className="text-[var(--text-muted)]">Operator:</label>
            <select
              value={operatorFilter}
              onChange={(e) => setOperatorFilter(e.target.value as typeof operatorFilter)}
              className="bg-[var(--bg-secondary)] border border-[var(--border)] rounded px-2 py-1"
            >
              <option value="all">All</option>
              <option value="llm_verdict">LLM verdict</option>
              <option value="auto">Auto (mechanical)</option>
            </select>
          </div>

          <button
            onClick={() => refetch()}
            className="text-sm px-3 py-1 rounded bg-[var(--bg-secondary)] border border-[var(--border)] hover:bg-[var(--bg-tertiary)]"
          >
            Refresh
          </button>

          {metrics && (
            <div className="ml-auto flex gap-4 text-sm text-[var(--text-muted)]">
              <span>
                <span className="text-[var(--text-primary)] font-semibold">{metrics.attempted}</span> attempted
              </span>
              <span>
                <span className="text-[var(--success)] font-semibold">{metrics.success}</span> ok
              </span>
              <span>
                <span className="text-[var(--warm)] font-semibold">{metrics.stale_races}</span> stale
              </span>
              <span>
                <span className="text-[var(--error)] font-semibold">
                  {metrics.parse_errors + metrics.http_errors}
                </span>{' '}
                err
              </span>
            </div>
          )}
        </div>
      </div>

      <div className="flex-1 overflow-auto px-6 py-4">
        {isLoading && <div className="text-[var(--text-muted)]">Loading decisions...</div>}
        {error != null && (
          <div className="text-[var(--error)]">Failed to load: {String((error as Error).message)}</div>
        )}
        {!isLoading && decisions.length === 0 && (
          <div className="text-[var(--text-muted)] italic">
            No dedup decisions recorded yet. They accumulate as memories are stored and merged.
          </div>
        )}

        <div className="space-y-2">
          {decisions.map((d) => (
            <div
              key={d.id}
              className="border border-[var(--border)] rounded p-3 bg-[var(--bg-secondary)] hover:bg-[var(--bg-tertiary)] transition-colors"
            >
              <div className="flex items-center gap-3 text-sm flex-wrap">
                <span className={`px-2 py-0.5 rounded border text-xs uppercase ${relationClasses(d.relation)}`}>
                  {d.relation}
                </span>
                <span className="text-[var(--text-muted)] text-xs font-mono">{d.operator}</span>
                <span className="text-[var(--text-muted)] text-xs">
                  conf {(d.confidence * 100).toFixed(0)}%
                </span>
                {d.embedding_score != null && (
                  <span className="text-[var(--text-muted)] text-xs">
                    sim {d.embedding_score.toFixed(2)}
                  </span>
                )}
                {d.conflict_detected && (
                  <span className="text-[var(--error)] text-xs">⚠ conflict</span>
                )}
                <span className="text-[var(--text-muted)] text-xs ml-auto">
                  {new Date(d.created_at).toLocaleString()}
                </span>
              </div>

              {d.reason && (
                <div className="mt-2 text-sm text-[var(--text-primary)]">{d.reason}</div>
              )}

              <div className="mt-2 flex gap-4 text-xs text-[var(--text-muted)] font-mono flex-wrap">
                {d.winner_id && <span>winner: {d.winner_id.slice(0, 12)}…</span>}
                {d.loser_id && <span>loser: {d.loser_id.slice(0, 12)}…</span>}
                {d.canonical_id && <span>canonical: {d.canonical_id.slice(0, 12)}…</span>}
              </div>

              {d.merged_summary && (
                <details className="mt-2 text-xs">
                  <summary className="cursor-pointer text-[var(--text-muted)] hover:text-[var(--text-primary)]">
                    Synthesized content
                  </summary>
                  <pre className="mt-1 whitespace-pre-wrap text-[var(--text-primary)] bg-[var(--bg-primary)] p-2 rounded">
                    {d.merged_summary}
                  </pre>
                </details>
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
