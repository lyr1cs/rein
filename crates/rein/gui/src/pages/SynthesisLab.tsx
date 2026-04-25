import { useEffect, useMemo, useState } from 'react';
import { useRecall } from '../hooks/useApi';
import SynthesisCard from '../components/SynthesisCard';
import type { RecallResult } from '../api/types';

/* ── helpers ─────────────────────────────────────────────────────── */

type TierFilter = 'all' | 'hot' | 'warm' | 'cold';

function timeAgo(iso: string): string {
  const diff = Date.now() - new Date(iso).getTime();
  const mins = Math.floor(diff / 60_000);
  if (mins < 1) return 'just now';
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs}h ago`;
  const days = Math.floor(hrs / 24);
  if (days < 30) return `${days}d ago`;
  const months = Math.floor(days / 30);
  return `${months}mo ago`;
}

function tierBadge(tier: 'hot' | 'warm' | 'cold') {
  switch (tier) {
    case 'hot':
      return { label: '\u{1F525} Hot', bg: 'bg-[var(--hot)]/20', text: 'text-[var(--hot)]' };
    case 'warm':
      return { label: 'Warm', bg: 'bg-[var(--warm)]/20', text: 'text-[var(--warm)]' };
    case 'cold':
      return { label: '❄️ Cold', bg: 'bg-[var(--cold)]/20', text: 'text-[var(--cold)]' };
  }
}

/* ── RecallCard ──────────────────────────────────────────────────── */

/**
 * Compact recall result card used in the Synthesis Lab raw column. Mirrors
 * the visual language of `MemoryCard` in Memories.tsx but trimmed to the
 * fields that matter for an A/B comparison: rank, score, tier, topic, and
 * an evidence-preview hover.
 *
 * We define this locally instead of importing MemoryCard so the lab page
 * can iterate independently — splitting out a shared card is a v0.25.2+
 * follow-up if both surfaces converge.
 */
function RecallCard({ rank, result }: { rank: number; result: RecallResult }) {
  const badge = tierBadge(result.tier);
  const isCold = result.tier === 'cold';

  return (
    <div
      className={`group relative bg-[var(--bg-secondary)] border border-[var(--border)] rounded-xl p-4 ${
        isCold ? 'opacity-60' : ''
      }`}
    >
      {/* Top row: rank, tier, time */}
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-[var(--border)] text-[var(--text-muted)]">
            #{rank}
          </span>
          <span className={`text-xs px-2 py-0.5 rounded ${badge.bg} ${badge.text}`}>
            {badge.label}
          </span>
        </div>
        <span className="text-xs text-[var(--text-muted)]">{timeAgo(result.updated_at)}</span>
      </div>

      {/* Summary */}
      <p className="text-sm text-[var(--text-primary)] leading-snug line-clamp-3 mb-3">
        {result.summary_short ?? result.summary}
      </p>

      {/* Bottom row: topic + score */}
      <div className="flex items-center justify-between gap-2 mb-2">
        <span className="text-xs px-2 py-0.5 rounded bg-[var(--accent)]/15 text-[var(--accent)] truncate max-w-[60%]">
          {result.topic}
        </span>
        <span
          className="text-[10px] font-mono text-[var(--new)]"
          title="Final fused score"
        >
          score {result.score.toFixed(3)}
        </span>
      </div>

      {/* Evidence preview (always visible in lab — this is the comparison
          ground truth, not a hover-easter-egg) */}
      {result.evidence_preview?.length ? (
        <div className="mt-3 rounded-md border border-[var(--border)] bg-[var(--bg-primary)]/40 p-2">
          <div className="mb-1 text-[10px] uppercase tracking-wider text-[var(--accent)]">
            Evidence Preview
          </div>
          <div className="space-y-1 text-xs text-[var(--text-secondary)]">
            {result.evidence_preview.slice(0, 3).map((line, idx) => (
              <div key={`${result.id}-ev-${idx}`} className="line-clamp-2 break-words">
                {line}
              </div>
            ))}
          </div>
        </div>
      ) : null}
    </div>
  );
}

/* ── SynthesisLab page ───────────────────────────────────────────── */

export default function SynthesisLab() {
  const [query, setQuery] = useState('');
  const [debouncedQuery, setDebouncedQuery] = useState('');
  const [limit, setLimit] = useState(10);
  const [tierFilter, setTierFilter] = useState<TierFilter>('all');

  // Debounce so each keystroke doesn't hammer the LLM. 400ms feels right for
  // a "Run on pause" lab — tighter than Memories.tsx because the page is
  // explicitly opt-in.
  useEffect(() => {
    const timer = setTimeout(() => setDebouncedQuery(query.trim()), 400);
    return () => clearTimeout(timer);
  }, [query]);

  const { data, isLoading, isFetching, error } = useRecall(debouncedQuery, {
    limit,
    synthesize: true,
  });

  const filteredResults = useMemo(() => {
    const results = data?.results;
    if (!results) return [];
    if (tierFilter === 'all') return results;
    return results.filter((r) => r.tier === tierFilter);
  }, [data, tierFilter]);

  const isInitialState = debouncedQuery.length === 0;

  return (
    <div className="flex h-full flex-col overflow-hidden">
      {/* Banner */}
      <div className="px-6 pt-5 pb-4 border-b border-[var(--border)]">
        <div className="flex items-baseline gap-3 mb-1">
          <h1 className="text-lg font-semibold text-[var(--text-primary)]">
            {'\u{1F9EA}'} Synthesis Lab
          </h1>
          <span className="text-xs text-[var(--text-muted)]">ARS Capability B</span>
        </div>
        <p className="text-xs text-[var(--text-muted)] leading-relaxed max-w-3xl">
          Compare raw recall against narrative synthesis. The right column shows
          the LLM's combined answer over the top results; the left shows the
          underlying memories with evidence previews. Toggle the feature in
          <code className="mx-1 px-1 py-0.5 rounded bg-[var(--bg-secondary)] text-[var(--text-secondary)] font-mono">
            [ars].recall_synthesis_enabled
          </code>
          .
        </p>
      </div>

      {/* Toolbar */}
      <div className="px-6 py-4 border-b border-[var(--border)] flex items-center gap-3 flex-wrap">
        <div className="relative flex-1 min-w-[280px]">
          <span className="absolute left-3 top-1/2 -translate-y-1/2 text-[var(--text-muted)] text-sm pointer-events-none">
            {'\u{1F50D}'}
          </span>
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Try: 'what did we decide about resummerize fuses?'"
            className="w-full bg-[var(--bg-secondary)] border border-[var(--border)] rounded-lg pl-9 pr-4 py-2 text-sm text-[var(--text-primary)] placeholder-[var(--text-muted)] focus:outline-none focus:border-[var(--accent)] transition-colors"
          />
        </div>

        <label className="flex items-center gap-2 text-xs text-[var(--text-muted)]">
          Limit
          <input
            type="number"
            min={1}
            max={50}
            value={limit}
            onChange={(e) => {
              const v = parseInt(e.target.value, 10);
              if (!Number.isNaN(v) && v > 0 && v <= 50) setLimit(v);
            }}
            className="w-16 bg-[var(--bg-secondary)] border border-[var(--border)] rounded-lg px-2 py-1.5 text-xs text-[var(--text-primary)] focus:outline-none focus:border-[var(--accent)] transition-colors font-mono"
          />
        </label>

        <div className="flex rounded-lg overflow-hidden border border-[var(--border)]">
          {(['all', 'hot', 'warm', 'cold'] as TierFilter[]).map((t) => (
            <button
              key={t}
              type="button"
              onClick={() => setTierFilter(t)}
              className={`px-3 py-1.5 text-xs capitalize transition-colors ${
                tierFilter === t
                  ? t === 'hot'
                    ? 'bg-[var(--hot)]/20 text-[var(--hot)]'
                    : t === 'warm'
                      ? 'bg-[var(--warm)]/20 text-[var(--warm)]'
                      : t === 'cold'
                        ? 'bg-[var(--cold)]/20 text-[var(--cold)]'
                        : 'bg-[var(--accent)]/20 text-[var(--accent)]'
                  : 'text-[var(--text-muted)] hover:bg-[var(--bg-secondary)]'
              }`}
            >
              {t}
            </button>
          ))}
        </div>

        <button
          type="button"
          onClick={() => setDebouncedQuery(query.trim())}
          disabled={query.trim().length === 0}
          className="ml-auto px-3 py-1.5 text-xs rounded-lg bg-[var(--accent)]/20 text-[var(--accent)] hover:bg-[var(--accent)]/30 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
          title="Force-run the query immediately (debounce auto-runs after 400ms pause)"
        >
          Run
        </button>

        {isFetching && !isLoading && (
          <span className="text-[10px] text-[var(--text-muted)] font-mono animate-pulse">
            refreshing…
          </span>
        )}
      </div>

      {/* Body — split-pane */}
      <div className="flex-1 overflow-hidden flex flex-col md:flex-row">
        {/* LEFT: raw recall */}
        <section className="flex-1 min-w-0 overflow-y-auto px-6 py-5 md:border-r border-b md:border-b-0 border-[var(--border)]">
          <div className="flex items-baseline justify-between mb-3">
            <h2 className="text-xs uppercase tracking-wider text-[var(--text-muted)]">
              Raw Recall
            </h2>
            <span className="text-[10px] text-[var(--text-muted)] font-mono">
              {isInitialState
                ? '—'
                : isLoading
                  ? 'loading…'
                  : `${filteredResults.length}/${data?.results.length ?? 0}`}
            </span>
          </div>

          {isInitialState ? (
            <div className="flex flex-col items-center justify-center py-20 text-[var(--text-muted)]">
              <div className="text-3xl mb-3">{'\u{1F50D}'}</div>
              <div className="text-sm">Enter a query to begin</div>
            </div>
          ) : isLoading ? (
            <div className="space-y-3">
              {Array.from({ length: 3 }).map((_, i) => (
                <div
                  key={i}
                  className="h-32 rounded-xl bg-[var(--bg-secondary)] border border-[var(--border)] animate-pulse"
                />
              ))}
            </div>
          ) : error ? (
            <div className="rounded-lg border border-[var(--hot)]/30 bg-[var(--hot)]/10 px-4 py-3 text-sm text-[var(--hot)]">
              Recall failed: {error instanceof Error ? error.message : 'unknown error'}
            </div>
          ) : filteredResults.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-20 text-[var(--text-muted)]">
              <div className="text-sm">No results match the current filter</div>
            </div>
          ) : (
            <div className="space-y-3">
              {filteredResults.map((r, idx) => (
                <RecallCard key={r.id} rank={idx + 1} result={r} />
              ))}
            </div>
          )}
        </section>

        {/* RIGHT: synthesis */}
        <section className="flex-1 min-w-0 overflow-y-auto px-6 py-5 bg-[var(--bg-primary)]/40">
          <div className="flex items-baseline justify-between mb-3">
            <h2 className="text-xs uppercase tracking-wider text-[var(--text-muted)]">
              AI Synthesis
            </h2>
            {data?.route && (
              <span
                className="text-[10px] text-[var(--text-muted)] font-mono"
                title="Recall route classifier verdict"
              >
                route: {data.route}
              </span>
            )}
          </div>

          {isInitialState ? (
            <div className="flex flex-col items-center justify-center py-20 text-[var(--text-muted)]">
              <div className="text-3xl mb-3">{'\u{1F9EA}'}</div>
              <div className="text-sm">Synthesis will appear here</div>
            </div>
          ) : isLoading ? (
            <div className="space-y-3">
              <div className="h-48 rounded-lg bg-[var(--bg-secondary)] border border-[var(--border)] animate-pulse" />
              <div className="h-24 rounded-lg bg-[var(--bg-secondary)] border border-[var(--border)] animate-pulse" />
            </div>
          ) : (
            <>
              <SynthesisCard outcome={data?.synthesis} />

              {/* Sources used — numbered attribution that maps back to the
                  Raw Recall ranks on the left. We deliberately don't try to
                  highlight inline citations in the prose; that's a v0.25.2+
                  enhancement once the backend emits structured spans. */}
              {data?.results && data.results.length > 0 && (
                <div className="mt-2 rounded-lg border border-[var(--border)] bg-[var(--bg-primary)]/60 p-3">
                  <div className="text-[10px] text-[var(--text-muted)] uppercase tracking-wider mb-2">
                    Sources
                  </div>
                  <ol className="space-y-1.5">
                    {data.results.map((r, idx) => (
                      <li
                        key={`src-${r.id}`}
                        className="flex items-start gap-2 text-xs text-[var(--text-secondary)]"
                      >
                        <span className="font-mono text-[var(--text-muted)] shrink-0">
                          #{idx + 1}
                        </span>
                        <span className="flex-1 min-w-0">
                          <span className="text-[var(--text-primary)]">{r.topic}</span>
                          <span className="text-[var(--text-muted)]"> — </span>
                          <span className="line-clamp-1">
                            {r.summary_short ?? r.summary}
                          </span>
                        </span>
                        <span className="font-mono text-[10px] text-[var(--text-muted)] shrink-0">
                          {r.id.slice(0, 8)}
                        </span>
                      </li>
                    ))}
                  </ol>
                </div>
              )}

              {/* Empty placeholder when there's nothing — neither prose nor
                  skip — to explain the silence. SynthesisCard renders null in
                  that branch, so the user would otherwise see a blank pane. */}
              {!data?.synthesis && !isLoading && debouncedQuery.length > 0 && (
                <div className="rounded-lg border border-[var(--border)] bg-[var(--bg-primary)]/60 p-3 text-xs text-[var(--text-muted)] italic">
                  No synthesis returned for this query.
                </div>
              )}
            </>
          )}
        </section>
      </div>
    </div>
  );
}
