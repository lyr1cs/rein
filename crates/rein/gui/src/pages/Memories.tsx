import { useState, useEffect, useCallback, useMemo } from 'react';
import { useQueryClient } from '@tanstack/react-query';
import { useMemoryDetail, useRecent, useRecall, useTopics } from '../hooks/useApi';
import { apiDelete } from '../api/client';
import SynthesisCard from '../components/SynthesisCard';
import { timeAgo } from '../utils/time';
import type {
  Memory,
  MemoryDetailResponse,
  RecallResult,
} from '../api/types';

/* ── helpers ─────────────────────────────────────────────────────── */

function tierBadge(tier: 'hot' | 'warm' | 'cold') {
  switch (tier) {
    case 'hot':
      return { label: '\u{1F525} Hot', bg: 'bg-[var(--hot)]/20', text: 'text-[var(--hot)]' };
    case 'warm':
      return { label: 'Warm', bg: 'bg-[var(--warm)]/20', text: 'text-[var(--warm)]' };
    case 'cold':
      return { label: '\u2744\uFE0F Cold', bg: 'bg-[var(--cold)]/20', text: 'text-[var(--cold)]' };
  }
}

function strengthColor(strength: number): string {
  if (strength >= 0.7) return 'var(--hot)';
  if (strength >= 0.4) return 'var(--warm)';
  return 'var(--cold)';
}

type TierFilter = 'all' | 'hot' | 'warm' | 'cold';
type SortMode = 'recent' | 'support' | 'strength' | 'relevance';

/* ── MemoryCard ──────────────────────────────────────────────────── */

function MemoryCard({ memory, onClick }: { memory: Memory | RecallResult; onClick: () => void }) {
  const badge = tierBadge(memory.tier);
  const isCold = memory.tier === 'cold';
  const recallMeta = 'score' in memory ? memory as RecallResult : null;
  const evidenceCount = recallMeta?.evidence_count ?? Math.max(memory.support_count - 1, 0);

  return (
    <button
      onClick={onClick}
      className={`group relative bg-[var(--bg-secondary)] border border-[var(--border)] rounded-xl p-4 text-left transition-all hover:border-[var(--accent)]/50 hover:shadow-[0_0_16px_var(--accent)/10] cursor-pointer w-full ${
        isCold ? 'opacity-60' : ''
      }`}
    >
      {/* Top row: tier + time */}
      <div className="flex items-center justify-between mb-2">
        <span className={`text-xs px-2 py-0.5 rounded ${badge.bg} ${badge.text}`}>
          {badge.label}
        </span>
        <span className="text-xs text-[var(--text-muted)]">{timeAgo(memory.updated_at)}</span>
      </div>

      {/* Summary (2-line clamp) */}
      <p className="text-sm text-[var(--text-primary)] leading-snug line-clamp-2 mb-3">
        {memory.summary_short ?? memory.summary}
      </p>

      {/* Bottom row: topic + importance */}
      <div className="flex items-center justify-between gap-2 mb-3">
        <span className="text-xs px-2 py-0.5 rounded bg-[var(--accent)]/15 text-[var(--accent)] truncate max-w-[60%]">
          {memory.topic}
        </span>
        <span className="text-xs text-[var(--text-muted)] capitalize shrink-0">
          {memory.importance}
        </span>
      </div>

      <div className="flex items-center gap-2 mb-3 text-[10px] text-[var(--text-muted)]">
        <span className="rounded bg-[var(--border)] px-2 py-0.5 font-mono">
          sup {memory.support_count}
        </span>
        {evidenceCount > 0 && (
          <span className="rounded bg-[var(--accent)]/12 px-2 py-0.5 text-[var(--accent)]">
            ev {evidenceCount}
          </span>
        )}
        {recallMeta && (
          <span className="rounded bg-[var(--success)]/12 px-2 py-0.5 text-[var(--success)]">
            {(recallMeta.confidence * 100).toFixed(0)}%
          </span>
        )}
      </div>

      {/* Strength bar */}
      <div className="w-full h-1.5 rounded-full bg-[var(--border)] overflow-hidden">
        <div
          className="h-full rounded-full transition-all"
          style={{
            width: `${Math.min(memory.strength * 100, 100)}%`,
            backgroundColor: strengthColor(memory.strength),
          }}
        />
      </div>

      {recallMeta?.evidence_preview?.length ? (
        <div className="pointer-events-none absolute left-3 right-3 top-full z-20 mt-2 hidden rounded-lg border border-[var(--border)] bg-[#0b1220]/95 p-3 text-xs text-[var(--text-secondary)] shadow-2xl group-hover:block">
          <div className="mb-1 text-[10px] uppercase tracking-wider text-[var(--accent)]">
            Evidence Preview
          </div>
          <div className="space-y-1">
            {recallMeta.evidence_preview.map((line) => (
              <div key={line} className="line-clamp-2 break-words">
                {line}
              </div>
            ))}
          </div>
        </div>
      ) : null}
    </button>
  );
}

/* ── DetailPanel ─────────────────────────────────────────────────── */

function DetailPanel({
  memory,
  detail,
  error,
  loading,
  onClose,
  onDelete,
}: {
  memory: Memory | RecallResult;
  detail: MemoryDetailResponse | null | undefined;
  error: string | null;
  loading: boolean;
  onClose: () => void;
  onDelete: (id: string) => void;
}) {
  const [confirmDelete, setConfirmDelete] = useState(false);
  const [showAllEvidence, setShowAllEvidence] = useState(false);
  const display = detail?.memory ?? memory;
  const badge = tierBadge(display.tier);
  const recallMeta = 'score' in memory ? memory as RecallResult : null;
  const evidence = detail?.evidence ?? [];
  // `evidence_total` is the un-truncated row count from the server;
  // `evidence` itself is preview-capped at 200. Falling back to the
  // preview length when the server didn't report a total keeps older
  // backends honest (the number we show always matches what we render).
  const evidenceTotal = detail?.evidence_total ?? evidence.length;
  const visibleEvidence = showAllEvidence ? evidence : evidence.slice(0, 3);

  return (
    <div className="fixed inset-y-0 right-0 w-[280px] bg-[var(--bg-secondary)] border-l border-[var(--border)] shadow-2xl z-50 flex flex-col animate-slide-in overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-[var(--border)]">
        <span className={`text-xs px-2 py-0.5 rounded ${badge.bg} ${badge.text}`}>
          {badge.label}
        </span>
        <button
          onClick={onClose}
          aria-label="Close detail"
          className="w-6 h-6 flex items-center justify-center rounded hover:bg-[var(--border)] text-[var(--text-muted)] transition-colors"
        >
          {'\u2715'}
        </button>
      </div>

      {/* Body (scrollable) */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {/* Summary */}
        <div>
          <div className="text-xs text-[var(--text-muted)] uppercase tracking-wider mb-1">Summary</div>
          <p className="text-sm text-[var(--text-primary)] leading-relaxed">{display.summary}</p>
        </div>

        {/* Full content */}
        <div>
          <div className="text-xs text-[var(--text-muted)] uppercase tracking-wider mb-1">Content</div>
          <p className="text-sm text-[var(--text-secondary)] leading-relaxed whitespace-pre-wrap break-words">
            {display.content}
          </p>
        </div>

        {/* Recall score */}
        {recallMeta && (
          <div className="grid grid-cols-3 gap-2">
            <div>
              <div className="text-xs text-[var(--text-muted)]">Score</div>
              <div className="text-sm font-mono text-[var(--new)]">{recallMeta.score.toFixed(3)}</div>
            </div>
            <div>
              <div className="text-xs text-[var(--text-muted)]">Confidence</div>
              <div className="text-sm font-mono text-[var(--success)]">{(recallMeta.confidence * 100).toFixed(0)}%</div>
            </div>
            <div>
              <div className="text-xs text-[var(--text-muted)]">Sources</div>
              <div className="text-sm font-mono text-[var(--warm)]">{recallMeta.sources_hit}/3</div>
            </div>
          </div>
        )}

        {/* Metadata grid */}
        <div className="grid grid-cols-2 gap-x-3 gap-y-2 text-xs">
          <div>
            <span className="text-[var(--text-muted)]">Layer</span>
            <div className="text-[var(--text-primary)]">{display.layer}</div>
          </div>
          <div>
            <span className="text-[var(--text-muted)]">Importance</span>
            <div className="text-[var(--text-primary)] capitalize">{display.importance}</div>
          </div>
          <div>
            <span className="text-[var(--text-muted)]">Strength</span>
            <div className="font-mono" style={{ color: strengthColor(display.strength) }}>
              {display.strength.toFixed(3)}
            </div>
          </div>
          <div>
            <span className="text-[var(--text-muted)]">Accesses</span>
            <div className="text-[var(--text-primary)] font-mono">{display.access_count}</div>
          </div>
          <div>
            <span className="text-[var(--text-muted)]">Source</span>
            <div className="text-[var(--text-primary)]">{display.source}</div>
          </div>
          <div>
            <span className="text-[var(--text-muted)]">Status</span>
            <div className="text-[var(--text-primary)]">{display.status}</div>
          </div>
          <div>
            <span className="text-[var(--text-muted)]">Support</span>
            <div className="text-[var(--text-primary)] font-mono">{display.support_count}</div>
          </div>
          <div>
            <span className="text-[var(--text-muted)]">Diversity</span>
            <div className="text-[var(--text-primary)] font-mono">{display.source_diversity.toFixed(2)}</div>
          </div>
          <div className="col-span-2">
            <span className="text-[var(--text-muted)]">Created</span>
            <div className="text-[var(--text-primary)]">{new Date(display.created_at).toLocaleString()}</div>
          </div>
          <div className="col-span-2">
            <span className="text-[var(--text-muted)]">Last Accessed</span>
            <div className="text-[var(--text-primary)]">{new Date(display.last_accessed).toLocaleString()}</div>
          </div>
        </div>

        {/* Keywords */}
        {display.keywords.length > 0 && (
          <div>
            <div className="text-xs text-[var(--text-muted)] uppercase tracking-wider mb-1.5">Keywords</div>
            <div className="flex flex-wrap gap-1.5">
              {display.keywords.map((kw) => (
                <span key={kw} className="text-xs px-2 py-0.5 rounded bg-[var(--border)] text-[var(--text-secondary)]">
                  {kw}
                </span>
              ))}
            </div>
          </div>
        )}

        <div>
          <div className="text-xs text-[var(--text-muted)] uppercase tracking-wider mb-1.5">Evidence</div>
          {loading ? (
            <div className="text-xs text-[var(--text-muted)]">Loading detail...</div>
          ) : error ? (
            <div className="text-xs text-[var(--hot)] break-words">Failed to load detail: {error}</div>
          ) : evidence.length > 0 ? (
            <div className="space-y-2">
              <div className="rounded-lg border border-[var(--accent)]/30 bg-[var(--accent)]/5 p-2">
                <div className="text-[10px] uppercase tracking-wider text-[var(--accent)] mb-1">
                  Canonical
                </div>
                <div className="text-xs text-[var(--text-primary)]">{display.summary}</div>
              </div>
              {visibleEvidence.map((item) => (
                <div key={item.id} className="rounded-lg border border-[var(--border)] bg-[var(--bg)]/50 p-2">
                  <div className="flex items-center justify-between gap-2 mb-1">
                    <span className="text-[10px] uppercase tracking-wider text-[var(--accent)]">{item.source_topic}</span>
                    <span className="text-[10px] text-[var(--text-muted)]">{new Date(item.imported_at).toLocaleDateString()}</span>
                  </div>
                  <div className="text-xs text-[var(--text-primary)] mb-1">{item.summary}</div>
                  <div className="text-xs text-[var(--text-secondary)] line-clamp-4 whitespace-pre-wrap break-words">
                    {item.content}
                  </div>
                </div>
              ))}
              {evidence.length > 3 && (
                <div className="flex items-center justify-between gap-2 pt-1">
                  <span className="text-[10px] text-[var(--text-muted)]">
                    {/* Honest count: report what's actually rendered
                     * (`visibleEvidence.length`), not the preview cap.
                     * Otherwise a collapsed view of a 543-row canonical
                     * would say "Showing 200 of 543" while only 3 cards
                     * are visible. */}
                    {visibleEvidence.length === evidenceTotal
                      ? `${evidenceTotal} evidence items`
                      : `Showing ${visibleEvidence.length} of ${evidenceTotal} evidence items`}
                  </span>
                  <button
                    onClick={() => setShowAllEvidence((v) => !v)}
                    className="text-xs text-[var(--accent)] hover:text-[var(--accent)]/80 transition-colors"
                  >
                    {showAllEvidence ? 'Show fewer' : 'Show all'}
                  </button>
                </div>
              )}
            </div>
          ) : (
            <div className="text-xs text-[var(--text-muted)]">No supporting evidence beyond the canonical record.</div>
          )}
        </div>

        {/* Related IDs */}
        {display.related_ids.length > 0 && (
          <div>
            <div className="text-xs text-[var(--text-muted)] uppercase tracking-wider mb-1.5">Related Memories</div>
            <div className="space-y-1">
              {display.related_ids.map((id) => (
                <div key={id} className="text-xs font-mono text-[var(--text-muted)] truncate">{id}</div>
              ))}
            </div>
          </div>
        )}

        {/* Concept IDs */}
        {display.concept_ids.length > 0 && (
          <div>
            <div className="text-xs text-[var(--text-muted)] uppercase tracking-wider mb-1.5">Concepts</div>
            <div className="space-y-1">
              {display.concept_ids.map((id) => (
                <div key={id} className="text-xs font-mono text-[var(--concept)] truncate">{id}</div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Footer actions */}
      <div className="flex gap-2 p-4 border-t border-[var(--border)]">
        <button
          className="flex-1 text-xs px-3 py-2 rounded-lg bg-[var(--accent)]/20 text-[var(--accent)] hover:bg-[var(--accent)]/30 transition-colors cursor-not-allowed opacity-50"
          disabled
          title="Coming soon"
        >
          Edit
        </button>
        {confirmDelete ? (
          <div className="flex-1 flex gap-1">
            <button
              onClick={() => onDelete(display.id)}
              className="flex-1 text-xs px-2 py-2 rounded-lg bg-red-500/20 text-red-400 hover:bg-red-500/30 transition-colors"
            >
              Confirm
            </button>
            <button
              onClick={() => setConfirmDelete(false)}
              className="flex-1 text-xs px-2 py-2 rounded-lg bg-[var(--border)] text-[var(--text-muted)] hover:bg-[var(--border)]/80 transition-colors"
            >
              Cancel
            </button>
          </div>
        ) : (
          <button
            onClick={() => setConfirmDelete(true)}
            className="flex-1 text-xs px-3 py-2 rounded-lg bg-red-500/10 text-red-400 hover:bg-red-500/20 transition-colors"
          >
            Delete
          </button>
        )}
      </div>
    </div>
  );
}

/* ── Memories page ───────────────────────────────────────────────── */

export default function Memories() {
  const queryClient = useQueryClient();
  const [query, setQuery] = useState('');
  const [debouncedQuery, setDebouncedQuery] = useState('');
  const [topicFilter, setTopicFilter] = useState('');
  const [tierFilter, setTierFilter] = useState<TierFilter>('all');
  const [sortMode, setSortMode] = useState<SortMode>('recent');
  const [selected, setSelected] = useState<(Memory | RecallResult) | null>(null);
  const [deleteError, setDeleteError] = useState<string | null>(null);
  // v0.25 ARS Cap B: opt-in narrative synthesis over the recall results. Off
  // by default — the LLM round-trip adds latency and only pays off for
  // multi-result queries, so we only flip it on when the user explicitly asks.
  const [synthesize, setSynthesize] = useState(false);
  const {
    data: selectedDetail,
    isLoading: selectedLoading,
    error: selectedDetailError,
  } = useMemoryDetail(selected?.id ?? null);

  // Debounce search query (300ms)
  useEffect(() => {
    const timer = setTimeout(() => setDebouncedQuery(query), 300);
    return () => clearTimeout(timer);
  }, [query]);

  const { data: topicsData } = useTopics();
  const { data: recentData, isLoading: recentLoading, refetch: refetchRecent } = useRecent(50);
  const { data: recallData, isLoading: recallLoading } = useRecall(debouncedQuery, {
    topic: topicFilter || undefined,
    limit: 50,
    synthesize,
  });

  const isSearching = debouncedQuery.length > 0;
  const isLoading = isSearching ? recallLoading : recentLoading;

  // Build the displayed list with tier filter applied
  const memories = useMemo(() => {
    let list: (Memory | RecallResult)[];
    if (isSearching && recallData) {
      list = recallData.results;
    } else if (!isSearching && recentData) {
      list = recentData.memories;
    } else {
      list = [];
    }
    // Apply tier filter
    if (tierFilter !== 'all') {
      list = list.filter((m) => m.tier === tierFilter);
    }
    // Apply topic filter for recent (recall already passes topic to API)
    if (!isSearching && topicFilter) {
      list = list.filter((m) => m.topic === topicFilter);
    }
    list = [...list].sort((a, b) => {
      switch (sortMode) {
        case 'support':
          return b.support_count - a.support_count || b.source_diversity - a.source_diversity;
        case 'strength':
          return b.strength - a.strength || b.support_count - a.support_count;
        case 'relevance':
          return ('score' in b ? b.score : 0) - ('score' in a ? a.score : 0);
        case 'recent':
        default:
          return new Date(b.updated_at).getTime() - new Date(a.updated_at).getTime();
      }
    });
    return list;
  }, [isSearching, recallData, recentData, tierFilter, topicFilter, sortMode]);

  const handleDelete = useCallback(
    async (id: string) => {
      try {
        await apiDelete(`/api/memories/${id}`);
        setSelected(null);
        setDeleteError(null);
        await Promise.all([
          refetchRecent(),
          queryClient.invalidateQueries({ queryKey: ['recent'] }),
          queryClient.invalidateQueries({ queryKey: ['recall'] }),
          queryClient.invalidateQueries({ queryKey: ['memory-detail'] }),
          // M3 (v0.26 cleanup): the topic dropdown is fed by `useTopics`,
          // which polls but doesn't react to mutations. Deleting the last
          // memory under a topic would silently leave that topic in the
          // dropdown until the next poll tick. Invalidate so the dropdown
          // re-syncs immediately.
          queryClient.invalidateQueries({ queryKey: ['topics'] }),
        ]);
      } catch (err) {
        // B6 — surface the failure instead of only logging to console so the
        // user sees why the item stayed visible after clicking Delete.
        const message =
          err instanceof Error
            ? err.message
            : typeof err === 'string'
              ? err
              : 'Delete failed';
        console.error('Delete failed:', err);
        setDeleteError(message);
      }
    },
    [queryClient, refetchRecent],
  );

  return (
    <div className="flex h-full relative">
      {/* Main area */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Search + filters bar */}
        <div className="px-6 pt-5 pb-4 space-y-3">
          {/* Search input */}
          <div className="relative">
            <span className="absolute left-3 top-1/2 -translate-y-1/2 text-[var(--text-muted)] text-sm pointer-events-none">
              {'\u{1F50D}'}
            </span>
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search memories..."
              className="w-full bg-[var(--bg-secondary)] border border-[var(--border)] rounded-lg pl-9 pr-4 py-2 text-sm text-[var(--text-primary)] placeholder-[var(--text-muted)] focus:outline-none focus:border-[var(--accent)] transition-colors"
            />
          </div>

          {/* Filter row */}
          <div className="flex items-center gap-3 flex-wrap">
            {/* Topic dropdown */}
            <select
              value={topicFilter}
              onChange={(e) => setTopicFilter(e.target.value)}
              className="bg-[var(--bg-secondary)] border border-[var(--border)] rounded-lg px-3 py-1.5 text-xs text-[var(--text-secondary)] focus:outline-none focus:border-[var(--accent)] transition-colors"
            >
              <option value="">All Topics</option>
              {topicsData?.topics.map((t) => (
                <option key={t} value={t}>{t}</option>
              ))}
            </select>

            {/* Tier buttons */}
            <div className="flex rounded-lg overflow-hidden border border-[var(--border)]">
              {(['all', 'hot', 'warm', 'cold'] as TierFilter[]).map((t) => (
                <button
                  key={t}
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

            {/* Results count */}
            <span className="text-xs text-[var(--text-muted)] ml-auto">
              {isLoading ? 'Loading...' : `${memories.length} memories`}
            </span>

            {/* Synthesize toggle (v0.25 ARS Cap B). Only meaningful when the
                user is actually searching — disabled in browse mode to keep the
                affordance honest. */}
            <label
              className={`flex items-center gap-1.5 text-xs select-none ${
                isSearching
                  ? 'text-[var(--text-secondary)] cursor-pointer'
                  : 'text-[var(--text-muted)] cursor-not-allowed opacity-60'
              }`}
              title="Use LLM to combine top results into one narrative answer."
            >
              <input
                type="checkbox"
                checked={synthesize}
                onChange={(e) => setSynthesize(e.target.checked)}
                disabled={!isSearching}
                className="accent-[var(--accent)] cursor-pointer disabled:cursor-not-allowed"
              />
              Synthesize results (LLM, slower)
            </label>

            <select
              value={sortMode}
              onChange={(e) => setSortMode(e.target.value as SortMode)}
              className="bg-[var(--bg-secondary)] border border-[var(--border)] rounded-lg px-3 py-1.5 text-xs text-[var(--text-secondary)] focus:outline-none focus:border-[var(--accent)] transition-colors"
            >
              <option value="recent">Sort: Recent</option>
              <option value="support">Sort: Support</option>
              <option value="strength">Sort: Strength</option>
              <option value="relevance">Sort: Relevance</option>
            </select>
          </div>
        </div>

        {/* Card grid */}
        <div className="flex-1 overflow-y-auto px-6 pb-6">
          {/* Synthesis card (v0.25 ARS Cap B). Mirrors the Graph "Current
              state" card pattern (rounded border, bg-primary/60 panel, muted
              uppercase header, leading-relaxed body, footer meta). Renders
              only when the user has the toggle on AND we have something to
              say — otherwise the recall flow looks identical to v0.24.
              v0.26: `recallId` is plumbed through so the dwell timer +
              click + thumb hooks can correlate feedback events with the
              originating recall request. */}
          {isSearching && synthesize && !recallLoading && (
            // `key` on `synthesis_id` forces a fresh component instance
            // per synthesis output: the dwell timer + thumb-vote state +
            // citation-feedback latch all reset cleanly when a new
            // synthesis arrives. Falling back to 'none' for the
            // legacy/skipped branches keeps the same instance across
            // re-renders that don't carry a new synthesis (so the skip
            // notice doesn't flicker).
            <SynthesisCard
              key={recallData?.synthesis?.synthesis_id ?? 'none'}
              outcome={recallData?.synthesis}
              recallId={recallData?.request_id}
            />
          )}

          {isLoading ? (
            <div className="flex items-center justify-center py-20 text-[var(--text-muted)]">
              Loading...
            </div>
          ) : memories.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-20 text-[var(--text-muted)]">
              <div className="text-3xl mb-3">{'\u{1F50D}'}</div>
              <div className="text-sm">No memories found</div>
              {debouncedQuery && (
                <div className="text-xs mt-1">Try a different search query</div>
              )}
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {memories.map((m) => (
                <MemoryCard key={m.id} memory={m} onClick={() => setSelected(m)} />
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Delete error banner — B6 #30 follow-up */}
      {deleteError && (
        <div className="fixed top-4 right-4 z-50 max-w-md rounded-lg border border-[var(--hot)]/30 bg-[var(--hot)]/10 px-4 py-3 text-sm text-[var(--hot)] shadow-lg">
          <div className="flex items-start gap-2">
            <span className="font-medium">Delete failed:</span>
            <span className="flex-1 break-words">{deleteError}</span>
            <button
              type="button"
              onClick={() => setDeleteError(null)}
              className="text-[var(--text-muted)] hover:text-[var(--text-primary)] -mt-0.5"
              aria-label="Dismiss error"
            >
              ×
            </button>
          </div>
        </div>
      )}

      {/* Detail slide-over panel */}
      {selected && (
        <>
          {/* Backdrop */}
          <div
            className="fixed inset-0 bg-black/30 z-40"
            onClick={() => setSelected(null)}
          />
          <DetailPanel
            memory={selected}
            detail={selectedDetail}
            error={selectedDetailError instanceof Error ? selectedDetailError.message : null}
            loading={selectedLoading}
            onClose={() => setSelected(null)}
            onDelete={handleDelete}
          />
        </>
      )}

      {/* Slide-in animation */}
      <style>{`
        @keyframes slide-in-right {
          from { transform: translateX(100%); }
          to { transform: translateX(0); }
        }
        .animate-slide-in {
          animation: slide-in-right 0.2s ease-out;
        }
      `}</style>
    </div>
  );
}
