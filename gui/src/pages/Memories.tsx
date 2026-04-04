import { useState, useEffect, useCallback, useMemo } from 'react';
import { useRecent, useRecall, useTopics } from '../hooks/useApi';
import { apiDelete } from '../api/client';
import type { Memory, RecallResult } from '../api/types';

/* ── helpers ─────────────────────────────────────────────────────── */

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
      return { label: '\u2744\uFE0F Cold', bg: 'bg-[var(--cold)]/20', text: 'text-[var(--cold)]' };
  }
}

function strengthColor(strength: number): string {
  if (strength >= 0.7) return 'var(--hot)';
  if (strength >= 0.4) return 'var(--warm)';
  return 'var(--cold)';
}

type TierFilter = 'all' | 'hot' | 'warm' | 'cold';

/* ── MemoryCard ──────────────────────────────────────────────────── */

function MemoryCard({ memory, onClick }: { memory: Memory | RecallResult; onClick: () => void }) {
  const badge = tierBadge(memory.tier);
  const isCold = memory.tier === 'cold';

  return (
    <button
      onClick={onClick}
      className={`bg-[var(--bg-secondary)] border border-[var(--border)] rounded-xl p-4 text-left transition-all hover:border-[var(--accent)]/50 hover:shadow-[0_0_16px_var(--accent)/10] cursor-pointer w-full ${
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
        {memory.summary}
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
    </button>
  );
}

/* ── DetailPanel ─────────────────────────────────────────────────── */

function DetailPanel({
  memory,
  onClose,
  onDelete,
}: {
  memory: Memory | RecallResult;
  onClose: () => void;
  onDelete: (id: string) => void;
}) {
  const [confirmDelete, setConfirmDelete] = useState(false);
  const badge = tierBadge(memory.tier);
  const isRecall = 'score' in memory;

  return (
    <div className="fixed inset-y-0 right-0 w-[280px] bg-[var(--bg-secondary)] border-l border-[var(--border)] shadow-2xl z-50 flex flex-col animate-slide-in overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-[var(--border)]">
        <span className={`text-xs px-2 py-0.5 rounded ${badge.bg} ${badge.text}`}>
          {badge.label}
        </span>
        <button
          onClick={onClose}
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
          <p className="text-sm text-[var(--text-primary)] leading-relaxed">{memory.summary}</p>
        </div>

        {/* Full content */}
        <div>
          <div className="text-xs text-[var(--text-muted)] uppercase tracking-wider mb-1">Content</div>
          <p className="text-sm text-[var(--text-secondary)] leading-relaxed whitespace-pre-wrap break-words">
            {memory.content}
          </p>
        </div>

        {/* Recall score */}
        {isRecall && (
          <div className="grid grid-cols-3 gap-2">
            <div>
              <div className="text-xs text-[var(--text-muted)]">Score</div>
              <div className="text-sm font-mono text-[var(--new)]">{(memory as RecallResult).score.toFixed(3)}</div>
            </div>
            <div>
              <div className="text-xs text-[var(--text-muted)]">Confidence</div>
              <div className="text-sm font-mono text-[var(--success)]">{((memory as RecallResult).confidence * 100).toFixed(0)}%</div>
            </div>
            <div>
              <div className="text-xs text-[var(--text-muted)]">Sources</div>
              <div className="text-sm font-mono text-[var(--warm)]">{(memory as RecallResult).sources_hit}/3</div>
            </div>
          </div>
        )}

        {/* Metadata grid */}
        <div className="grid grid-cols-2 gap-x-3 gap-y-2 text-xs">
          <div>
            <span className="text-[var(--text-muted)]">Layer</span>
            <div className="text-[var(--text-primary)]">{memory.layer}</div>
          </div>
          <div>
            <span className="text-[var(--text-muted)]">Importance</span>
            <div className="text-[var(--text-primary)] capitalize">{memory.importance}</div>
          </div>
          <div>
            <span className="text-[var(--text-muted)]">Strength</span>
            <div className="font-mono" style={{ color: strengthColor(memory.strength) }}>
              {memory.strength.toFixed(3)}
            </div>
          </div>
          <div>
            <span className="text-[var(--text-muted)]">Accesses</span>
            <div className="text-[var(--text-primary)] font-mono">{memory.access_count}</div>
          </div>
          <div>
            <span className="text-[var(--text-muted)]">Source</span>
            <div className="text-[var(--text-primary)]">{memory.source}</div>
          </div>
          <div>
            <span className="text-[var(--text-muted)]">Status</span>
            <div className="text-[var(--text-primary)]">{memory.status}</div>
          </div>
          <div className="col-span-2">
            <span className="text-[var(--text-muted)]">Created</span>
            <div className="text-[var(--text-primary)]">{new Date(memory.created_at).toLocaleString()}</div>
          </div>
          <div className="col-span-2">
            <span className="text-[var(--text-muted)]">Last Accessed</span>
            <div className="text-[var(--text-primary)]">{new Date(memory.last_accessed).toLocaleString()}</div>
          </div>
        </div>

        {/* Keywords */}
        {memory.keywords.length > 0 && (
          <div>
            <div className="text-xs text-[var(--text-muted)] uppercase tracking-wider mb-1.5">Keywords</div>
            <div className="flex flex-wrap gap-1.5">
              {memory.keywords.map((kw) => (
                <span key={kw} className="text-xs px-2 py-0.5 rounded bg-[var(--border)] text-[var(--text-secondary)]">
                  {kw}
                </span>
              ))}
            </div>
          </div>
        )}

        {/* Related IDs */}
        {memory.related_ids.length > 0 && (
          <div>
            <div className="text-xs text-[var(--text-muted)] uppercase tracking-wider mb-1.5">Related Memories</div>
            <div className="space-y-1">
              {memory.related_ids.map((id) => (
                <div key={id} className="text-xs font-mono text-[var(--text-muted)] truncate">{id}</div>
              ))}
            </div>
          </div>
        )}

        {/* Concept IDs */}
        {memory.concept_ids.length > 0 && (
          <div>
            <div className="text-xs text-[var(--text-muted)] uppercase tracking-wider mb-1.5">Concepts</div>
            <div className="space-y-1">
              {memory.concept_ids.map((id) => (
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
              onClick={() => onDelete(memory.id)}
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
  const [query, setQuery] = useState('');
  const [debouncedQuery, setDebouncedQuery] = useState('');
  const [topicFilter, setTopicFilter] = useState('');
  const [tierFilter, setTierFilter] = useState<TierFilter>('all');
  const [selected, setSelected] = useState<(Memory | RecallResult) | null>(null);

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
    return list;
  }, [isSearching, recallData, recentData, tierFilter, topicFilter]);

  const handleDelete = useCallback(
    async (id: string) => {
      try {
        await apiDelete(`/api/memories/${id}`);
        setSelected(null);
        refetchRecent();
      } catch (err) {
        console.error('Delete failed:', err);
      }
    },
    [refetchRecent],
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
          </div>
        </div>

        {/* Card grid */}
        <div className="flex-1 overflow-y-auto px-6 pb-6">
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
