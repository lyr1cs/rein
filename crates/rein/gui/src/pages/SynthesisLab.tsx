import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useRecall } from '../hooks/useApi';
import SynthesisCard from '../components/SynthesisCard';
import type { RecallResult } from '../api/types';
import { postSynthesisInteraction } from '../api/feedback';
import { timeAgo } from '../utils/time';
import {
  PRESET_QUERIES,
  clearRecentQueries,
  loadRecentQueries,
  saveRecentQuery,
} from '../utils/presets';

/* ── helpers ─────────────────────────────────────────────────────── */

type TierFilter = 'all' | 'hot' | 'warm' | 'cold';

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
function RecallCard({
  rank,
  result,
  flashing,
}: {
  rank: number;
  result: RecallResult;
  /**
   * `true` while a transient highlight ring is shown after a citation
   * click jumped the user to this card. The parent flips this off after
   * the flash duration so the ring decays without per-card timer state.
   */
  flashing?: boolean;
}) {
  const badge = tierBadge(result.tier);
  const isCold = result.tier === 'cold';

  return (
    <div
      // Stable anchor target consumed by `onCitationClick` below — the
      // backend emits 1-based ranks, the rank prop is also 1-based, so
      // `recall-rank-${rank}` is the contract both sides agree on.
      id={`recall-rank-${rank}`}
      data-rank={rank}
      className={`group relative bg-[var(--bg-secondary)] border border-[var(--border)] rounded-xl p-4 transition-shadow ${
        isCold ? 'opacity-60' : ''
      } ${flashing ? 'ring-2 ring-[var(--accent)] ring-offset-2 ring-offset-[var(--bg-primary)]' : ''}`}
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
  const [elapsedMs, setElapsedMs] = useState<number | null>(null);
  // Lazy-init from localStorage so SSR-style mount ordering doesn't trigger
  // an extra render. Cap and dedupe live in `presets.ts`.
  const [recentQueries, setRecentQueries] = useState<string[]>(() => loadRecentQueries());

  // Refs (NOT state) for the timing path so the running stopwatch never
  // triggers a re-render during the in-flight request. Only the completion
  // branch in the `isLoading` effect calls `setElapsedMs`.
  const startedAtRef = useRef<number | null>(null);
  const prevLoadingRef = useRef<boolean>(false);

  // Debounce so each keystroke doesn't hammer the LLM. 400ms feels right for
  // a "Run on pause" lab — tighter than Memories.tsx because the page is
  // explicitly opt-in. The recent-queries push happens here (not in a
  // separate effect on `debouncedQuery`) so the localStorage write fires
  // exactly when the query is committed, not as an effect-cascade.
  useEffect(() => {
    const timer = setTimeout(() => {
      const trimmed = query.trim();
      setDebouncedQuery(trimmed);
      if (trimmed.length > 0) {
        setRecentQueries((current) => saveRecentQuery(trimmed, current));
      }
    }, 400);
    return () => clearTimeout(timer);
  }, [query]);

  const { data, isLoading, isFetching, error } = useRecall(debouncedQuery, {
    limit,
    synthesize: true,
  });

  // Latency stopwatch. Two effects coordinate:
  //
  // 1. On request-identity change (`debouncedQuery` or `limit`), reset
  //    the start-time ref AND clear the stale badge. This catches the
  //    case Codex R3 G10 flagged: when the user types a new query while
  //    the previous request is still in flight, `isLoading` stays true
  //    and the loading-transition effect below never sees false→true,
  //    so without this reset the badge would measure from the previous
  //    request's start.
  //
  // 2. On loading transition, capture elapsed. Refs (not state) for the
  //    start time so the in-flight period doesn't carry an
  //    `elapsedMs=null → number` state change that would re-render
  //    mid-fetch.
  // External-state sync: imperatively reset the stopwatch ref + clear
  // the badge when the request identity changes (Codex R3 G10). The
  // setElapsedMs(null) call is a sync from external request-lifecycle
  // signal into React state — legitimate effect use per the react-hooks
  // guidance.
  useEffect(() => {
    if (debouncedQuery.length > 0) {
      startedAtRef.current = Date.now();
      setElapsedMs(null);
    }
  }, [debouncedQuery, limit]);

  // Capture elapsed on loading transitions. Sync of external "request
  // lifecycle" signal into React state.
  useEffect(() => {
    const prev = prevLoadingRef.current;
    if (!prev && isLoading) {
      // Cold-start case: no identity-change effect fired (initial mount
      // with non-empty query, or polling refetch). Stamp the start now.
      if (startedAtRef.current === null) {
        startedAtRef.current = Date.now();
      }
    } else if (prev && !isLoading) {
      const start = startedAtRef.current;
      if (start !== null) {
        setElapsedMs(Date.now() - start);
        startedAtRef.current = null;
      }
    }
    prevLoadingRef.current = isLoading;
  }, [isLoading]);

  const filteredResults = useMemo(() => {
    const results = data?.results;
    if (!results) return [];
    if (tierFilter === 'all') return results;
    return results.filter((r) => r.tier === tierFilter);
  }, [data, tierFilter]);

  const isInitialState = debouncedQuery.length === 0;
  const requestId = data?.request_id;
  const synthesisId = data?.synthesis?.synthesis_id;
  const sourceCount = data?.synthesis?.source_count;
  const synthesisChars = data?.synthesis?.synthesis?.length;

  // ── ImmediateRequery detection ──────────────────────────────────
  // When the user submits a fresh query while a previous synthesis is
  // still in scope, emit a single `immediate_requery` event with the
  // gap_ms from when the previous synthesis arrived to when the new
  // query was committed. The consumer applies a sliding threshold
  // server-side; we emit the raw gap, not a thresholded boolean.
  //
  // We track:
  //   - `lastSynthesisIdRef`: which synthesis_id is currently "in scope"
  //   - `lastSynthesisAtRef`: when its prose first arrived
  //   - `lastRecallIdRef`: paired correlation id for the POST
  //
  // The effect fires when `debouncedQuery` changes AND a previous
  // synthesis_id was tracked — see §5.2 of the v0.26 contract.
  const lastSynthesisIdRef = useRef<string | undefined>(undefined);
  const lastSynthesisAtRef = useRef<number | null>(null);
  const lastRecallIdRef = useRef<string | undefined>(undefined);
  const lastDebouncedQueryRef = useRef<string>('');

  // External-state sync: when a fresh synthesis lands, latch the new id +
  // arrival timestamp. We stamp arrival time once per synthesis_id so the
  // gap measures user reading time, not API round-trip jitter.
  useEffect(() => {
    if (
      synthesisId !== undefined &&
      synthesisId !== lastSynthesisIdRef.current
    ) {
      lastSynthesisIdRef.current = synthesisId;
      lastSynthesisAtRef.current = Date.now();
      lastRecallIdRef.current = requestId;
    }
  }, [synthesisId, requestId]);

  // External-state sync: detect query commit and emit ImmediateRequery
  // BEFORE the new synthesis arrives. We compare against the previous
  // committed query so initial-mount + rapid-typing pause both behave.
  useEffect(() => {
    const prevQuery = lastDebouncedQueryRef.current;
    if (
      debouncedQuery.length > 0 &&
      debouncedQuery !== prevQuery &&
      lastSynthesisIdRef.current !== undefined &&
      lastSynthesisAtRef.current !== null
    ) {
      const gap_ms = Date.now() - lastSynthesisAtRef.current;
      // Fire-and-forget — postSynthesisInteraction guards both ids.
      void postSynthesisInteraction(
        lastSynthesisIdRef.current,
        lastRecallIdRef.current,
        { kind: 'immediate_requery', gap_ms },
        typeof sourceCount === 'number' ||
        typeof synthesisChars === 'number'
          ? {
              ...(typeof sourceCount === 'number'
                ? { source_count: sourceCount }
                : {}),
              ...(typeof synthesisChars === 'number'
                ? { synthesis_chars: synthesisChars }
                : {}),
            }
          : undefined,
      );
      // Clear the latch so we only emit once per (prior synthesis,
      // new query) pair. The new synthesis will re-arm the refs on
      // arrival via the effect above.
      lastSynthesisIdRef.current = undefined;
      lastSynthesisAtRef.current = null;
      lastRecallIdRef.current = undefined;
    }
    lastDebouncedQueryRef.current = debouncedQuery;
    // synthesisId/sourceCount/synthesisChars omitted from deps: the
    // requery emission must NOT re-fire when a new synthesis lands,
    // only when the user commits a NEW query. They're read at the
    // moment of the requery for metadata fidelity.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [debouncedQuery]);
  const showBadge = !isInitialState && elapsedMs !== null;
  const latencyColor = error
    ? 'var(--hot)'
    : elapsedMs !== null && elapsedMs >= 4000
      ? 'var(--hot)'
      : elapsedMs !== null && elapsedMs >= 1500
        ? 'var(--warm)'
        : 'var(--success)';
  const badgeTooltip = error
    ? `Wall-clock latency (debounce + network + LLM). Request ID: ${requestId ?? 'n/a'} (failed)`
    : `Wall-clock latency (debounce + network + LLM). Request ID: ${requestId ?? 'n/a'}`;

  const handlePresetChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const value = e.target.value;
    if (value === '__clear_recent__') {
      clearRecentQueries();
      setRecentQueries([]);
      return;
    }
    if (value.length === 0) return;
    setQuery(value);
    // Note: the <select> is controlled with value="" so this is a no-op
    // visually, but it documents intent for future maintainers.
  };

  // Citation click → smooth-scroll the left RecallCard into view + flash a
  // 1.2s highlight ring. We track the flashing rank in state (not a class
  // mutation) so the highlight survives React re-renders mid-animation,
  // and we keep a `setTimeout` ID in a ref to cancel a pending clear when
  // the user clicks a second citation before the first one decays.
  const [flashRank, setFlashRank] = useState<number | null>(null);
  const flashTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  useEffect(
    () => () => {
      if (flashTimerRef.current !== null) {
        clearTimeout(flashTimerRef.current);
      }
    },
    [],
  );
  const handleCitationClick = useCallback(
    (rank: number) => {
      const target = document.getElementById(`recall-rank-${rank}`);
      if (target) {
        target.scrollIntoView({ behavior: 'smooth', block: 'center' });
      } else if (tierFilter !== 'all') {
        // Codex R5 G11: synthesis cites by unfiltered rank, so the
        // target card may be hidden by the current tier filter and
        // `getElementById` returns null. Clear the filter (the user
        // can re-apply if desired) and retry the scroll on the next
        // frame after React re-renders the now-visible card.
        setTierFilter('all');
        requestAnimationFrame(() => {
          document
            .getElementById(`recall-rank-${rank}`)
            ?.scrollIntoView({ behavior: 'smooth', block: 'center' });
        });
      }
      setFlashRank(rank);
      if (flashTimerRef.current !== null) {
        clearTimeout(flashTimerRef.current);
      }
      flashTimerRef.current = setTimeout(() => {
        setFlashRank((cur) => (cur === rank ? null : cur));
        flashTimerRef.current = null;
      }, 1200);
    },
    [tierFilter],
  );

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

        {/* Preset / recent queries — controlled with `value=""` so the
            select always snaps back to the placeholder after picking. */}
        <select
          aria-label="Preset queries"
          value=""
          onChange={handlePresetChange}
          className="bg-[var(--bg-secondary)] border border-[var(--border)] rounded-lg px-3 py-1.5 text-xs text-[var(--text-secondary)] focus:outline-none focus:border-[var(--accent)] transition-colors max-w-[14rem]"
          title="Pick a canned demo query or replay a recent one"
        >
          <option value="" disabled>
            — Try a preset —
          </option>
          <optgroup label="Presets">
            {PRESET_QUERIES.map((q) => (
              <option key={`preset-${q}`} value={q}>
                {q}
              </option>
            ))}
          </optgroup>
          {recentQueries.length > 0 && (
            <optgroup label="Recent">
              {recentQueries.map((q) => (
                <option key={`recent-${q}`} value={q}>
                  {q}
                </option>
              ))}
              <option value="__clear_recent__">Clear recent</option>
            </optgroup>
          )}
        </select>

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
              {filteredResults.map((r) => {
                // Rank is the unfiltered position in `data.results` so
                // citation anchors stay correct when a tier filter hides
                // some cards. The LLM scored against the full result list,
                // so [#3] in the prose must always point to the 3rd entry
                // of `data.results`, not the 3rd visible card.
                const unfilteredIdx = data?.results.findIndex((x) => x.id === r.id) ?? -1;
                const rank = unfilteredIdx + 1;
                return (
                  <RecallCard
                    key={r.id}
                    rank={rank}
                    result={r}
                    flashing={flashRank === rank}
                  />
                );
              })}
            </div>
          )}
        </section>

        {/* RIGHT: synthesis */}
        <section className="flex-1 min-w-0 overflow-y-auto px-6 py-5 bg-[var(--bg-primary)]/40">
          <div className="flex items-baseline justify-between mb-3">
            <div className="flex items-baseline">
              <h2 className="text-xs uppercase tracking-wider text-[var(--text-muted)]">
                AI Synthesis
              </h2>
              {showBadge && (
                <span
                  className="text-[10px] text-[var(--text-muted)] font-mono ml-2"
                  title={badgeTooltip}
                >
                  <span style={{ color: latencyColor }}>{elapsedMs}ms</span>
                  {requestId ? ` · ${requestId.slice(0, 8)}…` : ''}
                </span>
              )}
            </div>
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
              {/* See Memories.tsx for rationale: `key` on synthesis_id
                  drives a fresh instance per synthesis output so dwell +
                  thumb + click-feedback state can't leak across queries. */}
              <SynthesisCard
                key={data?.synthesis?.synthesis_id ?? 'none'}
                outcome={data?.synthesis}
                recallId={requestId}
                onCitationClick={handleCitationClick}
              />

              {/* Sources used — numbered attribution that maps back to the
                  Raw Recall ranks on the left. v0.25.2 added inline `[k]`
                  badges in SynthesisCard prose itself (driven by
                  `data.synthesis.citations`); this list remains as a
                  deterministic at-a-glance ledger of every source the LLM
                  saw, which the inline pills don't surface for sources
                  with zero citations. */}
              {data?.results && data.results.length > 0 && (
                <div className="mt-2 rounded-lg border border-[var(--border)] bg-[var(--bg-primary)]/60 p-3">
                  <div className="text-[10px] text-[var(--text-muted)] uppercase tracking-wider mb-2">
                    Sources
                  </div>
                  <ol className="space-y-1.5">
                    {data.results.map((r, idx) => {
                      const rank = idx + 1;
                      return (
                        <li key={`src-${r.id}`}>
                          <button
                            type="button"
                            onClick={() => handleCitationClick(rank)}
                            className="w-full flex items-start gap-2 text-xs text-[var(--text-secondary)] text-left rounded px-1 -mx-1 py-0.5 hover:bg-[var(--accent)]/10 transition-colors"
                            aria-label={`Jump to source #${rank}`}
                          >
                            <span className="font-mono text-[var(--text-muted)] shrink-0">
                              #{rank}
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
                          </button>
                        </li>
                      );
                    })}
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
