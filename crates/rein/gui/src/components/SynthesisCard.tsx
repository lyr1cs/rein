import { useCallback, useMemo, useRef, useState } from 'react';
import type { Citation, RecallSynthesisOutcome } from '../api/types';
import { postSynthesisInteraction } from '../api/feedback';
import { useDwellTimer } from '../hooks/useDwellTimer';

/**
 * Render synthesized prose with inline `[k]` citation badges anchored at
 * each citation's `span_end` (a CHAR offset, not a byte offset — see
 * `Citation` doc in `api/types.ts`).
 *
 * Citations sharing the same `span_end` (the LLM emitted `[#1][#3]`)
 * render as adjacent badges with no whitespace, preserving the visual
 * grouping while keeping each rank independently clickable.
 *
 * When `onCite` is `undefined` (the Memories.tsx caller doesn't wire a
 * scroll handler), badges fall back to a non-interactive `<span>` so they
 * stay visually consistent across surfaces but don't invite a dead click.
 *
 * v0.26: badges always emit a `clicked_source` feedback POST when
 * `synthesis_id` + `recall_id` are present, even on the non-scrolling
 * surface — the click signal is independent of scroll behaviour.
 */
function renderProseWithCitations(
  prose: string,
  citations: Citation[],
  onCite?: (rank: number) => void,
  onCiteFeedback?: (rank: number) => void,
) {
  if (citations.length === 0) {
    // Whitespace-pre-wrap to preserve linebreaks the LLM may have emitted —
    // matches the legacy `<div>{outcome.synthesis}</div>` rendering.
    return <span className="whitespace-pre-wrap break-words">{prose}</span>;
  }

  // Char-aware split. JS strings are UTF-16; `Array.from` materializes one
  // entry per code point, which is the same unit the Rust backend's
  // `chars().count()` uses. Joining with `''` reconstitutes the slice.
  // (For BMP-only text this matches `slice()`; for emoji / supplementary
  // plane chars the spread is the only safe approach.)
  const charArr = Array.from(prose);

  // Group citations by span_end so the markers we render are visually
  // adjacent. Sort defensively in case the backend ordering ever changes.
  const groups = new Map<number, Citation[]>();
  for (const c of citations) {
    const arr = groups.get(c.span_end);
    if (arr) {
      arr.push(c);
    } else {
      groups.set(c.span_end, [c]);
    }
  }
  const sortedOffsets = Array.from(groups.keys()).sort((a, b) => a - b);

  const nodes: React.ReactNode[] = [];
  let cursor = 0;
  sortedOffsets.forEach((offset, groupIdx) => {
    // Clamp to prose length defensively — a buggy backend emitting an
    // out-of-bounds offset must not throw at render time.
    const clamped = Math.max(cursor, Math.min(offset, charArr.length));
    if (clamped > cursor) {
      nodes.push(
        <span key={`prose-${groupIdx}`} className="whitespace-pre-wrap break-words">
          {charArr.slice(cursor, clamped).join('')}
        </span>,
      );
    }
    const group = groups.get(offset)!;
    group.forEach((cite, i) => {
      const sharedClass =
        'text-[9px] font-mono bg-[var(--accent)]/15 text-[var(--accent)] px-1 rounded ml-0.5 align-super';
      // Always render an interactive button when EITHER a scroll handler
      // is wired OR feedback emission is plumbed — the visible citation
      // signal is just as valuable as the scroll-jump on Memories.tsx
      // (where there's no fixed anchor to scroll to).
      const handleClick = () => {
        if (onCite) {
          onCite(cite.rank);
        }
        if (onCiteFeedback) {
          onCiteFeedback(cite.rank);
        }
      };
      const isInteractive = onCite || onCiteFeedback;
      if (isInteractive) {
        nodes.push(
          <button
            key={`cite-${groupIdx}-${i}`}
            type="button"
            onClick={handleClick}
            aria-label={`Source #${cite.rank}`}
            title={
              onCite
                ? `Jump to source #${cite.rank}`
                : `Source #${cite.rank}`
            }
            className={`${sharedClass} hover:bg-[var(--accent)]/30 cursor-pointer transition-colors`}
          >
            {cite.rank}
          </button>,
        );
      } else {
        // Non-interactive fallback — feedback is also off; keep the badge
        // purely decorative.
        nodes.push(
          <span
            key={`cite-${groupIdx}-${i}`}
            aria-label={`Source #${cite.rank}`}
            title={`Source #${cite.rank}`}
            className={sharedClass}
          >
            {cite.rank}
          </span>,
        );
      }
    });
    cursor = clamped;
  });
  if (cursor < charArr.length) {
    nodes.push(
      <span key="prose-tail" className="whitespace-pre-wrap break-words">
        {charArr.slice(cursor).join('')}
      </span>,
    );
  }
  return <>{nodes}</>;
}

/**
 * Reusable skip-state shell. Keeps the chrome identical across the four
 * "we have no prose to show" branches so the page layout stays stable
 * whether synthesis succeeded, was disabled, or fell short.
 */
function SkipNotice({ children }: { children: React.ReactNode }) {
  return (
    <div className="mb-4 rounded-lg border border-[var(--border)] bg-[var(--bg-primary)]/60 p-3">
      <div className="text-[10px] text-[var(--text-muted)] uppercase tracking-wider mb-1.5">
        AI Synthesis
      </div>
      <div className="text-xs text-[var(--text-muted)] italic">{children}</div>
    </div>
  );
}

/**
 * SynthesisCard — renders the result of v0.25 ARS Capability B
 * (recall-time LLM narrative synthesis), instrumented for v0.26 D direction
 * feedback emission (Viewed dwell, ClickedSource, ExplicitThumb).
 *
 * Visual language mirrors the Graph "Current state" card:
 *   - rounded border, `bg-[var(--bg-primary)]/60` panel
 *   - small uppercase muted header
 *   - leading-relaxed prose body (whitespace-preserved)
 *   - footer meta row (now includes thumb up/down affordance)
 *
 * Branching rules (skip flag → message):
 *   - undefined outcome              → render nothing
 *   - skipped_disabled               → "Synthesis disabled by operator"
 *   - skipped_adaptive_decision      → "Adaptive layer skipped synthesis…"
 *   - skipped_no_llm                 → "No LLM provider configured"
 *   - skipped_too_few_results        → "Too few recall results to synthesize"
 *   - empty/missing synthesis        → "No synthesis returned"
 *   - filled synthesis               → AI Synthesis panel + dwell + thumbs
 *
 * `onCitationClick` is optional — when provided (SynthesisLab) the inline
 * `[k]` badges become clickable scroll-to-source buttons; when omitted
 * (Memories.tsx) they degrade to non-scrolling buttons that still emit
 * `clicked_source` feedback (or pure spans when `recallId` is also missing).
 *
 * `recallId` is the request_id from `RecallMemoryOutput` — required for
 * any feedback POST (along with `outcome.synthesis_id`). Both undefined →
 * card is read-only (legacy backends pre-v0.26 D direction).
 */
export default function SynthesisCard({
  outcome,
  recallId,
  onCitationClick,
}: {
  outcome: RecallSynthesisOutcome | undefined;
  recallId?: string;
  onCitationClick?: (rank: number) => void;
}) {
  // Hooks MUST be called unconditionally — declare them before any early
  // returns. The dwell timer is a no-op when `synthesisId` is undefined
  // (the hook's internal `key` gate handles that).
  const cardRef = useRef<HTMLDivElement | null>(null);
  const [thumbState, setThumbState] = useState<'up' | 'down' | null>(null);

  const synthesisId = outcome?.synthesis_id;
  const sourceCount = outcome?.source_count;
  const synthesisChars = outcome?.synthesis?.length;

  // Build the metadata payload once per synthesis so dwell + click + thumb
  // events all carry a consistent diagnostic envelope. We only include
  // fields the server has actually surfaced — query_type and cluster_id
  // would require backend exposure beyond v0.26.0 (see deliverables note).
  // Memoized so the three feedback callbacks below get stable deps and
  // don't tear-down/recreate every render.
  const metadata = useMemo(
    () =>
      typeof sourceCount === 'number' || typeof synthesisChars === 'number'
        ? {
            ...(typeof sourceCount === 'number'
              ? { source_count: sourceCount }
              : {}),
            ...(typeof synthesisChars === 'number'
              ? { synthesis_chars: synthesisChars }
              : {}),
          }
        : undefined,
    [sourceCount, synthesisChars],
  );

  const handleDwellComplete = useCallback(
    (dwellMs: number) => {
      // Round to integer ms — the server expects u64. Floor not round so
      // partial-ms accumulation can never inflate the value.
      const dwell = Math.floor(dwellMs);
      void postSynthesisInteraction(
        synthesisId,
        recallId,
        { kind: 'viewed', dwell_ms: dwell },
        metadata,
      );
    },
    [synthesisId, recallId, metadata],
  );
  useDwellTimer(cardRef, synthesisId, handleDwellComplete);

  const handleCitationFeedback = useCallback(
    (rank: number) => {
      void postSynthesisInteraction(
        synthesisId,
        recallId,
        { kind: 'clicked_source', source_index: rank },
        metadata,
      );
    },
    [synthesisId, recallId, metadata],
  );

  const handleThumb = useCallback(
    (up: boolean) => {
      // UI: latch state immediately so the user sees the choice land.
      // Re-clicking the same thumb is a no-op (no second POST) — the
      // server only counts the first vote per (synthesis_id, user). We
      // also disallow flipping (up→down) here for the same reason; if
      // the user wants to revise they can re-query.
      setThumbState((prev) => (prev !== null ? prev : up ? 'up' : 'down'));
      if (thumbState === null) {
        void postSynthesisInteraction(
          synthesisId,
          recallId,
          { kind: 'explicit_thumb', up },
          metadata,
        );
      }
    },
    [synthesisId, recallId, metadata, thumbState],
  );

  if (!outcome) return null;

  // Skip-state banners — keep visual weight low so they read as status, not as
  // an answer. Use the same panel chrome so the page layout stays stable
  // whether synthesis succeeded, was disabled, or fell short.
  if (outcome.skipped_disabled) {
    return <SkipNotice>Synthesis disabled in [ars] config.</SkipNotice>;
  }

  if (outcome.skipped_adaptive_decision) {
    return (
      <SkipNotice>
        Adaptive layer skipped synthesis for this query (cluster history did
        not justify the LLM round-trip).
      </SkipNotice>
    );
  }

  if (outcome.skipped_no_llm) {
    return <SkipNotice>No LLM provider configured.</SkipNotice>;
  }

  if (outcome.skipped_too_few_results) {
    return (
      <SkipNotice>
        Too few recall results to synthesize (got {outcome.source_count}).
      </SkipNotice>
    );
  }

  // No skip flag, no prose → server returned the field but the LLM either
  // errored or produced nothing usable. Render an explicit "no synthesis
  // returned" notice instead of going silent — otherwise the SynthesisLab
  // right pane shows sources with no explanation, and Memories.tsx silently
  // hides the toggle outcome.
  if (!outcome.synthesis || outcome.synthesis.trim().length === 0) {
    return (
      <SkipNotice>
        No synthesis returned (LLM errored or produced empty output).
      </SkipNotice>
    );
  }

  const modelLabel = outcome.model_used ?? '—';
  // Feedback emission only lights up when BOTH ids are present — see §8
  // invariant 9. Older backends without these fields render a read-only
  // card (no thumbs, no click POST, no dwell) so we don't post bogus
  // events with empty correlation ids.
  const feedbackEnabled =
    typeof synthesisId === 'string' &&
    synthesisId.length > 0 &&
    typeof recallId === 'string' &&
    recallId.length > 0;

  return (
    <div
      ref={cardRef}
      className="mb-4 rounded-lg border border-[var(--border)] bg-[var(--bg-primary)]/60 p-3"
    >
      {/* Header: label + model chip */}
      <div className="flex items-center justify-between mb-2">
        <div className="text-[10px] text-[var(--text-muted)] uppercase tracking-wider">
          AI Synthesis
        </div>
        <span
          className="text-[10px] px-1.5 py-0.5 rounded bg-[var(--accent)]/15 text-[var(--accent)] font-mono"
          title="LLM backend that produced this synthesis"
        >
          {modelLabel}
        </span>
      </div>

      {/* Body: prose with inline citation badges. v0.25.2 wraps each
          cited claim with a click-to-scroll superscript pill linking back
          to the corresponding RecallCard rank. v0.26 also emits a
          `clicked_source` feedback event when the badge is interactive. */}
      <div className="text-sm text-[var(--text-secondary)] leading-relaxed">
        {renderProseWithCitations(
          outcome.synthesis,
          outcome.citations ?? [],
          onCitationClick,
          feedbackEnabled ? handleCitationFeedback : undefined,
        )}
      </div>

      {/* Footer meta */}
      <div className="mt-3 flex items-center justify-between text-[10px] text-[var(--text-muted)] gap-2">
        <span className="flex-shrink-0">
          {outcome.source_count}{' '}
          {outcome.source_count === 1 ? 'memory' : 'memories'} used
        </span>
        {feedbackEnabled && (
          <div
            className="flex items-center gap-1"
            role="group"
            aria-label="Rate this synthesis"
          >
            <button
              type="button"
              onClick={() => handleThumb(true)}
              disabled={thumbState !== null}
              aria-label={
                thumbState === 'up'
                  ? 'You rated this synthesis helpful'
                  : 'Rate this synthesis helpful'
              }
              aria-pressed={thumbState === 'up'}
              title="Helpful"
              className={`px-1.5 py-0.5 rounded transition-colors ${
                thumbState === 'up'
                  ? 'bg-[var(--success)]/20 text-[var(--success)]'
                  : 'text-[var(--text-muted)] hover:bg-[var(--accent)]/15 hover:text-[var(--accent)]'
              } ${thumbState === 'down' ? 'opacity-40 cursor-not-allowed' : ''}`}
            >
              {'\u{1F44D}'}
            </button>
            <button
              type="button"
              onClick={() => handleThumb(false)}
              disabled={thumbState !== null}
              aria-label={
                thumbState === 'down'
                  ? 'You rated this synthesis unhelpful'
                  : 'Rate this synthesis unhelpful'
              }
              aria-pressed={thumbState === 'down'}
              title="Not helpful"
              className={`px-1.5 py-0.5 rounded transition-colors ${
                thumbState === 'down'
                  ? 'bg-[var(--hot)]/20 text-[var(--hot)]'
                  : 'text-[var(--text-muted)] hover:bg-[var(--accent)]/15 hover:text-[var(--accent)]'
              } ${thumbState === 'up' ? 'opacity-40 cursor-not-allowed' : ''}`}
            >
              {'\u{1F44E}'}
            </button>
          </div>
        )}
        <span className="italic flex-shrink min-w-0 truncate text-right">
          Synthesized at recall time, may be incomplete
        </span>
      </div>
    </div>
  );
}
