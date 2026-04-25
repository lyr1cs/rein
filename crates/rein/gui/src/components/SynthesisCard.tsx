import type { Citation, RecallSynthesisOutcome } from '../api/types';

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
 */
function renderProseWithCitations(
  prose: string,
  citations: Citation[],
  onCite?: (rank: number) => void,
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
      if (onCite) {
        nodes.push(
          <button
            key={`cite-${groupIdx}-${i}`}
            type="button"
            onClick={() => onCite(cite.rank)}
            aria-label={`Source #${cite.rank}`}
            title={`Jump to source #${cite.rank}`}
            className={`${sharedClass} hover:bg-[var(--accent)]/30 cursor-pointer transition-colors`}
          >
            {cite.rank}
          </button>,
        );
      } else {
        // Non-interactive fallback — Memories.tsx surface where the recall
        // results aren't anchored anywhere we can scroll to.
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
 * SynthesisCard — renders the result of v0.25 ARS Capability B
 * (recall-time LLM narrative synthesis).
 *
 * Visual language mirrors the Graph "Current state" card:
 *   - rounded border, `bg-[var(--bg-primary)]/60` panel
 *   - small uppercase muted header
 *   - leading-relaxed prose body (whitespace-preserved)
 *   - footer meta row
 *
 * Branching rules:
 *   - undefined outcome  → render nothing
 *   - skipped_disabled   → muted notice "Synthesis disabled in [ars] config"
 *   - skipped_no_llm     → muted notice "No LLM provider configured"
 *   - skipped_too_few_results → muted notice w/ source count
 *   - empty synthesis + no skip flag → render nothing (defensive)
 *   - otherwise          → AI Synthesis panel with prose + footer meta
 *
 * `onCitationClick` is optional — when provided (SynthesisLab) the inline
 * `[k]` badges become clickable scroll-to-source buttons; when omitted
 * (Memories.tsx) they degrade to non-interactive `<span>`s so the visual
 * footnotes stay consistent across surfaces.
 */
export default function SynthesisCard({
  outcome,
  onCitationClick,
}: {
  outcome: RecallSynthesisOutcome | undefined;
  onCitationClick?: (rank: number) => void;
}) {
  if (!outcome) return null;

  // Skip-state banners — keep visual weight low so they read as status, not as
  // an answer. Use the same panel chrome so the page layout stays stable
  // whether synthesis succeeded, was disabled, or fell short.
  if (outcome.skipped_disabled) {
    return (
      <div className="mb-4 rounded-lg border border-[var(--border)] bg-[var(--bg-primary)]/60 p-3">
        <div className="text-[10px] text-[var(--text-muted)] uppercase tracking-wider mb-1.5">
          AI Synthesis
        </div>
        <div className="text-xs text-[var(--text-muted)] italic">
          Synthesis disabled in [ars] config.
        </div>
      </div>
    );
  }

  if (outcome.skipped_no_llm) {
    return (
      <div className="mb-4 rounded-lg border border-[var(--border)] bg-[var(--bg-primary)]/60 p-3">
        <div className="text-[10px] text-[var(--text-muted)] uppercase tracking-wider mb-1.5">
          AI Synthesis
        </div>
        <div className="text-xs text-[var(--text-muted)] italic">
          No LLM provider configured.
        </div>
      </div>
    );
  }

  if (outcome.skipped_too_few_results) {
    return (
      <div className="mb-4 rounded-lg border border-[var(--border)] bg-[var(--bg-primary)]/60 p-3">
        <div className="text-[10px] text-[var(--text-muted)] uppercase tracking-wider mb-1.5">
          AI Synthesis
        </div>
        <div className="text-xs text-[var(--text-muted)] italic">
          Too few recall results to synthesize (got {outcome.source_count}).
        </div>
      </div>
    );
  }

  // No skip flag, no prose → server returned the field but the LLM either
  // errored or produced nothing usable. Render an explicit "no synthesis
  // returned" notice instead of going silent — otherwise the SynthesisLab
  // right pane shows sources with no explanation, and Memories.tsx silently
  // hides the toggle outcome.
  if (!outcome.synthesis || outcome.synthesis.trim().length === 0) {
    return (
      <div className="mb-4 rounded-lg border border-[var(--border)] bg-[var(--bg-primary)]/60 p-3">
        <div className="text-[10px] text-[var(--text-muted)] uppercase tracking-wider mb-1.5">
          AI Synthesis
        </div>
        <div className="text-xs text-[var(--text-muted)] italic">
          No synthesis returned (LLM errored or produced empty output).
        </div>
      </div>
    );
  }

  const modelLabel = outcome.model_used ?? '—';

  return (
    <div className="mb-4 rounded-lg border border-[var(--border)] bg-[var(--bg-primary)]/60 p-3">
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
          to the corresponding RecallCard rank. The `whitespace-pre-wrap`
          + `break-words` are now applied per-text-segment by
          renderProseWithCitations so badges sit inline cleanly. */}
      <div className="text-sm text-[var(--text-secondary)] leading-relaxed">
        {renderProseWithCitations(
          outcome.synthesis,
          outcome.citations ?? [],
          onCitationClick,
        )}
      </div>

      {/* Footer meta */}
      <div className="mt-3 flex items-center justify-between text-[10px] text-[var(--text-muted)]">
        <span>
          {outcome.source_count} {outcome.source_count === 1 ? 'memory' : 'memories'} used
        </span>
        <span className="italic">Synthesized at recall time, may be incomplete</span>
      </div>
    </div>
  );
}
