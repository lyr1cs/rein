import type { RecallSynthesisOutcome } from '../api/types';

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
 */
export default function SynthesisCard({
  outcome,
}: {
  outcome: RecallSynthesisOutcome | undefined;
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

      {/* Body: prose */}
      <div className="text-sm text-[var(--text-secondary)] leading-relaxed whitespace-pre-wrap break-words">
        {outcome.synthesis}
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
