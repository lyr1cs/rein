import { useCallback, useMemo, useRef, useState } from 'react';
import { postConceptSummaryFeedback } from '../api/feedback';
import { useDwellTimer } from '../hooks/useDwellTimer';

/**
 * ConceptSummaryCard — renders a Cap A concept living-summary alongside the
 * v0.27 feedback hooks (dwell, click-source, explicit-thumb). Mirrors
 * `SynthesisCard` structurally so the M1 consumer on the backend sees a
 * uniform interaction envelope across Cap A + Cap B surfaces.
 *
 * `recallId` is a per-view correlation id minted by the parent page when
 * the concept is selected (Brain.tsx is not a recall surface). Both
 * `conceptId` and `recallId` must be non-empty strings for feedback emission
 * to light up — partial ids degrade the card to read-only so the consumer
 * never sees half-correlated events.
 *
 * Immediate-requery is a page-level concern (the parent compares concept
 * selections across time) and is intentionally NOT tracked here.
 */
export default function ConceptSummaryCard({
  conceptId,
  recallId,
  summary,
  sources,
  queryType,
  clusterId,
  revisionVersion,
  conceptSummaryId,
  livingSummaryId,
}: {
  conceptId: string;
  recallId: string;
  summary: string;
  sources?: string[];
  queryType?: string;
  clusterId?: number;
  revisionVersion?: number;
  conceptSummaryId?: string | null;
  livingSummaryId?: string | null;
}) {
  const cardRef = useRef<HTMLDivElement | null>(null);
  const [thumbState, setThumbState] = useState<'up' | 'down' | null>(null);

  const conceptChars = summary.length;

  const metadata = useMemo(
    () =>
      typeof queryType === 'string' ||
      typeof clusterId === 'number' ||
      typeof conceptChars === 'number' ||
      typeof revisionVersion === 'number'
        ? {
            ...(typeof queryType === 'string' ? { query_type: queryType } : {}),
            ...(typeof clusterId === 'number' ? { cluster_id: clusterId } : {}),
            ...(typeof conceptChars === 'number'
              ? { concept_chars: conceptChars }
              : {}),
            ...(typeof revisionVersion === 'number'
              ? { revision_version: revisionVersion }
              : {}),
          }
        : undefined,
    [queryType, clusterId, conceptChars, revisionVersion],
  );

  const feedbackEnabled =
    conceptId.length > 0 && recallId.length > 0 && summary.trim().length > 0;

  const handleDwellComplete = useCallback(
    (dwellMs: number) => {
      void postConceptSummaryFeedback(
        conceptId,
        recallId,
        { kind: 'viewed', dwell_ms: Math.floor(dwellMs) },
        metadata,
        { conceptSummaryId, livingSummaryId },
      );
    },
    [conceptId, recallId, metadata, conceptSummaryId, livingSummaryId],
  );
  // Key the dwell window on the (concept, recall) pair so re-selecting
  // the same concept under a fresh recallId emits a new Viewed event.
  const dwellKey = feedbackEnabled ? `${conceptId}::${recallId}` : undefined;
  useDwellTimer(cardRef, dwellKey, handleDwellComplete);

  const handleSourceClick = useCallback(
    (rank: number) => {
      void postConceptSummaryFeedback(
        conceptId,
        recallId,
        { kind: 'clicked_source', source_index: rank },
        metadata,
        { conceptSummaryId, livingSummaryId },
      );
    },
    [conceptId, recallId, metadata, conceptSummaryId, livingSummaryId],
  );

  const handleThumb = useCallback(
    (up: boolean) => {
      setThumbState((prev) => (prev !== null ? prev : up ? 'up' : 'down'));
      if (thumbState === null) {
        void postConceptSummaryFeedback(
          conceptId,
          recallId,
          { kind: 'explicit_thumb', up },
          metadata,
          { conceptSummaryId, livingSummaryId },
        );
      }
    },
    [conceptId, recallId, metadata, thumbState, conceptSummaryId, livingSummaryId],
  );

  if (summary.trim().length === 0) return null;

  return (
    <div
      ref={cardRef}
      className="mb-4 rounded-lg border border-[var(--border)] bg-[var(--bg-primary)]/60 p-3"
    >
      <div className="flex items-center justify-between mb-2">
        <div className="text-[10px] text-[var(--text-muted)] uppercase tracking-wider">
          Concept Summary
        </div>
        {typeof revisionVersion === 'number' && (
          <span
            className="text-[10px] px-1.5 py-0.5 rounded bg-[var(--accent)]/15 text-[var(--accent)] font-mono"
            title="Source revision the living summary was derived from"
          >
            rev #{revisionVersion}
          </span>
        )}
      </div>

      <div className="text-xs text-[var(--text-secondary)] leading-relaxed whitespace-pre-wrap break-words">
        {summary}
      </div>

      {sources && sources.length > 0 && (
        <div className="mt-3 flex flex-wrap gap-1">
          {sources.map((sid, idx) => {
            const rank = idx + 1;
            const label = `#${rank}`;
            const title = `Source memory ${sid.slice(0, 8)}`;
            if (feedbackEnabled) {
              return (
                <button
                  key={`src-${sid}-${idx}`}
                  type="button"
                  onClick={() => handleSourceClick(rank)}
                  aria-label={`Source ${rank}`}
                  title={title}
                  className="text-[10px] font-mono bg-[var(--accent)]/15 text-[var(--accent)] px-1.5 py-0.5 rounded hover:bg-[var(--accent)]/30 transition-colors"
                >
                  {label}
                </button>
              );
            }
            return (
              <span
                key={`src-${sid}-${idx}`}
                aria-label={`Source ${rank}`}
                title={title}
                className="text-[10px] font-mono bg-[var(--accent)]/15 text-[var(--accent)] px-1.5 py-0.5 rounded"
              >
                {label}
              </span>
            );
          })}
        </div>
      )}

      <div className="mt-3 flex items-center justify-between text-[10px] text-[var(--text-muted)] gap-2">
        <span className="flex-shrink-0">
          {sources?.length ?? 0}{' '}
          {(sources?.length ?? 0) === 1 ? 'source' : 'sources'}
        </span>
        {feedbackEnabled && (
          <div
            className="flex items-center gap-1"
            role="group"
            aria-label="Rate this concept summary"
          >
            <button
              type="button"
              onClick={() => handleThumb(true)}
              disabled={thumbState !== null}
              aria-label={
                thumbState === 'up'
                  ? 'You rated this concept summary helpful'
                  : 'Rate this concept summary helpful'
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
                  ? 'You rated this concept summary unhelpful'
                  : 'Rate this concept summary unhelpful'
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
          Auto-refreshed living summary
        </span>
      </div>
    </div>
  );
}
