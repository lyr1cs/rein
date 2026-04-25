import { useEffect, useRef } from 'react';

/**
 * Minimum dwell time (ms) below which a `Viewed` event is suppressed.
 * Filters React StrictMode double-mounts (which fire mount → unmount →
 * mount within ~1ms) and any "scroll past too fast" frames that
 * IntersectionObserver picks up. Per v0.26 contract §5.1 #1.
 *
 * Bootstrap value — v0.26.1+ may derive this from the `synthesis_feedback`
 * consumer's per-cluster dwell distribution.
 */
export const DWELL_MIN_MS = 250; // bootstrap; v0.26.1 → adaptive

/**
 * Reasonable IntersectionObserver threshold for "the synthesis card is
 * visibly engaging the user". A 25% threshold catches both compact + full
 * card layouts; tighter (50%+) excludes tall cards that never fully fit
 * the viewport on small screens.
 */
const INTERSECTION_THRESHOLD = 0.25;

/**
 * Track accumulated dwell on a DOM element and emit a single `dwell_ms`
 * value when the tracked element unmounts OR when `key` changes. The
 * stopwatch only ticks while the element is BOTH intersecting the
 * viewport AND the document is `visible` (per `document.visibilityState`)
 * — both gates are required to filter out background-tab / scrolled-off
 * false positives.
 *
 * Usage:
 * ```ts
 * const ref = useRef<HTMLDivElement>(null);
 * useDwellTimer(ref, synthesisId, (dwellMs) => emitViewed(dwellMs));
 * ```
 *
 * The `onDwellComplete` callback receives a single `dwell_ms` value once
 * per `key` (synthesis_id) lifecycle, gated by `dwell_ms > DWELL_MIN_MS`.
 * Caller is responsible for the synthesis_id provenance gate (§8
 * invariant 9) — this hook only knows about ms accumulation.
 *
 * Implementation notes:
 * - All accumulation lives in refs to avoid React re-renders during the
 *   visibility/intersection lifecycle.
 * - `key` (synthesis_id) change flushes the previous dwell BEFORE
 *   re-arming. This catches the case where the user's recall page swaps
 *   in a new synthesis without the card unmounting.
 * - Cleanup on unmount also flushes — covers tab close + route change.
 * - `visibilitychange` is global (document-level), `IntersectionObserver`
 *   is element-scoped. Both gates ANDed.
 */
export function useDwellTimer(
  ref: React.RefObject<HTMLElement | null>,
  key: string | undefined,
  onDwellComplete: (dwellMs: number) => void,
): void {
  // Refs for all mutable state — these never trigger re-renders.
  const accumulatedMsRef = useRef<number>(0);
  const visibleSinceRef = useRef<number | null>(null);
  const isIntersectingRef = useRef<boolean>(false);
  const isDocumentVisibleRef = useRef<boolean>(
    typeof document !== 'undefined'
      ? document.visibilityState === 'visible'
      : true,
  );
  // Snapshot the latest callback in a ref so the effect can stay
  // `[ref, key]`-only — otherwise an inline arrow on every render would
  // re-run the IO setup loop on each pass.
  const onDwellRef = useRef(onDwellComplete);
  useEffect(() => {
    onDwellRef.current = onDwellComplete;
  }, [onDwellComplete]);

  useEffect(() => {
    const node = ref.current;
    // No element to observe (yet) OR no key (yet) — bail; the hook will
    // re-arm once both materialize. Returning a no-op cleanup keeps the
    // effect symmetric.
    if (!node || !key) {
      return;
    }
    // Reset accumulator at the start of each new tracked window so the
    // previous synthesis_id's dwell doesn't bleed into this one.
    accumulatedMsRef.current = 0;
    visibleSinceRef.current = null;
    isIntersectingRef.current = false;

    /**
     * Both gates passed — start the stopwatch.
     */
    function startIfBothVisible() {
      if (
        isIntersectingRef.current &&
        isDocumentVisibleRef.current &&
        visibleSinceRef.current === null
      ) {
        visibleSinceRef.current = Date.now();
      }
    }

    /**
     * One gate dropped — accumulate elapsed and pause.
     */
    function pauseIfRunning() {
      if (visibleSinceRef.current !== null) {
        accumulatedMsRef.current += Date.now() - visibleSinceRef.current;
        visibleSinceRef.current = null;
      }
    }

    const io = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (entry.target === node) {
            isIntersectingRef.current = entry.isIntersecting;
            if (entry.isIntersecting) {
              startIfBothVisible();
            } else {
              pauseIfRunning();
            }
          }
        }
      },
      { threshold: INTERSECTION_THRESHOLD },
    );
    io.observe(node);

    function handleVisibilityChange() {
      const visible = document.visibilityState === 'visible';
      isDocumentVisibleRef.current = visible;
      if (visible) {
        startIfBothVisible();
      } else {
        pauseIfRunning();
      }
    }
    document.addEventListener('visibilitychange', handleVisibilityChange);

    return () => {
      io.disconnect();
      document.removeEventListener('visibilitychange', handleVisibilityChange);
      // Final flush — accumulate any in-flight dwell, then emit if past the
      // floor. This runs on synthesis_id change, route change, and unmount.
      pauseIfRunning();
      const total = accumulatedMsRef.current;
      accumulatedMsRef.current = 0;
      if (total > DWELL_MIN_MS) {
        // Stable callback ref — invoking the latest closure without
        // pulling it into the effect deps.
        onDwellRef.current(total);
      }
    };
  }, [ref, key]);
}
