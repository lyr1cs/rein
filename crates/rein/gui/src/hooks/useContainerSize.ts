import { useCallback, useEffect, useState } from 'react';
import { sizeFromEntries, type Size } from './containerSize';

/**
 * Size of a container element, kept current by a `ResizeObserver`.
 *
 * The observer is attached through a callback ref, so it follows the element
 * whatever render mounts it. A mount-only `useEffect` reading a `useRef`
 * could not: on a cold load the page returns its "Loading…" branch first,
 * the effect ran while the container did not exist, and nothing re-ran it
 * when the graph rendered — the canvas then kept its viewport-derived
 * initial size through every window, sidebar and detail-panel change.
 */
export function useContainerSize(
  initial: () => Size,
): [(node: HTMLDivElement | null) => void, Size] {
  const [node, setNode] = useState<HTMLDivElement | null>(null);
  const [size, setSize] = useState<Size>(initial);

  useEffect(() => {
    if (!node) return;
    const observer = new ResizeObserver((entries) => {
      setSize((current) => sizeFromEntries(entries, current));
    });
    // `observe()` delivers the element's current box in its first
    // notification, so the viewport estimate held as initial state is
    // replaced without a synchronous measurement here.
    observer.observe(node);
    return () => observer.disconnect();
  }, [node]);

  const ref = useCallback((el: HTMLDivElement | null) => setNode(el), []);
  return [ref, size];
}
