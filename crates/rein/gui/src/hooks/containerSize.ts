/**
 * Pure helpers behind `useContainerSize`, kept free of React and the DOM so
 * the sizing rules are unit-testable in the node test environment.
 */
export interface Size {
  width: number;
  height: number;
}

export interface SizeEntry {
  contentRect: { width: number; height: number };
}

/**
 * The size to adopt after a batch of resize-observer entries: the last entry
 * with a positive box wins; entries for a collapsed box (a hidden tab, a
 * container mid-transition) leave the current size alone.
 */
export function sizeFromEntries(entries: ReadonlyArray<SizeEntry>, current: Size): Size {
  let next = current;
  for (const entry of entries) {
    const width = Math.floor(entry.contentRect.width);
    const height = Math.floor(entry.contentRect.height);
    if (width > 0 && height > 0) {
      next = { width, height };
    }
  }
  return next;
}
