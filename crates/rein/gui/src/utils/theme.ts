/**
 * Shared color palettes used across multiple pages. Page-specific palettes
 * (e.g. event-type bars used only in Adaptive's Event Counts chart) stay
 * inline at their callsites — only colors that would otherwise be redefined
 * in two or more pages live here.
 *
 * Tier hex literals match the `--hot` / `--warm` / `--cold` / `--concept`
 * CSS custom properties in `index.css`. We use literal hex (not `var(--hot)`)
 * because canvas `ctx.fillStyle` does not resolve CSS custom properties —
 * Brain.tsx draws nodes on a `<canvas>`, so its fills must be raw colors.
 * Recharts SVG cells accept either, so Adaptive uses the same hex for
 * consistency.
 */
export const TIER_COLORS: Record<string, string> = {
  hot: '#f97316',
  warm: '#fbbf24',
  cold: '#3b82f6',
  concept: '#e2e8f0',
};

/**
 * Generic accent palette used for ordered category fills (e.g. learned alpha
 * bars in Adaptive, dedup-decision relation chips in Provenance). Indexed by
 * position; callers that want stable per-key colors should map their own
 * lookup table on top.
 */
export const ACCENT_PALETTE: string[] = [
  '#7c3aed', '#3b82f6', '#f97316', '#22d3ee', '#4ade80',
];
