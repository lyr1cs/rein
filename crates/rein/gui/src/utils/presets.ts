/**
 * Canned demo queries for the Synthesis Lab dropdown. Each one targets a real
 * topic the rein vault should have memories about, so a fresh operator can
 * smoke-test the recall + Cap B synthesis path without having to remember
 * project-specific phrasing.
 *
 * Static client-side list — no backend round-trip. Keep the count modest
 * (~7) so the `<select>` stays scannable. Wording is biased toward the kind
 * of "what did we decide" / "how does X work" phrasing the autonomous
 * router classifies as Episodic or Exploratory, since those are the routes
 * Cap B synthesis is most useful on.
 */
export const PRESET_QUERIES: readonly string[] = [
  'What did we decide about resummerize fuses?',
  'How does the v0.24 ARS rollout work?',
  "What's the difference between Cap A and Cap B synthesis?",
  "How is Brain View's force graph stabilized?",
  "What's the LLM concurrency budget?",
  'Which memories did the recent Codex audit find issues in?',
  "What's the evidence-aware recall pipeline?",
];

/**
 * localStorage key for the user's last few actually-run queries (debounced
 * final values, not every keystroke). Capped at `RECENT_QUERY_LIMIT`.
 */
export const RECENT_QUERIES_KEY = 'rein_synthesis_lab_recent';
export const RECENT_QUERY_LIMIT = 5;

export function loadRecentQueries(): string[] {
  try {
    const raw = localStorage.getItem(RECENT_QUERIES_KEY);
    if (!raw) return [];
    const parsed: unknown = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed.filter((v): v is string => typeof v === 'string').slice(0, RECENT_QUERY_LIMIT);
  } catch {
    // Corrupt JSON or storage disabled — silently fall back to no history.
    return [];
  }
}

export function saveRecentQuery(query: string, current: string[]): string[] {
  const trimmed = query.trim();
  if (trimmed.length === 0) return current;
  // Dedupe (case-sensitive — operators copy-paste their own phrasing).
  const next = [trimmed, ...current.filter((q) => q !== trimmed)].slice(0, RECENT_QUERY_LIMIT);
  try {
    localStorage.setItem(RECENT_QUERIES_KEY, JSON.stringify(next));
  } catch {
    // Storage quota / private mode — keep the in-memory list either way.
  }
  return next;
}

export function clearRecentQueries(): void {
  try {
    localStorage.removeItem(RECENT_QUERIES_KEY);
  } catch {
    // Same fall-through as save: ignore storage errors.
  }
}
