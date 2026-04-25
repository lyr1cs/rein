/**
 * Format an ISO-8601 timestamp as a relative "time ago" string. Shared by
 * Memories.tsx (card list) and Graph.tsx (living_summary card meta line) so
 * the whole app reads dates consistently. Defensive against future-dated
 * timestamps (clock skew, NTP rollback) — they collapse to "just now".
 */
export function timeAgo(iso: string): string {
  const diff = Date.now() - new Date(iso).getTime();
  if (!Number.isFinite(diff) || diff < 0) return 'just now';
  const mins = Math.floor(diff / 60_000);
  if (mins < 1) return 'just now';
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs}h ago`;
  const days = Math.floor(hrs / 24);
  if (days < 30) return `${days}d ago`;
  const months = Math.floor(days / 30);
  return `${months}mo ago`;
}
