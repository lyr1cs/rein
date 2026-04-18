import { useState, useMemo } from 'react';
import { useQuery } from '@tanstack/react-query';
import { apiGet } from '../api/client';

/* ── types ──────────────────────────────────────────────────────── */

interface TimelineEvent {
  type: 'episode' | 'memory';
  created_at: string;
  // Episode fields (when type=episode)
  id?: string;
  title?: string;
  outcome?: string;
  decisions?: string[];
  // Memory fields (when type=memory)
  summary?: string;
  topic?: string;
  tier?: 'hot' | 'warm' | 'cold';
  strength?: number;
}

interface TimelineResponse {
  events: TimelineEvent[];
}

/* ── helpers ─────────────────────────────────────────────────────── */

function tierBadge(tier: 'hot' | 'warm' | 'cold') {
  switch (tier) {
    case 'hot':
      return { label: 'Hot', bg: 'bg-[var(--hot)]/20', text: 'text-[var(--hot)]' };
    case 'warm':
      return { label: 'Warm', bg: 'bg-[var(--warm)]/20', text: 'text-[var(--warm)]' };
    case 'cold':
      return { label: 'Cold', bg: 'bg-[var(--cold)]/20', text: 'text-[var(--cold)]' };
  }
}

function formatDateTime(iso: string): { date: string; time: string } {
  const d = new Date(iso);
  return {
    date: d.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' }),
    time: d.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' }),
  };
}

function formatLocalDate(date: Date): string {
  const year = date.getFullYear();
  const month = String(date.getMonth() + 1).padStart(2, '0');
  const day = String(date.getDate()).padStart(2, '0');
  return `${year}-${month}-${day}`;
}

function defaultFrom(): string {
  const d = new Date();
  d.setDate(d.getDate() - 30);
  return formatLocalDate(d);
}

function defaultTo(): string {
  return formatLocalDate(new Date());
}

/* ── Timeline page ──────────────────────────────────────────────── */

export default function Timeline() {
  const [from, setFrom] = useState(defaultFrom);
  const [to, setTo] = useState(defaultTo);
  const [limit, setLimit] = useState(50);

  const { data, isLoading } = useQuery({
    queryKey: ['timeline', from, to, limit],
    queryFn: () => {
      const params = new URLSearchParams();
      // Send RFC3339 timestamps with the browser's UTC offset so the server
      // interprets range boundaries in the user's local time regardless of
      // server timezone. `from` spans from 00:00:00 local; `to` spans to 23:59:59.999 local.
      if (from) {
        const start = new Date(`${from}T00:00:00`);
        if (!isNaN(start.getTime())) params.set('from', start.toISOString());
      }
      if (to) {
        const end = new Date(`${to}T23:59:59.999`);
        if (!isNaN(end.getTime())) params.set('to', end.toISOString());
      }
      params.set('limit', String(limit));
      return apiGet<TimelineResponse>(`/api/timeline?${params}`);
    },
  });

  const events = useMemo(() => {
    if (!data?.events) return [];
    return [...data.events].sort(
      (a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime(),
    );
  }, [data]);

  return (
    <div className="flex flex-col h-full overflow-hidden">
      {/* Controls bar */}
      <div className="px-6 pt-5 pb-4 space-y-3">
        <div className="flex items-center gap-4 flex-wrap">
          {/* Date range */}
          <div className="flex items-center gap-2">
            <label className="text-xs text-[var(--text-muted)]">From</label>
            <input
              type="date"
              value={from}
              onChange={(e) => setFrom(e.target.value)}
              className="bg-[var(--bg-secondary)] border border-[var(--border)] rounded-lg px-3 py-1.5 text-xs text-[var(--text-secondary)] focus:outline-none focus:border-[var(--accent)] transition-colors"
            />
          </div>
          <div className="flex items-center gap-2">
            <label className="text-xs text-[var(--text-muted)]">To</label>
            <input
              type="date"
              value={to}
              onChange={(e) => setTo(e.target.value)}
              className="bg-[var(--bg-secondary)] border border-[var(--border)] rounded-lg px-3 py-1.5 text-xs text-[var(--text-secondary)] focus:outline-none focus:border-[var(--accent)] transition-colors"
            />
          </div>

          {/* Limit selector */}
          <div className="flex items-center gap-2">
            <label className="text-xs text-[var(--text-muted)]">Limit</label>
            <div className="flex rounded-lg overflow-hidden border border-[var(--border)]">
              {[20, 50, 100].map((n) => (
                <button
                  key={n}
                  onClick={() => setLimit(n)}
                  className={`px-3 py-1.5 text-xs transition-colors ${
                    limit === n
                      ? 'bg-[var(--accent)]/20 text-[var(--accent)]'
                      : 'text-[var(--text-muted)] hover:bg-[var(--bg-secondary)]'
                  }`}
                >
                  {n}
                </button>
              ))}
            </div>
          </div>

          {/* Count */}
          <span className="text-xs text-[var(--text-muted)] ml-auto">
            {isLoading ? 'Loading...' : `${events.length} events`}
          </span>
        </div>
      </div>

      {/* Event list */}
      <div className="flex-1 overflow-y-auto px-6 pb-6">
        {isLoading ? (
          <div className="flex items-center justify-center py-20 text-[var(--text-muted)]">
            Loading...
          </div>
        ) : events.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-20 text-[var(--text-muted)]">
            <div className="text-3xl mb-3">{'\u23F1\uFE0F'}</div>
            <div className="text-sm">No events in this time range</div>
          </div>
        ) : (
          <div className="space-y-3">
            {events.map((ev, i) => {
              const { date, time } = formatDateTime(ev.created_at);
              const isEpisode = ev.type === 'episode';
              const isMemory = ev.type === 'memory';

              return (
                <div key={i} className="flex gap-4">
                  {/* Date/time column */}
                  <div className="w-28 shrink-0 text-right pt-3">
                    <div className="text-xs text-[var(--text-secondary)]">{date}</div>
                    <div className="text-xs text-[var(--text-muted)]">{time}</div>
                  </div>

                  {/* Timeline line */}
                  <div className="flex flex-col items-center">
                    <div
                      className={`w-2.5 h-2.5 rounded-full mt-3.5 shrink-0 ${
                        isEpisode ? 'bg-[var(--cold)]' : 'bg-[var(--text-muted)]'
                      }`}
                    />
                    {i < events.length - 1 && (
                      <div className="w-px flex-1 bg-[var(--border)] mt-1" />
                    )}
                  </div>

                  {/* Event card */}
                  <div
                    className={`flex-1 bg-[var(--bg-secondary)] border rounded-xl p-4 mb-1 ${
                      isEpisode
                        ? 'border-l-[3px] border-l-[var(--cold)] border-t-[var(--border)] border-r-[var(--border)] border-b-[var(--border)]'
                        : 'border-l-[3px] border-l-[var(--text-muted)] border-t-[var(--border)] border-r-[var(--border)] border-b-[var(--border)]'
                    }`}
                  >
                    {isEpisode && (
                      <>
                        <div className="flex items-center gap-2 mb-2">
                          <span className="text-sm">{'\uD83C\uDFAC'}</span>
                          <span className="text-sm font-medium text-[var(--text-primary)]">
                            {ev.title || 'Untitled Episode'}
                          </span>
                        </div>
                        {ev.outcome && (
                          <p className="text-xs text-[var(--text-secondary)] mb-2 leading-relaxed">
                            {ev.outcome}
                          </p>
                        )}
                        {ev.decisions && ev.decisions.length > 0 && (
                          <div className="flex flex-wrap gap-1.5">
                            {ev.decisions.map((d, j) => (
                              <span
                                key={j}
                                className="text-xs px-2 py-0.5 rounded bg-[var(--cold)]/15 text-[var(--cold)]"
                              >
                                {d}
                              </span>
                            ))}
                          </div>
                        )}
                      </>
                    )}

                    {isMemory && (
                      <>
                        <div className="flex items-center gap-2 mb-2">
                          <span className="text-sm">{'\uD83E\uDDE0'}</span>
                          <span className="text-sm text-[var(--text-primary)] line-clamp-1">
                            {ev.summary || 'No summary'}
                          </span>
                        </div>
                        <div className="flex items-center gap-2">
                          {ev.topic && (
                            <span className="text-xs px-2 py-0.5 rounded bg-[var(--accent)]/15 text-[var(--accent)] truncate max-w-[40%]">
                              {ev.topic}
                            </span>
                          )}
                          {ev.tier && (() => {
                            const badge = tierBadge(ev.tier);
                            return (
                              <span className={`text-xs px-2 py-0.5 rounded ${badge.bg} ${badge.text}`}>
                                {badge.label}
                              </span>
                            );
                          })()}
                        </div>
                      </>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}
