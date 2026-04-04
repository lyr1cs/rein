import { useStats, useRecent, useTopics } from '../hooks/useApi';

function StatCard({ label, value, color }: { label: string; value: string | number; color: string }) {
  return (
    <div className="bg-[var(--bg-secondary)] border border-[var(--border)] rounded-xl p-4">
      <div className="text-xs text-[var(--text-muted)] uppercase tracking-wider mb-1">{label}</div>
      <div className="text-2xl font-semibold font-mono" style={{ color }}>{typeof value === 'number' ? value.toLocaleString() : value}</div>
    </div>
  );
}

export default function Dashboard() {
  const { data: stats, isLoading } = useStats();
  const { data: recentData } = useRecent(5);
  const { data: _topicsData } = useTopics();

  if (isLoading || !stats) {
    return <div className="flex items-center justify-center h-full text-[var(--text-muted)]">Loading...</div>;
  }

  return (
    <div className="p-6 max-w-6xl mx-auto">
      <h1 className="text-xl font-semibold mb-6">Dashboard</h1>

      {/* Stat cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
        <StatCard label="Total Memories" value={stats.total_memories} color="var(--new)" />
        <StatCard label="Concepts" value={stats.concept_count} color="var(--warm)" />
        <StatCard label="Topics" value={stats.topic_count} color="var(--accent)" />
        <StatCard label="Avg Strength" value={stats.avg_strength.toFixed(2)} color="var(--success)" />
        <StatCard label="LTM" value={stats.ltm_count} color="var(--hot)" />
        <StatCard label="STM" value={stats.stm_count} color="var(--cold)" />
        <StatCard label="Memoirs" value={stats.memoir_count} color="var(--concept)" />
        <StatCard label="Links" value={stats.link_count} color="var(--text-secondary)" />
      </div>

      {/* Recent memories */}
      <h2 className="text-sm font-semibold text-[var(--text-muted)] uppercase tracking-wider mb-3">Recent Memories</h2>
      <div className="space-y-2">
        {recentData?.memories.map((m) => (
          <div key={m.id} className="bg-[var(--bg-secondary)] border border-[var(--border)] rounded-lg p-3 flex items-start gap-3">
            <span className={`text-xs px-2 py-0.5 rounded ${
              m.tier === 'hot' ? 'bg-[var(--hot)]/20 text-[var(--hot)]' :
              m.tier === 'cold' ? 'bg-[var(--cold)]/20 text-[var(--cold)]' :
              'bg-[var(--warm)]/20 text-[var(--warm)]'
            }`}>
              {m.tier}
            </span>
            <div className="flex-1 min-w-0">
              <div className="text-sm truncate">{m.summary}</div>
              <div className="text-xs text-[var(--text-muted)] mt-1">{m.topic} · {new Date(m.created_at).toLocaleDateString()}</div>
            </div>
            <div className="text-xs text-[var(--text-muted)] font-mono">{m.strength.toFixed(2)}</div>
          </div>
        ))}
      </div>
    </div>
  );
}
