import { useState } from 'react';
import { useStats, useRecent, useTopics, useActivity } from '../hooks/useApi';
import {
  AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid,
} from 'recharts';

const ACTIVITY_RANGES = [
  { label: '7d', days: 7 },
  { label: '14d', days: 14 },
  { label: '30d', days: 30 },
  { label: '90d', days: 90 },
] as const;

function StatCard({ label, value, color }: { label: string; value: string | number; color: string }) {
  return (
    <div className="bg-[var(--bg-secondary)] border border-[var(--border)] rounded-xl p-4">
      <div className="text-xs text-[var(--text-muted)] uppercase tracking-wider mb-1">{label}</div>
      <div className="text-2xl font-semibold font-mono" style={{ color }}>{typeof value === 'number' ? value.toLocaleString() : value}</div>
    </div>
  );
}

function ActivityTooltip({ active, payload, label }: { active?: boolean; payload?: Array<{ value: number; name?: string; color?: string }>; label?: string }) {
  if (!active || !payload?.length) return null;
  return (
    <div className="bg-[#1e293b] border border-[#334155] rounded-lg px-3 py-2 text-xs">
      <div className="text-[var(--text-secondary)] mb-1">{label}</div>
      {payload.map((p, i) => (
        <div key={i} style={{ color: p.color }}>
          {p.name}: {p.value}
        </div>
      ))}
    </div>
  );
}

export default function Dashboard() {
  const [activityDays, setActivityDays] = useState(14);
  const { data: stats, isLoading } = useStats();
  const { data: recentData } = useRecent(5);
  const { data: _topicsData } = useTopics();
  const { data: activityData } = useActivity(activityDays);

  if (isLoading || !stats) {
    return <div className="flex items-center justify-center h-full text-[var(--text-muted)]">Loading...</div>;
  }

  // Format activity data for chart (short date labels)
  const chartData = (activityData?.activity ?? []).map((d) => ({
    ...d,
    label: d.date.slice(5), // "MM-DD"
  }));

  return (
    <div className="p-6 max-w-6xl mx-auto">
      <h1 className="text-xl font-semibold mb-6">Dashboard</h1>

      {/* Activity chart */}
      {chartData.length > 0 && (
        <div className="bg-[var(--bg-secondary)] border border-[var(--border)] rounded-xl p-4 mb-6">
          <div className="flex items-center justify-between mb-3">
            <div className="text-xs text-[var(--text-muted)] uppercase tracking-wider">Activity ({activityDays} days)</div>
            <div className="flex gap-1">
              {ACTIVITY_RANGES.map((r) => (
                <button
                  key={r.days}
                  onClick={() => setActivityDays(r.days)}
                  className={`px-2 py-0.5 text-[10px] rounded transition-colors ${
                    activityDays === r.days
                      ? 'bg-[var(--accent)] text-white'
                      : 'text-[var(--text-muted)] hover:text-[var(--text-secondary)] hover:bg-[var(--bg-tertiary)]'
                  }`}
                >
                  {r.label}
                </button>
              ))}
            </div>
          </div>
          <ResponsiveContainer width="100%" height={160}>
            <AreaChart data={chartData} margin={{ top: 5, right: 10, left: 0, bottom: 0 }}>
              <defs>
                <linearGradient id="recallGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#7c3aed" stopOpacity={0.3} />
                  <stop offset="95%" stopColor="#7c3aed" stopOpacity={0} />
                </linearGradient>
                <linearGradient id="storeGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#4ade80" stopOpacity={0.3} />
                  <stop offset="95%" stopColor="#4ade80" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis dataKey="label" tick={{ fill: '#64748b', fontSize: 10 }} axisLine={false} tickLine={false} />
              <YAxis tick={{ fill: '#64748b', fontSize: 10 }} axisLine={false} tickLine={false} />
              <Tooltip content={<ActivityTooltip />} />
              <Area type="monotone" dataKey="recalls" name="Recalls" stroke="#7c3aed" fill="url(#recallGrad)" strokeWidth={2} />
              <Area type="monotone" dataKey="stores" name="Stores" stroke="#4ade80" fill="url(#storeGrad)" strokeWidth={2} />
            </AreaChart>
          </ResponsiveContainer>
          <div className="flex gap-4 mt-2 text-[10px]">
            <div className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-full" style={{ background: '#7c3aed' }} /><span className="text-[var(--text-secondary)]">Recalls</span></div>
            <div className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-full" style={{ background: '#4ade80' }} /><span className="text-[var(--text-secondary)]">Stores</span></div>
          </div>
        </div>
      )}

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
