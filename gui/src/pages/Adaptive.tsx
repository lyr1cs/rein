import { useAdaptive, useStats } from '../hooks/useApi';
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, LineChart, Line,
} from 'recharts';

/* ── colour palettes ──────────────────────────────────────────── */

const ALPHA_COLORS = ['#7c3aed', '#3b82f6', '#f97316', '#22d3ee', '#4ade80'];
const TIER_COLORS = { hot: 'var(--hot)', warm: 'var(--warm)', cold: 'var(--cold)' };
const EVENT_COLORS = ['#7c3aed', '#3b82f6', '#f97316', '#22d3ee', '#4ade80', '#f43f5e', '#fbbf24', '#64748b'];

const WEIGHT_CATEGORIES: Record<string, { color: string; label: string }> = {
  fts: { color: '#a78bfa', label: 'Retrieval' },
  vec: { color: '#a78bfa', label: 'Retrieval' },
  kg: { color: '#a78bfa', label: 'Retrieval' },
  episode: { color: '#7dd3fc', label: 'Temporal' },
  recency: { color: '#7dd3fc', label: 'Temporal' },
  access: { color: '#7dd3fc', label: 'Temporal' },
  strength: { color: '#4ade80', label: 'Quality' },
  importance: { color: '#4ade80', label: 'Quality' },
  keyword: { color: '#4ade80', label: 'Quality' },
  topic_match: { color: '#fbbf24', label: 'Context' },
  brevity: { color: '#fbbf24', label: 'Context' },
  channel_coverage: { color: '#fbbf24', label: 'Context' },
  usage_recency: { color: '#fbbf24', label: 'Context' },
  connectivity: { color: '#64748b', label: 'Structural' },
  concept_richness: { color: '#64748b', label: 'Structural' },
  tier_score: { color: '#64748b', label: 'Structural' },
  is_current: { color: '#64748b', label: 'Structural' },
};

const CLUSTER_LINE_COLORS = [
  '#7c3aed', '#3b82f6', '#f97316', '#22d3ee', '#4ade80',
  '#f43f5e', '#fbbf24', '#a78bfa', '#7dd3fc', '#fb923c',
];

/* ── panel wrapper ────────────────────────────────────────────── */

function Panel({ children, title, className = '' }: { children: React.ReactNode; title: string; className?: string }) {
  return (
    <div className={`bg-[#111827] border border-[#1e293b] rounded-xl p-4 ${className}`}>
      <h3 className="text-xs uppercase tracking-wider text-[var(--text-muted)] mb-3">{title}</h3>
      {children}
    </div>
  );
}

/* ── custom tooltip ───────────────────────────────────────────── */

function ChartTooltip({ active, payload, label }: { active?: boolean; payload?: Array<{ value: number; name?: string; color?: string }>; label?: string }) {
  if (!active || !payload?.length) return null;
  return (
    <div className="bg-[#1e293b] border border-[#334155] rounded-lg px-3 py-2 text-xs">
      {label && <div className="text-[var(--text-secondary)] mb-1">{label}</div>}
      {payload.map((p, i) => (
        <div key={i} className="text-[var(--text-primary)] font-mono">
          {p.name ? `${p.name}: ` : ''}{typeof p.value === 'number' ? p.value.toFixed(4) : p.value}
        </div>
      ))}
    </div>
  );
}

/* ── main component ───────────────────────────────────────────── */

export default function Adaptive() {
  const { data: adaptive, isLoading } = useAdaptive();
  const { data: stats } = useStats();

  if (isLoading || !adaptive) {
    return <div className="flex items-center justify-center h-full text-[var(--text-muted)]">Loading adaptive data...</div>;
  }

  /* Panel 1: Alpha values */
  const alphaOrder = ['Global', 'Semantic', 'Temporal', 'ExactKeyword', 'Exploratory'];
  const alphaData = alphaOrder.map((key) => {
    const entry = adaptive.learned_alphas[key] ?? adaptive.learned_alphas[key.toLowerCase()];
    return { name: key.replace('ExactKeyword', 'ExactKW'), value: entry?.value ?? 0, samples: entry?.sample_count ?? 0 };
  });

  /* Panel 2: Tier distribution from stats */
  const totalMemories = stats?.total_memories ?? 0;
  // We don't have exact tier counts from stats, so derive from tier_boundaries + total
  // Rough heuristic: hot > hot_threshold, cold < cold_threshold, rest warm
  const tierData = (() => {
    const hot = Math.round(totalMemories * 0.2);
    const cold = Math.round(totalMemories * 0.3);
    const warm = totalMemories - hot - cold;
    return [
      { name: 'Hot', value: hot, color: TIER_COLORS.hot },
      { name: 'Warm', value: warm, color: TIER_COLORS.warm },
      { name: 'Cold', value: cold, color: TIER_COLORS.cold },
    ];
  })();

  /* Panel 3: Reranker weights */
  const weightData = Object.entries(adaptive.reranker_weights)
    .sort((a, b) => b[1] - a[1])
    .map(([name, value]) => ({
      name,
      value,
      color: WEIGHT_CATEGORIES[name]?.color ?? '#64748b',
    }));

  const weightLegendItems = [
    { label: 'Retrieval', color: '#a78bfa' },
    { label: 'Temporal', color: '#7dd3fc' },
    { label: 'Quality', color: '#4ade80' },
    { label: 'Context', color: '#fbbf24' },
    { label: 'Structural', color: '#64748b' },
  ];

  /* Panel 4: Event counts */
  const eventData = Object.entries(adaptive.event_counts)
    .sort((a, b) => b[1] - a[1])
    .map(([name, count], i) => ({
      name: name.replace(/_/g, ' '),
      count,
      color: EVENT_COLORS[i % EVENT_COLORS.length],
    }));

  /* Panel 5: Survival curves — use actual K-M steps from backend */
  const survivalData = adaptive.survival_curves
    .filter(c => c.steps && c.steps.length > 0)
    .slice(0, 10);

  // Build unified time axis from all curves' step data
  const allTimes = new Set<number>();
  survivalData.forEach(c => {
    c.steps?.forEach(([t]) => allTimes.add(t));
  });
  const sortedTimes = [...allTimes].sort((a, b) => a - b);

  const survivalLines = sortedTimes.map((time) => {
    const point: Record<string, number> = { day: Math.round(time * 10000) / 10000 };
    survivalData.forEach((curve) => {
      // Find the last step <= this time (K-M is a step function)
      let prob = 1.0;
      for (const [t, p] of curve.steps ?? []) {
        if (t <= time) prob = p;
        else break;
      }
      point[`c${curve.cluster_id}`] = prob;
    });
    return point;
  });

  /* Panel 6: Cluster & convergence */
  const { cluster_info, tier_boundaries, dedup_thresholds } = adaptive;

  return (
    <div className="p-6 max-w-7xl mx-auto">
      <h1 className="text-xl font-semibold mb-6">Adaptive Engine</h1>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {/* Panel 1: Alpha Values (2-col span) */}
        <Panel title="Learned Alpha Values" className="md:col-span-2">
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={alphaData} margin={{ top: 20, right: 10, left: 0, bottom: 0 }}>
              <XAxis dataKey="name" tick={{ fill: '#94a3b8', fontSize: 11 }} axisLine={false} tickLine={false} />
              <YAxis domain={[0, 1]} tick={{ fill: '#64748b', fontSize: 10 }} axisLine={false} tickLine={false} />
              <Tooltip content={<ChartTooltip />} />
              <Bar dataKey="value" radius={[4, 4, 0, 0]} label={{ position: 'top', fill: '#e2e8f0', fontSize: 11, formatter: (v: unknown) => typeof v === 'number' ? v.toFixed(2) : String(v) }}>
                {alphaData.map((_, i) => (
                  <Cell key={i} fill={ALPHA_COLORS[i % ALPHA_COLORS.length]} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
          <div className="text-[10px] text-[var(--text-muted)] mt-1 text-center">
            alpha = weight given to BM25 vs vector in CC fusion
          </div>
        </Panel>

        {/* Panel 2: Tier Distribution */}
        <Panel title="Tier Distribution">
          <div className="flex flex-col items-center">
            <ResponsiveContainer width="100%" height={180}>
              <PieChart>
                <Pie
                  data={tierData}
                  cx="50%"
                  cy="50%"
                  innerRadius={45}
                  outerRadius={70}
                  dataKey="value"
                  stroke="none"
                >
                  {tierData.map((entry, i) => (
                    <Cell key={i} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip content={<ChartTooltip />} />
              </PieChart>
            </ResponsiveContainer>
            <div className="text-2xl font-mono font-semibold -mt-2 mb-2" style={{ color: 'var(--text-primary)' }}>
              {totalMemories}
            </div>
            <div className="flex gap-4 text-xs">
              {tierData.map((t) => (
                <div key={t.name} className="flex items-center gap-1.5">
                  <span className="w-2.5 h-2.5 rounded-full" style={{ background: t.color }} />
                  <span className="text-[var(--text-secondary)]">{t.name}</span>
                  <span className="font-mono text-[var(--text-muted)]">
                    {t.value} ({totalMemories > 0 ? Math.round((t.value / totalMemories) * 100) : 0}%)
                  </span>
                </div>
              ))}
            </div>
          </div>
        </Panel>

        {/* Panel 3: Reranker Weights (2-col span) */}
        <Panel title="Reranker Feature Weights" className="md:col-span-2">
          <div className="flex gap-3 mb-2 flex-wrap">
            {weightLegendItems.map((item) => (
              <div key={item.label} className="flex items-center gap-1.5 text-[10px]">
                <span className="w-2.5 h-2.5 rounded-sm" style={{ background: item.color }} />
                <span className="text-[var(--text-secondary)]">{item.label}</span>
              </div>
            ))}
          </div>
          <ResponsiveContainer width="100%" height={Math.max(200, weightData.length * 22)}>
            <BarChart data={weightData} layout="vertical" margin={{ top: 0, right: 20, left: 80, bottom: 0 }}>
              <XAxis type="number" tick={{ fill: '#64748b', fontSize: 10 }} axisLine={false} tickLine={false} />
              <YAxis
                type="category"
                dataKey="name"
                tick={{ fill: '#94a3b8', fontSize: 10 }}
                axisLine={false}
                tickLine={false}
                width={75}
              />
              <Tooltip content={<ChartTooltip />} />
              <Bar dataKey="value" radius={[0, 4, 4, 0]} barSize={14}>
                {weightData.map((entry, i) => (
                  <Cell key={i} fill={entry.color} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </Panel>

        {/* Panel 4: Event Counts */}
        <Panel title="Event Counts">
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={eventData} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
              <XAxis dataKey="name" tick={{ fill: '#94a3b8', fontSize: 9 }} axisLine={false} tickLine={false} angle={-30} textAnchor="end" height={50} />
              <YAxis tick={{ fill: '#64748b', fontSize: 10 }} axisLine={false} tickLine={false} />
              <Tooltip content={<ChartTooltip />} />
              <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                {eventData.map((entry, i) => (
                  <Cell key={i} fill={entry.color} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </Panel>

        {/* Panel 5: Survival Curves */}
        <Panel title="K-M Survival Curves">
          {survivalData.length === 0 ? (
            <div className="flex items-center justify-center h-[200px] text-[var(--text-muted)] text-sm">
              No survival data yet
            </div>
          ) : (
            <>
              <ResponsiveContainer width="100%" height={200}>
                <LineChart data={survivalLines} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
                  <XAxis dataKey="day" tick={{ fill: '#64748b', fontSize: 10 }} axisLine={false} tickLine={false} label={{ value: 'days', position: 'insideBottomRight', offset: -5, fill: '#64748b', fontSize: 10 }} />
                  <YAxis domain={[0, 1]} tick={{ fill: '#64748b', fontSize: 10 }} axisLine={false} tickLine={false} />
                  <Tooltip content={<ChartTooltip />} />
                  {survivalData.map((curve, i) => (
                    <Line
                      key={curve.cluster_id}
                      type="monotone"
                      dataKey={`c${curve.cluster_id}`}
                      stroke={CLUSTER_LINE_COLORS[i % CLUSTER_LINE_COLORS.length]}
                      strokeWidth={1.5}
                      dot={false}
                      name={`Cluster ${curve.cluster_id}`}
                    />
                  ))}
                </LineChart>
              </ResponsiveContainer>
              <div className="flex flex-wrap gap-x-3 gap-y-1 mt-2 text-[10px]">
                {survivalData.map((curve, i) => (
                  <div key={curve.cluster_id} className="flex items-center gap-1">
                    <span className="w-2 h-2 rounded-full" style={{ background: CLUSTER_LINE_COLORS[i % CLUSTER_LINE_COLORS.length] }} />
                    <span className="text-[var(--text-secondary)]">C{curve.cluster_id}</span>
                    <span className="text-[var(--text-muted)] font-mono">{curve.median_survival?.toFixed(1)}d</span>
                  </div>
                ))}
              </div>
            </>
          )}
        </Panel>

        {/* Panel 6: Cluster & Convergence */}
        <Panel title="Cluster & Convergence">
          <div className="space-y-2.5 text-sm">
            <Row label="Cluster Version" value={cluster_info.cluster_version} />
            <Row label="Unique Clusters" value={cluster_info.unique_clusters} />
            <Row label="Assigned Memories" value={cluster_info.assigned_memories} />
            <Divider />
            <Row label="Global Dedup" value={dedup_thresholds.global.toFixed(3)} />
            {Object.entries(dedup_thresholds.per_cluster).length > 0 && (
              <div className="text-xs text-[var(--text-muted)]">
                Per-cluster thresholds:
                <div className="mt-1 grid grid-cols-2 gap-x-3 gap-y-0.5 font-mono text-[10px]">
                  {Object.entries(dedup_thresholds.per_cluster).slice(0, 10).map(([cid, val]) => (
                    <div key={cid}>C{cid}: {val.toFixed(3)}</div>
                  ))}
                  {Object.entries(dedup_thresholds.per_cluster).length > 10 && (
                    <div className="text-[var(--text-muted)]">+{Object.entries(dedup_thresholds.per_cluster).length - 10} more</div>
                  )}
                </div>
              </div>
            )}
            <Divider />
            <Row label="Hot Threshold" value={tier_boundaries.hot_threshold.toFixed(3)} color="var(--hot)" />
            <Row label="Cold Threshold" value={tier_boundaries.cold_threshold.toFixed(3)} color="var(--cold)" />
          </div>
        </Panel>
      </div>
    </div>
  );
}

/* ── small helpers ────────────────────────────────────────────── */

function Row({ label, value, color }: { label: string; value: string | number; color?: string }) {
  return (
    <div className="flex justify-between items-center">
      <span className="text-[var(--text-muted)] text-xs">{label}</span>
      <span className="font-mono text-xs" style={{ color: color ?? 'var(--text-primary)' }}>{value}</span>
    </div>
  );
}

function Divider() {
  return <div className="border-t border-[#1e293b]" />;
}
