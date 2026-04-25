import { useMemo } from 'react';
import { useAdaptive, useStats } from '../hooks/useApi';
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, LineChart, Line, LabelList,
} from 'recharts';
import { ACCENT_PALETTE, TIER_COLORS } from '../utils/theme';
import type { AdaptiveStatusSynthesis } from '../api/types';

/* ── colour palettes ──────────────────────────────────────────── */

// `ALPHA_COLORS` is the shared 5-stop accent palette; `EVENT_COLORS` is
// page-local because it adds three more chart-specific stops on top.
const ALPHA_COLORS = ACCENT_PALETTE;
const EVENT_COLORS = [...ACCENT_PALETTE, '#f43f5e', '#fbbf24', '#64748b'];

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
    const normalizedKey = key === 'ExactKeyword' ? 'exact' : key.toLowerCase();
    const entry = adaptive.learned_alphas[key] ?? adaptive.learned_alphas[normalizedKey];
    return { name: key.replace('ExactKeyword', 'ExactKW'), value: entry?.value ?? 0, samples: entry?.sample_count ?? 0 };
  });

  /* Panel 2: Tier distribution from stats (real counts from backend) */
  const tierData = [
    { name: 'Hot', value: stats?.hot_count ?? 0, color: TIER_COLORS.hot },
    { name: 'Warm', value: stats?.warm_count ?? 0, color: TIER_COLORS.warm },
    { name: 'Cold', value: stats?.cold_count ?? 0, color: TIER_COLORS.cold },
  ];

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
  const clusterProfiles = [...(adaptive.cluster_profiles ?? [])]
    .sort((a, b) => b.memory_count - a.memory_count)
    .slice(0, 8);

  /* Panel 7-9: Synthesis quality (v0.26 D direction) */
  const synthesisProjection = adaptive.synthesis;

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
              {stats?.total_memories ?? 0}
            </div>
            <div className="flex gap-4 text-xs">
              {tierData.map((t) => {
                const total = stats?.total_memories ?? 0;
                return (
                <div key={t.name} className="flex items-center gap-1.5">
                  <span className="w-2.5 h-2.5 rounded-full" style={{ background: t.color }} />
                  <span className="text-[var(--text-secondary)]">{t.name}</span>
                  <span className="font-mono text-[var(--text-muted)]">
                    {t.value} ({total > 0 ? Math.round((t.value / total) * 100) : 0}%)
                  </span>
                </div>
                );
              })}
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

        <Panel title="Admission & Promotion Decisions" className="md:col-span-2">
          {clusterProfiles.length === 0 ? (
            <div className="text-sm text-[var(--text-muted)]">No cluster decision data yet.</div>
          ) : (
            <div className="space-y-3">
              {clusterProfiles.map((cluster) => (
                <div key={cluster.cluster_id} className="rounded-lg border border-[#1e293b] bg-[#0b1220] p-3">
                  <div className="flex items-center justify-between mb-2">
                    <div className="text-xs uppercase tracking-wider text-[var(--text-secondary)]">
                      Cluster {cluster.cluster_id}
                    </div>
                    <div className="text-[10px] font-mono text-[var(--text-muted)]">
                      {cluster.memory_count} memories
                    </div>
                  </div>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-xs">
                    <Metric label="Avg strength" value={cluster.avg_strength.toFixed(3)} color="var(--success)" />
                    <Metric label="Dedup" value={cluster.dedup_threshold.toFixed(3)} color="var(--accent)" />
                    <Metric label="Admission" value={cluster.admission_threshold.toFixed(3)} color="var(--warm)" />
                    <Metric label="Promote @ access" value={cluster.promotion_threshold} color="var(--hot)" />
                  </div>
                  {cluster.median_survival != null && (
                    <div className="mt-2 text-[10px] text-[var(--text-muted)]">
                      Median survival: <span className="font-mono text-[var(--text-secondary)]">{cluster.median_survival.toFixed(1)}d</span>
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </Panel>

        {/* ── Synthesis Quality (v0.26 D direction) ──────────────────
            Observability surface for the `synthesis_feedback` consumer.
            Reads `adaptive.synthesis` (added in v0.26.0 by B_REST_MCP) and
            renders three small charts + a global summary. The whole
            block stays empty + shows a cold-start hint when no events
            have been recorded yet — which is the expected v0.26.0 state
            since the feature ships record-only and `[ars].recall_synthesis_enabled`
            stays `false` by default. */}
        <SynthesisQualitySection projection={synthesisProjection} />
      </div>
    </div>
  );
}

/* ── Synthesis Quality Panel (v0.26 D direction) ──────────────────── */

/**
 * Per-query-type chip color. Mirrors the autonomous-router taxonomy from
 * `search/classify.rs` (Episodic/Temporal/Preference/ExactKeyword/Semantic/
 * Exploratory). Unknown query_types fall back to the muted slate so the
 * chart legend never breaks on a backend that adds a new variant before
 * this table is updated.
 */
const QUERY_TYPE_COLORS: Record<string, string> = {
  Episodic: '#7c3aed',
  Temporal: '#7dd3fc',
  Preference: '#4ade80',
  ExactKeyword: '#fbbf24',
  Semantic: '#22d3ee',
  Exploratory: '#a78bfa',
  _global: '#64748b',
};

/**
 * Bin a list of dwell p50s into a discrete histogram so the chart axis
 * stays interpretable when clusters span 100ms → 30s. Buckets are
 * coarse-log so a glance shows "most synthesis dwells are sub-2s vs
 * mostly past-5s" without needing y-axis math.
 */
const DWELL_BUCKETS: Array<{ label: string; max: number }> = [
  { label: '<1s', max: 1000 },
  { label: '1-2s', max: 2000 },
  { label: '2-5s', max: 5000 },
  { label: '5-10s', max: 10000 },
  { label: '10s+', max: Number.POSITIVE_INFINITY },
];

function SynthesisQualitySection({
  projection,
}: {
  projection?: AdaptiveStatusSynthesis;
}) {
  // Normalize the per-cluster array — when the projection is undefined OR
  // both the bucket array AND the global rollup are empty we render a
  // dedicated cold-start state instead of empty charts.
  //
  // Memoized: a `??` fallback to a fresh `[]` on every render would
  // change identity and re-fire every downstream useMemo (queryTypes,
  // usefulBars, clickBars, dwellHistogram), so we pin the empty array
  // to one reference per polling tick.
  const byCluster = useMemo(
    () => projection?.by_cluster ?? [],
    [projection?.by_cluster],
  );
  const global = projection?.global ?? null;
  const hasData = byCluster.length > 0 || global !== null;

  // Distinct query types in the data — used for both the bar fill and the
  // legend chips. Stable order = first-seen (matches server projection).
  const queryTypes = useMemo(() => {
    const seen: string[] = [];
    for (const row of byCluster) {
      if (!seen.includes(row.query_type)) seen.push(row.query_type);
    }
    return seen;
  }, [byCluster]);

  const usefulBars = useMemo(
    () =>
      byCluster.map((row) => ({
        bucket: `C${row.cluster_id}/${row.query_type === '_global' ? 'global' : row.query_type}`,
        cluster_id: row.cluster_id,
        query_type: row.query_type,
        useful_rate: row.useful_rate,
        viewed_count: row.viewed_count,
      })),
    [byCluster],
  );

  const clickBars = useMemo(
    () =>
      byCluster.map((row) => ({
        bucket: `C${row.cluster_id}/${row.query_type === '_global' ? 'global' : row.query_type}`,
        clicked_source_rate: row.clicked_source_rate,
        query_type: row.query_type,
      })),
    [byCluster],
  );

  const dwellHistogram = useMemo(() => {
    const buckets = DWELL_BUCKETS.map((b) => ({ label: b.label, count: 0 }));
    for (const row of byCluster) {
      if (row.viewed_dwell_p50_ms === null) continue;
      const idx = DWELL_BUCKETS.findIndex(
        (b) => (row.viewed_dwell_p50_ms ?? 0) <= b.max,
      );
      if (idx >= 0) buckets[idx].count += 1;
    }
    return buckets;
  }, [byCluster]);

  if (!hasData) {
    return (
      <Panel title="Synthesis Quality" className="md:col-span-3">
        <div className="rounded-lg border border-dashed border-[#1e293b] bg-[#0b1220] p-6 text-center">
          <div className="text-sm text-[var(--text-secondary)] mb-1.5">
            Awaiting traffic
          </div>
          <div className="text-xs text-[var(--text-muted)] max-w-prose mx-auto leading-relaxed">
            Cap B (recall-time synthesis) may not be enabled yet, or no
            <code className="mx-1 px-1 py-0.5 rounded bg-[var(--bg-secondary)] text-[var(--text-secondary)] font-mono text-[10px]">
              SynthesisInteraction
            </code>
            events have been recorded. Toggle{' '}
            <code className="mx-1 px-1 py-0.5 rounded bg-[var(--bg-secondary)] text-[var(--text-secondary)] font-mono text-[10px]">
              [ars].recall_synthesis_enabled = true
            </code>{' '}
            and use the Synthesis Lab page to start producing feedback
            signals.
          </div>
        </div>
      </Panel>
    );
  }

  return (
    <>
      {/* Global summary — single big number + last consumed event watermark.
          When `global` is null but per-cluster data exists (degenerate
          backend), we show "—" rather than panicking. */}
      <Panel title="Synthesis Quality (Global)" className="md:col-span-1">
        <div className="flex flex-col items-start gap-1.5">
          <div className="text-[10px] uppercase tracking-wider text-[var(--text-muted)]">
            Useful Rate
          </div>
          <div className="text-3xl font-mono text-[var(--text-primary)]">
            {global !== null ? (global.useful_rate * 100).toFixed(0) + '%' : '—'}
          </div>
          {global !== null && (
            <>
              <div className="mt-2 text-[10px] text-[var(--text-muted)]">
                Total events
              </div>
              <div className="font-mono text-sm text-[var(--text-secondary)]">
                {global.total_events.toLocaleString()}
              </div>
              <div className="mt-2 text-[10px] text-[var(--text-muted)]">
                Last event id
              </div>
              <div className="font-mono text-xs text-[var(--text-muted)]">
                #{global.last_consumed_event_id.toLocaleString()}
              </div>
            </>
          )}
        </div>
      </Panel>

      {/* Per-cluster useful_rate bars. Color-coded by query_type so the
          eye can scan "Episodic clusters are doing well, Temporal less
          so" in one glance. Threshold of 0.5 is the bootstrap decision
          boundary — bars above the line vote-in synthesis for that
          cluster, bars below vote-out. */}
      <Panel title="Useful Rate (per cluster)" className="md:col-span-2">
        <div className="flex gap-3 mb-2 flex-wrap">
          {queryTypes.map((qt) => (
            <div key={qt} className="flex items-center gap-1.5 text-[10px]">
              <span
                className="w-2.5 h-2.5 rounded-sm"
                style={{
                  background: QUERY_TYPE_COLORS[qt] ?? QUERY_TYPE_COLORS._global,
                }}
              />
              <span className="text-[var(--text-secondary)]">
                {qt === '_global' ? 'global' : qt}
              </span>
            </div>
          ))}
        </div>
        <ResponsiveContainer width="100%" height={Math.max(180, byCluster.length * 22)}>
          <BarChart data={usefulBars} layout="vertical" margin={{ top: 0, right: 30, left: 60, bottom: 0 }}>
            <XAxis
              type="number"
              domain={[0, 1]}
              tick={{ fill: '#64748b', fontSize: 10 }}
              axisLine={false}
              tickLine={false}
            />
            <YAxis
              type="category"
              dataKey="bucket"
              tick={{ fill: '#94a3b8', fontSize: 10 }}
              axisLine={false}
              tickLine={false}
              width={55}
            />
            <Tooltip content={<ChartTooltip />} />
            <Bar dataKey="useful_rate" radius={[0, 4, 4, 0]} barSize={14}>
              <LabelList
                dataKey="viewed_count"
                position="right"
                formatter={(value: unknown) =>
                  typeof value === 'number' ? `n=${value}` : ''
                }
                fill="#64748b"
                fontSize={10}
              />
              {usefulBars.map((entry, i) => (
                <Cell
                  key={i}
                  fill={
                    QUERY_TYPE_COLORS[entry.query_type] ?? QUERY_TYPE_COLORS._global
                  }
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
        <div className="text-[10px] text-[var(--text-muted)] mt-1 text-center">
          Bootstrap threshold: 0.5 — clusters below this line route to the
          global synthesis flag in `decide_synthesize`
        </div>
      </Panel>

      {/* Dwell histogram — coarse log buckets (sub-1s → 10s+). Shape
          tells operator whether users are skimming or actually reading
          the synthesis. */}
      <Panel title="Dwell p50 distribution" className="md:col-span-1">
        <ResponsiveContainer width="100%" height={180}>
          <BarChart data={dwellHistogram} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
            <XAxis
              dataKey="label"
              tick={{ fill: '#94a3b8', fontSize: 10 }}
              axisLine={false}
              tickLine={false}
            />
            <YAxis tick={{ fill: '#64748b', fontSize: 10 }} axisLine={false} tickLine={false} allowDecimals={false} />
            <Tooltip content={<ChartTooltip />} />
            <Bar dataKey="count" radius={[4, 4, 0, 0]} fill={ACCENT_PALETTE[0]} />
          </BarChart>
        </ResponsiveContainer>
        <div className="text-[10px] text-[var(--text-muted)] mt-1 text-center">
          Cluster count by dwell-p50 bucket
        </div>
      </Panel>

      {/* Click-through rate per cluster. Different colour palette so it
          reads as a separate signal from useful_rate even when both
          panels stack. */}
      <Panel title="Click-through rate (per cluster)" className="md:col-span-2">
        <ResponsiveContainer width="100%" height={Math.max(180, byCluster.length * 22)}>
          <BarChart data={clickBars} layout="vertical" margin={{ top: 0, right: 20, left: 60, bottom: 0 }}>
            <XAxis
              type="number"
              domain={[0, 1]}
              tick={{ fill: '#64748b', fontSize: 10 }}
              axisLine={false}
              tickLine={false}
            />
            <YAxis
              type="category"
              dataKey="bucket"
              tick={{ fill: '#94a3b8', fontSize: 10 }}
              axisLine={false}
              tickLine={false}
              width={55}
            />
            <Tooltip content={<ChartTooltip />} />
            <Bar dataKey="clicked_source_rate" radius={[0, 4, 4, 0]} barSize={14} fill={ACCENT_PALETTE[2]} />
          </BarChart>
        </ResponsiveContainer>
        <div className="text-[10px] text-[var(--text-muted)] mt-1 text-center">
          Source citations clicked per viewed synthesis
        </div>
      </Panel>
    </>
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

function Metric({ label, value, color }: { label: string; value: string | number; color?: string }) {
  return (
    <div className="rounded-md bg-[#111827] px-2 py-2">
      <div className="text-[10px] uppercase tracking-wider text-[var(--text-muted)]">{label}</div>
      <div className="font-mono text-sm" style={{ color: color ?? 'var(--text-primary)' }}>
        {value}
      </div>
    </div>
  );
}
