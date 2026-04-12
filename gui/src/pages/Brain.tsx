import { useState, useEffect, useCallback, useRef, useMemo } from 'react';
import ForceGraph2D from 'react-force-graph-2d';
import type { ForceGraphMethods, LinkObject, NodeObject } from 'react-force-graph-2d';
import { apiGet } from '../api/client';
import { useMemoryDetail } from '../hooks/useApi';
import type { Memory, Concept, ConceptLink } from '../api/types';

/* ------------------------------------------------------------------ */
/*  Types                                                              */
/* ------------------------------------------------------------------ */

interface Memoir {
  id: string;
  name: string;
  description: string;
}

interface MemoirExport {
  memoir: Memoir;
  concepts: Concept[];
  links: ConceptLink[];
}

interface GraphNode {
  id: string;
  type: 'memory' | 'concept';
  label: string;
  tier?: 'hot' | 'warm' | 'cold';
  strength?: number;
  importance?: string;
  confidence?: number;
  cluster_id?: number | null;
  created_at: string;
  /* runtime – assigned by force-graph */
  x?: number;
  y?: number;
}

interface GraphLink {
  source: string;
  target: string;
  type: 'related' | 'concept_link' | 'memory_concept';
}

interface GraphData {
  nodes: GraphNode[];
  links: GraphLink[];
}

type BrainNode = NodeObject<GraphNode>;
type BrainLink = LinkObject<GraphNode, GraphLink>;
type GraphHandle = ForceGraphMethods<GraphNode, GraphLink> & { refresh?: () => void };
type LinkEndpoint = string | number | BrainNode | undefined;

function endpointId(endpoint: LinkEndpoint): string | null {
  if (endpoint == null) return null;
  if (typeof endpoint === 'object') {
    return endpoint.id == null ? null : String(endpoint.id);
  }
  return String(endpoint);
}

function endpointNode(endpoint: LinkEndpoint): BrainNode | null {
  return typeof endpoint === 'object' && endpoint !== null ? endpoint : null;
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : 'Failed to load data';
}

/* ------------------------------------------------------------------ */
/*  Tier colors                                                        */
/* ------------------------------------------------------------------ */

const TIER_COLORS: Record<string, string> = {
  hot: '#f97316',
  warm: '#fbbf24',
  cold: '#3b82f6',
  concept: '#e2e8f0',
};

/* ------------------------------------------------------------------ */
/*  Component                                                          */
/* ------------------------------------------------------------------ */

export default function Brain() {
  /* Data state */
  const [graphData, setGraphData] = useState<GraphData>({ nodes: [], links: [] });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  /* Interaction state */
  const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null);
  const [hoveredNode, setHoveredNode] = useState<GraphNode | null>(null);
  const [search, setSearch] = useState('');
  const [timeMax, setTimeMax] = useState(0);
  const [timeSlider, setTimeSlider] = useState(0);
  const { data: selectedMemoryDetail, isLoading: selectedMemoryLoading } = useMemoryDetail(
    selectedNode?.type === 'memory' ? selectedNode.id : null,
  );

  /* Refs */
  const fgRef = useRef<GraphHandle | undefined>(undefined);
  const containerRef = useRef<HTMLDivElement>(null);
  const [dimensions, setDimensions] = useState({ width: window.innerWidth - 48, height: window.innerHeight - 40 });


  /* ---- Fetch all data on mount ---- */
  useEffect(() => {
    let cancelled = false;

    async function load() {
      try {
        /* Fetch memories and memoirs in parallel */
        const [recentRes, memoirsRes] = await Promise.all([
          apiGet<{ memories: Memory[] }>('/api/recent?limit=200'),
          apiGet<{ memoirs: Memoir[] }>('/api/memoirs').catch(() => ({ memoirs: [] as Memoir[] })),
        ]);

        /* Fetch all memoir exports */
        const exports: MemoirExport[] = [];
        for (const m of memoirsRes.memoirs) {
          try {
            const exp = await apiGet<MemoirExport>(
              `/api/memoirs/${encodeURIComponent(m.name)}/export?format=json`,
            );
            exports.push(exp);
          } catch {
            /* skip failed memoir */
          }
        }

        if (cancelled) return;

        /* Build nodes */
        const nodeMap = new Map<string, GraphNode>();
        const memories = recentRes.memories;

        for (const mem of memories) {
          nodeMap.set(mem.id, {
            id: mem.id,
            type: 'memory',
            label: mem.summary || mem.topic || mem.id.slice(0, 8),
            tier: mem.tier,
            strength: mem.strength,
            importance: mem.importance,
            cluster_id: mem.cluster_id,
            created_at: mem.created_at,
          });
        }

        for (const exp of exports) {
          for (const c of exp.concepts) {
            if (!nodeMap.has(c.id)) {
              nodeMap.set(c.id, {
                id: c.id,
                type: 'concept',
                label: c.name,
                confidence: c.confidence,
                created_at: c.created_at || c.updated_at || new Date().toISOString(),
              });
            }
          }
        }

        /* Build links */
        const linkSet = new Set<string>();
        const links: GraphLink[] = [];

        function addLink(src: string, tgt: string, type: GraphLink['type']) {
          if (!nodeMap.has(src) || !nodeMap.has(tgt)) return;
          if (src === tgt) return;
          const key = `${src}|${tgt}|${type}`;
          if (linkSet.has(key)) return;
          linkSet.add(key);
          links.push({ source: src, target: tgt, type });
        }

        /* Memory related_ids */
        for (const mem of memories) {
          for (const rid of mem.related_ids) {
            addLink(mem.id, rid, 'related');
          }
          for (const cid of mem.concept_ids) {
            addLink(mem.id, cid, 'memory_concept');
          }
        }

        /* ConceptLink entries */
        for (const exp of exports) {
          for (const cl of exp.links) {
            addLink(cl.source_id, cl.target_id, 'concept_link');
          }
        }

        const nodes = Array.from(nodeMap.values());

        const now = Date.now();

        setGraphData({ nodes, links });
        setTimeMax(now);
        setTimeSlider(now);
        setLoading(false);
      } catch (err: unknown) {
        if (!cancelled) {
          setError(errorMessage(err));
          setLoading(false);
        }
      }
    }

    load();
    return () => { cancelled = true; };
  }, []);

  /* ---- Fit view on first render ---- */
  useEffect(() => {
    if (!loading && graphData.nodes.length > 0 && fgRef.current) {
      setTimeout(() => {
        fgRef.current?.zoomToFit?.(400, 60);
      }, 500);
    }
  }, [loading, graphData.nodes.length]);

  /* ---- Resize observer ---- */
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    // Set initial dimensions from actual container
    setDimensions({ width: el.clientWidth, height: el.clientHeight });
    const ro = new ResizeObserver((entries) => {
      for (const e of entries) {
        const w = Math.floor(e.contentRect.width);
        const h = Math.floor(e.contentRect.height);
        if (w > 0 && h > 0) {
          setDimensions({ width: w, height: h });
        }
      }
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  /* ---- Animation loop for hot node pulse ---- */
  useEffect(() => {
    /* Throttle to ~20fps to avoid burning CPU */
    const interval = setInterval(() => {
      fgRef.current?.refresh?.();
    }, 50);
    return () => {
      clearInterval(interval);
    };
  }, []);

  /* ---- Time-filtered data ---- */
  const filteredData = useMemo(() => {
    const cutoff = timeSlider || Number.POSITIVE_INFINITY;
    const visibleIds = new Set<string>();
    const nodes = graphData.nodes.filter((n) => {
      const t = new Date(n.created_at).getTime();
      if (t <= cutoff) {
        visibleIds.add(n.id);
        return true;
      }
      return false;
    });
    const links = graphData.links.filter((l) => {
      const src = endpointId(l.source);
      const tgt = endpointId(l.target);
      return src !== null && tgt !== null && visibleIds.has(src) && visibleIds.has(tgt);
    });
    return { nodes, links };
  }, [graphData, timeSlider]);

  /* ---- Search matching ---- */
  const searchLower = search.toLowerCase();
  const matchingIds = useMemo(() => {
    if (!searchLower) return null;
    return new Set(
      filteredData.nodes
        .filter((n) => n.label.toLowerCase().includes(searchLower))
        .map((n) => n.id),
    );
  }, [searchLower, filteredData.nodes]);

  /* ---- Connected ids for hover highlight ---- */
  const connectedIds = useMemo(() => {
    if (!hoveredNode) return null;
    const ids = new Set<string>();
    ids.add(hoveredNode.id);
    for (const l of filteredData.links) {
      const src = endpointId(l.source);
      const tgt = endpointId(l.target);
      if (src === hoveredNode.id && tgt !== null) ids.add(tgt);
      if (tgt === hoveredNode.id && src !== null) ids.add(src);
    }
    return ids;
  }, [hoveredNode, filteredData.links]);

  /* ---- Time range ---- */
  const timeMin = useMemo(() => {
    if (graphData.nodes.length === 0) return timeMax;
    let oldest = Number.POSITIVE_INFINITY;
    for (const n of graphData.nodes) {
      const t = new Date(n.created_at).getTime();
      if (t < oldest) oldest = t;
    }
    return Number.isFinite(oldest) ? oldest : timeMax;
  }, [graphData.nodes, timeMax]);

  /* ---- Node canvas renderer ---- */
  const paintNode = useCallback(
    (node: BrainNode, ctx: CanvasRenderingContext2D, globalScale: number) => {
      const n = node;
      const x = node.x as number;
      const y = node.y as number;
      if (x == null || y == null) return;

      const isMemory = n.type === 'memory';
      const tier = n.tier || 'cold';
      const baseR = isMemory
        ? 2 + (n.strength ?? 0.5) * 3
        : 2 + (n.confidence ?? 0.5) * 3;

      /* Dimming logic */
      const isSearchMatch = matchingIds === null || matchingIds.has(n.id);
      const isHoverConnected = connectedIds === null || connectedIds.has(n.id);
      const dimmed = (matchingIds !== null && !isSearchMatch) || (connectedIds !== null && !isHoverConnected);

      /* Search glow override */
      const isSearchHighlight = matchingIds !== null && matchingIds.has(n.id);

      /* ---- Outer glow ring ---- */
      if (!dimmed) {
        let glowColor: string;
        let glowRadius: number;
        let glowAlpha: number;

        if (isSearchHighlight) {
          glowColor = '#22d3ee';
          glowRadius = baseR * 1.8;
          glowAlpha = 0.5;
        } else if (isMemory) {
          if (tier === 'hot') {
            const pulse = 0.25 + 0.08 * Math.sin(Date.now() * 0.003);
            glowColor = TIER_COLORS.hot;
            glowRadius = baseR * 1.6;
            glowAlpha = pulse;
          } else if (tier === 'warm') {
            glowColor = TIER_COLORS.warm;
            glowRadius = baseR * 1.4;
            glowAlpha = 0.2;
          } else {
            glowColor = TIER_COLORS.cold;
            glowRadius = baseR * 1.3;
            glowAlpha = 0.15;
          }
        } else {
          glowColor = TIER_COLORS.concept;
          glowRadius = baseR * 1.5;
          glowAlpha = 0.2;
        }

        ctx.beginPath();
        ctx.arc(x, y, glowRadius, 0, 2 * Math.PI);
        ctx.fillStyle = glowColor;
        ctx.globalAlpha = glowAlpha;
        ctx.fill();
        ctx.globalAlpha = 1;
      }

      /* ---- Core shape ---- */
      const coreColor = dimmed
        ? 'rgba(100,116,139,0.2)'
        : isSearchHighlight
          ? '#22d3ee'
          : isMemory
            ? TIER_COLORS[tier] || TIER_COLORS.cold
            : TIER_COLORS.concept;

      if (isMemory) {
        /* Circle */
        ctx.beginPath();
        ctx.arc(x, y, baseR, 0, 2 * Math.PI);
        ctx.fillStyle = coreColor;
        ctx.fill();
      } else {
        /* Diamond (rotated square) */
        const s = baseR * 0.9;
        ctx.beginPath();
        ctx.moveTo(x, y - s);
        ctx.lineTo(x + s, y);
        ctx.lineTo(x, y + s);
        ctx.lineTo(x - s, y);
        ctx.closePath();
        ctx.fillStyle = coreColor;
        ctx.fill();
      }

      /* ---- Label ---- */
      if (globalScale > 1.5) {
        const fontSize = Math.min(12 / globalScale, 4);
        ctx.font = `${fontSize}px sans-serif`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'top';
        ctx.fillStyle = dimmed ? 'rgba(148,163,184,0.3)' : '#e2e8f0';
        const truncated = n.label.length > 20 ? n.label.slice(0, 20) + '...' : n.label;
        ctx.fillText(truncated, x, y + baseR + 1);
      }
    },
    [matchingIds, connectedIds],
  );

  /* ---- Pointer area ---- */
  const paintNodeArea = useCallback(
    (node: BrainNode, color: string, ctx: CanvasRenderingContext2D) => {
      const n = node;
      const r = 2 + ((n.type === 'memory' ? n.strength : n.confidence) ?? 0.5) * 3 + 2;
      ctx.beginPath();
      ctx.arc(node.x ?? 0, node.y ?? 0, r, 0, 2 * Math.PI);
      ctx.fillStyle = color;
      ctx.fill();
    },
    [],
  );

  /* ---- Link renderer ---- */
  const paintLink = useCallback(
    (link: BrainLink, ctx: CanvasRenderingContext2D, globalScale: number) => {
      const l = link;
      const src = endpointNode(l.source);
      const tgt = endpointNode(l.target);
      if (!src || !tgt) return;
      const srcX = src.x;
      const srcY = src.y;
      const tgtX = tgt.x;
      const tgtY = tgt.y;
      if (srcX == null || srcY == null || tgtX == null || tgtY == null) return;

      ctx.beginPath();
      ctx.moveTo(srcX, srcY);
      ctx.lineTo(tgtX, tgtY);
      const alpha = l.type === 'memory_concept' ? 0.4 : 0.3;
      ctx.strokeStyle = `rgba(124, 58, 237, ${alpha})`;
      ctx.lineWidth = 0.5 / globalScale;
      ctx.stroke();
    },
    [],
  );

  /* ---- Detail panel: find memory/concept data ---- */
  const selectedMemory = useMemo(() => {
    if (!selectedNode || selectedNode.type !== 'memory') return null;
    return selectedMemoryDetail?.memory ?? null;
  }, [selectedNode, selectedMemoryDetail]);

  const selectedConcept = useMemo(() => {
    if (!selectedNode || selectedNode.type !== 'concept') return null;
    return selectedNode;
  }, [selectedNode]);

  /* ---- Zoom controls ---- */
  const handleZoomIn = useCallback(() => {
    if (!fgRef.current) return;
    const scale = fgRef.current.zoom?.() ?? 1;
    fgRef.current.zoom?.(Math.min(scale * 1.4, 20));
  }, []);

  const handleZoomOut = useCallback(() => {
    if (!fgRef.current) return;
    const scale = fgRef.current.zoom?.() ?? 1;
    fgRef.current.zoom?.(Math.max(scale / 1.4, 0.1));
  }, []);

  const handleReset = useCallback(() => {
    fgRef.current?.zoomToFit?.(400, 60);
  }, []);

  /* ---- Slider date label ---- */
  const sliderDateLabel = useMemo(() => {
    return new Date(timeSlider).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
    });
  }, [timeSlider]);

  /* ---------------------------------------------------------------- */
  /*  Loading / Error states                                           */
  /* ---------------------------------------------------------------- */

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full text-[var(--text-muted)]">
        Loading neural map...
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center justify-center h-full text-[var(--text-muted)]">
        {error}
      </div>
    );
  }

  if (graphData.nodes.length === 0) {
    return (
      <div className="flex items-center justify-center h-full text-[var(--text-muted)]">
        <div className="text-center">
          <div className="text-lg mb-2">No memories or concepts found</div>
          <div className="text-sm">
            Store some memories with{' '}
            <span className="font-mono text-[var(--accent)]">rein_store</span> first.
          </div>
        </div>
      </div>
    );
  }

  /* ---------------------------------------------------------------- */
  /*  Render                                                           */
  /* ---------------------------------------------------------------- */

  return (
    <div className="relative w-full h-full overflow-hidden">
      {/* Graph container */}
      <div ref={containerRef} className="absolute inset-0">
        <ForceGraph2D<GraphNode, GraphLink>
          ref={fgRef}
          width={dimensions.width}
          height={dimensions.height}
          graphData={filteredData}
          backgroundColor="transparent"
          nodeCanvasObject={paintNode}
          nodeCanvasObjectMode={() => 'replace'}
          nodePointerAreaPaint={paintNodeArea}
          linkCanvasObject={paintLink}
          linkCanvasObjectMode={() => 'replace'}
          onNodeClick={(node) => setSelectedNode(node)}
          onNodeHover={(node) => setHoveredNode(node)}
          onBackgroundClick={() => setSelectedNode(null)}
          d3AlphaDecay={0.02}
          d3VelocityDecay={0.3}
          cooldownTime={4000}
          enableNodeDrag
        />
      </div>

      {/* Background gradient overlay (behind everything but the canvas handles its own bg) */}
      <div
        className="absolute inset-0 pointer-events-none"
        style={{
          background: 'radial-gradient(ellipse at 50% 50%, #0f172a 0%, #020617 70%)',
          zIndex: -1,
        }}
      />

      {/* Search (top-left) */}
      <div className="absolute top-4 left-4 z-10 flex items-center gap-2">
        <input
          type="text"
          placeholder="Search nodes..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          className="bg-[#0f172a]/80 backdrop-blur border border-[var(--border)] rounded-lg px-3 py-1.5 text-sm text-[var(--text-primary)] w-56 outline-none focus:border-[var(--accent)] placeholder:text-[var(--text-muted)]"
        />
        {search && (
          <button
            onClick={() => setSearch('')}
            className="text-xs text-[var(--text-muted)] hover:text-[var(--text-primary)] bg-[#0f172a]/80 backdrop-blur border border-[var(--border)] rounded-lg px-2 py-1.5"
          >
            Clear
          </button>
        )}
      </div>

      {/* Zoom controls (bottom-right) */}
      <div className="absolute bottom-20 right-4 z-10 flex flex-col gap-1">
        <button
          onClick={handleZoomIn}
          className="w-8 h-8 bg-[#0f172a]/80 backdrop-blur border border-[var(--border)] rounded-lg text-[var(--text-primary)] hover:border-[var(--accent)] text-sm font-mono flex items-center justify-center"
        >
          +
        </button>
        <button
          onClick={handleZoomOut}
          className="w-8 h-8 bg-[#0f172a]/80 backdrop-blur border border-[var(--border)] rounded-lg text-[var(--text-primary)] hover:border-[var(--accent)] text-sm font-mono flex items-center justify-center"
        >
          -
        </button>
        <button
          onClick={handleReset}
          className="w-8 h-8 bg-[#0f172a]/80 backdrop-blur border border-[var(--border)] rounded-lg text-[var(--text-primary)] hover:border-[var(--accent)] text-xs flex items-center justify-center"
          title="Reset view"
        >
          {'\u27F2'}
        </button>
      </div>

      {/* Legend (bottom-left) */}
      <div className="absolute bottom-20 left-4 z-10 bg-[#0f172a]/80 backdrop-blur border border-[var(--border)] rounded-lg p-3 text-xs space-y-1.5">
        <div className="flex items-center gap-2">
          <span className="w-2.5 h-2.5 rounded-full inline-block" style={{ backgroundColor: '#f97316' }} />
          <span className="text-[var(--text-secondary)]">Hot</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="w-2.5 h-2.5 rounded-full inline-block" style={{ backgroundColor: '#fbbf24' }} />
          <span className="text-[var(--text-secondary)]">Warm</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="w-2.5 h-2.5 rounded-full inline-block" style={{ backgroundColor: '#3b82f6' }} />
          <span className="text-[var(--text-secondary)]">Cold</span>
        </div>
        <div className="flex items-center gap-2">
          <span
            className="inline-block w-2.5 h-2.5"
            style={{
              backgroundColor: '#e2e8f0',
              transform: 'rotate(45deg)',
              borderRadius: '1px',
            }}
          />
          <span className="text-[var(--text-secondary)]">Concept</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="w-2.5 h-2.5 rounded-full inline-block" style={{ backgroundColor: '#22d3ee' }} />
          <span className="text-[var(--text-secondary)]">Match</span>
        </div>
      </div>

      {/* Time slider (bottom, full width) */}
      <div className="absolute bottom-0 left-0 right-0 z-10 px-4 pb-3 pt-1 bg-gradient-to-t from-[#020617]/90 to-transparent">
        <div className="flex items-center gap-3">
          <span className="text-[10px] text-[var(--text-muted)] font-mono whitespace-nowrap">
            {new Date(timeMin).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: '2-digit' })}
          </span>
          <input
            type="range"
            min={timeMin}
            max={timeMax}
            value={timeSlider}
            onChange={(e) => setTimeSlider(Number(e.target.value))}
            className="flex-1 h-1 accent-[var(--accent)] cursor-pointer"
          />
          <span className="text-[10px] text-[var(--text-muted)] font-mono whitespace-nowrap">
            {sliderDateLabel}
          </span>
        </div>
      </div>

      {/* Detail panel (right side) */}
      {selectedNode && (
        <div className="absolute top-0 right-0 bottom-14 w-[280px] z-20 border-l border-[var(--border)] bg-[#0f172a]/95 backdrop-blur overflow-y-auto p-4">
          {/* Close button */}
          <div className="flex items-start justify-between mb-3">
            <h3 className="text-sm font-semibold text-[var(--text-primary)] break-words leading-snug pr-2">
              {selectedNode.label}
            </h3>
            <button
              onClick={() => setSelectedNode(null)}
              className="text-[var(--text-muted)] hover:text-[var(--text-primary)] ml-1 shrink-0 text-sm"
            >
              x
            </button>
          </div>

          {/* Type badge */}
          <div className="flex items-center gap-2 mb-3">
            <span
              className="text-[10px] uppercase tracking-wider px-1.5 py-0.5 rounded font-medium"
              style={{
                backgroundColor:
                  selectedNode.type === 'memory'
                    ? `${TIER_COLORS[selectedNode.tier || 'cold']}22`
                    : 'rgba(226,232,240,0.1)',
                color:
                  selectedNode.type === 'memory'
                    ? TIER_COLORS[selectedNode.tier || 'cold']
                    : '#e2e8f0',
              }}
            >
              {selectedNode.type === 'memory' ? selectedNode.tier ?? 'memory' : 'concept'}
            </span>
          </div>

          {/* Memory details */}
          {selectedMemory && (
            <>
              {selectedMemory.importance && (
                <div className="text-xs text-[var(--text-muted)] mb-2">
                  Importance:{' '}
                  <span className="text-[var(--text-secondary)]">{selectedMemory.importance}</span>
                </div>
              )}

              {/* Strength bar */}
              {selectedMemory.strength != null && (
                <div className="mb-3">
                  <div className="text-[10px] text-[var(--text-muted)] uppercase tracking-wider mb-1">
                    Strength
                  </div>
                  <div className="w-full h-1.5 bg-[var(--border)] rounded-full overflow-hidden">
                    <div
                      className="h-full rounded-full"
                      style={{
                        width: `${Math.min(selectedMemory.strength * 100, 100)}%`,
                        backgroundColor: TIER_COLORS[selectedMemory.tier || 'cold'],
                      }}
                    />
                  </div>
                  <div className="text-[10px] text-[var(--text-muted)] mt-0.5">
                    {(selectedMemory.strength).toFixed(3)}
                  </div>
                </div>
              )}

              {selectedMemory.cluster_id != null && (
                <div className="text-xs text-[var(--text-muted)] mb-2">
                  Cluster:{' '}
                  <span className="text-[var(--text-secondary)] font-mono">{selectedMemory.cluster_id}</span>
                </div>
              )}

              <div className="text-xs text-[var(--text-muted)] mb-2">
                Support:{' '}
                <span className="text-[var(--text-secondary)] font-mono">{selectedMemory.support_count}</span>
              </div>

              <div className="text-xs text-[var(--text-muted)] mb-3">
                Diversity:{' '}
                <span className="text-[var(--text-secondary)] font-mono">{selectedMemory.source_diversity.toFixed(2)}</span>
              </div>

              <div className="mb-3">
                <div className="text-[10px] text-[var(--text-muted)] uppercase tracking-wider mb-1">
                  Evidence
                </div>
                {selectedMemoryLoading ? (
                  <div className="text-xs text-[var(--text-muted)]">Loading detail...</div>
                ) : selectedMemoryDetail?.evidence?.length ? (
                  <div className="space-y-2">
                    {selectedMemoryDetail.evidence.slice(0, 4).map((item) => (
                      <div key={item.id} className="rounded border border-[var(--border)] bg-[var(--bg-secondary)]/40 p-2">
                        <div className="text-[10px] uppercase tracking-wider text-[var(--accent)] mb-1">
                          {item.source_topic}
                        </div>
                        <div className="text-xs text-[var(--text-primary)] mb-1">{item.summary}</div>
                        <div className="text-[11px] text-[var(--text-secondary)] line-clamp-3 whitespace-pre-wrap break-words">
                          {item.content}
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="text-xs text-[var(--text-muted)]">
                    No supporting evidence beyond the canonical record.
                  </div>
                )}
              </div>
            </>
          )}

          {/* Concept details */}
          {selectedConcept && (
            <>
              {selectedConcept.confidence != null && (
                <div className="text-xs text-[var(--text-muted)] mb-2">
                  Confidence:{' '}
                  <span className="text-[var(--text-secondary)]">
                    {(selectedConcept.confidence * 100).toFixed(0)}%
                  </span>
                </div>
              )}
            </>
          )}

          {/* Created at */}
          <div className="text-xs text-[var(--text-muted)] mb-3">
            Created:{' '}
            <span className="text-[var(--text-secondary)] font-mono text-[10px]">
              {new Date(selectedNode.created_at).toLocaleString()}
            </span>
          </div>

          {/* ID */}
          <div className="text-[10px] text-[var(--text-muted)] font-mono break-all opacity-60">
            {selectedNode.id}
          </div>
        </div>
      )}

      {/* Node count indicator (top-right) */}
      <div className="absolute top-4 right-4 z-10 text-[10px] text-[var(--text-muted)] font-mono bg-[#0f172a]/80 backdrop-blur border border-[var(--border)] rounded-lg px-2 py-1">
        {filteredData.nodes.length} nodes / {filteredData.links.length} links
      </div>
    </div>
  );
}
