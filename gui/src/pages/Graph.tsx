import { useState, useEffect, useCallback, useRef, useMemo } from 'react';
import ForceGraph2D from 'react-force-graph-2d';
import type { LinkObject, NodeObject } from 'react-force-graph-2d';
import { apiGet } from '../api/client';
import type { Concept, ConceptLink } from '../api/types';

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
  name: string;
  definition: string;
  labels: string[];
  confidence: number;
  revision: number;
  source_memory_ids: string[];
  last_episode_id: string | null;
  val: number;                       // used by force-graph for node size
  created_at: string;
  updated_at: string;
  /* runtime */
  x?: number;
  y?: number;
}

interface GraphLink {
  source: string;
  target: string;
  relation: string;
  weight: number;
  valid_from: string | null;
  valid_until: string | null;
}

interface GraphData {
  nodes: GraphNode[];
  links: GraphLink[];
}

type ConceptNode = NodeObject<GraphNode>;
type ConceptLinkObject = LinkObject<GraphNode, GraphLink>;
type LinkEndpoint = string | number | ConceptNode | undefined;

function endpointId(endpoint: LinkEndpoint): string | null {
  if (endpoint == null) return null;
  if (typeof endpoint === 'object') {
    return endpoint.id == null ? null : String(endpoint.id);
  }
  return String(endpoint);
}

function endpointNode(endpoint: LinkEndpoint): ConceptNode | null {
  return typeof endpoint === 'object' && endpoint !== null ? endpoint : null;
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : 'Failed to load graph data';
}

function getPollingIntervalMs(): number {
  const saved = localStorage.getItem('rein_polling_interval');
  const seconds = saved ? parseInt(saved, 10) : 5;
  return Number.isFinite(seconds) ? seconds * 1000 : 5000;
}

/* ------------------------------------------------------------------ */
/*  Relation-type colors                                              */
/* ------------------------------------------------------------------ */

const RELATION_COLORS: Record<string, string> = {
  is_a:       '#7c3aed',
  has_a:      '#3b82f6',
  part_of:    '#22d3ee',
  depends_on: '#f97316',
  related_to: '#94a3b8',
  causes:     '#ef4444',
  uses:       '#4ade80',
  extends:    '#fbbf24',
  implements: '#a78bfa',
};

function relationColor(rel: string): string {
  const key = rel.toLowerCase().replace(/[\s-]+/g, '_');
  return RELATION_COLORS[key] ?? '#64748b';
}

/* ------------------------------------------------------------------ */
/*  Component                                                          */
/* ------------------------------------------------------------------ */

export default function Graph() {
  /* Memoir selector state */
  const [memoirs, setMemoirs] = useState<Memoir[]>([]);
  const [selectedMemoir, setSelectedMemoir] = useState<string>('');
  const [memoirLoading, setMemoirLoading] = useState(true);
  const [memoirError, setMemoirError] = useState('');

  /* Graph data */
  const [graphData, setGraphData] = useState<GraphData>({ nodes: [], links: [] });
  const [graphLoading, setGraphLoading] = useState(false);

  /* Interaction state */
  const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null);
  const [hoveredNode, setHoveredNode] = useState<GraphNode | null>(null);
  const [search, setSearch] = useState('');
  const [reloadVersion, setReloadVersion] = useState(0);

  const containerRef = useRef<HTMLDivElement>(null);
  const [dimensions, setDimensions] = useState(() => ({
    width: typeof window !== 'undefined' ? window.innerWidth - 96 : 800,
    height: typeof window !== 'undefined' ? window.innerHeight - 120 : 600,
  }));

  const refreshGraph = useCallback(() => {
    setGraphLoading(true);
    setReloadVersion((version) => version + 1);
  }, []);

  useEffect(() => {
    const intervalId = window.setInterval(refreshGraph, getPollingIntervalMs());
    return () => window.clearInterval(intervalId);
  }, [refreshGraph]);

  /* ---- Fetch memoirs on mount and on each refresh (polling / manual Refresh) ----
   *
   * Keyed on reloadVersion so creating/renaming/deleting a memoir elsewhere is
   * reflected next poll tick. On subsequent fetches we preserve the current
   * selection instead of auto-selecting the first memoir.
   */
  useEffect(() => {
    let cancelled = false;
    apiGet<{ memoirs: Memoir[] }>('/api/memoirs')
      .then((res) => {
        if (cancelled) return;
        setMemoirs(res.memoirs);
        if (res.memoirs.length > 0 && !selectedMemoir) {
          setGraphLoading(true);
          setSelectedNode(null);
          setSelectedMemoir(res.memoirs[0].name);
        }
        setMemoirError('');
      })
      .catch((err: unknown) => {
        if (!cancelled) setMemoirError(errorMessage(err));
      })
      .finally(() => {
        if (!cancelled) setMemoirLoading(false);
      });
    return () => { cancelled = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [reloadVersion]);

  /* ---- Fetch graph data when memoir changes ---- */
  useEffect(() => {
    if (!selectedMemoir) return;
    let cancelled = false;

    apiGet<MemoirExport>(`/api/memoirs/${encodeURIComponent(selectedMemoir)}/export?format=json`)
      .then((res) => {
        if (cancelled) return;
        const nodes: GraphNode[] = res.concepts.map((c) => ({
          id: c.id,
          name: c.name,
          definition: c.definition,
          labels: c.labels,
          confidence: c.confidence,
          revision: c.revision,
          source_memory_ids: c.source_memory_ids,
          last_episode_id: c.last_episode_id,
          val: Math.max(1, c.confidence * 4),
          created_at: c.created_at,
          updated_at: c.updated_at,
        }));
        const nodeIds = new Set(nodes.map((n) => n.id));
        const links: GraphLink[] = res.links
          .filter((l) => nodeIds.has(l.source_id) && nodeIds.has(l.target_id))
          .map((l) => ({
            source: l.source_id,
            target: l.target_id,
            relation: l.relation,
            weight: l.weight,
            valid_from: l.valid_from,
              valid_until: l.valid_until,
          }));
        setGraphData({ nodes, links });
        setSelectedNode((current) => {
          if (!current) return null;
          return nodes.find((node) => node.id === current.id) ?? null;
        });
      })
      .catch(() => {
        if (!cancelled) setGraphData({ nodes: [], links: [] });
      })
      .finally(() => {
        if (!cancelled) setGraphLoading(false);
      });

    return () => { cancelled = true; };
  }, [selectedMemoir, reloadVersion]);

  /* ---- Resize observer ---- */
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    // Initialize dimensions immediately
    setDimensions({ width: Math.floor(el.clientWidth), height: Math.floor(el.clientHeight) });
    const ro = new ResizeObserver((entries) => {
      for (const e of entries) {
        setDimensions({
          width: Math.floor(e.contentRect.width),
          height: Math.floor(e.contentRect.height),
        });
      }
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  /* ---- Search helpers ---- */
  const searchLower = search.toLowerCase();
  const matchingIds = useMemo(() => {
    if (!searchLower) return null;
    return new Set(
      graphData.nodes
        .filter((n) => n.name.toLowerCase().includes(searchLower) || n.labels.some((l) => l.toLowerCase().includes(searchLower)))
        .map((n) => n.id),
    );
  }, [searchLower, graphData.nodes]);

  /* ---- Connected node ids for hover highlight ---- */
  const connectedIds = useMemo(() => {
    if (!hoveredNode) return null;
    const ids = new Set<string>();
    ids.add(hoveredNode.id);
    for (const l of graphData.links) {
      const src = endpointId(l.source);
      const tgt = endpointId(l.target);
      if (src === hoveredNode.id && tgt !== null) ids.add(tgt);
      if (tgt === hoveredNode.id && src !== null) ids.add(src);
    }
    return ids;
  }, [hoveredNode, graphData.links]);

  /* ---- Node canvas renderer ---- */
  const paintNode = useCallback(
    (node: ConceptNode, ctx: CanvasRenderingContext2D, globalScale: number) => {
      const n = node;
      const r = Math.sqrt(n.val) * 3;
      const isMatch = matchingIds === null || matchingIds.has(n.id);
      const isHoverConnected = connectedIds === null || connectedIds.has(n.id);
      const dimmed = (matchingIds !== null && !isMatch) || (connectedIds !== null && !isHoverConnected);
      const x = node.x;
      const y = node.y;
      if (x == null || y == null) return;

      ctx.beginPath();
      ctx.arc(x, y, r, 0, 2 * Math.PI);

      if (matchingIds !== null && isMatch) {
        /* Glow for search matches */
        ctx.shadowColor = '#7c3aed';
        ctx.shadowBlur = 12;
        ctx.fillStyle = '#c4b5fd';
      } else if (dimmed) {
        ctx.fillStyle = 'rgba(100,116,139,0.25)';
      } else {
        ctx.fillStyle = '#e2e8f0';
      }
      ctx.fill();
      ctx.shadowBlur = 0;

      /* Label: only if zoomed in enough */
      if (globalScale > 1.2) {
        const fontSize = Math.min(12 / globalScale, 4);
        ctx.font = `${fontSize}px sans-serif`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'top';
        ctx.fillStyle = dimmed ? 'rgba(148,163,184,0.3)' : '#e2e8f0';
        ctx.fillText(n.name, x, y + r + 1);
      }
    },
    [matchingIds, connectedIds],
  );

  /* ---- Pointer area for click detection ---- */
  const paintNodeArea = useCallback(
    (node: ConceptNode, color: string, ctx: CanvasRenderingContext2D) => {
      const r = Math.sqrt(node.val) * 3;
      ctx.beginPath();
      ctx.arc(node.x ?? 0, node.y ?? 0, r, 0, 2 * Math.PI);
      ctx.fillStyle = color;
      ctx.fill();
    },
    [],
  );

  /* ---- Link renderer ---- */
  const paintLink = useCallback(
    (link: ConceptLinkObject, ctx: CanvasRenderingContext2D, globalScale: number) => {
      const l = link;
      const src = endpointNode(l.source);
      const tgt = endpointNode(l.target);
      if (!src || !tgt) return;
      const srcX = src.x;
      const srcY = src.y;
      const tgtX = tgt.x;
      const tgtY = tgt.y;
      if (srcX == null || srcY == null || tgtX == null || tgtY == null) return;

      const expired = l.valid_until != null;
      const color = relationColor(l.relation);

      ctx.beginPath();
      ctx.moveTo(srcX, srcY);
      ctx.lineTo(tgtX, tgtY);
      ctx.strokeStyle = expired ? `${color}66` : color;
      ctx.lineWidth = 0.5 / globalScale;
      if (expired) ctx.setLineDash([4 / globalScale, 4 / globalScale]);
      else ctx.setLineDash([]);
      ctx.stroke();
      ctx.setLineDash([]);
    },
    [],
  );

  /* ---- Detail panel helpers ---- */
  const linksFrom = useMemo(() => {
    if (!selectedNode) return [];
    return graphData.links.filter((l) => {
      const src = endpointId(l.source);
      return src === selectedNode.id;
    });
  }, [selectedNode, graphData.links]);

  const linksTo = useMemo(() => {
    if (!selectedNode) return [];
    return graphData.links.filter((l) => {
      const tgt = endpointId(l.target);
      return tgt === selectedNode.id;
    });
  }, [selectedNode, graphData.links]);

  const nodeNameById = useMemo(() => {
    const map = new Map<string, string>();
    for (const n of graphData.nodes) map.set(n.id, n.name);
    return map;
  }, [graphData.nodes]);

  function resolveName(idOrObj: LinkEndpoint): string {
    if (typeof idOrObj === 'object' && idOrObj !== null) {
      return idOrObj.name ?? endpointId(idOrObj) ?? '?';
    }
    if (idOrObj == null) return '?';
    return nodeNameById.get(String(idOrObj)) ?? String(idOrObj);
  }

  /* ---------------------------------------------------------------- */
  /*  Empty / loading states                                           */
  /* ---------------------------------------------------------------- */

  if (memoirLoading) {
    return <div className="flex items-center justify-center h-full text-[var(--text-muted)]">Loading memoirs...</div>;
  }

  if (memoirError) {
    return <div className="flex items-center justify-center h-full text-[var(--text-muted)]">{memoirError}</div>;
  }

  if (memoirs.length === 0) {
    return (
      <div className="flex items-center justify-center h-full text-[var(--text-muted)]">
        <div className="text-center">
          <div className="text-lg mb-2">No memoirs found</div>
          <div className="text-sm">Create one with <span className="font-mono text-[var(--accent)]">rein_memoir_create</span>.</div>
        </div>
      </div>
    );
  }

  /* ---------------------------------------------------------------- */
  /*  Render                                                           */
  /* ---------------------------------------------------------------- */

  return (
    <div className="flex flex-col h-full overflow-hidden">
      {/* Toolbar */}
      <div className="flex items-center gap-3 px-4 py-3 border-b border-[var(--border)] bg-[var(--bg-secondary)] shrink-0">
        {/* Memoir selector */}
        <label className="text-xs text-[var(--text-muted)] uppercase tracking-wider mr-1">Memoir</label>
        <select
          value={selectedMemoir}
          onChange={(e) => {
            setGraphLoading(true);
            setSelectedNode(null);
            setSelectedMemoir(e.target.value);
          }}
          className="bg-[var(--bg-primary)] border border-[var(--border)] rounded px-2 py-1 text-sm text-[var(--text-primary)] outline-none focus:border-[var(--accent)]"
        >
          {memoirs.map((m) => (
            <option key={m.id} value={m.name}>{m.name}</option>
          ))}
        </select>

        {/* Search */}
        <div className="flex-1" />
        <input
          type="text"
          placeholder="Search concepts..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          className="bg-[var(--bg-primary)] border border-[var(--border)] rounded px-3 py-1 text-sm text-[var(--text-primary)] w-56 outline-none focus:border-[var(--accent)] placeholder:text-[var(--text-muted)]"
        />
        {search && (
          <button
            onClick={() => setSearch('')}
            className="text-xs text-[var(--text-muted)] hover:text-[var(--text-primary)]"
          >
            Clear
          </button>
        )}
        <button
          onClick={refreshGraph}
          className="rounded border border-[var(--border)] px-2 py-1 text-xs text-[var(--text-muted)] hover:text-[var(--text-primary)] hover:border-[var(--accent)]"
        >
          Refresh
        </button>
      </div>

      {/* Main area: graph + optional detail panel */}
      <div className="flex flex-1 min-h-0">
        {/* Graph canvas */}
        <div ref={containerRef} className="flex-1 relative min-w-0 overflow-hidden" style={{ backgroundColor: '#050a15' }}>
          {graphLoading && (
            <div className="absolute inset-0 flex items-center justify-center z-10 text-[var(--text-muted)]">
              Loading graph...
            </div>
          )}

          {!graphLoading && graphData.nodes.length === 0 && (
            <div className="absolute inset-0 flex items-center justify-center z-10 text-[var(--text-muted)]">
              <div className="text-center">
                <div className="text-lg mb-2">This memoir has no concepts yet.</div>
              </div>
            </div>
          )}

          {graphData.nodes.length > 0 && (
            <ForceGraph2D<GraphNode, GraphLink>
              width={dimensions.width}
              height={dimensions.height}
              graphData={graphData}
              backgroundColor="#050a15"
              nodeCanvasObject={paintNode}
              nodeCanvasObjectMode={() => 'replace'}
              nodePointerAreaPaint={paintNodeArea}
              linkCanvasObject={paintLink}
              linkCanvasObjectMode={() => 'replace'}
              linkLabel={(l: ConceptLinkObject) => `${l.relation} (w=${l.weight.toFixed(2)})`}
              onNodeClick={(node) => setSelectedNode(node)}
              onNodeHover={(node) => setHoveredNode(node)}
              onBackgroundClick={() => setSelectedNode(null)}
              cooldownTime={3000}
              enableNodeDrag
            />
          )}
        </div>

        {/* Detail panel */}
        {selectedNode && (
          <div className="w-[280px] shrink-0 border-l border-[var(--border)] bg-[var(--bg-secondary)] overflow-y-auto p-4">
            {/* Header */}
            <div className="flex items-start justify-between mb-3">
              <h3 className="text-sm font-semibold text-[var(--text-primary)] break-words leading-snug">{selectedNode.name}</h3>
              <button
                onClick={() => setSelectedNode(null)}
                className="text-[var(--text-muted)] hover:text-[var(--text-primary)] ml-2 shrink-0"
              >
                x
              </button>
            </div>

            {/* Revision + confidence */}
            <div className="flex items-center gap-3 text-xs text-[var(--text-muted)] mb-3">
              <span>rev {selectedNode.revision}</span>
              <span>confidence {(selectedNode.confidence * 100).toFixed(0)}%</span>
            </div>

            {/* Definition */}
            <div className="text-xs text-[var(--text-secondary)] mb-4 leading-relaxed whitespace-pre-wrap">
              {selectedNode.definition || '(no definition)'}
            </div>

            {/* Labels */}
            {selectedNode.labels.length > 0 && (
              <div className="mb-4">
                <div className="text-[10px] text-[var(--text-muted)] uppercase tracking-wider mb-1">Labels</div>
                <div className="flex flex-wrap gap-1">
                  {selectedNode.labels.map((l) => (
                    <span key={l} className="text-[10px] px-1.5 py-0.5 rounded bg-[var(--accent)]/20 text-[var(--accent)]">{l}</span>
                  ))}
                </div>
              </div>
            )}

            {/* Source memory count */}
            <div className="text-xs text-[var(--text-muted)] mb-1">
              Source memories: <span className="text-[var(--text-secondary)]">{selectedNode.source_memory_ids?.length ?? 0}</span>
            </div>

            {/* Last episode */}
            <div className="text-xs text-[var(--text-muted)] mb-4">
              Last episode: <span className="text-[var(--text-secondary)] font-mono">{selectedNode.last_episode_id ? selectedNode.last_episode_id.slice(0, 8) : 'none'}</span>
            </div>

            {/* Links from */}
            {linksFrom.length > 0 && (
              <div className="mb-3">
                <div className="text-[10px] text-[var(--text-muted)] uppercase tracking-wider mb-1">Links from</div>
                <div className="space-y-1">
                  {linksFrom.map((l, i) => {
                    return (
                      <div key={i} className="text-xs flex items-center gap-1.5">
                        <span className="w-1.5 h-1.5 rounded-full shrink-0" style={{ backgroundColor: relationColor(l.relation) }} />
                        <span className="text-[var(--text-muted)]">{l.relation}</span>
                        <span className="text-[var(--text-secondary)] truncate">{resolveName(l.target)}</span>
                        {l.valid_until && <span className="text-[var(--text-muted)] text-[10px]">(expired)</span>}
                      </div>
                    );
                  })}
                </div>
              </div>
            )}

            {/* Links to */}
            {linksTo.length > 0 && (
              <div className="mb-3">
                <div className="text-[10px] text-[var(--text-muted)] uppercase tracking-wider mb-1">Links to</div>
                <div className="space-y-1">
                  {linksTo.map((l, i) => (
                    <div key={i} className="text-xs flex items-center gap-1.5">
                      <span className="w-1.5 h-1.5 rounded-full shrink-0" style={{ backgroundColor: relationColor(l.relation) }} />
                      <span className="text-[var(--text-secondary)] truncate">{resolveName(l.source)}</span>
                      <span className="text-[var(--text-muted)]">{l.relation}</span>
                      {l.valid_until && <span className="text-[var(--text-muted)] text-[10px]">(expired)</span>}
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Future button */}
            <button
              disabled
              className="w-full mt-4 text-xs py-1.5 rounded border border-[var(--border)] text-[var(--text-muted)] cursor-not-allowed"
            >
              View in Wiki (coming soon)
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
