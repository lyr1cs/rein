import { useState, useEffect, useCallback, useRef, useMemo } from 'react';
import ForceGraph2D from 'react-force-graph-2d';
import type { LinkObject, NodeObject } from 'react-force-graph-2d';
import { useQueryClient } from '@tanstack/react-query';
import { getConceptState } from '../api/client';
import { useMemoirs, useMemoirExport } from '../hooks/useApi';
import {
  endpointId as endpointIdGeneric,
  endpointNode as endpointNodeGeneric,
  mergeForceGraphData,
  type LinkEndpoint as LinkEndpointGeneric,
} from '../utils/forceGraph';
import { timeAgo } from '../utils/time';
import type { ConceptState } from '../api/types';

/* ------------------------------------------------------------------ */
/*  Types                                                              */
/* ------------------------------------------------------------------ */

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
// M4 (v0.26 cleanup): see Brain.tsx for the rationale — generics hoisted to
// `utils/forceGraph.ts`; the type alias + thin wrappers below keep
// call-sites readable.
type LinkEndpoint = LinkEndpointGeneric<ConceptNode>;
const endpointId = (e: LinkEndpoint) => endpointIdGeneric<ConceptNode>(e);
const endpointNode = (e: LinkEndpoint) => endpointNodeGeneric<ConceptNode>(e);

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : 'Failed to load graph data';
}

/* ------------------------------------------------------------------ */
/*  Relation-type colors                                              */
/* ------------------------------------------------------------------ */

// Keys must match the snake_case serde names of the server `Relation` enum in
// src/types/memory.rs (Relation::PartOf -> "part_of", etc). Before B6, most
// of these keys were aspirational ("is_a", "has_a", "causes", "uses",
// "extends", "implements") and never matched server output, so 6 of the 9
// real relations always rendered in the fallback grey.
const RELATION_COLORS: Record<string, string> = {
  part_of:         '#22d3ee',
  depends_on:      '#f97316',
  related_to:      '#94a3b8',
  contradicts:     '#ef4444',
  refines:         '#4ade80',
  alternative_to:  '#a78bfa',
  caused_by:       '#fbbf24',
  instance_of:     '#7c3aed',
  superseded_by:   '#3b82f6',
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
  const [selectedMemoir, setSelectedMemoir] = useState<string>('');

  /* Graph data — committed to state via the diff-merge effect below so
   * unchanged nodes/links keep object identity across react-query polls. */
  const [graphData, setGraphData] = useState<GraphData>({ nodes: [], links: [] });

  /* Interaction state */
  const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null);
  const [hoveredNode, setHoveredNode] = useState<GraphNode | null>(null);
  const [search, setSearch] = useState('');

  /* Concept state (v0.24 ARS Capability A): living_summary snapshot for the
   * currently-selected concept. Null when no node selected, the fetch is
   * in-flight, or the fetch failed — the UI treats all three as "hide the
   * section" per the silent-fallback contract. */
  const [conceptState, setConceptState] = useState<ConceptState | null>(null);

  const containerRef = useRef<HTMLDivElement>(null);
  const [dimensions, setDimensions] = useState(() => ({
    width: typeof window !== 'undefined' ? window.innerWidth - 96 : 800,
    height: typeof window !== 'undefined' ? window.innerHeight - 120 : 600,
  }));

  /* ---- Data fetching via react-query (H3 + M7) ----
   *
   * `useMemoirs()` polls the directory; `useMemoirExport(selectedMemoir)`
   * polls the active memoir's concepts + links. Both gate on tab visibility
   * (refetchIntervalInBackground=false default) and dedup with any other
   * consumer of the same query key. Manual Refresh invalidates both keys. */
  const queryClient = useQueryClient();
  const memoirsQuery = useMemoirs();
  const memoirs = useMemo(() => memoirsQuery.data?.memoirs ?? [], [memoirsQuery.data]);
  const exportQuery = useMemoirExport(selectedMemoir || null);

  const memoirLoading = memoirsQuery.isLoading;
  const memoirError = memoirsQuery.error ? errorMessage(memoirsQuery.error) : '';
  /* Show the loading overlay while the export is in-flight OR when the
   * underlying selected-memoir id is set but no data has arrived yet. */
  const graphLoading = exportQuery.isFetching && !exportQuery.data;

  /* Auto-select the first memoir on the first successful directory fetch.
   * Side-effect — must be in an effect, not inline derivation, so subsequent
   * polls don't override the user's manual selection. The setState calls
   * legitimately sync external (react-query) data into local UI state. */
  /* eslint-disable react-hooks/set-state-in-effect */
  useEffect(() => {
    if (!selectedMemoir && memoirs.length > 0) {
      setSelectedNode(null);
      setSelectedMemoir(memoirs[0].name);
    }
  }, [memoirs, selectedMemoir]);
  /* eslint-enable react-hooks/set-state-in-effect */

  const refreshGraph = useCallback(() => {
    queryClient.invalidateQueries({ queryKey: ['memoirs'] });
    queryClient.invalidateQueries({ queryKey: ['memoir-export'] });
  }, [queryClient]);

  /* Derive the "next" graph snapshot purely from the latest export response.
   * react-query's structural sharing keeps `exportQuery.data` referentially
   * stable when the poll surfaced no changes, so this useMemo is identity-
   * preserving for byte-equal polls and the diff-merge effect below skips. */
  const nextGraph = useMemo<GraphData>(() => {
    const data = exportQuery.data;
    if (!data) return { nodes: [], links: [] };
    const nodes: GraphNode[] = data.concepts.map((c) => ({
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
    const links: GraphLink[] = data.links
      .filter((l) => nodeIds.has(l.source_id) && nodeIds.has(l.target_id))
      .map((l) => ({
        source: l.source_id,
        target: l.target_id,
        relation: l.relation,
        weight: l.weight,
        valid_from: l.valid_from,
        valid_until: l.valid_until,
      }));
    return { nodes, links };
  }, [exportQuery.data]);

  /* Diff-merge so unchanged nodes/links keep object identity (otherwise
   * react-force-graph-2d reheats the d3-force simulation alpha=1 every poll
   * → "fireworks" 3s loop because cooldownTime (3000ms) is shorter than the
   * default poll interval (5000ms)). The effect only fires when `nextGraph`
   * actually changes — react-query's structural sharing makes identical poll
   * responses no-ops at this layer.
   *
   * The setState calls below intentionally sync external (react-query) data
   * into a locally-merged `graphData` whose object identities must persist
   * (Object.assign in `mergeForceGraphData` mutates d3-force's x/y/vx/vy
   * stamps), so `react-hooks/set-state-in-effect` is silenced for this and
   * the memoir-change reset effect that follows. */
  /* Track the memoir whose graph is currently in `graphData` so the merge
   * effect knows when to replace-from-empty (memoir change) vs preserve
   * identity (poll refetch on the same memoir). Replaces the earlier
   * separate "reset on selectedMemoir change" effect, which raced the
   * merge effect on cache-hit memoir switches and could leave the page
   * showing an empty graph for a memoir that actually had concepts. */
  const lastMemoirRef = useRef<string>('');
  /* eslint-disable react-hooks/set-state-in-effect */
  useEffect(() => {
    const memoirChanged = lastMemoirRef.current !== selectedMemoir;
    lastMemoirRef.current = selectedMemoir;

    if (!exportQuery.data) {
      // Fetch in flight, errored, or memoir cleared. Drop the previous
      // memoir's graph if the user just switched (so the wrong nodes don't
      // linger while B's fetch is in flight) or if the fetch errored.
      if (memoirChanged || exportQuery.isError) {
        setGraphData({ nodes: [], links: [] });
        setSelectedNode(null);
      }
      return;
    }

    setGraphData((prev) =>
      mergeForceGraphData(
        memoirChanged ? { nodes: [], links: [] } : prev,
        nextGraph,
        (l) => {
          const src = endpointId(l.source as LinkEndpoint) ?? '';
          const tgt = endpointId(l.target as LinkEndpoint) ?? '';
          return `${src}|${tgt}|${l.relation}`;
        },
      ),
    );
    setSelectedNode((current) => {
      if (memoirChanged) return null;
      if (!current) return null;
      return nextGraph.nodes.find((n) => n.id === current.id) ?? null;
    });
  }, [nextGraph, exportQuery.data, exportQuery.isError, selectedMemoir]);
  /* eslint-enable react-hooks/set-state-in-effect */

  /* ---- Fetch concept state when a node is selected ----
   *
   * v0.24 ARS Capability A: surface `living_summary` alongside the concept
   * detail. Any fetch failure silently collapses the section (no toast, no
   * placeholder) so the rest of the panel remains functional if the REST
   * endpoint is missing (e.g. pre-v0.24 backend). The setState calls clear
   * stale state before a new fetch and on selection clear — both are
   * external-store synchronization, not derived state. */
  /* eslint-disable react-hooks/set-state-in-effect */
  useEffect(() => {
    if (!selectedNode) {
      setConceptState(null);
      return;
    }
    let cancelled = false;
    setConceptState(null);
    getConceptState(selectedNode.id)
      .then((state) => {
        if (!cancelled) setConceptState(state);
      })
      .catch(() => {
        if (!cancelled) setConceptState(null);
      });
    return () => { cancelled = true; };
  }, [selectedNode]);
  /* eslint-enable react-hooks/set-state-in-effect */

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
            // graphLoading is now driven by `exportQuery.isFetching && !data`,
            // so it auto-flips true when the new memoir's export starts loading.
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
                aria-label="Close detail"
                className="text-[var(--text-muted)] hover:text-[var(--text-primary)] ml-2 shrink-0"
              >
                x
              </button>
            </div>

            {/* Current state — living_summary card (v0.24 ARS Capability A).
                Rendered only when the REST fetch returned a non-null summary;
                on null / fetch failure the section is hidden entirely. */}
            {conceptState?.living_summary && (
              <div className="mb-4 rounded-lg border border-[var(--border)] bg-[var(--bg-primary)]/60 p-3">
                <div className="text-[10px] text-[var(--text-muted)] uppercase tracking-wider mb-1.5">
                  Current state
                </div>
                <div className="text-xs text-[var(--text-secondary)] leading-relaxed whitespace-pre-wrap line-clamp-5">
                  {conceptState.living_summary}
                </div>
                {conceptState.living_summary_updated_at && (
                  <div className="text-[10px] text-[var(--text-muted)] mt-2">
                    Auto-refreshed {timeAgo(conceptState.living_summary_updated_at)}
                    {conceptState.living_summary_source_revision != null && (
                      <> from revision #{conceptState.living_summary_source_revision}</>
                    )}
                  </div>
                )}
              </div>
            )}

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
