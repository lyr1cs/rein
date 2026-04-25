/**
 * Generic shape of one endpoint of a force-graph link. d3-force mutates the
 * raw `string` source/target into the resolved node object after the first
 * simulation tick, so any helper that walks links must accept either form.
 *
 * M4 (v0.26 cleanup): hoisted out of `Brain.tsx` and `Graph.tsx` (both had
 * a near-identical `LinkEndpoint = string | number | TNode | undefined`
 * alias and matching `endpointId` / `endpointNode` helpers). Generic over
 * the node type so each page keeps its page-specific shape.
 */
export type LinkEndpoint<TNode extends { id: string } = { id: string }> =
  | string
  | number
  | TNode
  | undefined;

/**
 * Resolve an endpoint to its `id` string. Returns `null` for nullish or
 * id-less inputs so callers can early-bail without throwing.
 */
export function endpointId<TNode extends { id: string }>(
  endpoint: LinkEndpoint<TNode>,
): string | null {
  if (endpoint == null) return null;
  if (typeof endpoint === 'object') {
    return endpoint.id == null ? null : String(endpoint.id);
  }
  return String(endpoint);
}

/**
 * Resolve an endpoint to a node object (after d3-force has stamped it),
 * else `null`. Used for hover/highlight passes that need the live x/y
 * coords on the node.
 */
export function endpointNode<TNode extends { id: string }>(
  endpoint: LinkEndpoint<TNode>,
): TNode | null {
  return typeof endpoint === 'object' && endpoint !== null ? endpoint : null;
}

/**
 * Diff-merge new graph nodes/links into the previous graphData while preserving
 * object identity for unchanged entries. Required by react-force-graph-2d, which
 * detects new node identities and reheats the d3-force simulation alpha=1 ->
 * "fireworks" expansion every poll cycle.
 *
 * Mutates existing node/link objects in place (Object.assign) so d3-stamped
 * runtime fields like x/y/vx/vy/index persist across the merge.
 *
 * The `linkKey` callback must produce a stable string identifier for each link.
 * Brain.tsx keys by `${src}|${tgt}|${type}`; Graph.tsx keys by
 * `${src}|${tgt}|${relation}`. Endpoint extraction is the caller's job because
 * d3 mutates `source`/`target` from string IDs into node objects after the
 * first simulation tick.
 */
export function mergeForceGraphData<
  N extends { id: string },
  L extends { source: unknown; target: unknown },
>(
  prev: { nodes: N[]; links: L[] },
  next: { nodes: N[]; links: L[] },
  linkKey: (link: L) => string,
): { nodes: N[]; links: L[] } {
  const oldNodeById = new Map(prev.nodes.map((n) => [n.id, n]));
  const mergedNodes = next.nodes.map((n) => {
    const existing = oldNodeById.get(n.id);
    if (existing) {
      Object.assign(existing, n);
      return existing;
    }
    return n;
  });
  const oldLinkByKey = new Map(prev.links.map((l) => [linkKey(l), l]));
  const mergedLinks = next.links.map((l) => {
    const existing = oldLinkByKey.get(linkKey(l));
    if (existing) {
      Object.assign(existing, l);
      return existing;
    }
    return l;
  });
  return { nodes: mergedNodes, links: mergedLinks };
}
