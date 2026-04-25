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
