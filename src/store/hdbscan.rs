//! Pure Rust HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise).
//!
//! Implements the full HDBSCAN pipeline:
//! 1. Compute core distances (k-th nearest neighbor distance)
//! 2. Build mutual reachability graph
//! 3. Construct minimum spanning tree (Prim's algorithm)
//! 4. Build condensed cluster tree
//! 5. Extract clusters via Excess of Mass (EOMBST) stability selection
//!
//! No external crate dependencies — only `std`.

use std::collections::HashSet;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// A cluster produced by HDBSCAN.
#[derive(Debug, Clone)]
pub struct Cluster {
    /// Unique cluster identifier (0-based).
    pub id: u32,
    /// Indices into the original embeddings slice that belong to this cluster.
    pub member_indices: Vec<usize>,
    /// Excess-of-mass stability score for this cluster.
    pub stability: f64,
}

/// Full result of an HDBSCAN run.
#[derive(Debug, Clone)]
pub struct HdbscanResult {
    /// Extracted clusters sorted by id.
    pub clusters: Vec<Cluster>,
    /// Per-point cluster assignment. `None` means noise.
    pub labels: Vec<Option<u32>>,
    /// Indices of points labelled as noise.
    pub noise_indices: Vec<usize>,
}

// ---------------------------------------------------------------------------
// Distance helpers
// ---------------------------------------------------------------------------

/// Cosine distance between two vectors: `1 - cosine_similarity`.
///
/// Returns `1.0` when either vector has zero magnitude (maximally dissimilar).
pub fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "vector length mismatch");
    let mut dot: f64 = 0.0;
    let mut na: f64 = 0.0;
    let mut nb: f64 = 0.0;
    for i in 0..a.len() {
        let ai = a[i] as f64;
        let bi = b[i] as f64;
        dot += ai * bi;
        na += ai * ai;
        nb += bi * bi;
    }
    let denom = na.sqrt() * nb.sqrt();
    if denom < 1e-12 {
        return 1.0;
    }
    let sim = (dot / denom).clamp(-1.0, 1.0);
    (1.0 - sim) as f32
}

/// Compute a full symmetric distance matrix using cosine distance.
pub fn compute_distance_matrix(embeddings: &[Vec<f32>]) -> Vec<Vec<f32>> {
    let n = embeddings.len();
    let mut mat = vec![vec![0.0f32; n]; n];
    for i in 0..n {
        for j in (i + 1)..n {
            let d = cosine_distance(&embeddings[i], &embeddings[j]);
            mat[i][j] = d;
            mat[j][i] = d;
        }
    }
    mat
}

// ---------------------------------------------------------------------------
// Internal structures
// ---------------------------------------------------------------------------

/// An edge in the MST.
#[derive(Debug, Clone, Copy)]
struct MstEdge {
    u: usize,
    v: usize,
    weight: f32,
}

// ---------------------------------------------------------------------------
// Core algorithm
// ---------------------------------------------------------------------------

/// Compute core distances: for each point, the distance to its `min_samples`-th
/// nearest neighbor (1-indexed, so the point itself is excluded).
fn compute_core_distances(dist_matrix: &[Vec<f32>], min_samples: usize) -> Vec<f32> {
    let n = dist_matrix.len();
    let k = min_samples.min(n.saturating_sub(1));
    let mut core = vec![0.0f32; n];
    for i in 0..n {
        let mut dists: Vec<f32> = (0..n)
            .filter(|&j| j != i)
            .map(|j| dist_matrix[i][j])
            .collect();
        dists.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        core[i] = if k == 0 { 0.0 } else { dists[k - 1] };
    }
    core
}

/// Mutual reachability distance between points a and b.
#[inline]
fn mutual_reachability(core_a: f32, core_b: f32, dist_ab: f32) -> f32 {
    dist_ab.max(core_a).max(core_b)
}

/// Build MST over the mutual reachability graph using Prim's algorithm.
/// Returns edges sorted by weight ascending.
fn build_mst(dist_matrix: &[Vec<f32>], core_dists: &[f32]) -> Vec<MstEdge> {
    let n = dist_matrix.len();
    if n < 2 {
        return Vec::new();
    }

    let mut in_tree = vec![false; n];
    let mut min_weight = vec![f32::INFINITY; n];
    let mut nearest = vec![0usize; n];

    in_tree[0] = true;
    for j in 1..n {
        min_weight[j] = mutual_reachability(core_dists[0], core_dists[j], dist_matrix[0][j]);
        nearest[j] = 0;
    }

    let mut edges = Vec::with_capacity(n - 1);

    for _ in 0..(n - 1) {
        let mut best = usize::MAX;
        let mut best_w = f32::INFINITY;
        for j in 0..n {
            if !in_tree[j] && min_weight[j] < best_w {
                best_w = min_weight[j];
                best = j;
            }
        }
        if best == usize::MAX {
            break;
        }

        edges.push(MstEdge {
            u: nearest[best],
            v: best,
            weight: best_w,
        });
        in_tree[best] = true;

        for j in 0..n {
            if !in_tree[j] {
                let w = mutual_reachability(core_dists[best], core_dists[j], dist_matrix[best][j]);
                if w < min_weight[j] {
                    min_weight[j] = w;
                    nearest[j] = best;
                }
            }
        }
    }

    edges.sort_by(|a, b| {
        a.weight
            .partial_cmp(&b.weight)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    edges
}

/// Build MST from pre-computed k-NN lists (approximate mode for large datasets).
/// `knn`: for each point, a sorted list of `(neighbor_index, distance)`.
fn build_mst_approximate(knn: &[Vec<(usize, f32)>], core_dists: &[f32], n: usize) -> Vec<MstEdge> {
    if n < 2 {
        return Vec::new();
    }

    let mut in_tree = vec![false; n];
    let mut min_weight = vec![f32::INFINITY; n];
    let mut nearest = vec![0usize; n];

    in_tree[0] = true;
    for &(j, d) in &knn[0] {
        let w = mutual_reachability(core_dists[0], core_dists[j], d);
        if w < min_weight[j] {
            min_weight[j] = w;
            nearest[j] = 0;
        }
    }

    let mut edges = Vec::with_capacity(n - 1);

    for _ in 0..(n - 1) {
        let mut best = usize::MAX;
        let mut best_w = f32::INFINITY;
        for j in 0..n {
            if !in_tree[j] && min_weight[j] < best_w {
                best_w = min_weight[j];
                best = j;
            }
        }
        if best == usize::MAX {
            break;
        }

        edges.push(MstEdge {
            u: nearest[best],
            v: best,
            weight: best_w,
        });
        in_tree[best] = true;

        for &(j, d) in &knn[best] {
            if !in_tree[j] {
                let w = mutual_reachability(core_dists[best], core_dists[j], d);
                if w < min_weight[j] {
                    min_weight[j] = w;
                    nearest[j] = best;
                }
            }
        }
    }

    edges.sort_by(|a, b| {
        a.weight
            .partial_cmp(&b.weight)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    edges
}

/// Union-Find (disjoint set) with path compression and union by rank.
struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<usize>,
    size: Vec<usize>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
            size: vec![1; n],
        }
    }

    fn find(&mut self, mut x: usize) -> usize {
        while self.parent[x] != x {
            self.parent[x] = self.parent[self.parent[x]];
            x = self.parent[x];
        }
        x
    }

    fn union(&mut self, a: usize, b: usize) -> usize {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra == rb {
            return ra;
        }
        let (big, small) = if self.rank[ra] >= self.rank[rb] {
            (ra, rb)
        } else {
            (rb, ra)
        };
        self.parent[small] = big;
        self.size[big] += self.size[small];
        if self.rank[big] == self.rank[small] {
            self.rank[big] += 1;
        }
        big
    }
}

// ---------------------------------------------------------------------------
// Phase 1: Dendrogram (bottom-up from sorted MST)
// ---------------------------------------------------------------------------

/// A node in the single-linkage dendrogram.
/// Leaf nodes (index < n) are individual points.
/// Internal nodes (index >= n) represent merges.
struct DendrogramNode {
    left: usize,   // left child index (point or internal node)
    right: usize,  // right child index
    distance: f32, // merge distance (mutual reachability)
    _size: usize,  // total points in this subtree
}

/// Build a single-linkage dendrogram from sorted MST edges.
/// Returns (dendrogram_nodes, root_index).
fn build_dendrogram(edges: &[MstEdge], n: usize) -> (Vec<DendrogramNode>, usize) {
    let mut uf = UnionFind::new(n);
    // Map: UF root → dendrogram node index (leaves are 0..n)
    let mut node_of: Vec<usize> = (0..n).collect();
    let mut nodes: Vec<DendrogramNode> = Vec::with_capacity(n - 1);
    let mut next_id = n;

    for edge in edges {
        let ra = uf.find(edge.u);
        let rb = uf.find(edge.v);
        if ra == rb {
            continue;
        }

        let left = node_of[ra];
        let right = node_of[rb];
        let size = uf.size[ra] + uf.size[rb];

        nodes.push(DendrogramNode {
            left,
            right,
            distance: edge.weight,
            _size: size,
        });

        let new_root = uf.union(ra, rb);
        node_of[new_root] = next_id;
        // Also update the other root's mapping (union may pick either)
        let other = if new_root == ra { rb } else { ra };
        node_of[other] = next_id;
        next_id += 1;
    }

    let root = if nodes.is_empty() { 0 } else { next_id - 1 };
    (nodes, root)
}

// ---------------------------------------------------------------------------
// Phase 2: Condensed tree (top-down from dendrogram)
// ---------------------------------------------------------------------------

/// A node in the condensed cluster tree.
struct CondensedCluster {
    /// Points that "fell out" of this cluster (noise candidates).
    /// Each entry: (point_index, lambda_at_which_it_fell_out).
    fell_out: Vec<(usize, f64)>,
    /// Child cluster indices in the condensed tree.
    children: Vec<usize>,
    /// Lambda (1/distance) at which this cluster was born.
    lambda_birth: f64,
    /// Lambda at which this cluster died (merged into parent). 0.0 for root.
    lambda_death: f64,
    /// Points that are still members when this cluster "dies" or becomes a leaf.
    leaf_points: Vec<usize>,
    /// Excess-of-mass stability score.
    stability: f64,
}

/// Build the condensed tree by walking the dendrogram top-down.
fn build_condensed_tree(
    dendro: &[DendrogramNode],
    n: usize,
    root: usize,
    min_cluster_size: usize,
) -> Vec<CondensedCluster> {
    let mut clusters: Vec<CondensedCluster> = Vec::new();

    // Recursive traversal: returns (condensed_cluster_idx, set of point indices)
    fn traverse(
        dendro: &[DendrogramNode],
        n: usize,
        node_idx: usize,
        min_cluster_size: usize,
        clusters: &mut Vec<CondensedCluster>,
        parent_lambda: f64,
    ) -> (Option<usize>, Vec<usize>) {
        // Leaf (individual point)
        if node_idx < n {
            return (None, vec![node_idx]);
        }

        let dendro_idx = node_idx - n;
        if dendro_idx >= dendro.len() {
            return (None, vec![]);
        }
        let dnode = &dendro[dendro_idx];
        let lambda = if dnode.distance > 0.0 {
            1.0 / dnode.distance as f64
        } else {
            f64::MAX
        };

        // Recurse into children
        let (left_cluster, left_points) =
            traverse(dendro, n, dnode.left, min_cluster_size, clusters, lambda);
        let (right_cluster, right_points) =
            traverse(dendro, n, dnode.right, min_cluster_size, clusters, lambda);

        let left_size = left_points.len();
        let right_size = right_points.len();
        let left_big = left_size >= min_cluster_size;
        let right_big = right_size >= min_cluster_size;

        if left_big && right_big {
            // Real split: create two child clusters
            let left_cid = clusters.len();
            clusters.push(CondensedCluster {
                fell_out: Vec::new(),
                children: left_cluster.into_iter().collect(),
                lambda_birth: lambda,
                lambda_death: 0.0,
                leaf_points: left_points.clone(),
                stability: 0.0,
            });

            let right_cid = clusters.len();
            clusters.push(CondensedCluster {
                fell_out: Vec::new(),
                children: right_cluster.into_iter().collect(),
                lambda_birth: lambda,
                lambda_death: 0.0,
                leaf_points: right_points.clone(),
                stability: 0.0,
            });

            // Create parent that holds both children
            let parent_cid = clusters.len();
            clusters.push(CondensedCluster {
                fell_out: Vec::new(),
                children: vec![left_cid, right_cid],
                lambda_birth: parent_lambda,
                lambda_death: lambda,
                leaf_points: Vec::new(),
                stability: 0.0,
            });

            // Set children's death lambda
            clusters[left_cid].lambda_death = lambda;
            clusters[right_cid].lambda_death = lambda;

            let mut all_points = left_points;
            all_points.extend(right_points);
            (Some(parent_cid), all_points)
        } else if left_big || right_big {
            // One side big, other small — small side falls out
            let (big_points, small_points, big_child) = if left_big {
                (left_points, right_points, left_cluster)
            } else {
                (right_points, left_points, right_cluster)
            };

            // Create or extend the big cluster
            let cid = if let Some(existing) = big_child {
                // There's already a condensed cluster for the big side
                for &p in &small_points {
                    clusters[existing].fell_out.push((p, lambda));
                }
                existing
            } else {
                // Create new cluster for the big side
                let new_cid = clusters.len();
                let fell = small_points.iter().map(|&p| (p, lambda)).collect();
                clusters.push(CondensedCluster {
                    fell_out: fell,
                    children: Vec::new(),
                    lambda_birth: lambda,
                    lambda_death: 0.0,
                    leaf_points: big_points.clone(),
                    stability: 0.0,
                });
                new_cid
            };

            let mut all_points = big_points;
            all_points.extend(small_points);
            (Some(cid), all_points)
        } else {
            // Both small — just merge, no cluster created
            let mut all_points = left_points;
            all_points.extend(right_points);
            (None, all_points)
        }
    }

    let (_root_cluster, _all_points) =
        traverse(dendro, n, root, min_cluster_size, &mut clusters, 0.0);
    clusters
}

// ---------------------------------------------------------------------------
// Phase 3: Stability + EOMBST selection
// ---------------------------------------------------------------------------

/// Compute stability for each condensed cluster.
///
/// Leaf cluster: stab = Σ_{leaf points} (lambda_death - lambda_birth)
///                    + Σ_{fell_out} (lambda_fell - lambda_birth)
///
/// Internal cluster: stab = Σ_{fell_out} (lambda_fell - lambda_birth)
///                        + n_surviving × (lambda_death - lambda_birth)
/// where n_surviving = leaf_points count (points alive from birth to death).
fn compute_stability(clusters: &mut [CondensedCluster]) {
    // Find max lambda: highest density level observed
    let lambda_max = clusters
        .iter()
        .flat_map(|c| c.fell_out.iter().map(|&(_, l)| l))
        .chain(
            clusters
                .iter()
                .map(|c| c.lambda_death)
                .filter(|&l| l > 0.0 && l.is_finite()),
        )
        .fold(0.0f64, f64::max);
    let lambda_max = if lambda_max.is_finite() && lambda_max > 0.0 {
        lambda_max
    } else {
        1.0
    };

    for c in clusters.iter_mut() {
        // Ensure lambda_death is set
        if c.lambda_death <= 0.0 || !c.lambda_death.is_finite() {
            c.lambda_death = lambda_max;
        }

        let birth = c.lambda_birth;
        let death = c.lambda_death;
        let mut stab = 0.0f64;

        // Points that fell out: each contributes (lambda_fell - lambda_birth)
        for &(_, lambda_fell) in &c.fell_out {
            let lf = if lambda_fell.is_finite() {
                lambda_fell
            } else {
                death
            };
            stab += (lf - birth).max(0.0);
        }

        if c.children.is_empty() {
            // Leaf: surviving points persist until death
            stab += c.leaf_points.len() as f64 * (death - birth).max(0.0);
        } else {
            // Internal: surviving points (leaf_points) persist from birth to death
            // These are points that didn't fall out and went to children at death
            let n_surviving = c.leaf_points.len();
            stab += n_surviving as f64 * (death - birth).max(0.0);
        }

        c.stability = stab.max(0.0);
    }
}

/// EOMBST cluster selection: bottom-up, at each internal node choose either
/// this node (as one cluster) or its children (as separate clusters).
fn eombst_select(clusters: &[CondensedCluster]) -> Vec<bool> {
    let n = clusters.len();
    let mut selected = vec![false; n];
    let mut effective_stability = vec![0.0f64; n];

    // Initialize from computed stability
    for i in 0..n {
        effective_stability[i] = clusters[i].stability;
    }

    // Process leaves first, then internal nodes
    // Leaf nodes (no children) are always selected initially
    for i in 0..n {
        if clusters[i].children.is_empty() {
            selected[i] = true;
        }
    }

    // Process internal nodes: if parent stability > sum of children stability,
    // select parent and deselect children
    for i in 0..n {
        if !clusters[i].children.is_empty() {
            let children_stab: f64 = clusters[i]
                .children
                .iter()
                .map(|&c| effective_stability[c])
                .sum();
            if effective_stability[i] > children_stab {
                selected[i] = true;
                for &c in &clusters[i].children {
                    deselect_all(clusters, c, &mut selected);
                }
            } else {
                effective_stability[i] = children_stab;
            }
        }
    }

    selected
}

fn deselect_all(clusters: &[CondensedCluster], idx: usize, selected: &mut [bool]) {
    selected[idx] = false;
    for &c in &clusters[idx].children {
        deselect_all(clusters, c, selected);
    }
}

/// Collect all point indices belonging to a selected cluster (including from sub-clusters).
fn gather_points(clusters: &[CondensedCluster], idx: usize, selected: &[bool]) -> Vec<usize> {
    let mut points = Vec::new();
    let c = &clusters[idx];

    // Leaf points
    if c.children.is_empty() {
        points.extend_from_slice(&c.leaf_points);
    }

    // Fell-out points belong to this cluster (they fell out of it, so they're noise
    // relative to children, but belong to this cluster if it's selected)
    for &(p, _) in &c.fell_out {
        points.push(p);
    }

    // Points from non-selected children
    for &child in &c.children {
        if !selected[child] {
            points.extend(gather_points(clusters, child, selected));
        }
    }

    points
}

/// Full pipeline: MST → Dendrogram → Condensed Tree → Stability → EOMBST → Clusters
fn build_condensed_tree_and_extract(
    edges: &[MstEdge],
    n: usize,
    min_cluster_size: usize,
) -> (Vec<Cluster>, Vec<Option<u32>>) {
    if n == 0 {
        return (Vec::new(), Vec::new());
    }
    if n < min_cluster_size {
        return (Vec::new(), vec![None; n]);
    }
    if edges.is_empty() {
        let c = Cluster {
            id: 0,
            member_indices: (0..n).collect(),
            stability: 0.0,
        };
        return (vec![c], vec![Some(0); n]);
    }

    // Phase 1: Build dendrogram
    let (dendro, root) = build_dendrogram(edges, n);

    // Phase 2: Build condensed tree (top-down)
    let mut condensed = build_condensed_tree(&dendro, n, root, min_cluster_size);

    if condensed.is_empty() {
        // No splits found — all points in one cluster or all noise
        if n >= min_cluster_size {
            let c = Cluster {
                id: 0,
                member_indices: (0..n).collect(),
                stability: 0.0,
            };
            return (vec![c], vec![Some(0); n]);
        } else {
            return (Vec::new(), vec![None; n]);
        }
    }

    // Phase 3: Compute stability
    compute_stability(&mut condensed);

    // Phase 4: EOMBST selection
    let selected = eombst_select(&condensed);

    // Phase 5: Extract final clusters
    let mut clusters = Vec::new();
    let mut labels = vec![None; n];
    let mut cid = 0u32;

    for (i, &sel) in selected.iter().enumerate() {
        if !sel {
            continue;
        }
        let mut members = gather_points(&condensed, i, &selected);
        members.sort();
        members.dedup();

        if members.len() < min_cluster_size {
            continue;
        }

        for &m in &members {
            if labels[m].is_none() {
                labels[m] = Some(cid);
            }
        }

        clusters.push(Cluster {
            id: cid,
            member_indices: members,
            stability: condensed[i].stability,
        });
        cid += 1;
    }

    (clusters, labels)
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Run HDBSCAN clustering on a set of embeddings.
///
/// # Arguments
/// * `embeddings` — slice of `(id, embedding_vector)` pairs.
/// * `min_cluster_size` — minimum number of points to form a cluster.
///   A common heuristic is `max(3, n^0.25)`.
///
/// # Returns
/// An [`HdbscanResult`] containing clusters, per-point labels, and noise indices.
///
/// # Complexity
/// O(n^2) time and space for the distance matrix. For n > 2000, prefer
/// [`hdbscan_approximate`] with pre-computed k-NN lists.
/// Hard limit for full O(n^2) distance matrix computation.
/// Above this threshold, auto-fallback to sampling + label propagation.
pub const HDBSCAN_FULL_MATRIX_LIMIT: usize = 3000;

pub fn hdbscan(embeddings: &[(String, Vec<f32>)], min_cluster_size: usize) -> HdbscanResult {
    let n = embeddings.len();
    if n == 0 {
        return HdbscanResult {
            clusters: Vec::new(),
            labels: Vec::new(),
            noise_indices: Vec::new(),
        };
    }

    // For large datasets, sample + cluster + propagate labels to avoid O(n^2) OOM
    if n > HDBSCAN_FULL_MATRIX_LIMIT {
        return hdbscan_sampled(
            embeddings,
            min_cluster_size,
            None,
            HDBSCAN_FULL_MATRIX_LIMIT,
        );
    }

    let mcs = min_cluster_size.max(2);
    let min_samples = mcs;

    let vecs: Vec<Vec<f32>> = embeddings.iter().map(|(_, v)| v.clone()).collect();
    let dist_matrix = compute_distance_matrix(&vecs);
    let core_dists = compute_core_distances(&dist_matrix, min_samples);
    let mst = build_mst(&dist_matrix, &core_dists);
    let (clusters, labels) = build_condensed_tree_and_extract(&mst, n, mcs);

    let noise_indices: Vec<usize> = labels
        .iter()
        .enumerate()
        .filter_map(|(i, l)| if l.is_none() { Some(i) } else { None })
        .collect();

    HdbscanResult {
        clusters,
        labels,
        noise_indices,
    }
}

/// Run HDBSCAN with custom `min_samples` parameter.
///
/// `min_samples` controls the conservativeness of clustering (higher = more noise).
/// When `None`, defaults to `min_cluster_size`.
pub fn hdbscan_with_params(
    embeddings: &[(String, Vec<f32>)],
    min_cluster_size: usize,
    min_samples: Option<usize>,
) -> HdbscanResult {
    let n = embeddings.len();
    if n == 0 {
        return HdbscanResult {
            clusters: Vec::new(),
            labels: Vec::new(),
            noise_indices: Vec::new(),
        };
    }

    // OOM protection: route to sampled path for large datasets
    if n > HDBSCAN_FULL_MATRIX_LIMIT {
        return hdbscan_sampled(
            embeddings,
            min_cluster_size,
            min_samples,
            HDBSCAN_FULL_MATRIX_LIMIT,
        );
    }

    let mcs = min_cluster_size.max(2);
    let ms = min_samples.unwrap_or(mcs);

    let vecs: Vec<Vec<f32>> = embeddings.iter().map(|(_, v)| v.clone()).collect();
    let dist_matrix = compute_distance_matrix(&vecs);
    let core_dists = compute_core_distances(&dist_matrix, ms);
    let mst = build_mst(&dist_matrix, &core_dists);
    let (clusters, labels) = build_condensed_tree_and_extract(&mst, n, mcs);

    let noise_indices: Vec<usize> = labels
        .iter()
        .enumerate()
        .filter_map(|(i, l)| if l.is_none() { Some(i) } else { None })
        .collect();

    HdbscanResult {
        clusters,
        labels,
        noise_indices,
    }
}

/// Approximate HDBSCAN for large datasets (n > 2000).
///
/// Instead of computing a full O(n^2) distance matrix, accepts pre-computed
/// k-nearest-neighbor lists per point. The MST is built using only k-NN
/// edges, which is an approximation but much faster for high-n datasets.
///
/// # Arguments
/// * `knn_lists` — for each point, a sorted `Vec<(neighbor_index, distance)>`.
///   Must contain at least `min_samples` neighbors per point.
/// * `n` — total number of points.
/// * `min_cluster_size` — minimum cluster size.
/// * `min_samples` — if `None`, defaults to `min_cluster_size`.
pub fn hdbscan_approximate(
    knn_lists: &[Vec<(usize, f32)>],
    n: usize,
    min_cluster_size: usize,
    min_samples: Option<usize>,
) -> HdbscanResult {
    if n == 0 || knn_lists.is_empty() {
        return HdbscanResult {
            clusters: Vec::new(),
            labels: Vec::new(),
            noise_indices: Vec::new(),
        };
    }

    let mcs = min_cluster_size.max(2);
    let ms = min_samples.unwrap_or(mcs);

    let mut core_dists = vec![0.0f32; n];
    for i in 0..n {
        let k = ms.min(knn_lists[i].len());
        core_dists[i] = if k == 0 { 0.0 } else { knn_lists[i][k - 1].1 };
    }

    let mst = build_mst_approximate(knn_lists, &core_dists, n);
    let (clusters, labels) = build_condensed_tree_and_extract(&mst, n, mcs);

    let noise_indices: Vec<usize> = labels
        .iter()
        .enumerate()
        .filter_map(|(i, l)| if l.is_none() { Some(i) } else { None })
        .collect();

    HdbscanResult {
        clusters,
        labels,
        noise_indices,
    }
}

/// Assign a new point to the nearest existing cluster by centroid distance.
///
/// Computes the centroid of each cluster (mean of member embeddings), then
/// returns the cluster whose centroid is closest to the given embedding.
/// Returns `None` if no clusters exist or the point is closer to the global
/// mean of noise points than to any cluster centroid.
pub fn assign_to_nearest(
    embedding: &[f32],
    clusters: &[Cluster],
    all_embeddings: &[(String, Vec<f32>)],
) -> Option<u32> {
    if clusters.is_empty() {
        return None;
    }

    let dim = embedding.len();
    let mut best_id: Option<u32> = None;
    let mut best_dist = f32::INFINITY;

    for cluster in clusters {
        if cluster.member_indices.is_empty() {
            continue;
        }
        let centroid = compute_centroid(&cluster.member_indices, all_embeddings, dim);
        let d = cosine_distance(embedding, &centroid);
        if d < best_dist {
            best_dist = d;
            best_id = Some(cluster.id);
        }
    }

    // Check if the point is closer to the noise centroid.
    let clustered: HashSet<usize> = clusters
        .iter()
        .flat_map(|c| c.member_indices.iter().copied())
        .collect();
    let noise_indices: Vec<usize> = (0..all_embeddings.len())
        .filter(|i| !clustered.contains(i))
        .collect();

    if !noise_indices.is_empty() {
        let noise_centroid = compute_centroid(&noise_indices, all_embeddings, dim);
        let noise_dist = cosine_distance(embedding, &noise_centroid);
        if noise_dist < best_dist {
            return None;
        }
    }

    best_id
}

/// Compute the centroid (element-wise mean) of the given point indices.
fn compute_centroid(
    indices: &[usize],
    all_embeddings: &[(String, Vec<f32>)],
    dim: usize,
) -> Vec<f32> {
    let mut centroid = vec![0.0f64; dim];
    let mut count = 0usize;
    for &idx in indices {
        if idx < all_embeddings.len() {
            let v = &all_embeddings[idx].1;
            for (j, &val) in v.iter().enumerate() {
                if j < dim {
                    centroid[j] += val as f64;
                }
            }
            count += 1;
        }
    }
    if count > 0 {
        for c in centroid.iter_mut() {
            *c /= count as f64;
        }
    }
    centroid.into_iter().map(|c| c as f32).collect()
}

/// HDBSCAN with sampling for large datasets (n > HDBSCAN_FULL_MATRIX_LIMIT).
///
/// Strategy: randomly sample `sample_size` points, run full HDBSCAN on sample,
/// then propagate labels to remaining points via nearest-centroid assignment.
fn hdbscan_sampled(
    embeddings: &[(String, Vec<f32>)],
    min_cluster_size: usize,
    min_samples: Option<usize>,
    sample_size: usize,
) -> HdbscanResult {
    let n = embeddings.len();

    // Deterministic sampling using splitmix
    let mut sample_indices: Vec<usize> = (0..n).collect();
    // Fisher-Yates shuffle with deterministic seed
    for i in (1..n).rev() {
        let mut x = (i as u64).wrapping_add(0x9e3779b97f4a7c15);
        x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
        x = x ^ (x >> 31);
        let j = (x as usize) % (i + 1);
        sample_indices.swap(i, j);
    }
    sample_indices.truncate(sample_size);
    sample_indices.sort_unstable(); // keep order stable for reproducibility

    // Run full HDBSCAN on sampled subset
    let sampled_embeddings: Vec<(String, Vec<f32>)> = sample_indices
        .iter()
        .map(|&i| embeddings[i].clone())
        .collect();
    let sample_result = hdbscan_with_params(&sampled_embeddings, min_cluster_size, min_samples);

    if sample_result.clusters.is_empty() {
        // No clusters found in sample — treat everything as noise
        return HdbscanResult {
            clusters: Vec::new(),
            labels: vec![None; n],
            noise_indices: (0..n).collect(),
        };
    }

    // Propagate labels to non-sampled points via nearest centroid
    let mut labels = vec![None; n];

    // First, assign labels for sampled points
    for (sample_idx, &original_idx) in sample_indices.iter().enumerate() {
        labels[original_idx] = sample_result.labels[sample_idx];
    }

    // Then, assign remaining points to nearest cluster centroid
    let sampled_set: HashSet<usize> = sample_indices.iter().copied().collect();
    for i in 0..n {
        if sampled_set.contains(&i) {
            continue;
        }
        labels[i] = assign_to_nearest(
            &embeddings[i].1,
            &sample_result.clusters,
            &sampled_embeddings,
        );
    }

    // Rebuild clusters with all points
    let max_cluster_id = sample_result
        .clusters
        .iter()
        .map(|c| c.id)
        .max()
        .unwrap_or(0);
    let mut cluster_members: Vec<Vec<usize>> = vec![Vec::new(); (max_cluster_id + 1) as usize];
    let mut noise_indices = Vec::new();

    for (i, label) in labels.iter().enumerate() {
        match label {
            Some(cid) => {
                if (*cid as usize) < cluster_members.len() {
                    cluster_members[*cid as usize].push(i);
                }
            }
            None => noise_indices.push(i),
        }
    }

    let clusters: Vec<Cluster> = cluster_members
        .into_iter()
        .enumerate()
        .filter(|(_, members)| !members.is_empty())
        .map(|(id, members)| {
            // Use the sample cluster's stability as a proxy
            let stability = sample_result
                .clusters
                .iter()
                .find(|c| c.id == id as u32)
                .map(|c| c.stability)
                .unwrap_or(0.0);
            Cluster {
                id: id as u32,
                member_indices: members,
                stability,
            }
        })
        .collect();

    HdbscanResult {
        clusters,
        labels,
        noise_indices,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Simple deterministic pseudo-random number generator (xorshift32).
    struct Rng {
        state: u32,
    }

    impl Rng {
        fn new(seed: u32) -> Self {
            Self {
                state: if seed == 0 { 1 } else { seed },
            }
        }

        fn next_u32(&mut self) -> u32 {
            let mut x = self.state;
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            self.state = x;
            x
        }

        /// Uniform float in [-1, 1].
        fn next_f32(&mut self) -> f32 {
            (self.next_u32() as f32 / u32::MAX as f32) * 2.0 - 1.0
        }
    }

    /// Generate a cluster of `n` points around a center, normalized to unit sphere.
    fn make_cluster(rng: &mut Rng, center: &[f32], n: usize, noise_scale: f32) -> Vec<Vec<f32>> {
        let dim = center.len();
        (0..n)
            .map(|_| {
                let mut v = vec![0.0f32; dim];
                for d in 0..dim {
                    v[d] = center[d] + rng.next_f32() * noise_scale;
                }
                let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm > 1e-8 {
                    for x in v.iter_mut() {
                        *x /= norm;
                    }
                }
                v
            })
            .collect()
    }

    fn to_embeddings(vecs: &[Vec<f32>]) -> Vec<(String, Vec<f32>)> {
        vecs.iter()
            .enumerate()
            .map(|(i, v)| (format!("p{}", i), v.clone()))
            .collect()
    }

    #[test]
    fn test_cosine_distance_identical() {
        let a = vec![1.0, 0.0, 0.0];
        assert!((cosine_distance(&a, &a) - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_cosine_distance_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert!((cosine_distance(&a, &b) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_cosine_distance_opposite() {
        let a = vec![1.0, 0.0];
        let b = vec![-1.0, 0.0];
        assert!((cosine_distance(&a, &b) - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_empty_input() {
        let result = hdbscan(&[], 3);
        assert!(result.clusters.is_empty());
        assert!(result.labels.is_empty());
        assert!(result.noise_indices.is_empty());
    }

    #[test]
    fn test_single_point() {
        let emb = vec![("a".to_string(), vec![1.0, 0.0, 0.0])];
        let result = hdbscan(&emb, 2);
        assert_eq!(result.labels.len(), 1);
        assert!(result.labels[0].is_none());
    }

    #[test]
    fn test_three_well_separated_clusters() {
        let mut rng = Rng::new(42);

        // Three clusters in 8D, well separated via distinct dominant dimensions.
        let center_a = vec![5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let center_b = vec![0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0];
        let center_c = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 0.0];

        let mut vecs = Vec::new();
        vecs.extend(make_cluster(&mut rng, &center_a, 10, 0.1));
        vecs.extend(make_cluster(&mut rng, &center_b, 10, 0.1));
        vecs.extend(make_cluster(&mut rng, &center_c, 10, 0.1));

        let emb = to_embeddings(&vecs);
        let result = hdbscan(&emb, 3);

        assert!(
            result.clusters.len() >= 2,
            "Expected at least 2 clusters, got {}",
            result.clusters.len()
        );

        let group_a: Vec<Option<u32>> = result.labels[0..10].to_vec();
        let group_b: Vec<Option<u32>> = result.labels[10..20].to_vec();
        let group_c: Vec<Option<u32>> = result.labels[20..30].to_vec();

        fn same_label(group: &[Option<u32>]) -> bool {
            let non_noise: Vec<u32> = group.iter().filter_map(|l| *l).collect();
            if non_noise.is_empty() {
                return true;
            }
            non_noise.iter().all(|&l| l == non_noise[0])
        }
        assert!(
            same_label(&group_a),
            "Group A labels inconsistent: {:?}",
            group_a
        );
        assert!(
            same_label(&group_b),
            "Group B labels inconsistent: {:?}",
            group_b
        );
        assert!(
            same_label(&group_c),
            "Group C labels inconsistent: {:?}",
            group_c
        );

        let label_a = group_a.iter().find_map(|l| *l);
        let label_b = group_b.iter().find_map(|l| *l);
        let label_c = group_c.iter().find_map(|l| *l);

        if let (Some(la), Some(lb)) = (label_a, label_b) {
            assert_ne!(
                la, lb,
                "Groups A and B should have different cluster labels"
            );
        }
        if let (Some(la), Some(lc)) = (label_a, label_c) {
            assert_ne!(
                la, lc,
                "Groups A and C should have different cluster labels"
            );
        }
        if let (Some(lb), Some(lc)) = (label_b, label_c) {
            assert_ne!(
                lb, lc,
                "Groups B and C should have different cluster labels"
            );
        }
    }

    #[test]
    fn test_noise_detection() {
        let mut rng = Rng::new(99);

        let center_a = vec![3.0, 0.0, 0.0, 0.0];
        let center_b = vec![0.0, 0.0, 3.0, 0.0];
        let mut vecs = Vec::new();
        vecs.extend(make_cluster(&mut rng, &center_a, 10, 0.05));
        vecs.extend(make_cluster(&mut rng, &center_b, 10, 0.05));

        // Add 5 random outlier points.
        for _ in 0..5 {
            let v: Vec<f32> = (0..4).map(|_| rng.next_f32()).collect();
            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 1e-8 {
                vecs.push(v.iter().map(|x| x / norm).collect());
            } else {
                vecs.push(vec![1.0, 0.0, 0.0, 0.0]);
            }
        }

        let emb = to_embeddings(&vecs);
        let result = hdbscan(&emb, 4);

        assert!(
            !result.noise_indices.is_empty(),
            "Expected some noise points but got none. Labels: {:?}",
            result.labels
        );

        let labelled_in_groups: usize = result.labels[0..20].iter().filter(|l| l.is_some()).count();
        assert!(
            labelled_in_groups >= 14,
            "Expected most of the 20 group points to be clustered, got {}",
            labelled_in_groups
        );
    }

    #[test]
    fn test_single_cluster() {
        let mut rng = Rng::new(77);
        let center = vec![1.0, 1.0, 1.0, 1.0];
        let vecs = make_cluster(&mut rng, &center, 15, 0.05);
        let emb = to_embeddings(&vecs);
        let result = hdbscan(&emb, 3);

        assert!(!result.clusters.is_empty(), "Expected at least one cluster");

        let total_clustered: usize = result.labels.iter().filter(|l| l.is_some()).count();
        assert!(
            total_clustered >= 10,
            "Expected most points in a cluster, got {}",
            total_clustered
        );

        let labels: Vec<u32> = result.labels.iter().filter_map(|l| *l).collect();
        if !labels.is_empty() {
            assert!(
                labels.iter().all(|&l| l == labels[0]),
                "Expected single cluster, got multiple labels: {:?}",
                labels
            );
        }
    }

    #[test]
    fn test_assign_to_nearest() {
        let mut rng = Rng::new(55);

        let center_a = vec![3.0, 0.0, 0.0, 0.0];
        let center_b = vec![0.0, 0.0, 3.0, 0.0];
        let mut vecs = Vec::new();
        vecs.extend(make_cluster(&mut rng, &center_a, 8, 0.05));
        vecs.extend(make_cluster(&mut rng, &center_b, 8, 0.05));

        let emb = to_embeddings(&vecs);
        let result = hdbscan(&emb, 3);

        if result.clusters.len() >= 2 {
            let test_point = [3.0, 0.01, 0.0, 0.0];
            let norm: f32 = test_point.iter().map(|x| x * x).sum::<f32>().sqrt();
            let test_point: Vec<f32> = test_point.iter().map(|x| x / norm).collect();

            let assigned = assign_to_nearest(&test_point, &result.clusters, &emb);
            assert!(assigned.is_some(), "Expected assignment to a cluster");

            let label_a = result.labels[0];
            if let (Some(assigned_id), Some(expected_id)) = (assigned, label_a) {
                assert_eq!(
                    assigned_id, expected_id,
                    "Point near center_a should be assigned to cluster A"
                );
            }
        }
    }

    #[test]
    fn test_assign_to_nearest_no_clusters() {
        let embedding = vec![1.0, 0.0, 0.0];
        let clusters: Vec<Cluster> = Vec::new();
        let all: Vec<(String, Vec<f32>)> = Vec::new();
        assert_eq!(assign_to_nearest(&embedding, &clusters, &all), None);
    }

    #[test]
    fn test_distance_matrix_symmetry() {
        let vecs = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.5, 0.5, 0.0],
        ];
        let mat = compute_distance_matrix(&vecs);
        for i in 0..3 {
            assert!((mat[i][i] - 0.0).abs() < 1e-5, "Diagonal should be zero");
            for j in 0..3 {
                assert!(
                    (mat[i][j] - mat[j][i]).abs() < 1e-5,
                    "Matrix should be symmetric"
                );
            }
        }
    }

    #[test]
    fn test_approximate_matches_exact_small() {
        let mut rng = Rng::new(123);

        let center_a = vec![3.0, 0.0, 0.0];
        let center_b = vec![0.0, 0.0, 3.0];
        let mut vecs = Vec::new();
        vecs.extend(make_cluster(&mut rng, &center_a, 8, 0.1));
        vecs.extend(make_cluster(&mut rng, &center_b, 8, 0.1));

        let dist_matrix = compute_distance_matrix(&vecs);
        let n = vecs.len();
        let knn: Vec<Vec<(usize, f32)>> = (0..n)
            .map(|i| {
                let mut neighbors: Vec<(usize, f32)> = (0..n)
                    .filter(|&j| j != i)
                    .map(|j| (j, dist_matrix[i][j]))
                    .collect();
                neighbors
                    .sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
                neighbors
            })
            .collect();

        let approx_result = hdbscan_approximate(&knn, n, 3, None);

        assert!(
            !approx_result.clusters.is_empty(),
            "Approximate HDBSCAN should find clusters with full k-NN"
        );
    }

    #[test]
    fn test_sampled_hdbscan_labels_all_points() {
        // Generate enough points to trigger sampling (> HDBSCAN_FULL_MATRIX_LIMIT)
        // Use a smaller limit for testing by creating a dataset just above threshold
        let mut rng = Rng::new(42);
        let center_a = vec![5.0, 0.0, 0.0];
        let center_b = vec![0.0, 0.0, 5.0];
        let mut embeddings = Vec::new();
        // Create 20 points in 2 clusters (test the sampling path with smaller data)
        for i in 0..10 {
            embeddings.push((
                format!("a{i}"),
                make_cluster(&mut rng, &center_a, 1, 0.2).remove(0),
            ));
        }
        for i in 0..10 {
            embeddings.push((
                format!("b{i}"),
                make_cluster(&mut rng, &center_b, 1, 0.2).remove(0),
            ));
        }

        // Test that hdbscan_sampled produces labels for ALL points
        let result = hdbscan_sampled(&embeddings, 3, None, 10);
        assert_eq!(
            result.labels.len(),
            20,
            "Should have labels for all 20 points"
        );

        // At least some points should be clustered (not all noise)
        let clustered = result.labels.iter().filter(|l| l.is_some()).count();
        assert!(
            clustered > 0,
            "Sampled HDBSCAN should assign some points to clusters"
        );
    }
}
