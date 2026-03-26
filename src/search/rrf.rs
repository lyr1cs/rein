use std::collections::HashMap;

/// Reciprocal Rank Fusion - merges multiple ranked lists into one.
/// Each entry in ranked_lists is (results_vec, weight).
/// results_vec contains (id, score) pairs ordered by relevance.
/// k is the smoothing constant (default 60.0).
pub fn reciprocal_rank_fusion(
    ranked_lists: &[(Vec<(String, f32)>, f32)],
    k: f32,
) -> Vec<(String, f32)> {
    let mut scores: HashMap<String, f32> = HashMap::new();
    for (results, weight) in ranked_lists {
        for (rank, (id, _)) in results.iter().enumerate() {
            *scores.entry(id.clone()).or_default() += weight / (k + rank as f32 + 1.0);
        }
    }
    let mut merged: Vec<_> = scores.into_iter().collect();
    merged.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    merged
}

/// Convex Combination fusion — normalizes scores to [0,1] and blends with alpha.
/// score = alpha * sparse_norm + (1-alpha) * dense_norm
/// Based on Bruch et al. 2023: CC outperforms RRF with better sample efficiency.
pub fn convex_combination(
    fts_results: &[(String, f32)],
    vec_results: &[(String, f32)],
    alpha: f32,
) -> Vec<(String, f32)> {
    // Normalize each list to [0,1] using min-max normalization.
    // Singleton lists get score 1.0 (not 0.0) to preserve their contribution.
    fn normalize(results: &[(String, f32)]) -> Vec<(String, f32)> {
        if results.is_empty() { return vec![]; }
        let max = results.iter().map(|(_, s)| *s).fold(f32::NEG_INFINITY, f32::max);
        let min = results.iter().map(|(_, s)| *s).fold(f32::INFINITY, f32::min);
        let range = max - min;
        if range < 1e-6 {
            // All scores equal (including singleton) → assign 1.0 to all
            return results.iter().map(|(id, _)| (id.clone(), 1.0)).collect();
        }
        results.iter().map(|(id, s)| (id.clone(), (s - min) / range)).collect()
    }

    let fts_norm = normalize(fts_results);
    let vec_norm = normalize(vec_results);

    let mut scores: HashMap<String, f32> = HashMap::new();
    for (id, s) in &fts_norm {
        *scores.entry(id.clone()).or_default() += alpha * s;
    }
    for (id, s) in &vec_norm {
        *scores.entry(id.clone()).or_default() += (1.0 - alpha) * s;
    }

    let mut merged: Vec<_> = scores.into_iter().collect();
    merged.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    merged
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rrf_single_list() {
        let list = vec![
            ("a".to_string(), 10.0),
            ("b".to_string(), 5.0),
            ("c".to_string(), 1.0),
        ];
        let k = 60.0;
        let weight = 1.0;
        let result = reciprocal_rank_fusion(&[(list, weight)], k);

        assert_eq!(result.len(), 3);
        // rank 0: weight/(k+0+1) = 1/61
        assert!((result[0].1 - 1.0 / 61.0).abs() < 1e-6);
        // rank 1: weight/(k+1+1) = 1/62
        assert!((result[1].1 - 1.0 / 62.0).abs() < 1e-6);
        // rank 2: weight/(k+2+1) = 1/63
        assert!((result[2].1 - 1.0 / 63.0).abs() < 1e-6);
    }

    #[test]
    fn test_rrf_two_lists() {
        let list1 = vec![
            ("a".to_string(), 10.0),
            ("b".to_string(), 5.0),
        ];
        let list2 = vec![
            ("a".to_string(), 8.0),
            ("c".to_string(), 3.0),
        ];
        let k = 60.0;
        let result = reciprocal_rank_fusion(&[(list1, 1.0), (list2, 1.0)], k);

        // "a" appears in both lists at rank 0 => 1/61 + 1/61
        let a_score = result.iter().find(|(id, _)| id == "a").unwrap().1;
        let b_score = result.iter().find(|(id, _)| id == "b").unwrap().1;
        let c_score = result.iter().find(|(id, _)| id == "c").unwrap().1;

        assert!(a_score > b_score, "a in both lists should score higher than b in one");
        assert!(a_score > c_score, "a in both lists should score higher than c in one");
        assert!((a_score - 2.0 / 61.0).abs() < 1e-6);
    }

    #[test]
    fn test_rrf_disjoint() {
        let list1 = vec![("a".to_string(), 10.0)];
        let list2 = vec![("b".to_string(), 8.0)];
        let k = 60.0;
        let result = reciprocal_rank_fusion(&[(list1, 1.0), (list2, 1.0)], k);

        assert_eq!(result.len(), 2);
        // Both at rank 0 in their respective lists, same weight => same score
        assert!((result[0].1 - result[1].1).abs() < 1e-6);
    }

    #[test]
    fn test_rrf_empty() {
        let result = reciprocal_rank_fusion(&[], 60.0);
        assert!(result.is_empty());
    }

    #[test]
    fn test_cc_basic() {
        let fts = vec![
            ("a".to_string(), 10.0),
            ("b".to_string(), 5.0),
        ];
        let vec = vec![
            ("a".to_string(), 8.0),
            ("c".to_string(), 6.0),
        ];
        let result = convex_combination(&fts, &vec, 0.5);

        // "a" appears in both with high scores → should be #1
        assert_eq!(result[0].0, "a");
        // "a" gets: 0.5 * 1.0 (max in fts) + 0.5 * 1.0 (max in vec) = 1.0
        assert!((result[0].1 - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_cc_alpha_bias() {
        // Multiple items needed so normalization produces non-trivial scores
        let fts = vec![("x".to_string(), 10.0), ("z".to_string(), 2.0)];
        let vec = vec![("y".to_string(), 10.0), ("z".to_string(), 2.0)];
        // alpha=0.9 heavily favors FTS
        let result = convex_combination(&fts, &vec, 0.9);
        let x_score = result.iter().find(|(id, _)| id == "x").unwrap().1;
        let y_score = result.iter().find(|(id, _)| id == "y").unwrap().1;
        assert!(x_score > y_score, "alpha=0.9 should favor FTS (x={x_score}) over vec (y={y_score})");
    }

    #[test]
    fn test_cc_empty() {
        let result = convex_combination(&[], &[], 0.5);
        assert!(result.is_empty());
    }
}
