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
}
