use crate::types::Memory;

/// Calculate current memory strength using Ebbinghaus forgetting curve.
/// strength(t) = exp(-lambda_eff * days^beta)
/// lambda_eff = decay_lambda / (1 + access_count * 0.2)
pub fn calculate_strength(memory: &Memory) -> f64 {
    if memory.importance == crate::types::Importance::Critical {
        return 1.0; // Critical never decays
    }
    let days = (chrono::Utc::now() - memory.last_accessed).num_seconds() as f64 / 86400.0;
    if days <= 0.0 {
        return 1.0;
    }
    let lambda_eff = memory.decay_lambda / (1.0 + memory.access_count as f64 * 0.2);
    let beta = memory.layer.beta();
    (-lambda_eff * days.powf(beta)).exp()
}

/// Apply strength weighting to RRF score.
/// final_score = rrf_score * strength * (1 + access_count * 0.2)
pub fn apply_strength_weighting(rrf_score: f32, memory: &Memory) -> f32 {
    let strength = calculate_strength(memory);
    rrf_score * strength as f32 * (1.0 + memory.access_count as f32 * 0.2)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::*;
    use chrono::{Duration, Utc};

    fn make_memory(importance: Importance, layer: MemoryLayer, days_ago: i64, access_count: u32) -> Memory {
        Memory {
            id: "test".to_string(),
            layer,
            topic: "test".to_string(),
            summary: "test".to_string(),
            content: "test".to_string(),
            keywords: vec![],
            importance,
            source: Source::Manual,
            strength: 1.0,
            decay_lambda: 0.06 * importance.decay_factor(),
            access_count,
            superseded_by: None,
            related_ids: vec![],
            status: MemoryStatus::default(),
            embedding: None,
            created_at: Utc::now() - Duration::days(days_ago),
            updated_at: Utc::now(),
            last_accessed: Utc::now() - Duration::days(days_ago),
        }
    }

    #[test]
    fn test_critical_no_decay() {
        let mem = make_memory(Importance::Critical, MemoryLayer::LTM, 365, 0);
        let strength = calculate_strength(&mem);
        assert!((strength - 1.0).abs() < 1e-6, "Critical should always be 1.0, got {}", strength);
    }

    #[test]
    fn test_low_fast_decay() {
        let mem = make_memory(Importance::Low, MemoryLayer::STM, 90, 0);
        let strength = calculate_strength(&mem);
        assert!(strength < 0.1, "Low STM after 90 days should be very low, got {}", strength);
    }

    #[test]
    fn test_access_slows_decay() {
        let no_access = make_memory(Importance::Medium, MemoryLayer::STM, 30, 0);
        let with_access = make_memory(Importance::Medium, MemoryLayer::STM, 30, 10);

        let s_no = calculate_strength(&no_access);
        let s_with = calculate_strength(&with_access);

        assert!(
            s_with > s_no,
            "More access should slow decay: no_access={}, with_access={}",
            s_no,
            s_with
        );
    }

    #[test]
    fn test_fresh_memory_full_strength() {
        let mem = make_memory(Importance::Medium, MemoryLayer::STM, 0, 0);
        let strength = calculate_strength(&mem);
        assert!(
            (strength - 1.0).abs() < 0.01,
            "Fresh memory should be ~1.0, got {}",
            strength
        );
    }
}
