use crate::search::survival::SurvivalCurve;
use crate::types::Memory;

/// Calculate current memory strength using Ebbinghaus forgetting curve.
/// strength(t) = exp(-lambda_eff * days^beta)
/// lambda_eff = decay_lambda / (1 + access_count * 0.2)
pub fn calculate_strength(memory: &Memory) -> f64 {
    calculate_strength_with_curve(memory, None)
}

/// Calculate memory strength with optional Kaplan-Meier survival curve.
/// When a per-cluster survival curve is available and has sufficient data,
/// it replaces the fixed Ebbinghaus formula for data-driven decay.
pub fn calculate_strength_with_curve(memory: &Memory, curve: Option<&SurvivalCurve>) -> f64 {
    if memory.importance == crate::types::Importance::Critical {
        return 1.0; // Critical never decays
    }
    let days = (chrono::Utc::now() - memory.last_accessed).num_seconds() as f64 / 86400.0;
    if days <= 0.0 {
        return 1.0;
    }
    let lambda_eff = memory.decay_lambda / (1.0 + memory.access_count as f64 * 0.2);
    let beta = memory.layer.beta();
    let ebbinghaus = (-lambda_eff * days.powf(beta)).exp();

    // Use adaptive_strength for KM curve blending (cold-start safe)
    crate::search::survival::adaptive_strength(days, curve, ebbinghaus, 20, 50)
}

/// Recency boost: recent memories get higher scores.
/// 24h → +50%, 7 days → linearly decays to +0%, older → no boost.
pub fn recency_boost(memory: &Memory) -> f32 {
    let hours = (chrono::Utc::now() - memory.created_at).num_hours() as f64;
    if hours <= 24.0 {
        1.5
    } else if hours <= 168.0 {
        // Linear decay from 1.5 to 1.0 over 7 days
        1.0 + 0.5 * (1.0 - (hours - 24.0) / 144.0) as f32
    } else {
        1.0
    }
}

/// Apply strength weighting to RRF score with recency boost.
/// final_score = rrf_score * strength * (1 + access_count * 0.2) * recency_boost
pub fn apply_strength_weighting(rrf_score: f32, memory: &Memory) -> f32 {
    apply_strength_weighting_with_curve(rrf_score, memory, None)
}

/// Apply strength weighting with optional per-cluster survival curve.
pub fn apply_strength_weighting_with_curve(rrf_score: f32, memory: &Memory, curve: Option<&SurvivalCurve>) -> f32 {
    let strength = calculate_strength_with_curve(memory, curve);
    let recency = recency_boost(memory);
    rrf_score * strength as f32 * (1.0 + memory.access_count as f32 * 0.2) * recency
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
            concept_ids: vec![],
            status: MemoryStatus::default(),
            embedding: None,
            tier: MemoryTier::Warm,
            cluster_id: None,
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
