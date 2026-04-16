//! Kaplan-Meier survival analysis for data-driven memory decay.
//!
//! Replaces fixed Ebbinghaus decay curves with per-cluster survival curves
//! estimated from actual access patterns. Uses cold-start blending to
//! gracefully transition from Ebbinghaus fallback when data is sparse.

use serde::{Deserialize, Serialize};

/// A survival observation: time interval between two accesses.
#[derive(Debug, Clone)]
pub struct SurvivalInterval {
    /// Time between accesses in days (or time since last access if censored).
    pub duration_days: f64,
    /// `true` = was accessed again (event observed), `false` = censored (still waiting).
    pub is_event: bool,
}

/// Kaplan-Meier survival curve: a step function of (time, probability).
///
/// The curve starts at (0.0, 1.0) and decreases monotonically. Each step
/// corresponds to an observed event time where the survival probability drops.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SurvivalCurve {
    /// Sorted (time_days, survival_probability) pairs. Probability decreases over time.
    pub steps: Vec<(f64, f64)>,
    /// Number of uncensored observations used to build this curve.
    pub event_count: usize,
    /// Total observations (events + censored).
    pub total_count: usize,
    /// Median survival time (50% probability). `None` if curve never drops below 0.5.
    pub median_survival: Option<f64>,
}

impl SurvivalCurve {
    /// Get survival probability at a given time.
    ///
    /// Uses the step function directly for times within the observed range.
    /// For times beyond the last step, extrapolates using log-linear extension
    /// (constant hazard rate derived from the last two steps).
    pub fn probability_at(&self, days: f64) -> f64 {
        if days <= 0.0 {
            return 1.0;
        }
        if self.steps.is_empty() {
            return 1.0;
        }

        // Find the last step at or before `days`
        let mut prob = 1.0;
        let mut last_time = 0.0;
        let mut found = false;

        for &(t, p) in &self.steps {
            if t > days {
                break;
            }
            prob = p;
            last_time = t;
            found = true;
        }

        if !found {
            return 1.0;
        }

        // Check if we're within the observed range
        let last_step = self.steps.last().unwrap();
        if days <= last_step.0 {
            return prob;
        }

        // Extrapolate beyond the last step using log-linear extension.
        // Derive hazard rate from the last two steps (or from (0,1) if only one step).
        let (t_prev, p_prev, t_last, p_last) = if self.steps.len() >= 2 {
            let n = self.steps.len();
            let (t1, p1) = self.steps[n - 2];
            let (t2, p2) = self.steps[n - 1];
            (t1, p1, t2, p2)
        } else {
            (0.0, 1.0, self.steps[0].0, self.steps[0].1)
        };

        let dt = t_last - t_prev;
        // Degenerate curves: no time delta, zero survival, or tied steps (p_prev == p_last)
        // cannot produce a meaningful hazard. Fall back to the last known probability.
        if dt <= 0.0 || p_prev <= 0.0 || p_last <= 0.0 {
            return prob;
        }
        if (p_prev - p_last).abs() < f64::EPSILON {
            // Tied steps ⇒ ln(1) = 0 ⇒ hazard = 0 ⇒ no decay beyond the last point.
            return prob;
        }

        // Hazard rate: h = -ln(S(t_last)/S(t_prev)) / dt
        let ratio = p_last / p_prev;
        if !ratio.is_finite() || ratio <= 0.0 {
            return prob;
        }
        let hazard = -ratio.ln() / dt;
        if !hazard.is_finite() || hazard <= 0.0 {
            return prob;
        }

        // S(days) = S(t_last) * exp(-hazard * (days - t_last))
        let extrapolated = p_last * (-hazard * (days - last_time)).exp();
        if !extrapolated.is_finite() {
            return prob;
        }
        extrapolated.clamp(0.0, 1.0)
    }
}

/// Compute Kaplan-Meier survival curve from a set of intervals.
///
/// Returns `None` if no intervals provided. The estimator handles both
/// event and censored observations correctly: censored observations reduce
/// the risk set without producing a step in the curve.
pub fn kaplan_meier(intervals: &[SurvivalInterval]) -> Option<SurvivalCurve> {
    if intervals.is_empty() {
        return None;
    }

    // Sort by duration, events before censored at same time
    let mut sorted: Vec<_> = intervals.to_vec();
    sorted.sort_by(|a, b| {
        a.duration_days
            .partial_cmp(&b.duration_days)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                // Events before censored at the same time
                b.is_event.cmp(&a.is_event)
            })
    });

    let total_count = sorted.len();
    let event_count = sorted.iter().filter(|i| i.is_event).count();

    // Build the curve
    let mut steps = vec![(0.0, 1.0)];
    let mut survival = 1.0;
    let mut at_risk = total_count;
    let mut median_survival: Option<f64> = None;

    let mut i = 0;
    while i < sorted.len() {
        let t = sorted[i].duration_days;

        // Count events and censored at this exact time
        let mut events_at_t = 0usize;
        let mut censored_at_t = 0usize;

        while i < sorted.len() && (sorted[i].duration_days - t).abs() < 1e-12 {
            if sorted[i].is_event {
                events_at_t += 1;
            } else {
                censored_at_t += 1;
            }
            i += 1;
        }

        if events_at_t > 0 && at_risk > 0 {
            survival *= 1.0 - (events_at_t as f64 / at_risk as f64);
            steps.push((t, survival));

            if median_survival.is_none() && survival <= 0.5 {
                median_survival = Some(t);
            }
        }

        at_risk -= events_at_t + censored_at_t;
    }

    Some(SurvivalCurve {
        steps,
        event_count,
        total_count,
        median_survival,
    })
}

/// Convert a memory's access timestamps into survival intervals.
///
/// Each consecutive pair of timestamps produces an event interval (the memory
/// was accessed again). The final interval from the last access to `now` is
/// censored (we don't yet know if/when the next access will occur).
pub fn access_times_to_intervals(
    access_times: &[chrono::DateTime<chrono::Utc>],
    now: chrono::DateTime<chrono::Utc>,
) -> Vec<SurvivalInterval> {
    if access_times.is_empty() {
        return vec![];
    }

    let mut sorted: Vec<_> = access_times.to_vec();
    sorted.sort();

    let mut intervals = Vec::with_capacity(sorted.len());

    // Consecutive pairs are events (the memory was accessed again)
    for window in sorted.windows(2) {
        let duration = (window[1] - window[0]).num_seconds() as f64 / 86400.0;
        intervals.push(SurvivalInterval {
            duration_days: duration.max(0.0),
            is_event: true,
        });
    }

    // Last access to now is censored
    let last = sorted.last().unwrap();
    let censored_duration = (now - *last).num_seconds() as f64 / 86400.0;
    intervals.push(SurvivalInterval {
        duration_days: censored_duration.max(0.0),
        is_event: false,
    });

    intervals
}

/// Compute memory strength using survival curve with Ebbinghaus fallback.
///
/// Blends between Ebbinghaus (parametric) and survival curve (data-driven):
/// - Below `cold_start_min` observations: pure Ebbinghaus
/// - Between `cold_start_min` and `cold_start_max`: linear blend
/// - At or above `cold_start_max`: pure survival curve
///
/// This ensures stable behavior when data is sparse while transitioning
/// to the empirical curve as evidence accumulates.
pub fn adaptive_strength(
    days_since_last_access: f64,
    curve: Option<&SurvivalCurve>,
    ebbinghaus_strength: f64,
    cold_start_min: usize,
    cold_start_max: usize,
) -> f64 {
    let curve = match curve {
        Some(c) => c,
        None => return ebbinghaus_strength,
    };

    let n = curve.total_count;

    if n < cold_start_min {
        return ebbinghaus_strength;
    }

    let survival_strength = curve.probability_at(days_since_last_access);

    if n >= cold_start_max {
        return survival_strength;
    }

    // Linear blend: weight increases from 0.0 at cold_start_min to 1.0 at cold_start_max
    let denom = cold_start_max.saturating_sub(cold_start_min);
    if denom == 0 {
        return survival_strength;
    }
    let blend = (n - cold_start_min) as f64 / denom as f64;
    ebbinghaus_strength * (1.0 - blend) + survival_strength * blend
}

/// Derive an STM→LTM promotion threshold from a survival curve.
///
/// Clusters with longer median survival require more repeated accesses before promotion,
/// while fast-decaying clusters promote with fewer repeated accesses.
pub fn promotion_access_threshold(curve: &SurvivalCurve) -> u32 {
    // Default to 28.0 when median is None (insufficient data), which maps to
    // threshold=5, preserving backward compatibility with the legacy fixed threshold.
    let median = curve.median_survival.unwrap_or(28.0).clamp(1.0, 28.0);
    ((median / 7.0).ceil() as u32 + 1).clamp(2, 8)
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{Duration, Utc};

    #[test]
    fn test_kaplan_meier_uniform_events() {
        // 10 events at days 1..=10, no censoring
        let intervals: Vec<SurvivalInterval> = (1..=10)
            .map(|d| SurvivalInterval {
                duration_days: d as f64,
                is_event: true,
            })
            .collect();

        let curve = kaplan_meier(&intervals).unwrap();
        assert_eq!(curve.event_count, 10);
        assert_eq!(curve.total_count, 10);

        // At t=5: 5 events have occurred out of 10 subjects
        // S(1) = 9/10, S(2) = 9/10 * 8/9 = 8/10, ..., S(5) = 5/10 = 0.5
        let s5 = curve.probability_at(5.0);
        assert!((s5 - 0.5).abs() < 1e-6, "S(5) should be ~0.5, got {}", s5);
    }

    #[test]
    fn test_kaplan_meier_with_censoring() {
        // Mix of events and censored observations
        let intervals = vec![
            SurvivalInterval {
                duration_days: 1.0,
                is_event: true,
            },
            SurvivalInterval {
                duration_days: 2.0,
                is_event: false,
            }, // censored
            SurvivalInterval {
                duration_days: 3.0,
                is_event: true,
            },
            SurvivalInterval {
                duration_days: 4.0,
                is_event: false,
            }, // censored
            SurvivalInterval {
                duration_days: 5.0,
                is_event: true,
            },
        ];

        let curve = kaplan_meier(&intervals).unwrap();
        assert_eq!(curve.event_count, 3);
        assert_eq!(curve.total_count, 5);

        // t=1: at_risk=5, events=1 → S = 4/5 = 0.8
        // t=2: censored, at_risk drops to 3 (was 4, minus 1 censored)
        // t=3: at_risk=3, events=1 → S = 0.8 * 2/3 ≈ 0.5333
        // t=4: censored, at_risk drops to 1
        // t=5: at_risk=1, events=1 → S = 0.5333 * 0/1 = 0.0
        let s1 = curve.probability_at(1.0);
        assert!((s1 - 0.8).abs() < 1e-6, "S(1) should be 0.8, got {}", s1);

        let s3 = curve.probability_at(3.0);
        let expected = 0.8 * (2.0 / 3.0);
        assert!(
            (s3 - expected).abs() < 1e-6,
            "S(3) should be ~{}, got {}",
            expected,
            s3
        );
    }

    #[test]
    fn test_probability_at_interpolation() {
        let intervals: Vec<SurvivalInterval> = vec![2.0, 5.0, 8.0]
            .into_iter()
            .map(|d| SurvivalInterval {
                duration_days: d,
                is_event: true,
            })
            .collect();

        let curve = kaplan_meier(&intervals).unwrap();

        // Before any event: should be 1.0
        assert!((curve.probability_at(0.5) - 1.0).abs() < 1e-6);

        // Between events: step function, so S(3) = S(2)
        let s2 = curve.probability_at(2.0);
        let s3 = curve.probability_at(3.0);
        assert!(
            (s2 - s3).abs() < 1e-6,
            "Between steps, probability should be constant: S(2)={}, S(3)={}",
            s2,
            s3
        );

        // Monotonically decreasing at step points
        let s5 = curve.probability_at(5.0);
        let s8 = curve.probability_at(8.0);
        assert!(s2 > s5, "S(2)={} should be > S(5)={}", s2, s5);
        assert!(s5 > s8, "S(5)={} should be > S(8)={}", s5, s8);
    }

    #[test]
    fn test_probability_at_extrapolation() {
        // Use events + censored so the curve doesn't drop to 0.0
        let intervals = vec![
            SurvivalInterval {
                duration_days: 1.0,
                is_event: true,
            },
            SurvivalInterval {
                duration_days: 3.0,
                is_event: true,
            },
            SurvivalInterval {
                duration_days: 5.0,
                is_event: false,
            }, // censored keeps S > 0
            SurvivalInterval {
                duration_days: 5.0,
                is_event: false,
            },
        ];

        let curve = kaplan_meier(&intervals).unwrap();

        let s_last_step = *curve.steps.last().map(|(_, p)| p).unwrap();
        assert!(
            s_last_step > 0.0,
            "Last step should be > 0 for extrapolation test"
        );

        // Beyond last observation, extrapolation should still decrease
        let s_beyond = curve.probability_at(20.0);
        assert!(
            s_beyond < s_last_step,
            "Extrapolated S(20)={} should be < last step S={}",
            s_beyond,
            s_last_step
        );
        assert!(s_beyond >= 0.0, "Extrapolated probability should be >= 0");
    }

    #[test]
    fn test_access_times_to_intervals() {
        let now = Utc::now();
        let times = vec![
            now - Duration::days(20),
            now - Duration::days(15),
            now - Duration::days(10),
            now - Duration::days(5),
            now - Duration::days(1),
        ];

        let intervals = access_times_to_intervals(&times, now);

        // 4 event intervals + 1 censored
        assert_eq!(intervals.len(), 5);
        assert_eq!(
            intervals.iter().filter(|i| i.is_event).count(),
            4,
            "Should have 4 event intervals"
        );
        assert_eq!(
            intervals.iter().filter(|i| !i.is_event).count(),
            1,
            "Should have 1 censored interval"
        );

        // Last interval should be censored (~1 day)
        let last = intervals.last().unwrap();
        assert!(!last.is_event, "Last interval should be censored");
        assert!(
            (last.duration_days - 1.0).abs() < 0.01,
            "Last censored interval should be ~1 day, got {}",
            last.duration_days
        );

        // First event interval: ~5 days between day-20 and day-15
        assert!(
            (intervals[0].duration_days - 5.0).abs() < 0.01,
            "First interval should be ~5 days, got {}",
            intervals[0].duration_days
        );
    }

    #[test]
    fn test_adaptive_strength_pure_ebbinghaus() {
        // Below cold_start_min: pure Ebbinghaus
        let curve = SurvivalCurve {
            steps: vec![(0.0, 1.0), (5.0, 0.5)],
            event_count: 10,
            total_count: 10, // below min of 20
            median_survival: Some(5.0),
        };

        let strength = adaptive_strength(5.0, Some(&curve), 0.7, 20, 50);
        assert!(
            (strength - 0.7).abs() < 1e-6,
            "Below cold_start_min should use pure Ebbinghaus: got {}",
            strength
        );
    }

    #[test]
    fn test_adaptive_strength_pure_survival() {
        // At or above cold_start_max: pure survival
        let curve = SurvivalCurve {
            steps: vec![(0.0, 1.0), (5.0, 0.5)],
            event_count: 45,
            total_count: 50, // at max
            median_survival: Some(5.0),
        };

        let survival_prob = curve.probability_at(5.0);
        let strength = adaptive_strength(5.0, Some(&curve), 0.7, 20, 50);
        assert!(
            (strength - survival_prob).abs() < 1e-6,
            "At cold_start_max should use pure survival: got {} vs {}",
            strength,
            survival_prob
        );
    }

    #[test]
    fn test_promotion_access_threshold_tracks_median_survival() {
        let fast = SurvivalCurve {
            steps: vec![(0.0, 1.0), (3.0, 0.4)],
            event_count: 20,
            total_count: 25,
            median_survival: Some(3.0),
        };
        let slow = SurvivalCurve {
            steps: vec![(0.0, 1.0), (14.0, 0.4)],
            event_count: 20,
            total_count: 25,
            median_survival: Some(14.0),
        };

        assert!(promotion_access_threshold(&fast) < promotion_access_threshold(&slow));
    }

    #[test]
    fn test_adaptive_strength_blending() {
        // Between cold_start_min and cold_start_max: linear blend
        let curve = SurvivalCurve {
            steps: vec![(0.0, 1.0), (5.0, 0.4)],
            event_count: 30,
            total_count: 35, // between 20 and 50
            median_survival: Some(5.0),
        };

        let ebbinghaus = 0.7;
        let survival = curve.probability_at(5.0);
        let blend = (35 - 20) as f64 / (50 - 20) as f64; // 0.5
        let expected = ebbinghaus * (1.0 - blend) + survival * blend;

        let strength = adaptive_strength(5.0, Some(&curve), ebbinghaus, 20, 50);
        assert!(
            (strength - expected).abs() < 1e-6,
            "Blended strength should be {}, got {}",
            expected,
            strength
        );
    }

    #[test]
    fn test_adaptive_strength_no_curve() {
        // No curve: pure Ebbinghaus
        let strength = adaptive_strength(5.0, None, 0.7, 20, 50);
        assert!(
            (strength - 0.7).abs() < 1e-6,
            "No curve should return Ebbinghaus: got {}",
            strength
        );
    }

    #[test]
    fn test_empty_intervals_returns_none() {
        let result = kaplan_meier(&[]);
        assert!(result.is_none(), "Empty intervals should return None");
    }

    #[test]
    fn test_single_event_curve() {
        let intervals = vec![SurvivalInterval {
            duration_days: 3.0,
            is_event: true,
        }];

        let curve = kaplan_meier(&intervals).unwrap();
        assert_eq!(curve.event_count, 1);
        assert_eq!(curve.total_count, 1);
        assert_eq!(curve.steps.len(), 2); // (0, 1.0) and (3, 0.0)

        // S(0) = 1.0, S(3) = 0.0 (single subject, single event)
        assert!((curve.probability_at(0.0) - 1.0).abs() < 1e-6);
        assert!((curve.probability_at(3.0) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_median_survival_calculation() {
        // 10 events at days 1..=10: median should be 5 (where S drops to 0.5)
        let intervals: Vec<SurvivalInterval> = (1..=10)
            .map(|d| SurvivalInterval {
                duration_days: d as f64,
                is_event: true,
            })
            .collect();

        let curve = kaplan_meier(&intervals).unwrap();
        assert!(
            curve.median_survival.is_some(),
            "Should have a median survival"
        );
        assert!(
            (curve.median_survival.unwrap() - 5.0).abs() < 1e-6,
            "Median survival should be 5.0, got {}",
            curve.median_survival.unwrap()
        );
    }

    #[test]
    fn test_median_survival_none_when_never_below_half() {
        // All censored: curve never drops, so no median
        let intervals = vec![
            SurvivalInterval {
                duration_days: 1.0,
                is_event: false,
            },
            SurvivalInterval {
                duration_days: 2.0,
                is_event: false,
            },
        ];

        let curve = kaplan_meier(&intervals).unwrap();
        assert!(
            curve.median_survival.is_none(),
            "All-censored curve should have no median"
        );
    }

    #[test]
    fn test_access_times_to_intervals_empty() {
        let intervals = access_times_to_intervals(&[], Utc::now());
        assert!(intervals.is_empty());
    }

    #[test]
    fn test_access_times_to_intervals_single() {
        let now = Utc::now();
        let times = vec![now - Duration::days(3)];
        let intervals = access_times_to_intervals(&times, now);

        assert_eq!(intervals.len(), 1);
        assert!(
            !intervals[0].is_event,
            "Single access should produce censored interval"
        );
        assert!(
            (intervals[0].duration_days - 3.0).abs() < 0.01,
            "Duration should be ~3 days"
        );
    }

    #[test]
    fn survival_degenerate_tied_steps_returns_finite_probability() {
        // Curve with two tied steps (p_prev == p_last) must not produce NaN/Inf
        // from ln(1) / dt hazard computation.
        let curve = SurvivalCurve {
            steps: vec![(1.0, 0.5), (2.0, 0.5)],
            event_count: 2,
            total_count: 2,
            median_survival: None,
        };
        let p = curve.probability_at(10.0);
        assert!(p.is_finite(), "tied-step curve must return finite probability");
        assert!((0.0..=1.0).contains(&p));
    }

    #[test]
    fn survival_degenerate_zero_p_prev_returns_finite() {
        let curve = SurvivalCurve {
            steps: vec![(1.0, 0.0), (2.0, 0.0)],
            event_count: 2,
            total_count: 2,
            median_survival: None,
        };
        let p = curve.probability_at(10.0);
        assert!(p.is_finite());
        assert!((0.0..=1.0).contains(&p));
    }
}
