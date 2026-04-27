use std::collections::HashSet;
use std::sync::OnceLock;

use jieba_rs::Jieba;
use unicode_normalization::UnicodeNormalization;

use crate::extract::llm::ExtractorKind;
use crate::extract::temporal::{extract_temporal_rule_based, temporal_conflict, TemporalAnchor};
use crate::extract::triples::{extract_triples_rule_based, triple_overlap_score, Triple};
use crate::store::SqliteStore;
use crate::types::{Memory, MemoryStore, ReinResult};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LexicalDedupScore {
    pub jaccard: f32,
    pub containment: f32,
    pub score: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CandidateScore {
    pub memory_id: String,
    pub lexical: LexicalDedupScore,
    pub topic_variant_match: bool,
    pub cluster_match: bool,
    pub recency_days: i64,
    pub final_score: f32,
}

/// Normalize text for similarity comparison: lowercase + strip punctuation.
fn normalize_tokens(text: &str) -> HashSet<String> {
    let mut tokens: HashSet<String> = text
        .split_whitespace()
        .map(|t| {
            t.to_lowercase()
                .chars()
                .filter(|c| c.is_alphanumeric())
                .collect::<String>()
        })
        .filter(|t| !t.is_empty())
        .collect();

    if contains_cjk(text) {
        // HMM=true for better OOV recognition (matches extract_keywords_from_text);
        // aligning the two paths prevents threshold drift between dedup similarity
        // and keyword extraction on novel terminology (e.g. product names).
        let jieba_tokens = jieba()
            .cut(text, true)
            .into_iter()
            .map(normalize_token)
            .filter(|t| !t.is_empty())
            .filter(|t| t.chars().count() > 1 || t.chars().any(is_cjk));
        tokens.extend(jieba_tokens);

        let chars: Vec<char> = text
            .nfc()
            .flat_map(char::to_lowercase)
            .filter(|c| !c.is_whitespace())
            .filter(|c| c.is_alphanumeric() || is_cjk(*c))
            .collect();

        if chars.len() < 2 {
            if !chars.is_empty() {
                tokens.insert(chars.iter().collect());
            }
        } else {
            for window in chars.windows(2) {
                tokens.insert(window.iter().collect());
            }
        }
    }

    tokens
}

pub(crate) fn jieba() -> &'static Jieba {
    static INSTANCE: OnceLock<Jieba> = OnceLock::new();
    INSTANCE.get_or_init(Jieba::new)
}

fn normalize_token(text: &str) -> String {
    text.nfc()
        .flat_map(char::to_lowercase)
        .filter(|c| c.is_alphanumeric() || is_cjk(*c))
        .collect()
}

pub fn contains_cjk(text: &str) -> bool {
    text.chars().any(is_cjk)
}

pub(crate) fn is_cjk(ch: char) -> bool {
    matches!(
        ch as u32,
        0x3400..=0x4DBF     // CJK Extension A
            | 0x4E00..=0x9FFF // CJK Unified Ideographs
            | 0xF900..=0xFAFF // CJK Compatibility Ideographs
            | 0x20000..=0x2A6DF // CJK Extension B
            | 0x3040..=0x309F // Hiragana
            | 0x30A0..=0x30FF // Katakana
            | 0x31F0..=0x31FF // Katakana Phonetic Extensions
            | 0xFF65..=0xFF9F // Halfwidth Katakana
            | 0xAC00..=0xD7AF // Hangul syllables
    )
}

/// Jaccard similarity over token sets.
///
/// Returns `None` when comparison is undefined — specifically when the union
/// is empty (both sets empty). Returning `None` distinguishes "no basis for
/// comparison" from "definite zero overlap", which previously collided at 0.0
/// and caused empty-text inputs to look maximally dissimilar even though they
/// were simply un-tokenizable.
fn jaccard_from_sets(set_a: &HashSet<String>, set_b: &HashSet<String>) -> Option<f32> {
    let union = set_a.union(set_b).count();
    if union == 0 {
        return None;
    }
    let intersection = set_a.intersection(set_b).count();
    Some(intersection as f32 / union as f32)
}

/// Containment similarity: fraction of the smaller set covered by the intersection.
///
/// Returns `None` when either set is empty (smaller == 0) — there is no
/// "smaller side" to ground containment against. Callers must treat `None` as
/// "skip, do not dedup".
fn containment_from_sets(set_a: &HashSet<String>, set_b: &HashSet<String>) -> Option<f32> {
    let smaller = set_a.len().min(set_b.len());
    if smaller == 0 {
        return None;
    }
    Some(set_a.intersection(set_b).count() as f32 / smaller as f32)
}

/// Tokenize text for search/FTS purposes. Returns sorted tokens (deterministic order)
/// suitable for FTS queries. For CJK text, produces jieba segments + bigrams.
/// Used by Tantivy indexing, keyword extraction, etc.
pub fn tokenize_for_search(text: &str) -> Vec<String> {
    let mut tokens: Vec<String> = normalize_tokens(text).into_iter().collect();
    tokens.sort(); // deterministic order — HashSet iteration is random
    tokens
}

/// Tokenize text and return as a space-joined string suitable for Tantivy queries.
/// Order is deterministic (sorted).
pub fn tokenize_for_fts(text: &str) -> String {
    let tokens = tokenize_for_search(text);
    tokens.join(" ")
}

/// Extract meaningful keywords from text using jieba for CJK, whitespace for others.
/// Returns deduplicated keywords sorted by length (longer = more specific).
pub fn extract_keywords_from_text(text: &str, max_keywords: usize) -> Vec<String> {
    let raw: Vec<String> = if contains_cjk(text) {
        // Use jieba for meaningful word-level segmentation
        jieba()
            .cut(text, true) // HMM mode for better unknown word detection
            .into_iter()
            .map(normalize_token)
            .filter(|t| !t.is_empty() && t.chars().count() >= 2)
            .collect()
    } else {
        text.split_whitespace()
            .map(|t| {
                t.to_lowercase()
                    .chars()
                    .filter(|c| c.is_alphanumeric())
                    .collect::<String>()
            })
            .filter(|t| t.len() >= 3) // skip short words like "the", "is"
            .collect()
    };
    // Deduplicate with HashSet (handles non-adjacent duplicates), then sort by length
    let mut seen = HashSet::new();
    let mut keywords: Vec<String> = raw
        .into_iter()
        .filter(|kw| seen.insert(kw.clone()))
        .collect();
    keywords.sort_by_key(|keyword| std::cmp::Reverse(keyword.len()));
    keywords.truncate(max_keywords);
    keywords
}

/// Jaccard similarity between two texts (token-level, punctuation-stripped).
///
/// Preserves historical behavior (returns 0.0) for the public API — the
/// internal `jaccard_from_sets` now returns `None` when no comparison is
/// possible, and we collapse that to 0.0 here so scalar callers aren't
/// forced to change.  New callers should prefer [`jaccard_similarity_opt`].
pub fn jaccard_similarity(a: &str, b: &str) -> f32 {
    jaccard_similarity_opt(a, b).unwrap_or(0.0)
}

/// `Option` variant of [`jaccard_similarity`]: `None` means no tokens to
/// compare on either side — callers should treat this as "skip, do not dedup".
pub fn jaccard_similarity_opt(a: &str, b: &str) -> Option<f32> {
    jaccard_from_sets(&normalize_tokens(a), &normalize_tokens(b))
}

/// Containment similarity: what fraction of the shorter text is covered by the longer.
/// Better than Jaccard for dedup — a short summary of a longer text scores high.
pub fn containment_similarity(a: &str, b: &str) -> f32 {
    containment_similarity_opt(a, b).unwrap_or(0.0)
}

/// `Option` variant: `None` when either side has no tokens. Treat as "do not merge".
pub fn containment_similarity_opt(a: &str, b: &str) -> Option<f32> {
    containment_from_sets(&normalize_tokens(a), &normalize_tokens(b))
}

/// Combined similarity: max of Jaccard and Containment.
/// Tokenizes each input once (not 4x) — important for CJK where jieba is expensive.
///
/// Returns 0.0 when neither jaccard nor containment is defined (both sets empty).
pub fn similarity(a: &str, b: &str) -> f32 {
    let set_a = normalize_tokens(a);
    let set_b = normalize_tokens(b);
    let j = jaccard_from_sets(&set_a, &set_b);
    let c = containment_from_sets(&set_a, &set_b);
    match (j, c) {
        (Some(j), Some(c)) => j.max(c),
        (Some(v), None) | (None, Some(v)) => v,
        (None, None) => 0.0,
    }
}

pub fn lexical_score(a: &str, b: &str) -> LexicalDedupScore {
    let set_a = normalize_tokens(a);
    let set_b = normalize_tokens(b);
    // Preserve the legacy public contract (f32 fields) by collapsing None to 0.0
    // at the struct boundary.  Inside `check_dedup` we never enter the gray-zone
    // or merge paths when the raw option is None, because the resulting
    // `score == 0.0` stays below every dedup threshold.
    let jaccard = jaccard_from_sets(&set_a, &set_b).unwrap_or(0.0);
    let containment = containment_from_sets(&set_a, &set_b).unwrap_or(0.0);
    LexicalDedupScore {
        jaccard,
        containment,
        score: jaccard.max(containment),
    }
}

/// Directional check: does `new` strongly contain `old`?
///
/// Containment alone (max of the two directions) collapses asymmetry, so
/// `check_dedup` ends up treating "new is longer and subsumes old" the same as
/// "old is longer and subsumes new" — always returning `MergeInto(existing)`
/// even though the former case should `Supersede(existing)` (the longer new
/// record has strictly more information than the old one).
///
/// Returns `true` iff both (a) `old` is nearly fully covered by `new` and
/// (b) `new` is NOT fully covered by `old`. The asymmetry is what distinguishes
/// "new contains old with extras" from "both sides are near-duplicates".
pub(crate) fn new_strongly_contains_old(new_content: &str, old_content: &str) -> bool {
    let set_new = normalize_tokens(new_content);
    let set_old = normalize_tokens(old_content);
    if set_new.is_empty() || set_old.is_empty() {
        return false;
    }
    if set_new.len() <= set_old.len() {
        return false;
    }
    let intersection = set_new.intersection(&set_old).count() as f32;
    let old_covered = intersection / set_old.len() as f32;
    let new_covered = intersection / set_new.len() as f32;
    // "old is ~fully inside new" AND "new still has meaningful extra tokens"
    old_covered >= 0.85 && new_covered <= 0.7
}

pub fn normalize_topic_key(topic: &str) -> String {
    // Apply NFC normalization before lowercasing to handle composed vs decomposed
    // Unicode forms (e.g., "café" NFC vs "cafe\u{0301}" NFD produce the same key).
    let nfc: String = topic.trim().nfc().collect();
    let mut normalized = String::with_capacity(nfc.len());
    let mut prev_sep = false;
    for ch in nfc.to_lowercase().chars() {
        if ch.is_alphanumeric() {
            normalized.push(ch);
            prev_sep = false;
        } else if !prev_sep && !normalized.is_empty() {
            normalized.push('-');
            prev_sep = true;
        }
    }
    normalized.trim_matches('-').to_string()
}

pub fn topics_are_variants(left: &str, right: &str) -> bool {
    normalize_topic_key(left) == normalize_topic_key(right)
}

pub fn score_candidate(
    topic: &str,
    content: &str,
    candidate: &Memory,
    cluster_id: Option<u32>,
) -> CandidateScore {
    let lexical = lexical_score(content, &candidate.content);
    let topic_variant_match = topics_are_variants(topic, &candidate.topic);
    let cluster_match = cluster_id.is_some() && candidate.cluster_id == cluster_id;
    let recency_days = (chrono::Utc::now() - candidate.created_at).num_days();
    let mut final_score = lexical.score;
    if topic_variant_match {
        final_score += 0.05;
    }
    if cluster_match {
        final_score += 0.05;
    }
    CandidateScore {
        memory_id: candidate.id.clone(),
        lexical,
        topic_variant_match,
        cluster_match,
        recency_days,
        final_score: final_score.clamp(0.0, 1.0),
    }
}

pub fn gray_zone_lower_bound(best_sim: f32, llm_budget_available: bool) -> f32 {
    if (0.35..0.50).contains(&best_sim) && llm_budget_available {
        0.35
    } else {
        0.50
    }
}

fn should_escalate_gray_zone(
    best_score: &CandidateScore,
    best_sim: f32,
    llm_budget_available: bool,
) -> bool {
    let lower = gray_zone_lower_bound(best_sim, llm_budget_available);
    // Guard: lexical score must meet the dynamic lower bound (0.35 when budget
    // available and sim is in [0.35, 0.50), otherwise 0.50).
    if best_score.lexical.score < lower {
        return false;
    }
    best_sim >= lower
}

fn candidate_topics(store: &SqliteStore, topic: &str) -> ReinResult<Vec<String>> {
    let normalized = normalize_topic_key(topic);
    let mut topics = vec![topic.to_string()];
    for existing in store.list_topics()? {
        if existing != topic && normalize_topic_key(&existing) == normalized {
            topics.push(existing);
        }
    }
    Ok(topics)
}

/// What to do when storing a potentially duplicate memory.
#[derive(Debug, Clone)]
pub enum DedupAction {
    /// No duplicate found, create new memory.
    CreateNew,
    /// Similar content within time window, merge into existing memory.
    MergeInto(String),
    /// v0.27 Track 2 #6: ≥2 existing memories simultaneously matched the
    /// new memory above the merge threshold. The first `String` is the
    /// canonical winner id; the `Vec<String>` is the additional loser ids
    /// (length ≥1; if empty the caller should downgrade to `MergeInto`).
    /// Each loser will be folded into evidence + marked `superseded_by =
    /// winner_id` atomically inside a single savepoint.
    MergeIntoMany(String, Vec<String>),
    /// Similar content but older than time window, supersede old memory.
    Supersede(String),
    /// v0.27 Track 2 #8: text-similarity is high but temporal anchors
    /// conflict. Old memory is preserved as a historical version (status
    /// flips, but vec/Tantivy/HNSW are NOT torn down), and `version` is
    /// the next chain index. Currently degrades to `Supersede` semantics
    /// for memories pending v0.28+ schema work — see
    /// `apply_temporal_supersede` in `store/knowledge.rs`.
    TemporalSupersede(String, u32),
    /// Gray zone (0.5 <= sim < threshold): needs LLM judgment.
    /// Falls back to CreateNew if LLM unavailable.
    GrayZone(String, f32),
}

impl std::fmt::Display for DedupAction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DedupAction::CreateNew => write!(f, "CreateNew"),
            DedupAction::MergeInto(id) => write!(f, "MergeInto({id})"),
            DedupAction::MergeIntoMany(winner, losers) => {
                write!(f, "MergeIntoMany({winner}, +{} losers)", losers.len())
            }
            DedupAction::Supersede(id) => write!(f, "Supersede({id})"),
            DedupAction::TemporalSupersede(id, ver) => {
                write!(f, "TemporalSupersede({id}, v{ver})")
            }
            DedupAction::GrayZone(id, sim) => write!(f, "GrayZone({id}, {sim:.3})"),
        }
    }
}

/// Check for duplicate memories using FTS search and Jaccard similarity.
///
/// Given the store, a topic, and content text, search existing memories in that
/// topic using FTS. For the best match, compute Jaccard similarity.
/// - If > threshold and time diff < time_window_days -> MergeInto(id)
/// - If > threshold and time diff >= time_window_days -> Supersede(id)
/// - Otherwise -> CreateNew
///
/// v0.27 Track 2 layer (additive — does not change pre-v0.27 behavior unless
/// config knobs are tuned):
/// - **Triple-overlap upgrade (#5)**: gray-zone matches whose extracted
///   triple sets overlap ≥ `[dedup].triple_overlap_threshold` are upgraded
///   to `MergeInto` directly (skipping the LLM verdict path).
/// - **N-merge collection (#6)**: when ≥2 candidates exceed the merge
///   threshold, returns `MergeIntoMany(winner, losers)` instead of
///   `MergeInto(winner)` so all duplicates collapse in one savepoint.
/// - **Temporal supersede (#8, opt-in via `[dedup].temporal_supersede_enabled`)**:
///   high-similarity merges whose temporal anchors conflict are downgraded
///   to `TemporalSupersede(old, version)`.
///
/// Triple + temporal extraction is bounded: each is run **at most once**
/// per `check_dedup` call against the new content (cached locally).
pub fn check_dedup(
    store: &SqliteStore,
    topic: &str,
    summary: &str,
    content: &str,
    similarity_threshold: f32,
    time_window_days: i64,
    cluster_id: Option<u32>,
) -> ReinResult<DedupAction> {
    // Resolve embedding model name once per call to avoid repeated config reloads.
    let loaded_cfg = crate::config::ReinConfig::load().ok();
    let dedup_cfg = loaded_cfg
        .as_ref()
        .map(|c| c.dedup.clone())
        .unwrap_or_default();
    let intelligent_merge_enabled = loaded_cfg
        .as_ref()
        .map(|c| c.intelligent_merge.enabled)
        .unwrap_or(false);
    let embed_model = loaded_cfg
        .as_ref()
        .map(|c| c.embedding_model())
        .unwrap_or_default();

    let mut seen = HashSet::new();
    let mut candidates = Vec::new();

    // Channel 1: FTS-based candidates (within topic variants)
    // NOTE: FTS5 uses unicode61 which does NOT segment CJK, so we pass raw
    // whitespace tokens. For pure CJK without spaces, FTS may yield nothing —
    // that's OK because Channel 2 (embedding) covers this case.
    let query_tokens: Vec<&str> = content.split_whitespace().take(20).collect();
    if !query_tokens.is_empty() {
        let query = query_tokens.join(" ");
        for candidate_topic in candidate_topics(store, topic)? {
            for memory in store
                .search_fts(&query, Some(&candidate_topic), 8)?
                .into_iter()
                .filter(|m| m.superseded_by.is_none())
            {
                if seen.insert(memory.id.clone()) {
                    candidates.push(memory);
                }
            }
        }
    }

    // Channel 2: Embedding-based candidates (cross-topic semantic duplicates)
    // Critical for CJK text where FTS5 may return nothing.
    // Only uses cached embeddings — no API call on the hot path.
    if let Some(emb_candidates) =
        embedding_candidate_lookup(store, summary, content, &embed_model, topic)
    {
        for memory in emb_candidates
            .into_iter()
            .filter(|m| m.superseded_by.is_none())
        {
            if seen.insert(memory.id.clone()) {
                candidates.push(memory);
            }
        }
    }

    if candidates.is_empty() {
        return Ok(DedupAction::CreateNew);
    }

    // Find best match by lexical similarity
    let mut best_sim = 0.0f32;
    let mut best_memory = None;
    let mut best_candidate_score = None;

    for candidate in &candidates {
        let score = score_candidate(topic, content, candidate, cluster_id);
        if score.final_score > best_sim {
            best_sim = score.final_score;
            best_memory = Some(candidate);
            best_candidate_score = Some(score);
        }
    }

    // Also check embedding-based similarity for candidates that scored low lexically
    // but may be paraphrased duplicates (high vector, low lexical overlap).
    // Track best vector-only candidate separately to avoid auto-merging at low thresholds.
    let mut best_vec_sim = 0.0f32;
    let mut best_vec_memory: Option<&crate::types::Memory> = None;
    if !embed_model.is_empty() {
        for candidate in &candidates {
            if let Some(cosine) =
                embedding_cosine_check(store, summary, content, &candidate.id, &embed_model, topic)
            {
                if cosine > best_vec_sim {
                    best_vec_sim = cosine;
                    best_vec_memory = Some(candidate);
                }
            }
        }
    }

    // M6: Randomized threshold exploration (5% of the time, offset threshold by ±0.1)
    // This creates A/B test data for causal inference on optimal thresholds.
    let (effective_threshold, is_exploration) = m6_explore_threshold(similarity_threshold);

    // v0.27 Track 2 #6: Collect ALL candidates that exceed the merge threshold
    // for N-merge consideration. Sorted by final_score descending so the head
    // is the canonical winner and the tail is loser candidates. We only build
    // this when at least one candidate clears the threshold to keep the
    // sub-threshold path on the fast path it had pre-v0.27.
    let mut above_threshold: Vec<CandidateScore> = candidates
        .iter()
        .map(|c| score_candidate(topic, content, c, cluster_id))
        .filter(|s| s.final_score > effective_threshold)
        .collect();
    above_threshold.sort_by(|a, b| {
        b.final_score
            .partial_cmp(&a.final_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // v0.27 Track 2 — extractor + new-content triples / anchors are computed
    // at most once per `check_dedup` call (lazy, lazily memoized below).
    // Mirrors `embed_model` resolution above. Loading the extractor when
    // `[dedup]` features are off is still cheap (it's just config lookup +
    // an Option) and graceful-degrades when no LLM is configured.
    let extractor: Option<ExtractorKind> = build_dedup_extractor();
    let extractor_ref: Option<&ExtractorKind> = extractor.as_ref();
    let mut new_triples_cache: Option<Vec<Triple>> = None;
    let mut new_anchors_cache: Option<Vec<TemporalAnchor>> = None;

    // Vector-only candidates only auto-merge at a strong threshold (> 0.85).
    // Below that, they route to GrayZone for LLM review to avoid false-positive merges.
    //
    // v0.27 R8 P2 fix: when 2+ lexical candidates already exceed the merge
    // threshold, fall through to the N-merge path instead — the vector
    // shortcut would otherwise reduce the duplicate set to a single
    // MergeInto and silently skip MergeIntoMany.
    if best_vec_sim > 0.85 && above_threshold.len() < 2 {
        if let Some(memory) = best_vec_memory {
            let age_days = (chrono::Utc::now() - memory.created_at).num_days();
            if age_days < time_window_days && !new_strongly_contains_old(content, &memory.content) {
                // v0.27 #8: temporal-conflict downgrade.
                if let Some(supersede) = maybe_temporal_supersede(
                    store,
                    extractor_ref,
                    content,
                    memory,
                    &dedup_cfg,
                    &mut new_anchors_cache,
                ) {
                    return Ok(supersede);
                }
                return Ok(DedupAction::MergeInto(memory.id.clone()));
            } else {
                return Ok(DedupAction::Supersede(memory.id.clone()));
            }
        }
    } else if best_vec_sim > 0.60 && best_sim < effective_threshold {
        // v0.27 #5: gray-zone — attempt triple-overlap upgrade BEFORE routing
        // to LLM. v0.27 R2 P2 fix: skip the upgrade when
        // `[intelligent_merge].enabled = true` — the operator opted into
        // LLM-verdict gray-zone resolution, so a triple-overlap heuristic
        // must not bypass it (mirrors the lexical gray-zone branch below).
        if let Some(memory) = best_vec_memory {
            if !intelligent_merge_enabled {
                if let Some(upgrade) = maybe_triple_upgrade(
                    extractor_ref,
                    content,
                    memory,
                    &dedup_cfg,
                    &mut new_triples_cache,
                    time_window_days,
                ) {
                    // v0.27 R6 P2 fix: temporal-supersede check before
                    // triple-overlap MergeInto, mirroring the lexical
                    // gray-zone branch below (R4 fix). Otherwise a
                    // 2024-vs-2026 pair with high triple overlap would
                    // merge instead of supersede.
                    if let Some(supersede) = maybe_temporal_supersede(
                        store,
                        extractor_ref,
                        content,
                        memory,
                        &dedup_cfg,
                        &mut new_anchors_cache,
                    ) {
                        return Ok(supersede);
                    }
                    return Ok(upgrade);
                }
            }
            return Ok(DedupAction::GrayZone(memory.id.clone(), best_vec_sim));
        }
    }

    // v0.27 Track 2 #6: when ≥2 candidates exceed the merge threshold, return
    // MergeIntoMany so callers can collapse them in one savepoint. The head
    // of `above_threshold` is the winner; the rest (capped at
    // `n_merge_max_candidates - 1`) become losers.
    if above_threshold.len() >= 2 {
        let winner = above_threshold[0].clone();
        // v0.27 R3 P2 fix: honor the configured cap exactly. Operators can
        // set `n_merge_max_candidates = 0` or `1` to disable extra-loser
        // folding; the previous `.max(1)` floor would have forced one loser
        // through anyway. When `max_losers == 0` we fall through to the
        // standard MergeInto path below.
        let max_losers = dedup_cfg.n_merge_max_candidates.saturating_sub(1);
        let losers: Vec<String> = above_threshold
            .iter()
            .skip(1)
            .take(max_losers)
            .map(|s| s.memory_id.clone())
            .collect();

        // Find the winner Memory for temporal/age checks.
        let winner_memory = candidates
            .iter()
            .find(|c| c.id == winner.memory_id)
            .expect("winner came from candidates");

        let age_days = (chrono::Utc::now() - winner_memory.created_at).num_days();
        if is_exploration {
            m6_log_outcome(
                store,
                winner.final_score,
                effective_threshold,
                similarity_threshold,
                true,
            );
        }

        if age_days < time_window_days
            && !new_strongly_contains_old(content, &winner_memory.content)
        {
            // v0.27 #8: temporal-conflict downgrade.  When temporal supersede
            // fires we keep the existing MergeInto/Supersede semantics for
            // the winner only; the additional duplicates aren't folded in
            // (they're already separate memories — a future pass will
            // collapse them once the temporal chain stabilizes).
            if let Some(supersede) = maybe_temporal_supersede(
                store,
                extractor_ref,
                content,
                winner_memory,
                &dedup_cfg,
                &mut new_anchors_cache,
            ) {
                return Ok(supersede);
            }
            // v0.27 R6 P2 fix: filter losers by temporal compatibility
            // when temporal-supersede is enabled. Otherwise an N-merge
            // would deprecate a 2024 loser into a 2026 winner even though
            // the loser temporally conflicts with the new content. The
            // filter mirrors `maybe_temporal_supersede`'s gating: if the
            // loser's anchors conflict with new_anchors, keep it as a
            // separate memory (it'll surface in a subsequent dedup pass).
            let filtered_losers: Vec<String> = if dedup_cfg.temporal_supersede_enabled
                && new_anchors_cache.as_ref().is_some_and(|a| !a.is_empty())
            {
                let new_anchors = new_anchors_cache
                    .as_deref()
                    .unwrap_or(&[]);
                losers
                    .into_iter()
                    .filter(|loser_id| {
                        let Some(loser) = candidates.iter().find(|c| &c.id == loser_id) else {
                            return true;
                        };
                        let cand_anchors = extract_temporal_rule_based(
                            &loser.content,
                            loser.created_at,
                        );
                        if cand_anchors.is_empty() {
                            return true;
                        }
                        // Drop the loser when at least one new anchor has
                        // NO compatible counterpart on the loser side.
                        let conflict = new_anchors.iter().any(|a| {
                            cand_anchors.iter().all(|b| temporal_conflict(a, b))
                        });
                        !conflict
                    })
                    .collect()
            } else {
                losers
            };
            // Downgrade to MergeInto when no losers materialized (cap=1 or
            // similar pathological config, or temporal filter dropped all).
            if filtered_losers.is_empty() {
                return Ok(DedupAction::MergeInto(winner.memory_id.clone()));
            }
            return Ok(DedupAction::MergeIntoMany(winner.memory_id.clone(), filtered_losers));
        } else {
            return Ok(DedupAction::Supersede(winner.memory_id.clone()));
        }
    }

    if best_sim > effective_threshold {
        if let Some(memory) = best_memory {
            let age_days = (chrono::Utc::now() - memory.created_at).num_days();
            // Log exploration outcome for M6 learning
            if is_exploration {
                m6_log_outcome(
                    store,
                    best_sim,
                    effective_threshold,
                    similarity_threshold,
                    true,
                );
            }
            if age_days < time_window_days && !new_strongly_contains_old(content, &memory.content) {
                // v0.27 #8: temporal-conflict downgrade.
                if let Some(supersede) = maybe_temporal_supersede(
                    store,
                    extractor_ref,
                    content,
                    memory,
                    &dedup_cfg,
                    &mut new_anchors_cache,
                ) {
                    return Ok(supersede);
                }
                return Ok(DedupAction::MergeInto(memory.id.clone()));
            } else {
                return Ok(DedupAction::Supersede(memory.id.clone()));
            }
        }
    }

    // Gray zone: 0.5 <= sim < threshold — try embedding cosine first, then LLM
    // This avoids consuming LLM budget when embedding similarity is decisive.
    let llm_budget_available = m6_has_llm_budget(store);
    if let (Some(memory), Some(ref score)) = (best_memory, &best_candidate_score) {
        if should_escalate_gray_zone(score, best_sim, llm_budget_available) {
            // v0.27 #5: triple-overlap fast path BEFORE the embedding cosine
            // check. Triple overlap is a stronger semantic signal than vector
            // cosine for fact-equivalence and avoids the LLM round-trip in
            // many CJK / paraphrase cases.
            //
            // SKIP when `[intelligent_merge].enabled = true` — the operator
            // has opted into LLM-verdict gray-zone resolution; bypassing
            // that with a triple-jaccard heuristic would silently change
            // behavior on a release with intelligent_merge enabled. The
            // triple upgrade is for operators who DON'T have the heavier
            // LLM verdict path turned on. This preserves the v0.26.x
            // intelligent_merge contract (preflight verdict → mechanical
            // merge fallback) verbatim.
            if !intelligent_merge_enabled {
                if let Some(upgrade) = maybe_triple_upgrade(
                    extractor_ref,
                    content,
                    memory,
                    &dedup_cfg,
                    &mut new_triples_cache,
                    time_window_days,
                ) {
                    // v0.27 R4 P2 fix: temporal-supersede check must run
                    // BEFORE returning a triple-overlap MergeInto upgrade,
                    // otherwise an existing 2024 fact and an incoming 2026
                    // fact get merged instead of taking the supersede path.
                    if let Some(supersede) = maybe_temporal_supersede(
                        store,
                        extractor_ref,
                        content,
                        memory,
                        &dedup_cfg,
                        &mut new_anchors_cache,
                    ) {
                        return Ok(supersede);
                    }
                    return Ok(upgrade);
                }
            }
            // Try embedding-based resolution first (zero LLM cost)
            if let Some(embed_sim) =
                embedding_cosine_check(store, summary, content, &memory.id, &embed_model, topic)
            {
                if embed_sim > 0.85 {
                    // Embedding confirms strong match — treat as dedup
                    let age_days = (chrono::Utc::now() - memory.created_at).num_days();
                    if age_days < time_window_days
                        && !new_strongly_contains_old(content, &memory.content)
                    {
                        // v0.27 R4 P2 fix: gate the embedding-strong MergeInto
                        // path through temporal-supersede too — same reason
                        // as the triple-overlap branch above.
                        if let Some(supersede) = maybe_temporal_supersede(
                            store,
                            extractor_ref,
                            content,
                            memory,
                            &dedup_cfg,
                            &mut new_anchors_cache,
                        ) {
                            return Ok(supersede);
                        }
                        return Ok(DedupAction::MergeInto(memory.id.clone()));
                    } else {
                        return Ok(DedupAction::Supersede(memory.id.clone()));
                    }
                } else if embed_sim < 0.50 {
                    // Embedding confirms distinct — skip LLM
                    return Ok(DedupAction::CreateNew);
                }
                // 0.50..0.85 — embedding uncertain, fall through to LLM
            }
            // Embedding unavailable or uncertain — escalate to LLM
            if is_exploration {
                m6_log_outcome(
                    store,
                    best_sim,
                    effective_threshold,
                    similarity_threshold,
                    false,
                );
            }
            return Ok(DedupAction::GrayZone(memory.id.clone(), best_sim));
        }
    }

    // Log exploration non-match for control group
    if is_exploration && best_sim > 0.3 {
        m6_log_outcome(
            store,
            best_sim,
            effective_threshold,
            similarity_threshold,
            false,
        );
    }

    Ok(DedupAction::CreateNew)
}

/// Build an `ExtractorKind` from the `extract.dedup` resolved config for
/// v0.27 triple + temporal extraction. Returns `None` when no LLM is
/// configured (graceful degradation: the dedup decision falls back to
/// text-only signals, matching pre-v0.27 behavior).
///
/// v0.27.1: routes through `resolve_llm_for("extract.dedup")` so `[llm]`
/// inheritance applies. The resolver replicates the v0.26.x semantic of
/// "extract.dedup falls back to [extract]" when no dedicated section is
/// configured (see config.rs §8.1 fall-through chain).
fn build_dedup_extractor() -> Option<ExtractorKind> {
    let cfg = crate::config::ReinConfig::load().ok()?;
    let r = cfg.resolve_llm_for("extract.dedup").ok()?;
    match r.provider {
        crate::config::Provider::None => None,
        crate::config::Provider::Google => {
            // Codex R3 P2 fix — honor resolver's api_key_env.
            let api_key = r
                .api_key_env
                .as_deref()
                .and_then(|env_name| std::env::var(env_name).ok())
                .or_else(|| cfg.extract.google.api_key.clone())?;
            Some(ExtractorKind::Gemini(
                crate::extract::llm::GeminiExtractor::new(api_key, r.endpoint, r.model),
            ))
        }
        crate::config::Provider::Omlx => Some(ExtractorKind::Omlx(
            crate::extract::llm::OmlxExtractor::new(
                r.endpoint,
                r.model,
                cfg.extract.omlx.disable_thinking,
            ),
        )),
    }
}

/// v0.27 Track 2 #5: when text similarity is in the gray zone but extracted
/// triple sets agree strongly, upgrade to `MergeInto` directly (skip the
/// LLM verdict path).
///
/// Caching: `new_triples_cache` is mutated on first call so subsequent
/// invocations within the same `check_dedup` reuse the result. This bounds
/// triple extraction to at most ONE LLM call per dedup decision regardless
/// of how many candidates we evaluate.
///
/// Returns `None` (no upgrade) when:
///   - the new content yields no triples (LLM degraded or no facts present);
///   - the candidate's content yields no triples;
///   - the overlap score is below `triple_overlap_threshold`.
fn maybe_triple_upgrade(
    _extractor: Option<&ExtractorKind>,
    new_content: &str,
    candidate: &Memory,
    dedup_cfg: &crate::config::DedupConfig,
    new_triples_cache: &mut Option<Vec<Triple>>,
    time_window_days: i64,
) -> Option<DedupAction> {
    // v0.27 R1 P2 fix: dedup runs INSIDE `BEGIN IMMEDIATE` (sqlite write
    // lock). Calling LLM-mode `extract_triples` here would hold the write
    // lock for the duration of a network call, blocking concurrent writes.
    // Rule-based extraction is fast and deterministic; LLM-driven triple
    // extraction is deferred to a preflight pass in v0.27.1.
    let new_triples: &[Triple] = match new_triples_cache {
        Some(t) => t.as_slice(),
        None => {
            let extracted = extract_triples_rule_based(new_content);
            *new_triples_cache = Some(extracted);
            new_triples_cache.as_deref().unwrap_or(&[])
        }
    };
    if new_triples.is_empty() {
        return None;
    }

    let cand_triples = extract_triples_rule_based(&candidate.content);
    if cand_triples.is_empty() {
        return None;
    }

    let overlap = triple_overlap_score(new_triples, &cand_triples);
    if overlap >= dedup_cfg.triple_overlap_threshold as f32 {
        // v0.27 R11 P2 fix: respect the same age/containment guard as the
        // surrounding lexical/embedding branches. Stale candidates beyond
        // `time_window_days`, OR cases where the new content strongly
        // contains the old, take the `Supersede` path; otherwise the
        // triple-overlap signal upgrades to `MergeInto`.
        let age_days = (chrono::Utc::now() - candidate.created_at).num_days();
        let action = if age_days >= time_window_days
            || new_strongly_contains_old(new_content, &candidate.content)
        {
            DedupAction::Supersede(candidate.id.clone())
        } else {
            DedupAction::MergeInto(candidate.id.clone())
        };
        tracing::debug!(
            candidate_id = %candidate.id,
            overlap = overlap,
            threshold = dedup_cfg.triple_overlap_threshold,
            ?action,
            "v0.27 #5: triple-overlap upgrade"
        );
        Some(action)
    } else {
        None
    }
}

/// v0.27 Track 2 #8: when text similarity is high (the caller is about to
/// `MergeInto`) but the extracted temporal anchors conflict, downgrade to
/// `TemporalSupersede(old, version)`.
///
/// Returns `None` when:
///   - `[dedup].temporal_supersede_enabled = false` (default — preserves
///     pre-v0.27 behavior);
///   - either side has no temporal anchors (insufficient signal);
///   - no anchor pair conflicts (`temporal_conflict` is the boolean check).
///
/// Caching: same pattern as `maybe_triple_upgrade` — at most one LLM call
/// per dedup decision for new-content anchor extraction.
fn maybe_temporal_supersede(
    store: &SqliteStore,
    _extractor: Option<&ExtractorKind>,
    new_content: &str,
    candidate: &Memory,
    dedup_cfg: &crate::config::DedupConfig,
    new_anchors_cache: &mut Option<Vec<TemporalAnchor>>,
) -> Option<DedupAction> {
    if !dedup_cfg.temporal_supersede_enabled {
        return None;
    }
    let now = chrono::Utc::now();
    // v0.27 R1 P2 fix: same write-lock concern as `maybe_triple_upgrade`.
    // Rule-based extraction only inside the dedup transaction; LLM-mode
    // temporal extraction deferred to a preflight pass in v0.27.1.
    let new_anchors: &[TemporalAnchor] = match new_anchors_cache {
        Some(a) => a.as_slice(),
        None => {
            // The new memory hasn't been stored yet — its conceptual
            // "created_at" is `now`, so relative phrases ("yesterday")
            // resolve against `now` for the new side.
            let extracted = extract_temporal_rule_based(new_content, now);
            *new_anchors_cache = Some(extracted);
            new_anchors_cache.as_deref().unwrap_or(&[])
        }
    };
    if new_anchors.is_empty() {
        return None;
    }

    // v0.27 R4 P2 fix: resolve the candidate's relative anchors against
    // ITS `created_at`, not the current instant. Otherwise an old memory
    // saying "yesterday" would slide forward every day and falsely match a
    // new memory that also says "yesterday" — historical-vs-current
    // conflicts get missed. The new content is parsed with `now` (it
    // hasn't been stored yet) but the candidate uses its frozen timestamp.
    let cand_anchors = extract_temporal_rule_based(&candidate.content, candidate.created_at);
    if cand_anchors.is_empty() {
        return None;
    }

    // v0.27 R1 P2 fix: gate on "anchors without compatible counterparts"
    // rather than "any conflicting cross-pair". Two memories that BOTH
    // mention 2024 AND 2026 (a shared timeline) should NOT temporal-
    // supersede each other — the 2024-vs-2026 cross-pair conflicts but
    // each anchor has an identical match. Conflict only when at least one
    // new anchor has NO compatible (= non-conflicting) counterpart.
    let conflict = new_anchors.iter().any(|a| {
        cand_anchors
            .iter()
            .all(|b| temporal_conflict(a, b))
    });
    if !conflict {
        return None;
    }

    // Compute next chain version. memory schema doesn't carry an explicit
    // chain version, so we approximate it by counting how many memories
    // already supersede this candidate's canonical (transitively) — gives a
    // monotonic value good enough for audit-log purposes. v0.28+ will
    // formalize this against a memory_revisions table.
    let canonical = store.canonical_id_for(&candidate.id).unwrap_or_else(|_| {
        candidate.id.clone()
    });
    let version: u32 = store
        .conn()
        .query_row(
            "SELECT COALESCE(MAX(rev), 0) FROM (
                 SELECT 0 AS rev
                 UNION ALL
                 SELECT 1 + COUNT(*) AS rev
                   FROM memory_canonical_state
                  WHERE canonical_id = ?1 AND memory_id != ?1
             )",
            rusqlite::params![&canonical],
            |row| row.get::<_, i64>(0),
        )
        .map(|v| v.max(1) as u32)
        .unwrap_or(1);

    tracing::debug!(
        candidate_id = %candidate.id,
        version = version,
        "v0.27 #8: temporal conflict — MergeInto → TemporalSupersede"
    );
    Some(DedupAction::TemporalSupersede(candidate.id.clone(), version))
}

/// Look up embedding-based candidates for cross-topic dedup (zero API cost).
/// Only uses cached embeddings — never triggers an embedding API call.
/// Accepts a pre-resolved `model` name to avoid re-loading config on every call.
fn embedding_candidate_lookup(
    store: &SqliteStore,
    summary: &str,
    content: &str,
    model: &str,
    topic: &str,
) -> Option<Vec<crate::types::Memory>> {
    let enriched = crate::embed::prepend_metadata(topic, summary, content);
    let emb = crate::embed::EmbedCache::get(store.conn(), &enriched, model)
        .ok()
        .flatten()?;
    let results = crate::store::vec::search_vec(store.conn(), &emb, None, 5).ok()?;
    // A1: use adaptive global threshold minus margin as pre-filter floor.
    // This ensures candidates below the fixed 0.70 are not silently dropped
    // when the per-cluster dedup threshold is lower.
    let floor = crate::store::adaptive::AdaptiveState::restore_snapshot(store.conn())
        .map(|s| (s.get_dedup_threshold(None) as f64 - 0.10).max(0.40))
        .unwrap_or(0.60);
    let mut memories = Vec::new();
    for (id, distance) in results {
        let sim = 1.0 - distance as f64;
        if sim < floor {
            break; // Below meaningful similarity
        }
        if let Ok(m) = store.get(&id) {
            memories.push(m);
        }
    }
    if memories.is_empty() {
        None
    } else {
        Some(memories)
    }
}

/// Try to resolve gray zone using cached embeddings (zero LLM cost).
/// Returns cosine similarity if both embeddings are in cache, None otherwise.
/// Accepts a pre-resolved `model` name to avoid re-loading config on every call.
fn embedding_cosine_check(
    store: &SqliteStore,
    summary: &str,
    content: &str,
    candidate_id: &str,
    model: &str,
    topic: &str,
) -> Option<f32> {
    let enriched = crate::embed::prepend_metadata(topic, summary, content);
    let new_emb = crate::embed::EmbedCache::get(store.conn(), &enriched, model)
        .ok()
        .flatten()?;
    // Check if candidate has a stored embedding
    let cand_emb: Vec<f32> = {
        let blob: Vec<u8> = store
            .conn()
            .query_row(
                "SELECT embedding FROM vec_memories WHERE id = ?1",
                rusqlite::params![candidate_id],
                |row| row.get(0),
            )
            .ok()?;
        blob.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    };
    if new_emb.len() != cand_emb.len() || new_emb.is_empty() {
        return None;
    }
    // Cosine similarity
    let dot: f32 = new_emb.iter().zip(&cand_emb).map(|(a, b)| a * b).sum();
    let norm_a: f32 = new_emb.iter().map(|a| a * a).sum::<f32>().sqrt();
    let norm_b: f32 = cand_emb.iter().map(|b| b * b).sum::<f32>().sqrt();
    if norm_a < f32::EPSILON || norm_b < f32::EPSILON {
        return None;
    }
    Some(dot / (norm_a * norm_b))
}

/// M6: LLM judgment budget — allow up to 10 LLM dedup calls per hour.
/// Uses metadata table for cross-process budget sharing (multiple hook processes
/// may call check_dedup concurrently).
fn m6_has_llm_budget(store: &SqliteStore) -> bool {
    const MAX_LLM_CALLS_PER_HOUR: i64 = 10;

    let now_secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64;
    let current_hour = now_secs / 3600;

    let conn = store.conn();

    // Atomic increment-and-read in a single SQL statement.
    // Resets counter when the hour changes. Returns the new call count.
    let new_calls: i64 = conn.query_row(
        "INSERT INTO metadata (key, value)
         VALUES ('m6_llm_budget', json_object('hour', ?1, 'calls', 1))
         ON CONFLICT(key) DO UPDATE SET value = CASE
           WHEN CAST(json_extract(value, '$.hour') AS INTEGER) = ?1
           THEN json_object('hour', ?1, 'calls', CAST(json_extract(value, '$.calls') AS INTEGER) + 1)
           ELSE json_object('hour', ?1, 'calls', 1)
         END
         RETURNING CAST(json_extract(value, '$.calls') AS INTEGER)",
        rusqlite::params![current_hour],
        |row| row.get(0),
    ).unwrap_or(MAX_LLM_CALLS_PER_HOUR + 1);

    new_calls <= MAX_LLM_CALLS_PER_HOUR
}

/// M6: Randomized threshold exploration.
/// With 5% probability, offset the threshold by a random amount in [-0.1, +0.1].
/// Returns (effective_threshold, is_exploration).
fn m6_explore_threshold(base_threshold: f32) -> (f32, bool) {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);

    // Start at 1 to avoid count=0 which always hashes to 0 (a multiple of 20)
    let count = COUNTER.fetch_add(1, Ordering::Relaxed).wrapping_add(1);
    // Deterministic pseudo-random: explore on ~5% of calls
    let hash = count
        .wrapping_mul(0x9e3779b97f4a7c15)
        .wrapping_add(0x517cc1b727220a95);
    let explore = hash.is_multiple_of(20); // 5% probability

    if !explore {
        return (base_threshold, false);
    }

    // Random offset in [-0.1, +0.1]
    let offset_bits = ((hash >> 16) % 201) as f32 / 1000.0 - 0.1; // [-0.100, +0.100]
    let effective = (base_threshold + offset_bits).clamp(0.30, 0.95);
    (effective, true)
}

/// M6: Log threshold exploration outcome as feedback event.
fn m6_log_outcome(
    store: &SqliteStore,
    sim: f32,
    used_threshold: f32,
    base_threshold: f32,
    was_dedup: bool,
) {
    let payload = serde_json::json!({
        "similarity": sim,
        "threshold_used": used_threshold,
        "threshold_base": base_threshold,
        "offset": used_threshold - base_threshold,
        "was_dedup": was_dedup,
    });
    let _ = crate::store::adaptive::emit_event(
        store.conn(),
        crate::store::adaptive::FeedbackEvent {
            event_type: crate::store::adaptive::EventType::ParamUpdate,
            request_id: None,
            memory_id: None,
            concept_id: None,
            query: Some(format!("m6_explore:{sim:.3}")),
            query_type: Some("threshold_exploration".to_string()),
            topic: None,
            payload: Some(payload),
        },
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Importance, MemoryLayer, MemoryStatus, MemoryTier, Source};
    use chrono::Utc;

    fn test_memory(topic: &str, content: &str) -> Memory {
        Memory {
            id: ulid::Ulid::new().to_string(),
            layer: MemoryLayer::LTM,
            topic: topic.to_string(),
            summary: content.chars().take(32).collect(),
            content: content.to_string(),
            keywords: vec![],
            importance: Importance::High,
            source: Source::Manual,
            strength: 1.0,
            decay_lambda: 0.02,
            access_count: 0,
            superseded_by: None,
            canonical_id: None,
            support_count: 1,
            merge_count: 0,
            dedup_confidence: 1.0,
            source_diversity: 1.0,
            contradiction_score: 0.0,
            related_ids: vec![],
            concept_ids: vec![],
            status: MemoryStatus::Active,
            embedding: None,
            tier: MemoryTier::Warm,
            cluster_id: Some(7),
            archival_summary: None,
            archival_summary_at: None,
            archival_summary_version: None,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            last_accessed: Utc::now(),
        }
    }

    #[test]
    fn test_jaccard_identical() {
        let text = "the quick brown fox jumps over the lazy dog";
        assert!((jaccard_similarity(text, text) - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_jaccard_disjoint() {
        let a = "alpha beta gamma delta";
        let b = "one two three four";
        assert!((jaccard_similarity(a, b) - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_jaccard_partial() {
        // 2 shared tokens out of 6 unique tokens
        let a = "apple banana cherry";
        let b = "apple banana date";
        let sim = jaccard_similarity(a, b);
        // intersection = {apple, banana} = 2, union = {apple, banana, cherry, date} = 4
        // 2/4 = 0.5
        assert!((sim - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_jaccard_empty() {
        assert!((jaccard_similarity("", "") - 0.0).abs() < f32::EPSILON);
        assert!((jaccard_similarity("hello", "") - 0.0).abs() < f32::EPSILON);
        assert!((jaccard_similarity("", "world") - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn dedup_both_empty_returns_no_comparison() {
        // Both texts yield no tokens — comparison is undefined.
        assert!(jaccard_similarity_opt("", "").is_none());
        assert!(containment_similarity_opt("", "").is_none());
        // Non-alphanumeric whitespace-only texts are likewise undefined.
        assert!(jaccard_similarity_opt("   ", "\n\t").is_none());
        assert!(containment_similarity_opt("   ", "\n\t").is_none());
    }

    #[test]
    fn dedup_one_empty_returns_no_comparison() {
        // One side has tokens, the other does not.
        // Jaccard has a defined union (the non-empty side), so it DOES return Some(0.0).
        // Containment is still undefined — there's no "smaller side" to ground it.
        assert_eq!(jaccard_similarity_opt("hello world", ""), Some(0.0));
        assert_eq!(jaccard_similarity_opt("", "hello world"), Some(0.0));
        assert!(containment_similarity_opt("hello world", "").is_none());
        assert!(containment_similarity_opt("", "hello world").is_none());
    }

    #[test]
    fn test_jaccard_strips_punctuation() {
        // "pool" vs "pool." should match after stripping punctuation
        let a = "database connection pool";
        let b = "database connection pool.";
        assert!((jaccard_similarity(a, b) - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_containment_subset() {
        // Short text is fully contained in longer text
        let long = "Fixed OOM bug by closing database connection pool properly";
        let short = "Fixed OOM bug by closing database connection pool";
        let sim = containment_similarity(long, short);
        assert!(sim > 0.95, "containment should be ~1.0, got {sim}");
    }

    #[test]
    fn test_containment_disjoint() {
        let a = "alpha beta gamma";
        let b = "one two three";
        assert!((containment_similarity(a, b) - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_similarity_picks_best() {
        // Jaccard is low (0.65) but containment is high (1.0)
        let a = "Fixed OOM bug by closing database connection pool properly. The issue was that connections were not being released back to the pool after query execution.";
        let b = "Fixed OOM bug by closing database connection pool. Connections were not released back after query.";
        let sim = similarity(a, b);
        assert!(
            sim > 0.70,
            "similarity should be > 0.70 (containment dominates), got {sim}"
        );
    }

    #[test]
    fn test_topic_variant_match() {
        assert!(topics_are_variants(
            "Docker Deployment",
            "docker-deployment"
        ));
        assert!(topics_are_variants(
            "docker_deployment",
            "docker deployment"
        ));
        assert!(!topics_are_variants("Docker Deployment", "CP2K"));
    }

    #[test]
    fn test_score_candidate_boosts_variant_and_cluster() {
        let candidate = test_memory("docker-deployment", "Use docker compose for local stack");
        let scored = score_candidate(
            "Docker Deployment",
            "Use docker compose for the local stack",
            &candidate,
            Some(7),
        );
        assert!(scored.topic_variant_match);
        assert!(scored.cluster_match);
        assert!(scored.final_score >= scored.lexical.score);
    }

    #[test]
    fn test_topic_variant_nfc_nfd_equivalence() {
        // NFC: precomposed "é" (U+00E9)
        let nfc = "caf\u{00e9}-notes";
        // NFD: decomposed "e" + combining acute (U+0301)
        let nfd = "cafe\u{0301}-notes";
        assert!(
            topics_are_variants(nfc, nfd),
            "NFC and NFD forms of the same topic should be treated as variants"
        );
        assert_eq!(normalize_topic_key(nfc), normalize_topic_key(nfd));
    }

    #[test]
    fn test_topic_variant_non_latin() {
        assert!(
            !topics_are_variants("记忆管理", "記憶管理"),
            "different CJK characters should not match"
        );
        assert!(topics_are_variants("メモリ管理", "メモリ管理"));
    }

    #[test]
    fn test_gray_zone_lower_bound() {
        assert!((gray_zone_lower_bound(0.42, true) - 0.35).abs() < f32::EPSILON);
        assert!((gray_zone_lower_bound(0.42, false) - 0.50).abs() < f32::EPSILON);
        assert!((gray_zone_lower_bound(0.60, true) - 0.50).abs() < f32::EPSILON);
    }

    #[test]
    fn test_gray_zone_requires_stronger_lexical_signal() {
        let store = SqliteStore::in_memory().unwrap();

        store
            .store(test_memory(
                "docker",
                "alpha beta gamma delta epsilon zeta eta theta iota",
            ))
            .unwrap();

        let action = check_dedup(
            &store,
            "docker",
            "alpha beta gamma delta kappa lambda mu nu xi",
            "alpha beta gamma delta kappa lambda mu nu xi",
            0.70,
            7,
            None,
        )
        .unwrap();

        assert!(
            matches!(action, DedupAction::CreateNew),
            "weak gray-zone lexical matches should not escalate to LLM"
        );
    }

    #[test]
    fn test_cjk_similarity_uses_bigrams() {
        let a = "数据库连接池修复";
        let b = "数据库连接池问题修复";
        let sim = similarity(a, b);
        assert!(
            sim > 0.4,
            "CJK bigram fallback should produce meaningful similarity, got {sim}"
        );
    }

    #[test]
    fn test_cjk_similarity_distinguishes_unrelated_text() {
        let a = "数据库连接池修复";
        let b = "图神经网络训练";
        let sim = similarity(a, b);
        assert!(
            sim < 0.3,
            "unrelated CJK strings should stay low similarity, got {sim}"
        );
    }

    #[test]
    fn test_cjk_tokenization_includes_jieba_words() {
        let tokens = normalize_tokens("数据库连接池修复");
        assert!(
            tokens.contains("连接池") || tokens.contains("数据库"),
            "jieba-rs tokens should be present alongside n-grams: {tokens:?}"
        );
    }

    #[test]
    fn test_japanese_similarity_via_bigrams() {
        // jieba is Chinese-specific, but bigrams still work for Japanese
        let a = "データベース接続プール";
        let b = "データベース接続の問題";
        let sim = similarity(a, b);
        assert!(
            sim > 0.3,
            "Japanese text should get meaningful similarity via bigrams, got {sim}"
        );
    }

    #[test]
    fn test_korean_similarity_via_bigrams() {
        let a = "데이터베이스 연결 풀";
        let b = "데이터베이스 연결 문제";
        let sim = similarity(a, b);
        assert!(
            sim > 0.3,
            "Korean text should get meaningful similarity via bigrams, got {sim}"
        );
    }

    #[test]
    fn test_mixed_cjk_ascii_similarity() {
        let a = "使用 Docker 部署应用程序";
        let b = "使用 Docker 部署服务";
        let sim = similarity(a, b);
        assert!(
            sim > 0.4,
            "mixed CJK+ASCII text should produce meaningful similarity, got {sim}"
        );
    }

    #[test]
    fn test_single_cjk_char() {
        let tokens = normalize_tokens("人");
        assert!(
            !tokens.is_empty(),
            "single CJK character should produce at least one token"
        );
    }

    #[test]
    fn new_strongly_contains_old_detects_richer_replacement() {
        // Old summary is fully inside the new, which also carries new information.
        let old = "docker compose yaml for deployment";
        let new = "docker compose yaml for deployment with healthchecks restart policies and secret mounts";
        assert!(new_strongly_contains_old(new, old));
        // Reverse direction must NOT trip the check.
        assert!(!new_strongly_contains_old(old, new));
    }

    #[test]
    fn new_strongly_contains_old_ignores_near_duplicates() {
        let a = "alpha beta gamma delta";
        let b = "alpha beta gamma epsilon";
        assert!(!new_strongly_contains_old(a, b));
        assert!(!new_strongly_contains_old(b, a));
    }

    #[test]
    fn new_strongly_contains_old_handles_empty_inputs() {
        assert!(!new_strongly_contains_old("", "anything"));
        assert!(!new_strongly_contains_old("anything", ""));
        assert!(!new_strongly_contains_old("", ""));
    }

    // ─── v0.27 Track 2 — Agent E (DedupAction extensions) ───────────────────────

    /// Format-style smoke check: every variant prints something useful (the
    /// `Display` impl covers logging + tracing/audit). Future enum
    /// extensions get an immediate compile-fail reminder via the exhaustive
    /// match in `Display`.
    #[test]
    fn v027_dedup_action_display_covers_every_variant() {
        let cases = [
            DedupAction::CreateNew,
            DedupAction::MergeInto("X".into()),
            DedupAction::MergeIntoMany("W".into(), vec!["L1".into(), "L2".into()]),
            DedupAction::Supersede("O".into()),
            DedupAction::TemporalSupersede("O".into(), 3),
            DedupAction::GrayZone("C".into(), 0.65),
        ];
        for a in &cases {
            let rendered = format!("{a}");
            assert!(!rendered.is_empty(), "Display output empty for {a:?}");
        }
    }

    /// E3: when ≥2 candidates exceed the merge threshold,
    /// `check_dedup` returns `MergeIntoMany(winner, [losers...])`.
    ///
    /// FTS5 uses implicit-AND, so the seeded candidates must contain
    /// every token the new content emits. We seed identical content so
    /// FTS retrieval matches and final_score is uniformly high.
    #[test]
    fn v027_n_merge_returned_when_multiple_candidates_exceed_threshold() {
        let store = SqliteStore::in_memory().unwrap();

        // Three identical-content memories. FTS will find all three when
        // queried with the same shared content.
        let shared = "deploys docker compose healthchecks restart";
        let id1 = store.store(test_memory("deploys", shared)).unwrap();
        let id2 = store.store(test_memory("deploys", shared)).unwrap();
        let id3 = store.store(test_memory("deploys", shared)).unwrap();

        // Pre-condition: FTS retrieval finds all three.
        let fts_hits = store.search_fts(shared, Some("deploys"), 8).unwrap();
        assert_eq!(
            fts_hits.len(),
            3,
            "FTS must surface all 3 seeded memories; got {}",
            fts_hits.len()
        );

        let action = check_dedup(
            &store,
            "deploys",
            "deploys summary",
            shared,
            0.30,
            7,
            None,
        )
        .unwrap();

        let DedupAction::MergeIntoMany(winner, losers) = action else {
            panic!("expected MergeIntoMany, got {action:?}");
        };
        let all_ids = [id1, id2, id3];
        assert!(
            all_ids.contains(&winner),
            "winner {winner} must be one of {all_ids:?}"
        );
        assert_eq!(
            losers.len(),
            2,
            "with 3 above-threshold candidates we expect 2 losers (winner + 2)"
        );
        assert!(
            losers.iter().all(|l| all_ids.contains(l) && l != &winner),
            "every loser must be in the candidate set and != winner"
        );
    }

    /// E3: `n_merge_max_candidates` caps the loser count.
    ///
    /// We can't override config inside this unit test (the function loads
    /// `[dedup]` from disk; defaults to `5`). The cap is enforced via
    /// `n_merge_max_candidates.saturating_sub(1).max(1)` and `take(...)`,
    /// so seeding more candidates than the cap and asserting `losers.len()
    /// == 4` exercises the bound.
    #[test]
    fn v027_n_merge_capped_at_n_merge_max_candidates_minus_one() {
        let store = SqliteStore::in_memory().unwrap();

        // Seed 7 identical memories. FTS retrieval cap is 8, so all
        // 7 should be candidates; default cap (5) → 4 losers.
        let shared = "stack docker compose service cache queue metrics";
        for _ in 0..7 {
            store.store(test_memory("stack", shared)).unwrap();
        }

        // Pre-condition: FTS surfaces all 7.
        let fts_hits = store.search_fts(shared, Some("stack"), 8).unwrap();
        assert_eq!(
            fts_hits.len(),
            7,
            "FTS must surface all 7 seeded memories; got {}",
            fts_hits.len()
        );

        let action = check_dedup(&store, "stack", "stack summary", shared, 0.30, 7, None).unwrap();

        let DedupAction::MergeIntoMany(_, losers) = action else {
            panic!("expected MergeIntoMany, got {action:?}");
        };
        // Default cap n_merge_max_candidates=5 allows up to 4 losers.
        assert_eq!(
            losers.len(),
            4,
            "losers must be capped at n_merge_max_candidates-1 (4); got {}",
            losers.len()
        );
    }

    /// E2: the `maybe_triple_upgrade` helper returns `MergeInto` when
    /// triple-overlap is high. We exercise it directly here because
    /// `check_dedup`-level triple gating depends on a configured LLM
    /// (rule-based fallback may or may not produce overlap depending on
    /// the input text).
    #[test]
    fn v027_triple_upgrade_promotes_grayzone_to_mergeinto() {
        let candidate = test_memory("prefs", "I prefer tabs");
        let mut cache: Option<Vec<Triple>> = None;
        let cfg = crate::config::DedupConfig::default(); // threshold 0.7

        // Rule-based extraction on "I prefer tabs" yields (user, prefers,
        // tabs); same on the new content. Overlap = 1.0 → upgrade.
        let action = maybe_triple_upgrade(
            None, // rule-based fallback
            "I prefer tabs",
            &candidate,
            &cfg,
            &mut cache,
            30, // time_window_days — fresh candidate stays MergeInto
        );
        assert!(
            matches!(action, Some(DedupAction::MergeInto(_))),
            "high triple overlap must upgrade to MergeInto, got {action:?}"
        );

        // And the cache was populated so a second call wouldn't re-extract.
        assert!(
            cache.as_ref().is_some_and(|c| !c.is_empty()),
            "new_triples_cache must be populated on first call"
        );
    }

    /// E2: triple-overlap below threshold returns `None` (no upgrade).
    /// Distinct subjects → zero overlap.
    #[test]
    fn v027_triple_upgrade_returns_none_below_threshold() {
        let candidate = test_memory("prefs", "Alice prefers tabs");
        let mut cache: Option<Vec<Triple>> = None;
        let cfg = crate::config::DedupConfig::default();

        // "user prefers spaces" ≠ "Alice prefers tabs" → triple overlap is
        // 0 (objects differ AND subjects differ after pronoun normalization).
        let action = maybe_triple_upgrade(
            None,
            "I prefer spaces",
            &candidate,
            &cfg,
            &mut cache,
            30,
        );
        assert!(
            action.is_none(),
            "low triple overlap must NOT upgrade, got {action:?}"
        );
    }

    /// E4: `temporal_supersede_enabled = false` (the default) is the
    /// kill-switch — no temporal-conflict checks run, so a high-sim merge
    /// stays a `MergeInto`.
    #[test]
    fn v027_temporal_supersede_respects_disable_flag() {
        let store = SqliteStore::in_memory().unwrap();
        let candidate = test_memory("X", "candidate text");
        let mut cache: Option<Vec<TemporalAnchor>> = None;
        let cfg = crate::config::DedupConfig {
            temporal_supersede_enabled: false,
            ..Default::default()
        };

        // Even with text that has anchors, the disabled flag short-circuits.
        let action = maybe_temporal_supersede(
            &store,
            None,
            "this happened on 2024-01-15",
            &candidate,
            &cfg,
            &mut cache,
        );
        assert!(
            action.is_none(),
            "disabled flag must short-circuit; got {action:?}"
        );
    }

    /// E4: when enabled and anchors conflict → `TemporalSupersede(id, ver)`.
    #[test]
    fn v027_temporal_supersede_fires_on_conflict() {
        let store = SqliteStore::in_memory().unwrap();
        // Candidate carries 2024 anchor.
        let candidate = test_memory("react", "in 2024 we used React for the dashboard");
        store.store(candidate.clone()).unwrap();

        let mut cache: Option<Vec<TemporalAnchor>> = None;
        let cfg = crate::config::DedupConfig {
            temporal_supersede_enabled: true,
            ..Default::default()
        };

        // Incoming says 2026 → conflicts with 2024.
        let action = maybe_temporal_supersede(
            &store,
            None,
            "in 2026 we used Solid for the dashboard",
            &candidate,
            &cfg,
            &mut cache,
        );
        match action {
            Some(DedupAction::TemporalSupersede(id, ver)) => {
                assert_eq!(id, candidate.id);
                assert!(ver >= 1, "version must be >= 1, got {ver}");
            }
            other => panic!("expected TemporalSupersede, got {other:?}"),
        }
    }

    /// E1: variant smoke — `MergeIntoMany` with empty losers vec is a
    /// pathological-but-valid construction. `Display` shouldn't panic.
    #[test]
    fn v027_merge_into_many_with_empty_losers_renders_safely() {
        let action = DedupAction::MergeIntoMany("W".into(), vec![]);
        let rendered = format!("{action}");
        assert!(rendered.contains("0 losers"), "expected '0 losers', got {rendered}");
    }
}
