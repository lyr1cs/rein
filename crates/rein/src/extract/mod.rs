pub mod dedup;
pub mod hooks;
pub mod intelligent_merge;
pub mod llm;
pub mod patterns;
pub mod postprocess;

pub use dedup::{
    check_dedup, containment_similarity, extract_keywords_from_text, jaccard_similarity,
    similarity, tokenize_for_fts, tokenize_for_search, topics_are_variants, DedupAction,
};
pub use hooks::{hook_compact, hook_post, hook_prompt, hook_stop};
#[cfg(feature = "test-support")]
pub use llm::MockExtractor;
pub use llm::{
    create_extractor, extract_full_with_fallback, extract_with_fallback, llm_dedup_verdict,
    DedupVerdict, EpisodeSummary, ExtractedConcept, ExtractedLink, ExtractedMemory,
    ExtractionResult, ExtractorKind,
};
pub use patterns::{extract_facts, score_sentence};
