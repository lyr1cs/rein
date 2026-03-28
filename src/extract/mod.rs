pub mod dedup;
pub mod hooks;
pub mod llm;
pub mod patterns;
pub mod postprocess;

pub use dedup::{check_dedup, containment_similarity, jaccard_similarity, similarity, DedupAction};
pub use hooks::{hook_compact, hook_post, hook_prompt, hook_stop};
pub use llm::{
    create_extractor, extract_full_with_fallback, extract_with_fallback,
    ExtractedConcept, ExtractedLink, ExtractedMemory, ExtractionResult, ExtractorKind,
    EpisodeSummary,
};
pub use patterns::{extract_facts, score_sentence};
