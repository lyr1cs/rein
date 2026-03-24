pub mod dedup;
pub mod hooks;
pub mod patterns;

pub use dedup::{check_dedup, jaccard_similarity, DedupAction};
pub use hooks::{hook_compact, hook_post, hook_prompt};
pub use patterns::{extract_facts, score_sentence};
