pub mod dedup;
pub mod hooks;
pub mod patterns;

pub use dedup::{check_dedup, containment_similarity, jaccard_similarity, similarity, DedupAction};
pub use hooks::{hook_compact, hook_post, hook_prompt, hook_stop};
pub use patterns::{extract_facts, score_sentence};
