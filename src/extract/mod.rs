pub mod dedup;
// pub mod patterns;  // Task 8
// pub mod hooks;     // Task 8

pub use dedup::{check_dedup, jaccard_similarity, DedupAction};
