//! Test-only scripted `Embedder` for exercising the real `run_vec_dedup`
//! + HNSW + sqlite-vec pipeline from integration tests without a live
//! API key. Feature-gated behind `test-support` so it is absent from
//! production binaries.
//!
//! Design mirrors `extract::llm::MockExtractor`:
//! - A FIFO response queue consumed one element per `embed` call.
//! - `Err` variants surface as `ReinError::Config`, distinguishable from
//!   successful responses and matching the shape callers see from real
//!   network backends.
//! - `embed_batch` consumes one queued response per input so tests can
//!   seed a single coherent response sequence regardless of whether
//!   callers batch.
//! - An `AtomicUsize` call counter lets tests assert exact invocation
//!   counts without needing to hold a shared reference to the mock.

#![cfg(feature = "test-support")]

use crate::types::error::{ReinError, ReinResult};
use crate::types::traits::Embedder;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

pub struct MockEmbedder {
    model: String,
    dims: usize,
    responses: Mutex<VecDeque<Result<Vec<f32>, String>>>,
    call_count: AtomicUsize,
}

impl MockEmbedder {
    /// Build a mock that returns `responses` in FIFO order. `Err` variants
    /// become `ReinError::Config` errors.
    pub fn with_responses(dims: usize, responses: Vec<Result<Vec<f32>, String>>) -> Self {
        Self {
            model: "mock-embedder".to_string(),
            dims,
            responses: Mutex::new(responses.into()),
            call_count: AtomicUsize::new(0),
        }
    }

    /// Convenience: mock that always returns the same vector for every
    /// call. The vector is cloned per call.
    pub fn with_fixed_vector(dims: usize, vector: Vec<f32>) -> Self {
        // We'll pre-seed a large response queue and refresh it lazily via
        // `next_response` below, but a simpler approach is to accept that
        // most tests need only a few calls; seed 64 copies up front.
        let responses = std::iter::repeat_with(|| Ok(vector.clone()))
            .take(64)
            .collect();
        Self::with_responses(dims, responses)
    }

    /// Convenience: mock that always errors (simulates persistent API outage).
    pub fn with_persistent_error(dims: usize, message: impl Into<String>) -> Self {
        let msg = message.into();
        let responses = std::iter::repeat_with(|| Err(msg.clone()))
            .take(64)
            .collect();
        Self::with_responses(dims, responses)
    }

    pub fn call_count(&self) -> usize {
        self.call_count.load(Ordering::Relaxed)
    }

    fn next_response(&self) -> ReinResult<Vec<f32>> {
        self.call_count.fetch_add(1, Ordering::Relaxed);
        let next = self
            .responses
            .lock()
            .expect("MockEmbedder mutex poisoned")
            .pop_front();
        match next {
            Some(Ok(v)) => Ok(v),
            Some(Err(e)) => Err(ReinError::Config(format!(
                "mock embedder scripted error: {e}"
            ))),
            None => Err(ReinError::Config(
                "mock embedder: response queue exhausted".to_string(),
            )),
        }
    }
}

impl Embedder for MockEmbedder {
    fn model_name(&self) -> &str {
        &self.model
    }

    fn dimensions(&self) -> usize {
        self.dims
    }

    async fn embed(&self, _text: &str) -> ReinResult<Vec<f32>> {
        self.next_response()
    }

    async fn embed_batch(&self, texts: &[&str]) -> ReinResult<Vec<Vec<f32>>> {
        // Consume one response per input; if any fails the whole batch
        // fails (matching how a real upstream typically fails a batch
        // rather than partially succeeding).
        let mut out = Vec::with_capacity(texts.len());
        for _ in texts {
            out.push(self.next_response()?);
        }
        Ok(out)
    }
}
