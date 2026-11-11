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
    /// Returned once the scripted queue is empty (`always`).
    fallback: Option<Vec<f32>>,
    /// `embed_batch` fails when the request's total characters exceed this.
    max_batch_chars: Option<usize>,
    /// `embed_batch` answers with a rate-limit error above this many chars.
    rate_limit_over_chars: Option<usize>,
    /// `embed_batch` fails when any text contains this marker.
    poison_marker: Option<String>,
    /// Set by `reject_everything_from_now_on`: every `embed_batch` fails.
    reject_all: std::sync::atomic::AtomicBool,
    /// `embed_batch` requests still answered before `reject_all` turns on.
    batches_before_reject_all: Mutex<Option<usize>>,
    /// HTTP status the marker rejection is reported with (default: an
    /// error with no status, like a transport failure).
    poison_status: Option<reqwest::StatusCode>,
    /// The first N `embed_batch` calls answer with a rate-limit error.
    rate_limited_batches: Mutex<usize>,
    /// Per-text character counts of every `embed_batch` request received.
    batch_log: Mutex<Vec<Vec<usize>>>,
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
            fallback: None,
            max_batch_chars: None,
            rate_limit_over_chars: None,
            poison_marker: None,
            reject_all: std::sync::atomic::AtomicBool::new(false),
            batches_before_reject_all: Mutex::new(None),
            poison_status: None,
            rate_limited_batches: Mutex::new(0),
            batch_log: Mutex::new(Vec::new()),
        }
    }

    /// Mock that returns `vector` for every text, without a queue limit.
    pub fn always(dims: usize, vector: Vec<f32>) -> Self {
        let mut mock = Self::with_responses(dims, Vec::new());
        mock.fallback = Some(vector);
        mock
    }

    /// Fail any `embed_batch` request whose texts total more than `chars`
    /// characters (a provider request-size / quota ceiling).
    pub fn rejecting_batches_over(mut self, chars: usize) -> Self {
        self.max_batch_chars = Some(chars);
        self
    }

    /// Answer any `embed_batch` request over `chars` characters with a
    /// rate-limit error (a request that alone exceeds a per-minute quota).
    pub fn rate_limiting_batches_over(mut self, chars: usize) -> Self {
        self.rate_limit_over_chars = Some(chars);
        self
    }

    /// Fail any `embed_batch` request holding a text that contains `marker`.
    pub fn rejecting_texts_containing(mut self, marker: impl Into<String>) -> Self {
        self.poison_marker = Some(marker.into());
        self
    }

    /// From this call on every `embed_batch` request is rejected (a key
    /// revoked while a job is running).
    pub fn reject_everything_from_now_on(&self) {
        self.reject_all.store(true, Ordering::Relaxed);
    }

    /// Answer `count` more `embed_batch` requests, then reject every one.
    pub fn rejecting_after_batches(self, count: usize) -> Self {
        *self
            .batches_before_reject_all
            .lock()
            .expect("MockEmbedder mutex poisoned") = Some(count);
        self
    }

    /// Report marker rejections as a provider response with `status`.
    pub fn rejecting_with_status(mut self, status: reqwest::StatusCode) -> Self {
        self.poison_status = Some(status);
        self
    }

    /// Answer the first `count` `embed_batch` requests with a rate-limit
    /// error shaped like a real provider 429.
    pub fn rate_limiting_first_batches(mut self, count: usize) -> Self {
        self.rate_limited_batches = Mutex::new(count);
        self
    }

    /// Per-text character counts of every `embed_batch` request, in order.
    pub fn batch_char_sizes(&self) -> Vec<Vec<usize>> {
        self.batch_log
            .lock()
            .expect("MockEmbedder mutex poisoned")
            .clone()
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
            None => match &self.fallback {
                Some(vector) => Ok(vector.clone()),
                None => Err(ReinError::Config(
                    "mock embedder: response queue exhausted".to_string(),
                )),
            },
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
        let sizes: Vec<usize> = texts.iter().map(|t| t.chars().count()).collect();
        let total: usize = sizes.iter().sum();
        self.batch_log
            .lock()
            .expect("MockEmbedder mutex poisoned")
            .push(sizes);
        let rate_limited = {
            let mut left = self
                .rate_limited_batches
                .lock()
                .expect("MockEmbedder mutex poisoned");
            let hit = *left > 0;
            *left = left.saturating_sub(1);
            hit
        };
        if rate_limited || self.rate_limit_over_chars.is_some_and(|max| total > max) {
            return Err(ReinError::Embedding(crate::types::provider_status_error(
                "mock batch embed API",
                reqwest::StatusCode::TOO_MANY_REQUESTS,
                "",
            )));
        }
        if self.max_batch_chars.is_some_and(|max| total > max) {
            return Err(ReinError::Embedding(format!(
                "mock embedder: request of {total} chars exceeds the provider limit"
            )));
        }
        {
            let mut left = self
                .batches_before_reject_all
                .lock()
                .expect("MockEmbedder mutex poisoned");
            match left.as_mut() {
                Some(0) => self.reject_all.store(true, Ordering::Relaxed),
                Some(n) => *n -= 1,
                None => {}
            }
        }
        let reject_all = self.reject_all.load(Ordering::Relaxed);
        let poisoned = self
            .poison_marker
            .as_ref()
            .is_some_and(|marker| texts.iter().any(|t| t.contains(marker.as_str())));
        {
            if reject_all || poisoned {
                return Err(ReinError::Embedding(match self.poison_status {
                    Some(status) => {
                        crate::types::provider_status_error("mock batch embed API", status, "")
                    }
                    None => "mock embedder: request holds a text the provider rejects".to_string(),
                }));
            }
        }
        let mut out = Vec::with_capacity(texts.len());
        for _ in texts {
            out.push(self.next_response()?);
        }
        Ok(out)
    }
}
