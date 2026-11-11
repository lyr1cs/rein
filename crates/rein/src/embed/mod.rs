pub mod cache;
pub mod gemini;
pub mod openai_compat;
pub mod unembeddable;

pub use cache::EmbedCache;
pub use gemini::GeminiEmbedder;
pub use openai_compat::{OpenAiCompatibleEmbedder, OpenAiCompatibleOptions};

#[cfg(feature = "test-support")]
pub use mock::MockEmbedder;

#[cfg(feature = "test-support")]
pub mod mock;

use crate::types::error::ReinResult;
use crate::types::traits::Embedder;

/// Prepend topic and summary metadata to text before embedding.
/// This improves retrieval by encoding context into the vector.
pub fn prepend_metadata(topic: &str, summary: &str, text: &str) -> String {
    format!("topic:{} | {} | {}", topic, summary, text)
}

/// Character budget for one bulk document-embedding request.
///
/// A request sized by item count alone can carry megabytes of text when a
/// few of its memories are very large, and a provider rejects such a
/// request as a whole (per-minute token quota, body limits, read timeout),
/// ordinary memories included. The budget is counted in characters because
/// the corpus is CJK-heavy (roughly one token per character). It is a
/// starting point, not a tuned limit: a request the provider still rejects
/// is split further.
pub const BULK_EMBED_CHAR_BUDGET: usize = 100_000;

/// How long to sit out a rate-limit response. Embedding quotas are
/// per-minute windows; a sub-second retry lands in the same window.
const BULK_RATE_LIMIT_WAIT: std::time::Duration = std::time::Duration::from_secs(65);

/// Rate-limit waits a session may spend without a success in between.
const BULK_MAX_RATE_LIMIT_WAITS: usize = 5;

/// Failed multi-text requests in a row (no success in between) after which
/// the provider is treated as unavailable. Isolating one bad text in a full
/// page takes about log2(page) such failures; an outage fails every request.
const BULK_MAX_CONSECUTIVE_FAILURES: usize = 24;

/// Time allowed for one document-batch request carrying `texts`.
///
/// A client-wide timeout sized for a single short text also covers reading
/// the response body, and a full batch at 3072 dimensions answers with
/// several megabytes; a single very large text takes long to upload. The
/// allowance therefore grows from the interactive `base` with what the
/// request carries (one second per text, one per 50,000 characters) up to
/// `max`. Small batches stay close to `base`, so a provider that hangs
/// costs a periodic pass seconds, not minutes.
pub(crate) fn batch_request_timeout(
    texts: &[&str],
    base: std::time::Duration,
    max: std::time::Duration,
) -> std::time::Duration {
    let chars: usize = texts.iter().map(|t| t.chars().count()).sum();
    let extra_secs = texts.len() as u64 + (chars / 50_000) as u64;
    base.saturating_add(std::time::Duration::from_secs(extra_secs))
        .min(max)
}

/// Text sent to tell "the provider refuses this content" from "the provider
/// refuses this caller" (see [`BulkEmbedSession::embed`]).
const REFUSAL_PROBE_TEXT: &str = "hello";

/// Why one text of a bulk request has no embedding.
#[derive(Debug)]
pub struct BulkEmbedFailure {
    pub error: crate::types::error::ReinError,
    /// The provider answered a request holding only this text with a
    /// client-error status: the text itself is what it refuses (too long
    /// for the model, unacceptable content), so sending it again unchanged
    /// gives the same answer. False for rate limits, timeouts, outages and
    /// failures shared by a multi-text request.
    pub rejected_alone: bool,
}

pub type BulkEmbedResult = Result<Vec<f32>, BulkEmbedFailure>;

/// Bulk document embedding for backfill / reindex loops, one session per run.
///
/// [`embed`](Self::embed) returns one result per input text:
/// - requests are packed by [`BULK_EMBED_CHAR_BUDGET`], so a very large text
///   travels alone instead of sinking its neighbours;
/// - a failed request is split in half and retried down to single texts, so
///   one text the provider cannot embed fails alone;
/// - a rate-limit response waits out the quota window before the retry.
///
/// When the provider looks unavailable, or the rate limit does not clear,
/// the session degrades: one attempt per packed request, no splitting and
/// no waiting, so an outage costs about as many requests as there are
/// pages. It never stops trying, and the first success restores the full
/// behaviour, so texts that cannot be embedded never keep later ones from
/// being attempted.
pub struct BulkEmbedSession<'a, E: Embedder> {
    embedder: &'a E,
    char_budget: usize,
    rate_limit_wait: std::time::Duration,
    max_rate_limit_waits: usize,
    rate_limit_waits_left: usize,
    max_consecutive_failures: usize,
    consecutive_failures: usize,
    degraded: Option<&'static str>,
    any_success: bool,
    probed: bool,
    /// Lone texts refused since the last success.
    refusals_since_success: usize,
}

impl<'a, E: Embedder> BulkEmbedSession<'a, E> {
    pub fn new(embedder: &'a E) -> Self {
        Self {
            embedder,
            char_budget: BULK_EMBED_CHAR_BUDGET,
            rate_limit_wait: BULK_RATE_LIMIT_WAIT,
            max_rate_limit_waits: BULK_MAX_RATE_LIMIT_WAITS,
            rate_limit_waits_left: BULK_MAX_RATE_LIMIT_WAITS,
            max_consecutive_failures: BULK_MAX_CONSECUTIVE_FAILURES,
            consecutive_failures: 0,
            degraded: None,
            any_success: false,
            probed: false,
            refusals_since_success: 0,
        }
    }

    /// Whether the provider has embedded at least one request in this
    /// session.
    pub fn any_success(&self) -> bool {
        self.any_success
    }

    /// Whether the provider embeds a short plain text right now.
    ///
    /// A refusal of a lone text ([`BulkEmbedFailure::rejected_alone`]) is a
    /// verdict on that text only if the provider was working when it
    /// refused. Some providers answer a revoked key, or an overload, with
    /// the same status as unacceptable input; from that moment every text
    /// is "refused". Callers therefore ask this once, after their last
    /// request, before they treat any refusal of the run as lasting: a
    /// provider that still embeds the probe was refusing content, one that
    /// does not was refusing the caller.
    pub async fn provider_still_embeds(&mut self) -> bool {
        match self.embedder.embed_batch(&[REFUSAL_PROBE_TEXT]).await {
            Ok(vectors) if vectors.len() == 1 => {
                self.note_success();
                true
            }
            _ => false,
        }
    }

    fn note_success(&mut self) {
        self.consecutive_failures = 0;
        self.rate_limit_waits_left = self.max_rate_limit_waits;
        self.degraded = None;
        self.any_success = true;
        self.refusals_since_success = 0;
    }

    /// For callers that run inside a periodic pass holding a lock, where
    /// the next pass is the retry: never sleep on a rate limit, and stop
    /// splitting at the first request the provider does not answer
    /// (timeout, transport error, 5xx). A request the provider refuses is
    /// still split, so a refused text is isolated; an unavailable provider
    /// costs one request per pack instead of a slow walk down to single
    /// texts.
    pub fn for_periodic_pass(mut self) -> Self {
        self.max_rate_limit_waits = 0;
        self.rate_limit_waits_left = 0;
        self.max_consecutive_failures = 1;
        self
    }

    #[cfg(test)]
    fn with_limits(mut self, char_budget: usize, rate_limit_wait: std::time::Duration) -> Self {
        self.char_budget = char_budget;
        self.rate_limit_wait = rate_limit_wait;
        self
    }

    /// Why the session is currently sending single unsplit attempts, if it is.
    pub fn degraded(&self) -> Option<&'static str> {
        self.degraded
    }

    /// Embed `texts`, returning one result per text in input order.
    pub async fn embed(&mut self, texts: &[&str]) -> Vec<BulkEmbedResult> {
        use crate::types::error::ReinError;

        let mut out: Vec<Option<BulkEmbedResult>> = texts.iter().map(|_| None).collect();
        // Ranges still to send, next one on top.
        let mut stack = pack_by_char_budget(texts, self.char_budget);
        stack.reverse();
        // Single texts already retried once after a rate-limit wait.
        let mut retried_single: std::collections::HashSet<usize> = std::collections::HashSet::new();

        while let Some(range) = stack.pop() {
            let slice = &texts[range.clone()];
            let err = match self.embedder.embed_batch(slice).await {
                Ok(vectors) if vectors.len() == slice.len() => {
                    self.note_success();
                    for (slot, vector) in out[range].iter_mut().zip(vectors) {
                        *slot = Some(Ok(vector));
                    }
                    continue;
                }
                Ok(vectors) => ReinError::Embedding(format!(
                    "provider returned {} embeddings for {} inputs",
                    vectors.len(),
                    slice.len()
                )),
                Err(e) => e,
            };

            // A refusal may be about the caller, not the content: some
            // providers answer a bad API key with the same status as
            // unacceptable input. One short plain text settles it, asked at
            // the first refusal before anything was embedded and again
            // whenever a long run of lone texts has been refused with no
            // success in between (a key revoked mid-run). If the provider
            // refuses the probe too, it refuses everything, and splitting
            // requests down to single texts would only multiply the
            // refusals.
            let lone_refusal =
                range.len() == 1 && crate::types::error::is_provider_rejection_error(&err);
            if lone_refusal {
                self.refusals_since_success += 1;
            }
            let probe_due = self.degraded.is_none()
                && crate::types::error::is_provider_rejection_error(&err)
                && ((!self.any_success && !self.probed)
                    || self.refusals_since_success >= BULK_MAX_CONSECUTIVE_FAILURES);
            if probe_due {
                self.probed = true;
                if !self.provider_still_embeds().await {
                    self.degraded = Some(
                        "provider refuses every request; check the embedding API key and endpoint",
                    );
                }
            }

            let lone = range.len() == 1;
            let mut split = !lone;
            if self.degraded.is_some() {
                split = false;
            } else if crate::types::error::is_rate_limit_error(&err) {
                if self.rate_limit_waits_left == 0 {
                    self.degraded = Some("provider rate limit did not clear");
                    split = false;
                } else if !lone || retried_single.insert(range.start) {
                    self.rate_limit_waits_left -= 1;
                    tracing::warn!(
                        texts = range.len(),
                        wait_secs = self.rate_limit_wait.as_secs(),
                        "bulk embed: provider rate limit, waiting for the quota window"
                    );
                    tokio::time::sleep(self.rate_limit_wait).await;
                    if lone {
                        stack.push(range);
                        continue;
                    }
                }
            } else if !lone && !crate::types::error::is_provider_rejection_error(&err) {
                // Only a multi-text request failing without an answer from
                // the provider (transport error, timeout, 5xx) can mean it
                // is down. A lone text failing says something about that
                // text, and a 4xx is the provider refusing this request:
                // neither may stop the splitting that isolates bad texts.
                self.consecutive_failures += 1;
                if self.consecutive_failures >= self.max_consecutive_failures {
                    self.degraded = Some("provider kept failing");
                    split = false;
                }
            }

            if split {
                let mid = range.start + range.len() / 2;
                stack.push(mid..range.end);
                stack.push(range.start..mid);
            } else if lone {
                let rejected_alone = crate::types::error::is_provider_rejection_error(&err);
                out[range.start] = Some(Err(BulkEmbedFailure {
                    error: err,
                    rejected_alone,
                }));
            } else {
                // Every text of an unsplit request shares its failure.
                let message = err.to_string();
                for slot in &mut out[range] {
                    *slot = Some(Err(BulkEmbedFailure {
                        error: ReinError::Embedding(message.clone()),
                        rejected_alone: false,
                    }));
                }
            }
        }

        out.into_iter()
            .map(|slot| {
                slot.unwrap_or_else(|| {
                    Err(BulkEmbedFailure {
                        error: ReinError::Embedding(
                            "bulk embed produced no result for this text".into(),
                        ),
                        rejected_alone: false,
                    })
                })
            })
            .collect()
    }
}

/// Texts per bulk request. Providers cap the items of one batch call
/// (Gemini `batchEmbedContents` takes 100); the OpenAI-compatible embedder
/// applies its own per-vendor `max_batch` below this.
const BULK_MAX_TEXTS_PER_REQUEST: usize = 100;

/// Split `texts` into consecutive ranges of at most
/// [`BULK_MAX_TEXTS_PER_REQUEST`] texts whose total character count stays
/// within `budget`. A text larger than the budget gets a range of its own.
fn pack_by_char_budget(texts: &[&str], budget: usize) -> Vec<std::ops::Range<usize>> {
    let mut ranges = Vec::new();
    let mut start = 0usize;
    let mut used = 0usize;
    for (i, text) in texts.iter().enumerate() {
        let chars = text.chars().count();
        if i > start
            && (used.saturating_add(chars) > budget || i - start >= BULK_MAX_TEXTS_PER_REQUEST)
        {
            ranges.push(start..i);
            start = i;
            used = 0;
        }
        used = used.saturating_add(chars);
    }
    if start < texts.len() {
        ranges.push(start..texts.len());
    }
    ranges
}

/// Enum dispatch wrapper for all embedding backends.
/// Needed because async traits are not dyn-safe.
pub enum EmbedderKind {
    Gemini(GeminiEmbedder),
    OpenAiCompatible(OpenAiCompatibleEmbedder),
    /// Test-only scripted embedder. Lets integration tests drive the
    /// real `run_vec_dedup` / HNSW / sqlite-vec pipelines end-to-end
    /// without a live API key. Gated behind `test-support` so it never
    /// ships in release binaries. See `embed/mock.rs` for the
    /// scripted-response contract.
    #[cfg(feature = "test-support")]
    Mock(MockEmbedder),
}

impl Embedder for EmbedderKind {
    fn model_name(&self) -> &str {
        match self {
            Self::Gemini(e) => e.model_name(),
            Self::OpenAiCompatible(e) => e.model_name(),
            #[cfg(feature = "test-support")]
            Self::Mock(e) => e.model_name(),
        }
    }

    fn dimensions(&self) -> usize {
        match self {
            Self::Gemini(e) => e.dimensions(),
            Self::OpenAiCompatible(e) => e.dimensions(),
            #[cfg(feature = "test-support")]
            Self::Mock(e) => e.dimensions(),
        }
    }

    async fn embed(&self, text: &str) -> ReinResult<Vec<f32>> {
        match self {
            Self::Gemini(e) => e.embed(text).await,
            Self::OpenAiCompatible(e) => e.embed(text).await,
            #[cfg(feature = "test-support")]
            Self::Mock(e) => e.embed(text).await,
        }
    }

    async fn embed_batch(&self, texts: &[&str]) -> ReinResult<Vec<Vec<f32>>> {
        match self {
            Self::Gemini(e) => e.embed_batch(texts).await,
            Self::OpenAiCompatible(e) => e.embed_batch(texts).await,
            #[cfg(feature = "test-support")]
            Self::Mock(e) => e.embed_batch(texts).await,
        }
    }

    fn asymmetric(&self) -> bool {
        match self {
            Self::Gemini(e) => e.asymmetric(),
            Self::OpenAiCompatible(e) => e.asymmetric(),
            #[cfg(feature = "test-support")]
            Self::Mock(e) => e.asymmetric(),
        }
    }

    async fn embed_query(&self, text: &str) -> ReinResult<Vec<f32>> {
        match self {
            Self::Gemini(e) => e.embed_query(text).await,
            Self::OpenAiCompatible(e) => e.embed_query(text).await,
            #[cfg(feature = "test-support")]
            Self::Mock(e) => e.embed_query(text).await,
        }
    }

    async fn embed_query_batch(&self, texts: &[&str]) -> ReinResult<Vec<Vec<f32>>> {
        match self {
            Self::Gemini(e) => e.embed_query_batch(texts).await,
            Self::OpenAiCompatible(e) => e.embed_query_batch(texts).await,
            #[cfg(feature = "test-support")]
            Self::Mock(e) => e.embed_query_batch(texts).await,
        }
    }
}

/// Create an embedder from config. Returns None if provider is "none" or API key is missing.
pub fn create_embedder(config: &crate::config::ReinConfig) -> Option<EmbedderKind> {
    use crate::config::Provider;
    match config.embedding_provider() {
        Provider::Google => {
            let api_key = config.embedding.google.api_key.as_ref()?;
            if let Err(msg) = gemini::validate_endpoint(&config.embedding.google.endpoint) {
                // Fail closed: no embedder at all (recall falls back to FTS)
                // rather than one that would post the key over plain HTTP.
                tracing::warn!("{msg}; embedding disabled");
                return None;
            }
            Some(EmbedderKind::Gemini(GeminiEmbedder::new(
                api_key.clone(),
                config.embedding.google.endpoint.clone(),
                config.embedding.google.model.clone(),
                config.embedding.dimensions,
            )))
        }
        Provider::Omlx => {
            let name = config.embedding_provider_name();
            let preset = crate::vendors::vendor_preset(&name);
            let table = &config.embedding.omlx;
            let endpoint = resolve_embedding_endpoint(&name, &table.endpoint, preset)?;
            let api_key_env = table
                .api_key_env
                .clone()
                .filter(|s| !s.trim().is_empty())
                .or_else(|| preset.and_then(|p| p.api_key_env).map(|s| s.to_string()));
            let api_key = api_key_env.as_deref().and_then(|n| config.secret(n));
            if api_key_env.is_some() && api_key.is_none() {
                // Same contract as Google: no key → embedding disabled (doctor /
                // providers report it; recall falls back to FTS).
                return None;
            }
            let auth_header = table
                .auth_header
                .as_deref()
                .and_then(crate::vendors::AuthHeader::parse)
                .or_else(|| preset.map(|p| p.auth_header))
                .unwrap_or(crate::vendors::AuthHeader::Bearer);
            let encoding = table
                .encoding
                .as_deref()
                .and_then(crate::vendors::EmbeddingEncoding::parse)
                .or_else(|| preset.map(|p| p.encoding))
                .unwrap_or(crate::vendors::EmbeddingEncoding::Float);
            Some(EmbedderKind::OpenAiCompatible(
                OpenAiCompatibleEmbedder::new(
                    endpoint,
                    table.model.clone(),
                    config.embedding.dimensions,
                    OpenAiCompatibleOptions {
                        api_key,
                        auth_header,
                        send_dimensions: table.send_dimensions,
                        query_input_type: table.query_input_type.clone(),
                        document_input_type: table.document_input_type.clone(),
                        input_type_field: preset
                            .map(|p| p.input_type_field.to_string())
                            .unwrap_or_else(|| "input_type".to_string()),
                        dimensions_field: preset
                            .map(|p| p.dimensions_field.to_string())
                            .unwrap_or_else(|| "dimensions".to_string()),
                        encoding,
                        max_batch: table
                            .max_batch
                            .or_else(|| preset.map(|p| p.max_batch))
                            .unwrap_or(256),
                    },
                ),
            ))
        }
        Provider::None => None,
    }
}

/// The compiled default (`http://localhost:8000/v1`) counts as "unset" for a
/// named vendor so `provider = "openai"` never posts to localhost by accident.
/// Returns `None` (embedding disabled) when a vendor needs an explicit
/// endpoint and none was given.
pub fn resolve_embedding_endpoint(
    name: &str,
    configured: &str,
    preset: Option<&crate::vendors::VendorPreset>,
) -> Option<String> {
    let is_compiled_default = configured == "http://localhost:8000/v1";
    match preset {
        Some(p) if p.name != "omlx" => {
            if !is_compiled_default && !configured.trim().is_empty() {
                Some(configured.trim_end_matches('/').to_string())
            } else if let Some(ep) = p.embeddings_endpoint {
                Some(ep.to_string())
            } else {
                tracing::warn!(
                    provider = name,
                    "embedding provider needs an explicit [embedding.openai].endpoint; embedding disabled"
                );
                None
            }
        }
        _ => Some(configured.trim_end_matches('/').to_string()),
    }
}

#[cfg(all(test, feature = "test-support"))]
mod tests {
    use super::*;

    #[tokio::test]
    async fn embed_query_defaults_to_embed() {
        let e = EmbedderKind::Mock(MockEmbedder::with_fixed_vector(3, vec![1.0, 2.0, 3.0]));
        assert!(!e.asymmetric());
        assert_eq!(e.embed_query("q").await.unwrap(), vec![1.0, 2.0, 3.0]);
        assert_eq!(e.embed_query_batch(&["a", "b"]).await.unwrap().len(), 2);
    }

    fn text_of(len: usize) -> String {
        "x".repeat(len)
    }

    #[test]
    fn packing_respects_the_char_budget_and_isolates_oversized_texts() {
        let small = text_of(10);
        let huge = text_of(500);
        let texts: Vec<&str> = vec![&small, &small, &huge, &small, &small, &small];
        assert_eq!(
            pack_by_char_budget(&texts, 25),
            vec![0..2, 2..3, 3..5, 5..6]
        );
        assert!(pack_by_char_budget(&[], 25).is_empty());
        // CJK counts by character, not by byte.
        let cjk = "记忆".repeat(5);
        assert_eq!(pack_by_char_budget(&[&cjk, &cjk], 20), vec![0..2]);
    }

    // GitHub #5: a few very large memories must not cost the ordinary
    // memories on the same page their vectors.
    #[tokio::test]
    async fn bulk_session_embeds_small_texts_despite_oversized_neighbours() {
        let small = text_of(1_000);
        let huge = text_of(600_000);
        let mut owned: Vec<&str> = vec![small.as_str(); 100];
        owned[7] = &huge;
        owned[55] = &huge;
        // The provider refuses any request over 500k characters.
        let mock = MockEmbedder::always(3, vec![1.0, 2.0, 3.0]).rejecting_batches_over(500_000);
        let mut session = BulkEmbedSession::new(&mock);

        let results = session.embed(&owned).await;

        assert_eq!(results.len(), 100);
        for (i, result) in results.iter().enumerate() {
            assert_eq!(result.is_ok(), i != 7 && i != 55, "text {i}");
        }
        assert!(session.degraded().is_none());
        // No request mixed an oversized text with other texts.
        for sizes in mock.batch_char_sizes() {
            assert!(sizes.len() == 1 || sizes.iter().all(|&c| c == 1_000));
        }
    }

    #[tokio::test]
    async fn bulk_session_isolates_one_failing_text_by_splitting() {
        let texts: Vec<String> = (0..16)
            .map(|i| {
                if i == 11 {
                    "poison".to_string()
                } else {
                    format!("t{i}")
                }
            })
            .collect();
        let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
        let mock = MockEmbedder::always(2, vec![0.5, 0.5]).rejecting_texts_containing("poison");
        let mut session = BulkEmbedSession::new(&mock);

        let results = session.embed(&refs).await;

        for (i, result) in results.iter().enumerate() {
            assert_eq!(result.is_ok(), i != 11, "text {i}");
        }
        // 16 → 8 → 4 → 2 → 1: a handful of requests, not one per text.
        assert!(mock.batch_char_sizes().len() <= 9);
    }

    #[tokio::test]
    async fn bulk_session_marks_only_lone_provider_rejections_as_unembeddable() {
        let texts = ["good a", "poison", "good b"];
        let mock = MockEmbedder::always(2, vec![0.5, 0.5])
            .rejecting_texts_containing("poison")
            .rejecting_with_status(reqwest::StatusCode::BAD_REQUEST);
        let results = BulkEmbedSession::new(&mock).embed(&texts).await;
        assert!(results[0].is_ok() && results[2].is_ok());
        assert!(results[1].as_ref().unwrap_err().rejected_alone);

        // Statuses about the caller, not the text, are not refusals.
        for status in [
            reqwest::StatusCode::UNAUTHORIZED,
            reqwest::StatusCode::FORBIDDEN,
            reqwest::StatusCode::NOT_FOUND,
        ] {
            let mock = MockEmbedder::always(2, vec![0.5, 0.5])
                .rejecting_texts_containing("poison")
                .rejecting_with_status(status);
            let results = BulkEmbedSession::new(&mock).embed(&texts).await;
            assert!(!results[1].as_ref().unwrap_err().rejected_alone, "{status}");
        }

        // A failure with no provider status (timeout, transport) is not a
        // verdict on the text.
        let mock = MockEmbedder::always(2, vec![0.5, 0.5]).rejecting_texts_containing("poison");
        let results = BulkEmbedSession::new(&mock).embed(&texts).await;
        assert!(!results[1].as_ref().unwrap_err().rejected_alone);
    }

    #[tokio::test]
    async fn bulk_session_for_periodic_pass_fails_a_rate_limited_request_at_once() {
        let mock = MockEmbedder::always(2, vec![0.5, 0.5]).rate_limiting_first_batches(1);
        let mut session = BulkEmbedSession::new(&mock).for_periodic_pass();

        let results = session.embed(&["a", "b"]).await;
        assert!(results.iter().all(|r| r.is_err()));
        assert_eq!(mock.batch_char_sizes().len(), 1);

        // The next call succeeds and the session is whole again.
        assert!(session.embed(&["a", "b"]).await.iter().all(|r| r.is_ok()));
        assert!(session.degraded().is_none());
    }

    // A provider that refuses every request (a revoked key reported as a
    // client error) is recognised after one probe: no walk down to single
    // texts, and no refusal counts as a verdict on a text.
    #[tokio::test]
    async fn bulk_session_does_not_split_when_the_provider_refuses_everything() {
        let mock = MockEmbedder::always(2, vec![0.5, 0.5])
            .rejecting_texts_containing("")
            .rejecting_with_status(reqwest::StatusCode::BAD_REQUEST);
        let mut session = BulkEmbedSession::new(&mock);
        let texts: Vec<String> = (0..100).map(|i| format!("text {i}")).collect();
        let refs: Vec<&str> = texts.iter().map(String::as_str).collect();

        let results = session.embed(&refs).await;

        assert!(results.iter().all(|r| r.is_err()));
        assert!(!session.any_success());
        assert!(session.degraded().is_some());
        // The page and the probe; a second page costs one more request.
        assert_eq!(mock.batch_char_sizes().len(), 2);
        assert!(session.embed(&refs).await.iter().all(|r| r.is_err()));
        assert_eq!(mock.batch_char_sizes().len(), 3);
    }

    // The provider works, then starts refusing everything (a key revoked
    // mid-run). The session notices after a bounded run of refusals, stops
    // splitting, and reports that the provider no longer embeds — so the
    // caller does not take those refusals as verdicts on the texts.
    #[tokio::test]
    async fn bulk_session_notices_a_provider_that_starts_refusing_everything() {
        let mock = MockEmbedder::always(2, vec![0.5, 0.5])
            .rejecting_with_status(reqwest::StatusCode::BAD_REQUEST);
        let mut session = BulkEmbedSession::new(&mock);
        assert!(session.embed(&["first page"]).await[0].is_ok());
        mock.reject_everything_from_now_on();

        let texts: Vec<String> = (0..200).map(|i| format!("text {i}")).collect();
        let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
        let before = mock.batch_char_sizes().len();
        let results = session.embed(&refs).await;

        assert!(results.iter().all(|r| r.is_err()));
        assert!(session.degraded().is_some());
        let sent = mock.batch_char_sizes().len() - before;
        assert!(sent < 120, "bounded, not a full walk of 200 texts: {sent}");
        assert!(session.any_success());
        assert!(!session.provider_still_embeds().await);
    }

    #[tokio::test]
    async fn bulk_session_for_periodic_pass_does_not_split_an_unanswered_request() {
        // No status: the provider did not answer (timeout / transport).
        let mock = MockEmbedder::always(2, vec![0.5, 0.5]).rejecting_texts_containing("t");
        let mut session = BulkEmbedSession::new(&mock).for_periodic_pass();
        let texts: Vec<String> = (0..8).map(|i| format!("t{i}")).collect();
        let refs: Vec<&str> = texts.iter().map(String::as_str).collect();

        let results = session.embed(&refs).await;

        assert!(results.iter().all(|r| r.is_err()));
        assert_eq!(
            mock.batch_char_sizes().len(),
            1,
            "one request, no splitting"
        );
        assert!(!session.any_success());
    }

    #[test]
    fn batch_timeout_grows_with_the_request_and_is_capped() {
        use std::time::Duration;
        let base = Duration::from_secs(5);
        let max = Duration::from_secs(60);
        let small = ["a"; 8];
        assert_eq!(
            batch_request_timeout(&small, base, max),
            Duration::from_secs(13)
        );
        let full = ["a"; 100];
        assert_eq!(batch_request_timeout(&full, base, max), max);
        let huge = "x".repeat(600_000);
        assert_eq!(
            batch_request_timeout(&[huge.as_str()], base, max),
            Duration::from_secs(18)
        );
    }

    #[tokio::test]
    async fn packing_caps_the_number_of_texts_per_request() {
        let texts: Vec<&str> = vec!["x"; 250];
        assert_eq!(
            pack_by_char_budget(&texts, usize::MAX),
            vec![0..100, 100..200, 200..250]
        );
    }

    #[tokio::test]
    async fn bulk_session_waits_out_a_rate_limit_then_succeeds() {
        let mock = MockEmbedder::always(2, vec![0.5, 0.5]).rate_limiting_first_batches(1);
        let mut session =
            BulkEmbedSession::new(&mock).with_limits(1_000, std::time::Duration::ZERO);

        let results = session.embed(&["a", "b", "c", "d"]).await;

        assert!(results.iter().all(|r| r.is_ok()));
        // One wait was spent, and the success that followed refilled it.
        assert_eq!(mock.batch_char_sizes().len(), 3);
        assert_eq!(session.rate_limit_waits_left, BULK_MAX_RATE_LIMIT_WAITS);
        assert!(session.degraded().is_none());
    }

    #[tokio::test]
    async fn bulk_session_degrades_when_the_rate_limit_never_clears() {
        let mock = MockEmbedder::always(2, vec![0.5, 0.5]).rate_limiting_first_batches(usize::MAX);
        let mut session =
            BulkEmbedSession::new(&mock).with_limits(1_000, std::time::Duration::ZERO);
        let texts: Vec<String> = (0..64).map(|i| format!("t{i}")).collect();
        let refs: Vec<&str> = texts.iter().map(String::as_str).collect();

        let results = session.embed(&refs).await;

        assert!(results.iter().all(|r| r.is_err()));
        assert!(session.degraded().is_some());
        let sent = mock.batch_char_sizes().len();
        assert!(
            sent <= 2 * BULK_MAX_RATE_LIMIT_WAITS + 2,
            "sent {sent} requests"
        );
        // A later page costs one request, not a wait or a split.
        assert!(session.embed(&["later", "page"]).await[0].is_err());
        assert_eq!(mock.batch_char_sizes().len(), sent + 1);
    }

    #[tokio::test]
    async fn bulk_session_degrades_during_an_outage_and_recovers() {
        let mock = MockEmbedder::always(2, vec![0.5, 0.5]).rejecting_texts_containing("down");
        let mut session = BulkEmbedSession::new(&mock);
        let texts: Vec<String> = (0..100).map(|i| format!("down{i}")).collect();
        let refs: Vec<&str> = texts.iter().map(String::as_str).collect();

        let results = session.embed(&refs).await;

        assert!(results.iter().all(|r| r.is_err()));
        assert_eq!(session.degraded(), Some("provider kept failing"));
        let sent = mock.batch_char_sizes().len();
        // Bounded: nowhere near one request per text plus every split.
        assert!(sent < 100, "sent {sent} requests");

        // Still failing: the next page is one request.
        assert!(session.embed(&["down a", "down b"]).await[1].is_err());
        assert_eq!(mock.batch_char_sizes().len(), sent + 1);

        // Provider back: the page is embedded and the session is whole again.
        assert!(session
            .embed(&["up a", "up b"])
            .await
            .iter()
            .all(|r| r.is_ok()));
        assert!(session.degraded().is_none());
    }

    // Many adjacent texts the provider rejects one by one (each over the
    // request ceiling) must not keep the texts after them from being
    // embedded in the same run.
    #[tokio::test]
    async fn bulk_session_gets_past_a_long_run_of_unembeddable_texts() {
        let huge = text_of(2_000);
        let mut refs: Vec<&str> = vec![huge.as_str(); BULK_MAX_CONSECUTIVE_FAILURES + 16];
        refs.extend(["good a", "good b", "good c"]);
        let mock = MockEmbedder::always(2, vec![0.5, 0.5]).rejecting_batches_over(1_500);
        let mut session =
            BulkEmbedSession::new(&mock).with_limits(1_000, std::time::Duration::ZERO);

        let results = session.embed(&refs).await;

        let bad = BULK_MAX_CONSECUTIVE_FAILURES + 16;
        assert!(results[..bad].iter().all(|r| r.is_err()));
        assert!(results[bad..].iter().all(|r| r.is_ok()));
        assert!(session.degraded().is_none());
    }

    // Small texts that share requests with their neighbours and are each
    // refused by the provider (HTTP 400) are isolated however many there
    // are; the good texts around them are embedded in the same run.
    #[tokio::test]
    async fn bulk_session_isolates_a_long_run_of_small_rejected_texts() {
        let mut texts: Vec<String> = (0..60).map(|i| format!("poison {i}")).collect();
        texts.insert(0, "good first".to_string());
        texts.insert(30, "good middle".to_string());
        texts.push("good last".to_string());
        let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
        let mock = MockEmbedder::always(2, vec![0.5, 0.5])
            .rejecting_texts_containing("poison")
            .rejecting_with_status(reqwest::StatusCode::BAD_REQUEST);
        let mut session = BulkEmbedSession::new(&mock);

        let results = session.embed(&refs).await;

        for (text, result) in texts.iter().zip(&results) {
            assert_eq!(result.is_ok(), text.starts_with("good"), "{text}");
        }
        assert!(session.degraded().is_none());
    }

    // The same when each of them answers with a rate limit of its own: the
    // waits run out, and the texts after them are still sent.
    #[tokio::test]
    async fn bulk_session_gets_past_a_long_run_of_rate_limited_texts() {
        let huge = text_of(2_000);
        let lone_429s = 2 * BULK_MAX_RATE_LIMIT_WAITS;
        let mut refs: Vec<&str> = vec![huge.as_str(); lone_429s];
        refs.extend(["good a", "good b", "good c"]);
        let mock = MockEmbedder::always(2, vec![0.5, 0.5]).rate_limiting_batches_over(1_500);
        let mut session =
            BulkEmbedSession::new(&mock).with_limits(1_000, std::time::Duration::ZERO);

        let results = session.embed(&refs).await;

        assert!(results[..lone_429s].iter().all(|r| r.is_err()));
        assert!(results[lone_429s..].iter().all(|r| r.is_ok()));
        assert!(session.degraded().is_none());
    }

    #[test]
    fn named_vendor_replaces_compiled_localhost_default() {
        let p = crate::vendors::vendor_preset("perplexity");
        assert_eq!(
            resolve_embedding_endpoint("perplexity", "http://localhost:8000/v1", p).as_deref(),
            Some("https://api.perplexity.ai/v1")
        );
        assert_eq!(
            resolve_embedding_endpoint("perplexity", "https://proxy.example/v1/", p).as_deref(),
            Some("https://proxy.example/v1")
        );
        let omlx = crate::vendors::vendor_preset("omlx");
        assert_eq!(
            resolve_embedding_endpoint("omlx", "http://localhost:8000/v1", omlx).as_deref(),
            Some("http://localhost:8000/v1")
        );
        assert!(resolve_embedding_endpoint(
            "azure",
            "http://localhost:8000/v1",
            crate::vendors::vendor_preset("azure")
        )
        .is_none());
    }

    #[test]
    fn create_embedder_refuses_remote_http_google_endpoint() {
        let base = "[embedding]\nprovider = \"google\"\ndimensions = 8\n[embedding.google]\nmodel = \"gemini-embedding-001\"\napi_key = \"AIzaSyTESTKEY00000000003f2a\"\n";
        let cfg = crate::config::ReinConfig::load_from_str(&format!(
            "{base}endpoint = \"http://evil.example:8080\"\n"
        ))
        .unwrap();
        assert!(create_embedder(&cfg).is_none());
        let cfg = crate::config::ReinConfig::load_from_str(&format!(
            "{base}endpoint = \"https://generativelanguage.googleapis.com\"\n"
        ))
        .unwrap();
        assert!(create_embedder(&cfg).is_some());
        let cfg = crate::config::ReinConfig::load_from_str(&format!(
            "{base}endpoint = \"http://127.0.0.1:8080\"\n"
        ))
        .unwrap();
        assert!(create_embedder(&cfg).is_some());
    }

    #[test]
    fn create_embedder_disabled_when_vendor_key_missing() {
        std::env::remove_var("OPENAI_API_KEY");
        let mut cfg = crate::config::ReinConfig::load_from_str(
            "[embedding]\nprovider = \"openai\"\ndimensions = 8\n[embedding.openai]\nmodel = \"text-embedding-3-small\"\n",
        )
        .unwrap();
        assert!(create_embedder(&cfg).is_none());
        cfg.secrets
            .insert("OPENAI_API_KEY".into(), "sk-live".into());
        let e = create_embedder(&cfg).expect("embedder with key from secrets file");
        assert_eq!(e.model_name(), "text-embedding-3-small");
        assert_eq!(cfg.embedding_model(), "openai:text-embedding-3-small");
    }
}
