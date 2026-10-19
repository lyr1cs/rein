//! OpenAI-compatible `/embeddings` backend. Serves local servers (omlx,
//! Ollama, LM Studio, vLLM) and hosted vendors (OpenAI, Perplexity,
//! OpenRouter, Voyage, Jina, Mistral, …) through one request shape; vendor
//! differences are carried in [`OpenAiCompatibleOptions`].

use crate::types::error::{ReinError, ReinResult};
use crate::types::traits::Embedder;
use crate::vendors::{AuthHeader, EmbeddingEncoding};
use base64::Engine;
use reqwest::Client;
use serde_json::{json, Value};
use std::time::Duration;

/// Vendor-specific request / response behaviour. `Default` is the
/// unauthenticated local server (pre-v1.5 `omlx`).
#[derive(Debug, Clone)]
pub struct OpenAiCompatibleOptions {
    pub api_key: Option<String>,
    pub auth_header: AuthHeader,
    pub send_dimensions: bool,
    pub query_input_type: Option<String>,
    pub document_input_type: Option<String>,
    pub input_type_field: String,
    pub dimensions_field: String,
    pub encoding: EmbeddingEncoding,
    pub max_batch: usize,
}

impl Default for OpenAiCompatibleOptions {
    fn default() -> Self {
        Self {
            api_key: None,
            auth_header: AuthHeader::Bearer,
            send_dimensions: false,
            query_input_type: None,
            document_input_type: None,
            input_type_field: "input_type".to_string(),
            dimensions_field: "dimensions".to_string(),
            encoding: EmbeddingEncoding::Float,
            max_batch: 256,
        }
    }
}

pub struct OpenAiCompatibleEmbedder {
    client: Client,
    endpoint: String,
    model: String,
    dimensions: usize,
    options: OpenAiCompatibleOptions,
}

impl OpenAiCompatibleEmbedder {
    pub fn new(
        endpoint: String,
        model: String,
        dimensions: usize,
        options: OpenAiCompatibleOptions,
    ) -> Self {
        // Builder failure returning `Client::default()` would silently drop the
        // timeout and let a hung endpoint stall recall; surface it instead.
        let client = Client::builder()
            .timeout(Duration::from_secs(10))
            .build()
            .expect(
                "reqwest client build failed for openai-compatible embedder (likely TLS backend)",
            );
        let options = OpenAiCompatibleOptions {
            max_batch: options.max_batch.max(1),
            ..options
        };
        Self {
            client,
            endpoint,
            model,
            dimensions,
            options,
        }
    }

    pub fn endpoint(&self) -> &str {
        &self.endpoint
    }

    fn authed(&self, rb: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        match &self.options.api_key {
            Some(key) => rb.header(
                self.options.auth_header.header_name(),
                self.options.auth_header.header_value(key),
            ),
            None => rb,
        }
    }

    fn body(&self, texts: &[&str], input_type: Option<&str>) -> Value {
        let mut body = json!({ "model": &self.model, "input": texts });
        if self.options.send_dimensions {
            body[self.options.dimensions_field.as_str()] = json!(self.dimensions);
        }
        if let Some(role) = input_type {
            body[self.options.input_type_field.as_str()] = json!(role);
        }
        if self.options.encoding == EmbeddingEncoding::Base64Int8 {
            body["encoding_format"] = json!("base64_int8");
        }
        body
    }

    /// One HTTP round trip for ≤ `max_batch` texts; output is in input order.
    async fn request(&self, texts: &[&str], input_type: Option<&str>) -> ReinResult<Vec<Vec<f32>>> {
        let url = format!("{}/embeddings", self.endpoint);
        let resp = self
            .authed(self.client.post(&url))
            .json(&self.body(texts, input_type))
            .send()
            .await?;
        let status = resp.status();
        let text_body = resp.text().await?;
        if !status.is_success() {
            return Err(ReinError::Embedding(crate::types::provider_status_error(
                "OpenAI-compatible embed API",
                status,
                &text_body,
            )));
        }
        let parsed: Value =
            serde_json::from_str(&text_body).map_err(|e| ReinError::Embedding(e.to_string()))?;
        let data = parsed["data"]
            .as_array()
            .ok_or_else(|| ReinError::Embedding("missing data array".into()))?;
        if data.len() != texts.len() {
            return Err(ReinError::Embedding(format!(
                "embedding count mismatch: sent {}, got {}",
                texts.len(),
                data.len()
            )));
        }
        let mut indexed: Vec<(usize, Vec<f32>)> = Vec::with_capacity(data.len());
        for (pos, item) in data.iter().enumerate() {
            let index = item["index"].as_u64().map(|i| i as usize).unwrap_or(pos);
            let vector = decode_embedding(&item["embedding"], self.options.encoding)?;
            if vector.len() != self.dimensions {
                return Err(ReinError::Embedding(format!(
                    "dimension mismatch: expected {} dimensions, got {}",
                    self.dimensions,
                    vector.len()
                )));
            }
            indexed.push((index, vector));
        }
        indexed.sort_by_key(|(i, _)| *i);
        Ok(indexed.into_iter().map(|(_, v)| v).collect())
    }

    async fn embed_many(
        &self,
        texts: &[&str],
        input_type: Option<&str>,
    ) -> ReinResult<Vec<Vec<f32>>> {
        let mut out = Vec::with_capacity(texts.len());
        for chunk in texts.chunks(self.options.max_batch) {
            out.extend(self.request(chunk, input_type).await?);
        }
        Ok(out)
    }
}

/// Turn `data[i].embedding` into f32s. Arrays are accepted under every
/// encoding (some vendors ignore `encoding_format`); a string is decoded
/// only when the configured encoding says how.
pub fn decode_embedding(value: &Value, encoding: EmbeddingEncoding) -> ReinResult<Vec<f32>> {
    if let Some(values) = value.as_array() {
        return Ok(values
            .iter()
            .map(|v| v.as_f64().unwrap_or(0.0) as f32)
            .collect());
    }
    let Some(text) = value.as_str() else {
        return Err(ReinError::Embedding("missing data[].embedding".into()));
    };
    match encoding {
        EmbeddingEncoding::Float => Err(ReinError::Embedding(
            crate::types::unparseable_output_error(
                "embedding (expected a float array; set encoding = \"base64_int8\" for int8 vendors)",
                text,
            ),
        )),
        EmbeddingEncoding::Base64Int8 => {
            let bytes = base64::engine::general_purpose::STANDARD
                .decode(text.trim())
                .map_err(|e| ReinError::Embedding(format!("base64_int8 decode failed: {e}")))?;
            Ok(bytes.into_iter().map(|b| (b as i8) as f32).collect())
        }
    }
}

impl Embedder for OpenAiCompatibleEmbedder {
    fn model_name(&self) -> &str {
        &self.model
    }

    fn dimensions(&self) -> usize {
        self.dimensions
    }

    fn asymmetric(&self) -> bool {
        self.options.query_input_type.is_some() || self.options.document_input_type.is_some()
    }

    async fn embed(&self, text: &str) -> ReinResult<Vec<f32>> {
        let mut v = self
            .embed_many(&[text], self.options.document_input_type.as_deref())
            .await?;
        v.pop()
            .ok_or_else(|| ReinError::Embedding("empty embedding response".into()))
    }

    async fn embed_batch(&self, texts: &[&str]) -> ReinResult<Vec<Vec<f32>>> {
        self.embed_many(texts, self.options.document_input_type.as_deref())
            .await
    }

    async fn embed_query(&self, text: &str) -> ReinResult<Vec<f32>> {
        let mut v = self
            .embed_many(&[text], self.options.query_input_type.as_deref())
            .await?;
        v.pop()
            .ok_or_else(|| ReinError::Embedding("empty embedding response".into()))
    }

    async fn embed_query_batch(&self, texts: &[&str]) -> ReinResult<Vec<Vec<f32>>> {
        self.embed_many(texts, self.options.query_input_type.as_deref())
            .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stub_http::StubServer;
    use crate::vendors::EmbeddingEncoding;

    fn float_resp(vecs: &[Vec<f32>]) -> String {
        let data: Vec<_> = vecs
            .iter()
            .enumerate()
            .map(|(i, v)| json!({"object": "embedding", "index": i, "embedding": v}))
            .collect();
        json!({"object": "list", "data": data}).to_string()
    }

    fn opts() -> OpenAiCompatibleOptions {
        OpenAiCompatibleOptions::default()
    }

    #[tokio::test]
    async fn request_body_and_auth_for_openai_style() {
        let stub = StubServer::start(vec![(200, float_resp(&[vec![0.1, 0.2]]))]);
        let e = OpenAiCompatibleEmbedder::new(
            stub.url.clone(),
            "text-embedding-3-large".into(),
            2,
            OpenAiCompatibleOptions {
                api_key: Some("sk-1".into()),
                send_dimensions: true,
                ..opts()
            },
        );
        let v = e.embed("hello").await.unwrap();
        assert_eq!(v, vec![0.1, 0.2]);
        let req = &stub.requests()[0];
        assert_eq!(req.path, "/embeddings");
        assert_eq!(req.header("authorization"), Some("Bearer sk-1"));
        let body = req.json();
        assert_eq!(body["model"], "text-embedding-3-large");
        assert_eq!(body["input"], json!(["hello"]));
        assert_eq!(body["dimensions"], 2);
        assert!(body.get("input_type").is_none());
        assert!(body.get("encoding_format").is_none());
    }

    #[tokio::test]
    async fn voyage_uses_output_dimension_field() {
        let stub = StubServer::start(vec![(200, float_resp(&[vec![1.0, 2.0]]))]);
        let e = OpenAiCompatibleEmbedder::new(
            stub.url.clone(),
            "voyage-4".into(),
            2,
            OpenAiCompatibleOptions {
                send_dimensions: true,
                dimensions_field: "output_dimension".into(),
                ..opts()
            },
        );
        e.embed("x").await.unwrap();
        let body = stub.requests()[0].json();
        assert_eq!(body["output_dimension"], 2);
        assert!(body.get("dimensions").is_none());
    }

    #[tokio::test]
    async fn local_omlx_sends_no_auth_and_no_dimensions() {
        let stub = StubServer::start(vec![(200, float_resp(&[vec![1.0]]))]);
        let e = OpenAiCompatibleEmbedder::new(stub.url.clone(), "default".into(), 1, opts());
        e.embed("x").await.unwrap();
        let req = &stub.requests()[0];
        assert!(req.header("authorization").is_none());
        assert!(req.json().get("dimensions").is_none());
    }

    #[tokio::test]
    async fn query_and_document_roles_use_the_configured_field() {
        let stub = StubServer::start(vec![
            (200, float_resp(&[vec![1.0]])),
            (200, float_resp(&[vec![1.0]])),
        ]);
        let e = OpenAiCompatibleEmbedder::new(
            stub.url.clone(),
            "jina-embeddings-v3".into(),
            1,
            OpenAiCompatibleOptions {
                query_input_type: Some("retrieval.query".into()),
                document_input_type: Some("retrieval.passage".into()),
                input_type_field: "task".into(),
                ..opts()
            },
        );
        assert!(e.asymmetric());
        e.embed_query("q").await.unwrap();
        e.embed("d").await.unwrap();
        let reqs = stub.requests();
        assert_eq!(reqs[0].json()["task"], "retrieval.query");
        assert_eq!(reqs[1].json()["task"], "retrieval.passage");
    }

    #[tokio::test]
    async fn int8_vendors_get_encoding_format_and_are_decoded() {
        use base64::Engine;
        let b64 = base64::engine::general_purpose::STANDARD.encode([127u8, 0x80, 0, 5]);
        let body = json!({"data": [{"index": 0, "embedding": b64}]}).to_string();
        let stub = StubServer::start(vec![(200, body)]);
        let e = OpenAiCompatibleEmbedder::new(
            stub.url.clone(),
            "pplx-embed-v1-0.6b".into(),
            4,
            OpenAiCompatibleOptions {
                encoding: EmbeddingEncoding::Base64Int8,
                ..opts()
            },
        );
        assert_eq!(e.embed("x").await.unwrap(), vec![127.0, -128.0, 0.0, 5.0]);
        assert_eq!(stub.requests()[0].json()["encoding_format"], "base64_int8");
    }

    #[test]
    fn decode_base64_int8() {
        use base64::Engine;
        let bytes: Vec<u8> = vec![127u8, 0x80, 0, 5]; // 127, -128, 0, 5
        let b64 = base64::engine::general_purpose::STANDARD.encode(&bytes);
        let v = decode_embedding(&Value::String(b64), EmbeddingEncoding::Base64Int8).unwrap();
        assert_eq!(v, vec![127.0, -128.0, 0.0, 5.0]);
    }

    #[test]
    fn decode_rejects_string_when_float_expected_and_accepts_arrays_always() {
        let err = decode_embedding(&Value::String("AAA=".into()), EmbeddingEncoding::Float)
            .unwrap_err()
            .to_string();
        assert!(err.contains("expected a float array"), "{err}");
        assert!(!err.contains("AAA="), "{err}");
        let arr = json!([1.0, 2.0]);
        assert_eq!(
            decode_embedding(&arr, EmbeddingEncoding::Base64Int8).unwrap(),
            vec![1.0, 2.0]
        );
    }

    #[tokio::test]
    async fn dimension_mismatch_is_an_error() {
        let stub = StubServer::start(vec![(200, float_resp(&[vec![1.0, 2.0, 3.0]]))]);
        let e = OpenAiCompatibleEmbedder::new(stub.url.clone(), "m".into(), 2, opts());
        let err = e.embed("x").await.unwrap_err().to_string();
        assert!(err.contains("expected 2 dimensions, got 3"), "{err}");
    }

    #[tokio::test]
    async fn embed_batch_chunks_to_max_batch_and_preserves_order() {
        let stub = StubServer::start(vec![
            (200, float_resp(&[vec![1.0], vec![2.0]])),
            (200, float_resp(&[vec![3.0]])),
        ]);
        let e = OpenAiCompatibleEmbedder::new(
            stub.url.clone(),
            "m".into(),
            1,
            OpenAiCompatibleOptions {
                max_batch: 2,
                ..opts()
            },
        );
        let out = e.embed_batch(&["a", "b", "c"]).await.unwrap();
        assert_eq!(out, vec![vec![1.0], vec![2.0], vec![3.0]]);
        let reqs = stub.requests();
        assert_eq!(reqs.len(), 2);
        assert_eq!(reqs[0].json()["input"].as_array().unwrap().len(), 2);
        assert_eq!(reqs[1].json()["input"].as_array().unwrap().len(), 1);
    }

    #[tokio::test]
    async fn responses_are_reordered_by_index() {
        let body = json!({"data": [
            {"index": 1, "embedding": [2.0]},
            {"index": 0, "embedding": [1.0]}
        ]})
        .to_string();
        let stub = StubServer::start(vec![(200, body)]);
        let e = OpenAiCompatibleEmbedder::new(stub.url.clone(), "m".into(), 1, opts());
        assert_eq!(
            e.embed_batch(&["a", "b"]).await.unwrap(),
            vec![vec![1.0], vec![2.0]]
        );
    }

    #[tokio::test]
    async fn http_error_is_content_free() {
        let stub = StubServer::start(vec![(401, r#"{"error":"secret body"}"#.to_string())]);
        let e = OpenAiCompatibleEmbedder::new(stub.url.clone(), "m".into(), 1, opts());
        let err = e.embed("x").await.unwrap_err().to_string();
        assert!(err.contains("401"), "{err}");
        assert!(!err.contains("secret body"), "{err}");
    }

    #[test]
    fn both_indexes_use_cosine_so_unnormalized_int8_vectors_are_safe() {
        // Guard: Perplexity int8 vectors are unnormalized and only valid under
        // cosine. If either index ever switches metric, this must fail.
        let hnsw = include_str!("../store/hnsw.rs");
        assert!(hnsw.contains("MetricKind::Cos"), "HNSW must stay cosine");
        let schema = include_str!("../store/schema.rs");
        assert!(
            schema.contains("distance_metric=cosine"),
            "vec0 must stay cosine"
        );
    }
}
