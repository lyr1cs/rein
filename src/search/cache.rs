use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

struct CacheEntry<T> {
    value: T,
    created: Instant,
}

/// Simple in-memory TTL cache. Thread-safe via Mutex.
pub struct TtlCache<T> {
    entries: HashMap<String, CacheEntry<T>>,
    ttl: Duration,
    max_size: usize,
}

impl<T: Clone> TtlCache<T> {
    pub fn new(ttl: Duration, max_size: usize) -> Self {
        Self {
            entries: HashMap::new(),
            ttl,
            max_size,
        }
    }

    pub fn get(&self, key: &str) -> Option<T> {
        let entry = self.entries.get(key)?;
        if entry.created.elapsed() > self.ttl {
            return None;
        }
        Some(entry.value.clone())
    }

    pub fn put(&mut self, key: String, value: T) {
        // Evict expired entries if at capacity
        if self.entries.len() >= self.max_size {
            let ttl = self.ttl;
            self.entries.retain(|_, e| e.created.elapsed() <= ttl);
        }
        // If still at capacity, evict oldest
        if self.entries.len() >= self.max_size {
            if let Some(oldest_key) = self
                .entries
                .iter()
                .min_by_key(|(_, e)| e.created)
                .map(|(k, _)| k.clone())
            {
                self.entries.remove(&oldest_key);
            }
        }
        self.entries.insert(
            key,
            CacheEntry {
                value,
                created: Instant::now(),
            },
        );
    }
}

// ---------------------------------------------------------------------------
// Global cache instances
// ---------------------------------------------------------------------------

static EXPAND_CACHE: OnceLock<Mutex<TtlCache<Vec<String>>>> = OnceLock::new();
static RERANK_CACHE: OnceLock<Mutex<TtlCache<Vec<f32>>>> = OnceLock::new();

/// Get the global expansion cache (24h TTL, 500 entries max).
pub fn expand_cache() -> &'static Mutex<TtlCache<Vec<String>>> {
    EXPAND_CACHE.get_or_init(|| Mutex::new(TtlCache::new(Duration::from_secs(86400), 500)))
}

/// Get the global reranker cache (1h TTL, 200 entries max).
pub fn rerank_cache() -> &'static Mutex<TtlCache<Vec<f32>>> {
    RERANK_CACHE.get_or_init(|| Mutex::new(TtlCache::new(Duration::from_secs(3600), 200)))
}

// ---------------------------------------------------------------------------
// Shared HTTP clients (connection pool reuse)
// ---------------------------------------------------------------------------

static HTTP_CLIENT_10S: OnceLock<reqwest::Client> = OnceLock::new();
static HTTP_CLIENT_15S: OnceLock<reqwest::Client> = OnceLock::new();
static HTTP_CLIENT_20S: OnceLock<reqwest::Client> = OnceLock::new();

/// Shared HTTP client with 10s timeout (expansion: Gemini).
pub fn http_client_10s() -> &'static reqwest::Client {
    HTTP_CLIENT_10S.get_or_init(|| {
        reqwest::Client::builder()
            .timeout(Duration::from_secs(10))
            .pool_max_idle_per_host(4)
            .build()
            .unwrap_or_default()
    })
}

/// Shared HTTP client with 15s timeout (expansion: OMLX, rerank: Gemini).
pub fn http_client_15s() -> &'static reqwest::Client {
    HTTP_CLIENT_15S.get_or_init(|| {
        reqwest::Client::builder()
            .timeout(Duration::from_secs(15))
            .pool_max_idle_per_host(4)
            .build()
            .unwrap_or_default()
    })
}

/// Shared HTTP client with 20s timeout (rerank: OMLX).
pub fn http_client_20s() -> &'static reqwest::Client {
    HTTP_CLIENT_20S.get_or_init(|| {
        reqwest::Client::builder()
            .timeout(Duration::from_secs(20))
            .pool_max_idle_per_host(4)
            .build()
            .unwrap_or_default()
    })
}

// ---------------------------------------------------------------------------
// Cache key hashing
// ---------------------------------------------------------------------------

/// Hash a cache key using a simple FNV-like approach (fast, not cryptographic).
pub fn cache_key(parts: &[&str]) -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut hasher = DefaultHasher::new();
    for part in parts {
        part.hash(&mut hasher);
    }
    format!("{:016x}", hasher.finish())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_hit_miss() {
        let mut cache = TtlCache::new(Duration::from_secs(60), 10);
        cache.put("key1".to_string(), vec!["a".to_string()]);
        assert!(cache.get("key1").is_some());
        assert!(cache.get("key2").is_none());
    }

    #[test]
    fn test_cache_eviction() {
        let mut cache = TtlCache::new(Duration::from_secs(60), 2);
        cache.put("a".to_string(), vec![1.0]);
        cache.put("b".to_string(), vec![2.0]);
        cache.put("c".to_string(), vec![3.0]); // should evict "a"
        assert!(cache.get("a").is_none());
        assert!(cache.get("c").is_some());
    }

    #[test]
    fn test_cache_key_deterministic() {
        let k1 = cache_key(&["hello", "world"]);
        let k2 = cache_key(&["hello", "world"]);
        assert_eq!(k1, k2);
        let k3 = cache_key(&["world", "hello"]);
        assert_ne!(k1, k3);
    }
}
