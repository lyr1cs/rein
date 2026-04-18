//! OpsRuntime — surface-aware handle passed to every `#[op]` method.
//!
//! v0.21 A1 introduces unified ops where the same handler serves CLI, MCP,
//! and REST. Runtime carries the active surface and lazily opens per-request
//! store handles via `with_store`.

use std::sync::Arc;
use std::sync::atomic::AtomicUsize;

use crate::config::ReinConfig;
use crate::store::SqliteStore;
use crate::types::ReinResult;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SurfaceKind {
    Cli,
    Mcp,
    Rest,
}

pub struct OpsRuntime {
    pub config: Arc<ReinConfig>,
    // Reserved for Phase 1.5+ observability (count ops that bypass the store).
    #[allow(dead_code)]
    pub(crate) non_store_count: AtomicUsize,
    pub(crate) surface: SurfaceKind,
}

impl OpsRuntime {
    pub fn for_cli(config: Arc<ReinConfig>) -> Self {
        Self {
            config,
            non_store_count: AtomicUsize::new(0),
            surface: SurfaceKind::Cli,
        }
    }

    pub fn for_mcp(config: Arc<ReinConfig>) -> Self {
        Self {
            config,
            non_store_count: AtomicUsize::new(0),
            surface: SurfaceKind::Mcp,
        }
    }

    pub fn for_rest(config: Arc<ReinConfig>) -> Self {
        Self {
            config,
            non_store_count: AtomicUsize::new(0),
            surface: SurfaceKind::Rest,
        }
    }

    pub fn config(&self) -> &ReinConfig {
        &self.config
    }

    pub fn surface(&self) -> SurfaceKind {
        self.surface
    }

    pub fn remote(&self) -> bool {
        matches!(self.surface, SurfaceKind::Mcp | SurfaceKind::Rest)
    }

    pub fn dry_run(&self) -> bool {
        // v0.21: dry_run is handled per-op for now; runtime-level plumbing deferred.
        false
    }

    pub fn with_store<F, R>(&self, f: F) -> ReinResult<R>
    where
        F: FnOnce(&SqliteStore) -> ReinResult<R>,
    {
        let store = self.config.open_store()?;
        f(&store)
    }
}
