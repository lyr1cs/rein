//! OpsRuntime — surface-aware handle passed to every `#[op]` method.
//!
//! v0.21 A1 introduces unified ops where the same handler serves CLI, MCP,
//! and REST. Runtime carries the active surface and lazily opens per-request
//! store handles via `with_store`.

use std::sync::Arc;
use std::sync::atomic::{AtomicI32, AtomicUsize, Ordering};

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
    // Ops like `doctor` signal a non-zero exit code by calling `set_exit_code`.
    // The CLI inventory dispatcher reads it via `take_exit_code` after invoke
    // and calls `std::process::exit` so CI shell scripts still see 1 on failure.
    // MCP/REST surfaces ignore the field.
    pub(crate) exit_code: AtomicI32,
}

impl OpsRuntime {
    pub fn for_cli(config: Arc<ReinConfig>) -> Self {
        Self {
            config,
            non_store_count: AtomicUsize::new(0),
            surface: SurfaceKind::Cli,
            exit_code: AtomicI32::new(0),
        }
    }

    pub fn for_mcp(config: Arc<ReinConfig>) -> Self {
        Self {
            config,
            non_store_count: AtomicUsize::new(0),
            surface: SurfaceKind::Mcp,
            exit_code: AtomicI32::new(0),
        }
    }

    pub fn for_rest(config: Arc<ReinConfig>) -> Self {
        Self {
            config,
            non_store_count: AtomicUsize::new(0),
            surface: SurfaceKind::Rest,
            exit_code: AtomicI32::new(0),
        }
    }

    /// CLI-only: ops that want the process to exit with a specific code call
    /// this before returning. `doctor` uses it to preserve the historical
    /// `rein doctor` exit-1-on-failure contract that CI scripts depend on.
    /// MCP/REST surfaces also store the value but the adapters ignore it.
    pub fn set_exit_code(&self, code: i32) {
        self.exit_code.store(code, Ordering::Relaxed);
    }

    /// Read and clear the pending exit code. Returns `None` when no op
    /// requested an exit (the common case).
    pub fn take_exit_code(&self) -> Option<i32> {
        let code = self.exit_code.swap(0, Ordering::Relaxed);
        (code != 0).then_some(code)
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
