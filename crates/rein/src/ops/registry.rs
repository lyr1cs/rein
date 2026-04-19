#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OperationKind {
    Cli,
    Mcp,
    Rest,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Operation {
    pub kind: OperationKind,
    pub category: &'static str,
    pub name: &'static str,
}

impl Operation {
    pub const fn new(kind: OperationKind, category: &'static str, name: &'static str) -> Self {
        Self {
            kind,
            category,
            name,
        }
    }
}

macro_rules! op {
    ($kind:ident, $category:expr, $name:expr) => {
        Operation::new(OperationKind::$kind, $category, $name)
    };
}

pub const CLI_OPERATIONS: &[Operation] = &[
    op!(Cli, "server", "rein serve"),
    op!(Cli, "memory", "rein store"),
    op!(Cli, "ingest", "rein ingest"),
    op!(Cli, "memory", "rein recall"),
    op!(Cli, "memory", "rein forget"),
    op!(Cli, "memory", "rein update"),
    op!(Cli, "memory", "rein topics"),
    op!(Cli, "diagnostics", "rein stats"),
    op!(Cli, "diagnostics", "rein health"),
    op!(Cli, "diagnostics", "rein doctor"),
    op!(Cli, "knowledge", "rein consolidate"),
    op!(Cli, "maintenance", "rein dedup"),
    op!(Cli, "maintenance", "rein intelligent-merge-try"),
    op!(Cli, "maintenance", "rein cleanup"),
    op!(Cli, "maintenance", "rein migrate"),
    op!(Cli, "integration", "rein init"),
    op!(Cli, "memory", "rein recent"),
    op!(Cli, "memory", "rein canonicals"),
    op!(Cli, "memory", "rein evidence"),
    op!(Cli, "maintenance", "rein dedup-log"),
    op!(Cli, "maintenance", "rein gc"),
    op!(Cli, "knowledge", "rein organize"),
    op!(Cli, "knowledge", "rein dedup-concepts"),
    op!(Cli, "memory", "rein export"),
    op!(Cli, "knowledge", "rein upgrade"),
    op!(Cli, "index", "rein warmup"),
    op!(Cli, "diagnostics", "rein config"),
    op!(Cli, "adaptive", "rein adaptive-status"),
    op!(Cli, "worker", "rein worker"),
    op!(Cli, "hooks", "rein hook"),
    op!(Cli, "service", "rein dashboard"),
    op!(Cli, "service", "rein gui"),
    op!(Cli, "service", "rein proxy"),
];

pub const MCP_OPERATIONS: &[Operation] = &[
    op!(Mcp, "memory", "rein_recall"),
    op!(Mcp, "memory", "rein_store"),
    op!(Mcp, "session", "rein_ingest_session"),
    op!(Mcp, "memory", "rein_update"),
    op!(Mcp, "memory", "rein_forget"),
    op!(Mcp, "memory", "rein_list_topics"),
    op!(Mcp, "memory", "rein_stats"),
    op!(Mcp, "memory", "rein_health"),
    op!(Mcp, "maintenance", "rein_consolidate"),
    op!(Mcp, "maintenance", "rein_dedup"),
    op!(Mcp, "maintenance", "rein_cleanup"),
    op!(Mcp, "memory", "rein_recent"),
    op!(Mcp, "maintenance", "rein_gc"),
    op!(Mcp, "knowledge", "rein_organize"),
    op!(Mcp, "knowledge", "rein_memoir_create"),
    op!(Mcp, "knowledge", "rein_memoir_list"),
    op!(Mcp, "knowledge", "rein_memoir_show"),
    op!(Mcp, "knowledge", "rein_memoir_add_concept"),
    op!(Mcp, "knowledge", "rein_memoir_refine"),
    op!(Mcp, "knowledge", "rein_memoir_search"),
    op!(Mcp, "knowledge", "rein_memoir_search_all"),
    op!(Mcp, "knowledge", "rein_memoir_link"),
    op!(Mcp, "knowledge", "rein_memoir_inspect"),
    op!(Mcp, "knowledge", "rein_memoir_export"),
    op!(Mcp, "knowledge", "rein_dedup_concepts"),
    op!(Mcp, "adaptive", "rein_adaptive_status"),
    // phantom rebalance: rein_dedup_concepts moved from phantom→real inventory (+1);
    // rein_upgrade added as placeholder for the forthcoming upgrade MCP surface.
    op!(Mcp, "knowledge", "rein_upgrade"),
    op!(Mcp, "memory", "rein_canonicals"),
    op!(Mcp, "memory", "rein_evidence"),
    op!(Mcp, "knowledge", "rein_timeline"),
    op!(Mcp, "knowledge", "rein_concept_history"),
];

pub const REST_OPERATIONS: &[Operation] = &[
    op!(Rest, "metrics", "GET /api/stats"),
    op!(Rest, "metrics", "GET /api/activity"),
    op!(Rest, "memory", "GET /api/topics"),
    op!(Rest, "memory", "GET /api/recent"),
    op!(Rest, "adaptive", "GET /api/adaptive"),
    op!(Rest, "metrics", "GET /api/dedup_decisions"),
    op!(Rest, "metrics", "GET /api/intelligent_merge_metrics"),
    op!(Rest, "health", "GET /api/health"),
    op!(Rest, "diagnostics", "GET /api/doctor"),
    op!(Rest, "diagnostics", "POST /api/doctor"),
    op!(Rest, "session", "POST /api/ingest_session"),
    op!(Rest, "session", "POST /api/session"),
    op!(Rest, "session", "DELETE /api/session"),
    op!(Rest, "memory", "GET /api/recall_stream"),
    op!(Rest, "memory", "GET /api/memories"),
    op!(Rest, "memory", "GET /api/memories/{id}"),
    op!(Rest, "knowledge", "GET /api/memoirs"),
    op!(Rest, "knowledge", "GET /api/memoirs/{name}"),
    op!(Rest, "timeline", "GET /api/timeline"),
    op!(Rest, "timeline", "GET /api/episodes"),
    op!(Rest, "artifacts", "GET /api/artifacts"),
    op!(Rest, "artifacts", "GET /api/artifacts/{id}"),
    op!(Rest, "memory", "DELETE /api/memories/{id}"),
    op!(Rest, "metrics", "GET /api/version"),
    op!(Rest, "memory", "GET /api/canonicals"),
    op!(Rest, "memory", "GET /api/evidence"),
    op!(Rest, "maintenance", "POST /api/gc"),
    op!(Rest, "maintenance", "POST /api/dedup"),
    op!(Rest, "knowledge", "POST /api/dedup_concepts"),
    op!(Rest, "knowledge", "POST /api/organize"),
];

pub const ALL_OPERATIONS: &[Operation] = &[
    CLI_OPERATIONS[0],
    CLI_OPERATIONS[1],
    CLI_OPERATIONS[2],
    CLI_OPERATIONS[3],
    CLI_OPERATIONS[4],
    CLI_OPERATIONS[5],
    CLI_OPERATIONS[6],
    CLI_OPERATIONS[7],
    CLI_OPERATIONS[8],
    CLI_OPERATIONS[9],
    CLI_OPERATIONS[10],
    CLI_OPERATIONS[11],
    CLI_OPERATIONS[12],
    CLI_OPERATIONS[13],
    CLI_OPERATIONS[14],
    CLI_OPERATIONS[15],
    CLI_OPERATIONS[16],
    CLI_OPERATIONS[17],
    CLI_OPERATIONS[18],
    CLI_OPERATIONS[19],
    CLI_OPERATIONS[20],
    CLI_OPERATIONS[21],
    CLI_OPERATIONS[22],
    CLI_OPERATIONS[23],
    CLI_OPERATIONS[24],
    CLI_OPERATIONS[25],
    CLI_OPERATIONS[26],
    CLI_OPERATIONS[27],
    CLI_OPERATIONS[28],
    CLI_OPERATIONS[29],
    CLI_OPERATIONS[30],
    CLI_OPERATIONS[31],
    CLI_OPERATIONS[32],
    MCP_OPERATIONS[0],
    MCP_OPERATIONS[1],
    MCP_OPERATIONS[2],
    MCP_OPERATIONS[3],
    MCP_OPERATIONS[4],
    MCP_OPERATIONS[5],
    MCP_OPERATIONS[6],
    MCP_OPERATIONS[7],
    MCP_OPERATIONS[8],
    MCP_OPERATIONS[9],
    MCP_OPERATIONS[10],
    MCP_OPERATIONS[11],
    MCP_OPERATIONS[12],
    MCP_OPERATIONS[13],
    MCP_OPERATIONS[14],
    MCP_OPERATIONS[15],
    MCP_OPERATIONS[16],
    MCP_OPERATIONS[17],
    MCP_OPERATIONS[18],
    MCP_OPERATIONS[19],
    MCP_OPERATIONS[20],
    MCP_OPERATIONS[21],
    MCP_OPERATIONS[22],
    MCP_OPERATIONS[23],
    MCP_OPERATIONS[24],
    MCP_OPERATIONS[25],
    MCP_OPERATIONS[26],
    MCP_OPERATIONS[27],
    MCP_OPERATIONS[28],
    MCP_OPERATIONS[29],
    MCP_OPERATIONS[30],
    REST_OPERATIONS[0],
    REST_OPERATIONS[1],
    REST_OPERATIONS[2],
    REST_OPERATIONS[3],
    REST_OPERATIONS[4],
    REST_OPERATIONS[5],
    REST_OPERATIONS[6],
    REST_OPERATIONS[7],
    REST_OPERATIONS[8],
    REST_OPERATIONS[9],
    REST_OPERATIONS[10],
    REST_OPERATIONS[11],
    REST_OPERATIONS[12],
    REST_OPERATIONS[13],
    REST_OPERATIONS[14],
    REST_OPERATIONS[15],
    REST_OPERATIONS[16],
    REST_OPERATIONS[17],
    REST_OPERATIONS[18],
    REST_OPERATIONS[19],
    REST_OPERATIONS[20],
    REST_OPERATIONS[21],
    REST_OPERATIONS[22],
    REST_OPERATIONS[23],
    REST_OPERATIONS[24],
    REST_OPERATIONS[25],
    REST_OPERATIONS[26],
    REST_OPERATIONS[27],
    REST_OPERATIONS[28],
    REST_OPERATIONS[29],
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct OperationCounts {
    pub cli: usize,
    pub mcp: usize,
    pub rest: usize,
}

pub fn cli_operations() -> &'static [Operation] {
    CLI_OPERATIONS
}

pub fn mcp_operations() -> &'static [Operation] {
    MCP_OPERATIONS
}

pub fn rest_operations() -> &'static [Operation] {
    REST_OPERATIONS
}

pub fn counts() -> OperationCounts {
    OperationCounts {
        cli: CLI_OPERATIONS.len(),
        mcp: MCP_OPERATIONS.len(),
        rest: REST_OPERATIONS.len(),
    }
}

/// Compile-time assertion that ALL_OPERATIONS is kept in sync with the three
/// per-kind arrays. If any of CLI/MCP/REST grows but ALL_OPERATIONS is not
/// extended, this fires at build time rather than drifting silently.
const _: () = {
    let expected = CLI_OPERATIONS.len() + MCP_OPERATIONS.len() + REST_OPERATIONS.len();
    assert!(
        ALL_OPERATIONS.len() == expected,
        "ALL_OPERATIONS length must equal CLI_OPERATIONS + MCP_OPERATIONS + REST_OPERATIONS"
    );
};

pub fn operations(kind: OperationKind) -> &'static [Operation] {
    match kind {
        OperationKind::Cli => CLI_OPERATIONS,
        OperationKind::Mcp => MCP_OPERATIONS,
        OperationKind::Rest => REST_OPERATIONS,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_counts_match_expected_values() {
        assert_eq!(cli_operations().len(), 33);
        // Phase 2.3: evidence phantom→real inventory (+1). rein_concept_history added as
        // phantom rebalance (same pattern as rein_timeline in Task 9). 29→30.
        // Phase 2.3 Task 4: rein_dedup_concepts phantom→real inventory (+1).
        // rein_upgrade added as phantom rebalance. 30→31.
        assert_eq!(mcp_operations().len(), 31);
        // Phase 2.2 (H5 body-JSON landed): POST /api/ingest_session added
        // and POST /api/doctor moved from legacy to inventory. Both counted
        // here via the registry source-of-truth list.
        // Phase 2.3: GET /api/canonicals added (canonicals op migrated to #[op]).
        // Phase 2.3: GET /api/evidence added (evidence op migrated to #[op]).
        // Phase 2.3: POST /api/gc added (gc op migrated to #[op] inventory).
        // Phase 2.3: POST /api/dedup added (dedup op migrated to #[op] inventory).
        // Phase 2.3: POST /api/dedup_concepts added (dedup_concepts migrated to #[op] inventory).
        // Phase 2.3: POST /api/organize added (organize op migrated to #[op] inventory).
        assert_eq!(rest_operations().len(), 30);
        assert_eq!(
            counts(),
            OperationCounts {
                cli: 33,
                mcp: 31,
                rest: 30
            }
        );
    }

    #[test]
    fn registry_contains_representative_entries() {
        assert!(cli_operations().iter().any(|op| op.name == "rein doctor"));
        assert!(mcp_operations()
            .iter()
            .any(|op| op.name == "rein_memoir_export"));
        assert!(rest_operations()
            .iter()
            .any(|op| op.name == "POST /api/doctor"));
    }
}
