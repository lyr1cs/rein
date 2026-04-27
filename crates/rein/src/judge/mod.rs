//! v0.27.1 E direction — judge subsystem.
//!
//! Hosts the runtime LLM judge contract (7 invariants, J1-J7) plus its
//! atomic call-cap reservation primitive (J2 `reserve_call`). The actual
//! worker pulling jobs from the queue + emitting `SynthesisLlmJudge` /
//! `ConceptSummaryLlmJudge` events lives in
//! [`crate::ops::llm_judge_worker`].
//!
//! See `docs/superpowers/specs/2026-04-27-v0.27.1-runtime-llm-judge-design.md`
//! sections §3 (schema), §4 (judge contract), §5 (pipeline matrix), and
//! §6 (M1 consumer) for the design rationale.

pub mod contract;
