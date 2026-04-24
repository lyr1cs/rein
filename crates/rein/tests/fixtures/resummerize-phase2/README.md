# Phase 2 Fixture Corpus

## Provenance

- Drawn from the live production database `~/.rein/memories.db` on 2026-04-23.
- Important constraint: the live DB had `0` canonicals with `length(content) >= 5000`, so a strict >5 KB Phase 2 production-only corpus was not available.
- Fallback used here: the longest safe on-domain production canonicals available at the time, plus their real `memory_evidence` rows, yielding 30 production-derived cases across 6 categories.
- Inventory snapshot: >=5000=0, >=4000=0, >=3000=7, >=2000=17, >=1000=36.
- Invariant names in `expected_must_pass` were taken from `crates/rein/src/compression/contract.rs` (`no_new_facts`, `length_bounded`, `temporal_anchors_preserved`, `cjk_integrity`, `code_block_preserved`) rather than the informal labels in the task prompt.

## Category Stats

| Category | Cases | Avg canonical length | Avg evidence rows | Notes |
|---|---:|---:|---:|---|
| `cjk_large` | 5 | 2794.2 | 1.00 | Production recap with heavy CJK density, mixed technical prose, and release or implementation anchors. |
| `release_rollup` | 5 | 2441.4 | 1.60 | Release/status rollup with commit hashes, shipped scopes, and artifact or version metadata. |
| `planning_dense` | 5 | 2472.0 | 1.00 | Planning memo with ordered tasks, weekly sequencing, and explicit implementation constraints. |
| `architecture_policy` | 5 | 1623.4 | 1.00 | Architecture or policy rationale capturing constraints, trade-offs, and preferred future direction. |
| `audit_rollup` | 5 | 1645.2 | 1.00 | Audit/review summary covering findings, fixes, and verification notes across multiple files or subsystems. |
| `mixed_transcript` | 5 | 1479.6 | 1.20 | Transcript-style canonical with assistant narration, plan updates, and structured bullets from live work. |

## Sanitization Applied

- Replaced email addresses with `user@example.com` style placeholders.
- Replaced API-key / token patterns (`AIza...`, `sk-...`, `gho_...`, JWT-like strings) with `<REDACTED_KEY>`.
- Replaced absolute local filesystem paths with `<LOCAL_PATH>`.
- Rewrote GitHub repo-owner URLs from the production account to `https://github.com/example/rein/...` while preserving repository/path structure.
- Replaced IP-like literals with `<REDACTED_IP>` and password-like literals with `<REDACTED_SECRET>` when encountered.
- Left code, dates, commit hashes, version strings, relative file paths, and CJK text intact because they are useful signal for resummerize evaluation.

## Excluded Production Canonicals

- `01KNXQDFVPB3H17NKQ9PWPVT15` / `nrm-cover-v16-wide-film-iteration` — Excluded: off-domain image-generation workflow with dense local-path detail and weaker relevance to rein resummerize behavior.
- `01KPMR3FC7WG40MXSKWSNMMRRX` / `surge-20260420-hardening` — Excluded: off-domain personal network configuration material; useful technically, but not close to rein memory shapes.
- `01KPMR46JZQXVW4X63QC68HSXR` / `surge-rule-design-principles` — Excluded: same off-domain personal network-config regime as above.
- `01KNSMZTASMYE6NJTNP491CD2T` / `network-access` — Excluded: contained personal remote-access infrastructure plus credential-like and IP-like material; redaction would over-destroy structure.
- `01KP8P2Y7TTZD28F8J04YMVGFT` / `rein-documentation-status` — Excluded: redundant with stronger release/status rollups already selected and shorter than adjacent candidates.
- `01KPT8RXXXRFBDXC4380TKSQJ1` / `resummerize` — Excluded: short derivative summary overlapping more detailed v0.22/v0.23 planning canonicals.

## Notes

- No eval pipeline, `rein-eval`, `cargo build`, or `cargo test` was run as part of this corpus generation.
- Existing seed fixtures under `crates/rein/tests/fixtures/resummerize/` were not modified.
- Because the live DB had no >5 KB canonicals, this directory should be treated as a realistic fallback corpus, not as proof that Phase 2 production volume is already present in the database.
