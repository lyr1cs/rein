<!--
Thanks for contributing to rein!

Before opening: please make sure you've read CONTRIBUTING.md, especially the
DCO sign-off requirement and the pre-flight checklist below.
-->

## Summary

<!-- One or two sentences: what does this PR change and why? -->

## Related

- Closes #
- Related discussion / RFC / issue:

## Type

<!-- Check what applies. -->

- [ ] `feat` — new capability or behavior
- [ ] `fix` — bug fix (regression, incorrect behavior, audit finding)
- [ ] `docs` — documentation only
- [ ] `chore` — tooling / CI / dependency / build
- [ ] `refactor` — internal restructuring, no behavior change
- [ ] `test` — adding or fixing tests
- [ ] `release(vX.Y.Z)` — release prep
- [ ] Other (describe):

## Pre-flight checklist

- [ ] I ran `cargo test --workspace --no-fail-fast --features test-support` and tests pass
- [ ] I ran `cargo clippy --workspace --all-targets -- -D warnings` and lint is clean
- [ ] I ran `cargo fmt --all -- --check` and formatting is clean
- [ ] If I touched `crates/rein/gui/`, I also ran `npm run lint` and `npm run build` in `crates/rein/gui/`
- [ ] **All commits are signed off** (`git commit -s`) per [CONTRIBUTING.md](../CONTRIBUTING.md#developer-certificate-of-origin-dco)
- [ ] My commit messages follow the conventional-commit style used in the existing git log
- [ ] If I added or changed a public API (CLI, MCP, REST, config keys), I updated the relevant docs under `docs/manual/` or `docs/reference/`

## Implementation notes

<!--
What's the high-level approach? What invariants did you preserve? Any
non-obvious tradeoffs?

For changes that touch the adaptive engine (M1-M6, ARS), persistence schema,
or the recall pipeline, please call out:
- new/changed event-sourced state (and whether it requires the 5-invariant
  treatment: watermark filter + applied-prefix bump + replay-drain + CAS
  merge + per-consumer offset)
- new/changed snapshot blob keys
- backward compatibility for existing snapshots
-->

## Test plan

<!--
What did you actually run to convince yourself this is correct?
Specific test names / fixtures are more useful than "all tests pass".
For audit-style hardening PRs, link the codex review rounds you ran
(if any).
-->

## Operator-visible changes

<!--
If this PR changes anything an operator running `rein doctor` or
`rein dashboard` would notice — new config keys, default flips, schema
additions, new warnings — list them here. Otherwise write "none".
-->
