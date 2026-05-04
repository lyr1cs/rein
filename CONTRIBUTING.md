# Contributing to rein

Thanks for your interest in improving rein.

## Copyright and License

The rein project copyright is held by **Eric Lee** (the project maintainer).
Contributions you submit under DCO 1.1 (see below) are licensed to the
project under AGPL-3.0-or-later; you retain copyright over your own
contributions, and the project's right to dual-license future versions
extends only to contributions covered by your DCO sign-off.

rein is licensed under **AGPL-3.0-or-later**. By contributing you agree that your contribution will be distributed under the same license.

## Developer Certificate of Origin (DCO)

To keep the project's copyright provenance clean and to preserve the option to dual-license rein in the future (e.g., a commercial license alongside AGPL), every commit must be **signed off** under the [Developer Certificate of Origin 1.1](https://developercertificate.org/).

A sign-off is a single line at the end of your commit message:

```
Signed-off-by: Your Name <your.email@example.com>
```

(GitHub users who prefer to keep their real email private can use the
GitHub-provided no-reply form: `<id>+<username>@users.noreply.github.com`.)

`git` can add it automatically:

```bash
git commit -s -m "fix: short description"
```

Or globally:

```bash
git config --global format.signoff true
```

By signing off, you certify the [DCO 1.1 text](https://developercertificate.org/) — in plain English: that you wrote the contribution (or have the right to submit it under the project's license), and that you understand it will be public and recorded forever in the git history.

PRs without a `Signed-off-by` will not be merged.

## Workflow

1. Open an issue first for non-trivial work — saves both of us time
2. Fork the repo, branch from `master`
3. Run the full pre-flight before pushing:
   ```bash
   cargo test --workspace --no-fail-fast --features test-support
   cargo clippy --workspace --all-targets -- -D warnings
   cargo fmt --all -- --check
   ```
4. For bigger changes: include or update tests, and run a `codex review --uncommitted` round if you have it (not required, but appreciated)
5. Open a PR against `master` with a short description and the test plan you ran

## Commit style

Conventional-commit-ish: `feat:` / `fix:` / `docs:` / `chore:` / `refactor:` / `test:` / `release(vX.Y.Z):` ... — match what's already in the git log.

## Reporting issues

Use the GitHub issue tracker. For security issues, email the project owner directly rather than opening a public issue.
