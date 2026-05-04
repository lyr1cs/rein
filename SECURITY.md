# Security Policy

Thanks for taking the time to responsibly disclose any issues you find.

## Supported Versions

Rein follows semantic versioning. The latest minor release receives security
fixes; older releases are best-effort. As of this writing:

| Version | Supported           |
|---------|---------------------|
| 0.28.x  | ✅ active            |
| < 0.28  | ❌ no longer patched |

When a fix lands it ships as a patch release on the latest minor (e.g.,
v0.28.x+1) and is announced in the GitHub Release notes.

## Reporting a Vulnerability

**Please do not open a public GitHub issue for suspected security
vulnerabilities.** Public disclosure before a fix is available puts other
operators at risk.

Use one of the following private channels instead:

- **GitHub Security Advisories** (preferred):
  <https://github.com/lyr1cs/rein/security/advisories/new>

  This is a private, encrypted channel between you and the maintainers.
  GitHub coordinates disclosure with you, lets us prepare a fix in a
  private fork, and assigns a CVE if appropriate.

- **Direct contact**: open a minimal issue saying "I'd like to report
  privately, please reach out" and a maintainer will follow up off-list.

## What to Include

A useful report typically contains:

1. **Affected version(s)** — `rein --version` output, or commit SHA if you
   built from source.
2. **Threat model** — what role does the attacker have? (Local user,
   network attacker against `rein serve`, malicious memory store, malicious
   LLM upstream via the proxy, etc.)
3. **Reproduction** — minimal commands or a small test case. If a
   `cargo test` style reproducer is possible, that is ideal.
4. **Impact** — what does an attacker gain? Read access to memories,
   write access, code execution, denial of service, exfiltration through
   the LLM proxy, etc.
5. **Mitigations you tried** — config flags, environment, network setup.

## Response Timeline

- **Acknowledgement**: within **3 business days** of report.
- **Initial triage**: within **7 business days** — confirming or rejecting
  the issue, requesting more information, or proposing a fix direction.
- **Fix and disclosure**: depends on severity and complexity. We aim for
  a coordinated disclosure window of **30–90 days** from acknowledgement
  to public release. Critical issues are accelerated.

## Scope

In scope:

- Code in this repository (`crates/rein/`, `crates/rein-macros/`).
- Default-shipped configuration (`crates/rein/config/default.toml`).
- Documented operator workflows (`rein serve`, `rein doctor`,
  `rein hook *`, etc.).

Out of scope (please report to the upstream project instead):

- Vulnerabilities in third-party dependencies (use
  [RustSec advisory database](https://rustsec.org/) and the upstream
  project tracker).
- Vulnerabilities in LLM providers (Google, OpenAI, Anthropic, etc.).
- Vulnerabilities in MCP clients (Claude Code, Codex, Cursor, etc.).

If you are unsure whether a given issue is in scope, ask via the private
channel above; we'd rather receive a triage question than miss a real
problem.

## Hardening Guidance

For operators running `rein serve` on a network, see
[`docs/manual/07-security.md`](docs/manual/07-security.md) for the current
default-deny posture, host/origin guard, token requirements, and AGPL
network-use considerations.

## Acknowledgements

We list reporters in release notes (with permission) when their report
leads to a security fix. If you would like to remain anonymous, just say
so in your report.
