#!/usr/bin/env python3
"""Convert THIRD_PARTY_NOTICES.yaml (cargo-bundle-licenses output) to a
human-readable Markdown file.

Run via scripts/generate-third-party-notices.sh — that script handles the
cargo-bundle-licenses invocation and then calls this converter.
"""

import os
import sys
from collections import Counter

try:
    import yaml
except ImportError:
    print("error: PyYAML not installed. Run:", file=sys.stderr)
    print("  pip3 install pyyaml", file=sys.stderr)
    sys.exit(1)

INPUT = "THIRD_PARTY_NOTICES.yaml"
OUTPUT = "THIRD_PARTY_NOTICES.md"


def main():
    if not os.path.exists(INPUT):
        print(
            f"error: {INPUT} not found. Run scripts/generate-third-party-notices.sh "
            f"to regenerate it first.",
            file=sys.stderr,
        )
        sys.exit(1)

    with open(INPUT, "r") as f:
        data = yaml.safe_load(f)

    deps = data["third_party_libraries"]

    license_counts = Counter()
    for d in deps:
        license_counts[d.get("license", "(unknown)")] += 1

    md_lines = [
        "# Third-Party Notices",
        "",
        f"This file lists the {len(deps)} third-party Rust crates that rein "
        "depends on (transitive dependencies of `crates/rein` and "
        "`crates/rein-macros`), along with their licenses and the full text "
        "of each license as required by their respective terms.",
        "",
        "Generated automatically by [`cargo-bundle-licenses`]"
        "(https://crates.io/crates/cargo-bundle-licenses) "
        "(see `scripts/generate-third-party-notices.sh` for the regenerate "
        "command).",
        "",
        "rein itself is licensed under AGPL-3.0-or-later — see the "
        "[`LICENSE`](LICENSE) file in the project root.",
        "",
        "## License distribution",
        "",
        "| Count | License |",
        "|------:|---------|",
    ]
    for lic, n in sorted(license_counts.items(), key=lambda x: -x[1]):
        md_lines.append(f"| {n} | `{lic}` |")

    md_lines.extend(
        [
            "",
            "## Dependency index",
            "",
            "Sorted alphabetically. Click a row to jump to its full license "
            "text below.",
            "",
            "| Crate | Version | License | Repository |",
            "|-------|--------:|---------|------------|",
        ]
    )

    for d in sorted(deps, key=lambda x: x["package_name"]):
        name = d["package_name"]
        ver = d.get("package_version", "")
        lic = d.get("license", "(unknown)")
        repo = d.get("repository") or ""
        repo_link = f"[link]({repo})" if repo else "—"
        anchor = f"#{name.replace('_', '-').lower()}"
        md_lines.append(
            f"| [`{name}`]({anchor}) | {ver} | `{lic}` | {repo_link} |"
        )

    md_lines.extend(
        [
            "",
            "---",
            "",
            "## Full license text per dependency",
            "",
        ]
    )

    for d in sorted(deps, key=lambda x: x["package_name"]):
        name = d["package_name"]
        ver = d.get("package_version", "")
        lic = d.get("license", "(unknown)")
        repo = d.get("repository") or ""
        md_lines.append(f"### {name}")
        md_lines.append("")
        md_lines.append(f"- **Version**: {ver}")
        md_lines.append(f"- **License**: {lic}")
        if repo:
            md_lines.append(f"- **Repository**: {repo}")
        md_lines.append("")
        for lic_entry in d.get("licenses", []):
            lic_id = lic_entry.get("license", "?")
            text = lic_entry.get("text") or "(no license text bundled)"
            md_lines.append(
                f"<details><summary><code>{lic_id}</code> license text</summary>"
            )
            md_lines.append("")
            md_lines.append("```")
            md_lines.append(text.rstrip())
            md_lines.append("```")
            md_lines.append("")
            md_lines.append("</details>")
            md_lines.append("")

    md = "\n".join(md_lines)
    with open(OUTPUT, "w") as f:
        f.write(md)

    print(f"==> Wrote {OUTPUT}: {os.path.getsize(OUTPUT)} bytes ({len(deps)} deps)")


if __name__ == "__main__":
    main()
