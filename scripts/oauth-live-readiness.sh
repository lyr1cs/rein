#!/usr/bin/env bash
set -euo pipefail

PUBLIC_URL="${1:-}"
if [[ -z "${PUBLIC_URL}" ]]; then
  echo "usage: $0 https://rein.example.com" >&2
  exit 64
fi

PUBLIC_URL="${PUBLIC_URL%/}"
TMPDIR="$(mktemp -d)"
trap 'rm -rf "${TMPDIR}"' EXIT

metadata="${TMPDIR}/metadata.json"
resource="${TMPDIR}/resource.json"
challenge_headers="${TMPDIR}/mcp.headers"
challenge_body="${TMPDIR}/mcp.body"

curl -fsS "${PUBLIC_URL}/.well-known/oauth-authorization-server" -o "${metadata}"
python3 - "${metadata}" "${PUBLIC_URL}" <<'PY'
import json
import sys

path, public_url = sys.argv[1], sys.argv[2]
try:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
except json.JSONDecodeError as exc:
    raise SystemExit(f"authorization-server metadata is not JSON; endpoint is not OAuth-ready: {exc}")

required = [
    "issuer",
    "authorization_endpoint",
    "token_endpoint",
    "registration_endpoint",
    "revocation_endpoint",
    "response_types_supported",
    "grant_types_supported",
    "code_challenge_methods_supported",
    "token_endpoint_auth_methods_supported",
]
missing = [key for key in required if key not in data]
if missing:
    raise SystemExit(f"metadata missing keys: {missing}")
if data["issuer"].rstrip("/") != public_url:
    raise SystemExit(f"issuer mismatch: {data['issuer']!r} != {public_url!r}")
if "S256" not in data["code_challenge_methods_supported"]:
    raise SystemExit("metadata does not advertise PKCE S256")
PY

curl -fsS "${PUBLIC_URL}/.well-known/oauth-protected-resource/mcp" -o "${resource}"
python3 - "${resource}" "${PUBLIC_URL}" <<'PY'
import json
import sys

path, public_url = sys.argv[1], sys.argv[2]
try:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
except json.JSONDecodeError as exc:
    raise SystemExit(f"protected-resource metadata is not JSON; endpoint is not OAuth-ready: {exc}")

expected_resource = f"{public_url}/mcp"
if data.get("resource") != expected_resource:
    raise SystemExit(f"resource mismatch: {data.get('resource')!r} != {expected_resource!r}")
if public_url not in data.get("authorization_servers", []):
    raise SystemExit("protected resource metadata does not point at issuer")
PY

set +e
status="$(
  curl -sS \
    -D "${challenge_headers}" \
    -o "${challenge_body}" \
    -w "%{http_code}" \
    -H 'content-type: application/json' \
    -H 'accept: application/json, text/event-stream' \
    --data '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"oauth-live-readiness","version":"0"}}}' \
    "${PUBLIC_URL}/mcp"
)"
curl_exit=$?
set -e

if [[ "${curl_exit}" -ne 0 ]]; then
  cat "${challenge_body}" >&2 || true
  exit "${curl_exit}"
fi
if [[ "${status}" != "401" ]]; then
  echo "expected unauthenticated /mcp to return 401, got ${status}" >&2
  cat "${challenge_body}" >&2 || true
  exit 1
fi
if ! grep -qi 'www-authenticate: Bearer .*resource_metadata=.*\.well-known/oauth-protected-resource/mcp' "${challenge_headers}"; then
  echo "missing OAuth protected-resource WWW-Authenticate challenge" >&2
  cat "${challenge_headers}" >&2
  exit 1
fi

echo "oauth live readiness ok: ${PUBLIC_URL}"
