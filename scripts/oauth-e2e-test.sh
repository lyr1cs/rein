#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMPDIR="$(mktemp -d)"
PORT="${REIN_OAUTH_E2E_PORT:-18730}"
PID=""

cleanup() {
  if [[ -n "${PID}" ]]; then
    kill "${PID}" >/dev/null 2>&1 || true
    wait "${PID}" >/dev/null 2>&1 || true
  fi
  rm -rf "${TMPDIR}"
}
trap cleanup EXIT

cat >"${TMPDIR}/config.toml" <<EOF
[database]
path = "${TMPDIR}/memories.db"

[server]
sse_enabled = true
sse_bind = "127.0.0.1"
sse_port = ${PORT}
auth = "oauth"
public_url = "http://127.0.0.1:${PORT}"
background_warmup = false
EOF

(cd "${ROOT}" && cargo build -p rein >/dev/null)
REIN_HTTP_TOKEN="oauth-e2e-owner-token" REIN_CONFIG="${TMPDIR}/config.toml" "${ROOT}/target/debug/rein" serve --sse >"${TMPDIR}/rein.log" 2>&1 &
PID="$!"

python3 - "$PORT" "$TMPDIR" <<'PY'
import base64
import hashlib
import html
import http.client
import json
import os
import re
import secrets
import sys
import time
import urllib.parse

port = int(sys.argv[1])
tmp = sys.argv[2]
base = f"127.0.0.1:{port}"

def request(method, path, body=None, headers=None, follow=False):
    headers = dict(headers or {})
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    conn.request(method, path, body=body, headers=headers)
    resp = conn.getresponse()
    data = resp.read()
    status = resp.status
    out_headers = {k.lower(): v for k, v in resp.getheaders()}
    conn.close()
    if follow and status in (301, 302, 303, 307, 308):
        raise AssertionError("unexpected redirect follow request")
    return status, out_headers, data

deadline = time.time() + 20
while True:
    try:
        status, _, _ = request("GET", "/.well-known/oauth-authorization-server")
        if status == 200:
            break
    except OSError:
        pass
    if time.time() > deadline:
        print(open(os.path.join(tmp, "rein.log")).read(), file=sys.stderr)
        raise SystemExit("server did not start")
    time.sleep(0.2)

status, _, data = request(
    "POST",
    "/oauth/register",
    json.dumps({
        "client_name": "e2e-claude",
        "redirect_uris": ["http://localhost/callback"],
        "grant_types": ["authorization_code", "refresh_token"],
        "token_endpoint_auth_method": "none",
    }),
    {"content-type": "application/json"},
)
assert status == 201, (status, data)
client = json.loads(data)
client_id = client["client_id"]

mcp_body = json.dumps({"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"oauth-e2e","version":"0"}}})
status, headers, data = request(
    "POST",
    "/mcp",
    mcp_body,
    {"content-type": "application/json", "accept": "application/json, text/event-stream"},
)
assert status == 401, (status, data[:200])
challenge = headers.get("www-authenticate", "")
assert '/.well-known/oauth-protected-resource/mcp' in challenge, challenge
status, _, data = request("GET", "/.well-known/oauth-protected-resource/mcp")
assert status == 200, (status, data)
resource_metadata = json.loads(data)
assert resource_metadata["resource"] == f"http://127.0.0.1:{port}/mcp", resource_metadata

status, headers, data = request(
    "POST",
    "/api/session",
    "{}",
    {
        "content-type": "application/json",
        "x-rein-action": "1",
        "authorization": "Bearer oauth-e2e-owner-token",
    },
)
assert status == 200, (status, data)
owner_cookie = headers["set-cookie"].split(";", 1)[0]

verifier = base64.urlsafe_b64encode(secrets.token_bytes(48)).decode().rstrip("=")
challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).decode().rstrip("=")
state = "state-" + secrets.token_urlsafe(8)
qs = urllib.parse.urlencode({
    "client_id": client_id,
    "redirect_uri": "http://localhost/callback",
    "response_type": "code",
    "code_challenge": challenge,
    "code_challenge_method": "S256",
    "state": state,
})
status, headers, data = request("GET", f"/oauth/authorize?{qs}", headers={"cookie": owner_cookie})
assert status == 200, (status, data[:200])
csrf_cookie = headers["set-cookie"].split(";", 1)[0]
csrf = re.search(rb'name="csrf_token" value="([^"]+)"', data).group(1).decode()

form = urllib.parse.urlencode({
    "client_id": client_id,
    "redirect_uri": "http://localhost/callback",
    "code_challenge": challenge,
    "state": state,
    "csrf_token": html.unescape(csrf),
    "action": "allow",
})
status, headers, _ = request(
    "POST",
    "/oauth/authorize",
    form,
    {"content-type": "application/x-www-form-urlencoded", "cookie": f"{owner_cookie}; {csrf_cookie}"},
)
assert status == 302, status
location = headers["location"]
parsed = urllib.parse.urlparse(location)
params = urllib.parse.parse_qs(parsed.query)
assert params["state"][0] == state
code = params["code"][0]

token_form = urllib.parse.urlencode({
    "grant_type": "authorization_code",
    "client_id": client_id,
    "redirect_uri": "http://localhost/callback",
    "code": code,
    "code_verifier": verifier,
})
status, _, data = request("POST", "/oauth/token", token_form, {"content-type": "application/x-www-form-urlencoded"})
assert status == 200, (status, data)
tokens = json.loads(data)
access = tokens["access_token"]
refresh = tokens["refresh_token"]

mcp_headers = {
    "content-type": "application/json",
    "accept": "application/json, text/event-stream",
    "authorization": f"Bearer {access}",
}
status, _, data = request("POST", "/mcp", mcp_body, mcp_headers)
assert status == 200, (status, data[:200])

refresh_form = urllib.parse.urlencode({
    "grant_type": "refresh_token",
    "client_id": client_id,
    "refresh_token": refresh,
})
status, _, data = request("POST", "/oauth/token", refresh_form, {"content-type": "application/x-www-form-urlencoded"})
assert status == 200, (status, data)
tokens2 = json.loads(data)
access2 = tokens2["access_token"]

revoke_form = urllib.parse.urlencode({
    "client_id": client_id,
    "token": access2,
    "token_type_hint": "access_token",
})
status, _, data = request("POST", "/oauth/revoke", revoke_form, {"content-type": "application/x-www-form-urlencoded"})
assert status == 200, (status, data)

mcp_headers["authorization"] = f"Bearer {access2}"
status, _, _ = request("POST", "/mcp", mcp_body, mcp_headers)
assert status == 401, status

print("oauth e2e ok")
PY
