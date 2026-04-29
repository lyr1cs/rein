#!/usr/bin/env bash
set -euo pipefail

# Smoke the experimental websocket-enabled ChatGPT-login loopback path.
# Keep `codexsubp_provider.toml.tmpl` as the single source of truth.
PROMPT=${1:-"Reply with exactly OK."}
PROXY_URL=${REIN_PROXY_URL:-http://127.0.0.1:8690}
TMP_WORKDIR=""
if [[ -n "${REIN_SMOKE_WORKDIR:-}" ]]; then
  WORKDIR=$REIN_SMOKE_WORKDIR
  mkdir -p "$WORKDIR"
else
  WORKDIR=$(mktemp -d "${TMPDIR:-/tmp}/rein-smoke-codexsubp-ws.XXXXXX")
  TMP_WORKDIR=$WORKDIR
fi
cleanup() {
  if [[ -n "$TMP_WORKDIR" ]]; then
    rm -rf "$TMP_WORKDIR"
  fi
}
trap cleanup EXIT
SANDBOX_ARGS=()
if [[ "${REIN_SMOKE_ALLOW_DANGEROUS_SANDBOX:-0}" == "1" ]]; then
  SANDBOX_ARGS+=(--dangerously-bypass-approvals-and-sandbox)
fi
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
SOH=$'\x01'
# v0.27.4 codex R5 P1 + R7 P3: env_http_headers — see
# smoke_codexsubp.sh for the matching block.
PROVIDER_OVERRIDE=$(sed \
  -e "s#__PROVIDER_KEY__#rein_sub_proxy_ws#g" \
  -e "s#__PROVIDER_NAME__#Rein Subscription Proxy WS#g" \
  -e "s${SOH}__PROXY_URL__${SOH}${PROXY_URL//\\/\\\\}${SOH}g" \
  -e "s#__SUPPORTS_WEBSOCKETS__#true#g" \
  "$SCRIPT_DIR/codexsubp_provider.toml.tmpl")

REIN_SUB_PROXY_WS=1 codex exec \
  -C "$WORKDIR" \
  -c "$PROVIDER_OVERRIDE" \
  -c "model_provider=\"rein_sub_proxy_ws\"" \
  -c "chatgpt_base_url=\"$PROXY_URL\"" \
  "${SANDBOX_ARGS[@]}" \
  "$PROMPT" < /dev/null
