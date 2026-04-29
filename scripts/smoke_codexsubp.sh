#!/usr/bin/env bash
set -euo pipefail

# Smoke the ChatGPT-login loopback entrypoint against a running rein proxy.
# Keep `codexsubp_provider.toml.tmpl` as the single source of truth.
PROMPT=${1:-"Reply with exactly OK."}
PROXY_URL=${REIN_PROXY_URL:-http://127.0.0.1:8690}
TMP_WORKDIR=""
if [[ -n "${REIN_SMOKE_WORKDIR:-}" ]]; then
  WORKDIR=$REIN_SMOKE_WORKDIR
  mkdir -p "$WORKDIR"
else
  WORKDIR=$(mktemp -d "${TMPDIR:-/tmp}/rein-smoke-codexsubp.XXXXXX")
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
# v0.27.4 codex R5 P1 + R7 P3: the template uses Codex CLI's
# `env_http_headers` form, which reads `REIN_PROXY_TOKEN` at invocation
# time. We don't substitute the token literally here. Operators running
# against an authenticated proxy must set REIN_PROXY_TOKEN; operators
# running with `[proxy].allow_unauthenticated_loopback = true` don't
# need to set anything, and `env_http_headers` simply omits the header
# when the env var is unset.
PROVIDER_OVERRIDE=$(sed "s${SOH}__PROXY_URL__${SOH}${PROXY_URL//\\/\\\\}${SOH}g" "$SCRIPT_DIR/codexsubp_provider.toml.tmpl")
PROVIDER_OVERRIDE=$(printf '%s' "$PROVIDER_OVERRIDE" \
  | sed \
      -e "s#__PROVIDER_KEY__#rein_sub_proxy#g" \
      -e "s#__PROVIDER_NAME__#Rein Subscription Proxy#g" \
      -e "s#__SUPPORTS_WEBSOCKETS__#false#g")

codex exec \
  -C "$WORKDIR" \
  -c "$PROVIDER_OVERRIDE" \
  -c "model_provider=\"rein_sub_proxy\"" \
  -c "chatgpt_base_url=\"$PROXY_URL\"" \
  "${SANDBOX_ARGS[@]}" \
  "$PROMPT" < /dev/null
