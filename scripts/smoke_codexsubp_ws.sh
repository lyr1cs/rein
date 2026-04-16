#!/usr/bin/env bash
set -euo pipefail

# Smoke the experimental websocket-enabled ChatGPT-login loopback path.
# Keep `codexsubp_provider.toml.tmpl` as the single source of truth.
PROMPT=${1:-"Reply with exactly OK."}
PROXY_URL=${REIN_PROXY_URL:-http://127.0.0.1:8690}
WORKDIR=${REIN_SMOKE_WORKDIR:-$(pwd)}
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROVIDER_OVERRIDE=$(sed \
  -e "s#__PROVIDER_KEY__#rein_sub_proxy_ws#g" \
  -e "s#__PROVIDER_NAME__#Rein Subscription Proxy WS#g" \
  -e "s#__PROXY_URL__#${PROXY_URL//\\/\\\\}#g" \
  -e "s#__SUPPORTS_WEBSOCKETS__#true#g" \
  "$SCRIPT_DIR/codexsubp_provider.toml.tmpl")

REIN_SUB_PROXY_WS=1 codex exec \
  -C "$WORKDIR" \
  -c "$PROVIDER_OVERRIDE" \
  -c "model_provider=\"rein_sub_proxy_ws\"" \
  -c "chatgpt_base_url=\"$PROXY_URL\"" \
  --dangerously-bypass-approvals-and-sandbox \
  "$PROMPT" < /dev/null
