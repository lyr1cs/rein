#!/usr/bin/env bash
set -euo pipefail

# Smoke the ChatGPT-login loopback entrypoint against a running rein proxy.
# Keep `codexsubp_provider.toml.tmpl` as the single source of truth.
PROMPT=${1:-"Reply with exactly OK."}
PROXY_URL=${REIN_PROXY_URL:-http://127.0.0.1:8690}
WORKDIR=${REIN_SMOKE_WORKDIR:-$(pwd)}
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROVIDER_OVERRIDE=$(sed "s#__PROXY_URL__#${PROXY_URL//\\/\\\\}#g" "$SCRIPT_DIR/codexsubp_provider.toml.tmpl")
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
  --dangerously-bypass-approvals-and-sandbox \
  "$PROMPT" < /dev/null
