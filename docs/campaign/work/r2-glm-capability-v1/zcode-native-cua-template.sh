#!/usr/bin/env bash
set -euo pipefail
# TEMPLATE ONLY. Do not execute until the active provider registry explicitly lists glm-5.3-flash.
# No credentials are read by this template; the host must already have its approved auth binding.
SYNTHETIC_WORKSPACE=${SYNTHETIC_WORKSPACE:?set to an isolated synthetic workspace}
PROMPT_FILE=${PROMPT_FILE:?set to a synthetic diagnostic prompt file}
ZCODE=${ZCODE:-/opt/ZCode/resources/glm/zcode.cjs}
MODEL_TARGET=${MODEL_TARGET:-zai-coding-plan/glm-5.3-flash}
GLM_FLASH_CONFIRMED=${GLM_FLASH_CONFIRMED:?set to 1 only after a host registry/catalog check}
[ "$GLM_FLASH_CONFIRMED" = 1 ]
CONFIG="$SYNTHETIC_WORKSPACE/.zcode/config.json"

# The pinned 0.16.5 parser accepts --disallowed-tools; it does not accept the help-only
# --settings/--allowed-tools/--max-turns/--permission-mode options. Put the exact target and
# permission config in "$CONFIG" using the requested-target placeholder as a starting shape.
# This read-only check makes model selection explicit without passing unsupported --model/--target
# flags (the parser's --target mode conflicts with --prompt).
python3 - "$CONFIG" "$MODEL_TARGET" <<'EOFVALID'
import json, sys
with open(sys.argv[1], encoding='utf-8') as f:
    cfg = json.load(f)
main = cfg.get('model', {}).get('main', {})
want_provider, want_model = sys.argv[2].split('/', 1)
if main.get('provider') != want_provider or main.get('model') != want_model:
    raise SystemExit('model.main does not select the requested target')
EOFVALID

timeout --signal=TERM --kill-after=20s 600s \
  node "$ZCODE" \
    --cwd "$SYNTHETIC_WORKSPACE" \
    --prompt "$(cat "$PROMPT_FILE")" \
    --surface terminal \
    --mode plan \
    --disallowed-tools \
      Bash Edit Write \
      mcp__computer_use__open_application \
      mcp__computer_use__left_click mcp__computer_use__double_click mcp__computer_use__triple_click \
      mcp__computer_use__right_click mcp__computer_use__middle_click mcp__computer_use__scroll \
      mcp__computer_use__left_click_drag mcp__computer_use__mouse_move \
      mcp__computer_use__left_mouse_down mcp__computer_use__type mcp__computer_use__set_value \
      mcp__computer_use__select_text mcp__computer_use__key mcp__computer_use__hold_key \
      mcp__computer_use__perform_action mcp__computer_use__write_clipboard \
      mcp__node_repl__js mcp__node_repl__js_reset mcp__node_repl__js_add_node_module_dir
