#!/usr/bin/env bash
# Cursor stop hook: weekly workflow review, Monday auto-viewed, then exit.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
STATE_FILE="$REPO_ROOT/.cursor/agent-workflow-state.json"
HOOK_JSON="$(cat)"

read_state() {
  if [[ -f "$STATE_FILE" ]]; then
    cat "$STATE_FILE"
  else
    echo '{}'
  fi
}

days_since() {
  local then_date="${1:-}"
  if [[ -z "$then_date" ]]; then
    echo 999
    return
  fi
  local then_sec now_sec
  then_sec=$(date -j -f "%Y-%m-%d" "$then_date" "+%s" 2>/dev/null || date -d "$then_date" "+%s" 2>/dev/null || echo 0)
  now_sec=$(date "+%s")
  echo $(( (now_sec - then_sec) / 86400 ))
}

emit_followup() {
  local msg="$1"
  python3 -c 'import json,sys; print(json.dumps({"followup_message": sys.argv[1]}))' "$msg"
}

STATE_JSON="$(read_state)"

LAST_REVIEW=$(python3 -c "
import json,sys
d=json.loads(sys.argv[1] or '{}')
print(d.get('last_workflow_review_prompt_at',''))
" "$STATE_JSON")

PENDING=$(python3 -c "
import json,sys
d=json.loads(sys.argv[1] or '{}')
print('1' if d.get('pending_approval') else '0')
" "$STATE_JSON")

# Weekly stub when due and not already pending from this week
if [[ "$PENDING" != "1" ]] && [[ "$(days_since "$LAST_REVIEW")" -ge 7 ]]; then
  RESULT=$(cd "$REPO_ROOT/.cursor/hooks" && python3 weekly_workflow_report.py --write-stub 2>/dev/null || echo '{}')
  if echo "$RESULT" | python3 -c "import json,sys; d=json.load(sys.stdin); sys.exit(0 if d.get('path') else 1)" 2>/dev/null; then
    emit_followup "Weekly workflow review is ready at .cursor/workflow-review/latest.md. Read it and the daily/ digests, synthesize top 3–5 recommended changes for AGENTS.md and relevant .cursor/rules/*.mdc files as fenced unified diffs in the Proposed diffs section. Compare with the sources appendix. Summarize in chat, then ask: Apply these changes? (yes / no / edit). Do not edit AGENTS.md or .mdc until the user approves."
    exit 0
  fi
fi

if [[ "$PENDING" == "1" ]]; then
  emit_followup "A weekly workflow review is pending at .cursor/workflow-review/latest.md. Summarize themes and top recommendations, then ask: Apply these changes? (yes / no / edit). Do not edit AGENTS.md or .mdc rules until approved."
  exit 0
fi

# Auto mark Monday suggestion viewed after structured kickoff in Agent chat
MARK_RESULT=$(cd "$REPO_ROOT/.cursor/hooks" && echo "$HOOK_JSON" | python3 monday_workflow_suggestions.py --mark-viewed-from-stop 2>/dev/null || echo '{}')
if echo "$MARK_RESULT" | python3 -c "import json,sys; d=json.load(sys.stdin); sys.exit(0 if d.get('ok') else 1)" 2>/dev/null; then
  echo '{}'
  exit 0
fi

echo '{}'
