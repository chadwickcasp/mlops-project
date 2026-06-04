# Monday workflow suggestions

Weekly slots for template-informed workflow ideas. **Not** the same as
[workflow-review/latest.md](../workflow-review/latest.md) (rule-change approval).

## Schedule

- **Monday 09:00 local** (launchd): creates `YYYY-MM-DD-monday.md` stub + `unviewed` queue entry.
- **Your Agent chat**: use [KICKOFF.md](KICKOFF.md) to generate 1 big + 3 small proposals.
- **Auto viewed**: `afterAgentResponse` detects required headings in the reply; `stop` marks the week **viewed** (no CLI step). Transcripts are often redacted, so detection uses the live response hook—not the saved transcript alone.

## Queue (FIFO)

| Status | Meaning |
|--------|---------|
| `unviewed` | Stub exists; kickoff not completed (or backlog) |
| `viewed` | Structured kickoff reply was delivered in Agent mode |

**sessionStart** surfaces the **oldest unviewed** stub; when all are viewed, the **newest** week.

State: `.cursor/workflow-suggestions/queue.json` (gitignored).

External artifact directories: `.cursor/workflow-paths.json` (see `workflow-paths.example.json`).

## Commands

```bash
REPO="/Users/chadcasper/Documents/MLOps Zoomcamp/mlops-project"

# Launchd-equivalent (any day, for testing)
/usr/bin/python3 "$REPO/.cursor/hooks/monday_workflow_suggestions.py" --due

# Force rebuild today's stub
/usr/bin/python3 "$REPO/.cursor/hooks/monday_workflow_suggestions.py" --due --force

# Manual fallback only
/usr/bin/python3 "$REPO/.cursor/hooks/monday_workflow_suggestions.py" --mark-viewed 2026-06-04

# Inspect active queue entry
/usr/bin/python3 "$REPO/.cursor/hooks/monday_workflow_suggestions.py" --get-active
```

## Manual test (full)

1. Phase A: `--due` (above)
2. Phase B: Agent chat with [KICKOFF.md](KICKOFF.md)
3. Phase C: verify `queue.json` shows `viewed` after Agent `stop` (automatic)
