# Workflow review (daily → weekly)

Automated digests from Cursor agent transcripts and git activity, rolling up into an
approval-gated weekly report for `AGENTS.md` and `.cursor/rules/*.mdc`.

Monday **workflow suggestions** (template ideas) live separately in
[../workflow-suggestions/README.md](../workflow-suggestions/README.md).

## Layout

| Path | Retention |
|------|-----------|
| `daily/YYYY-MM-DD.md` | Current ISO week only (Mon–Sun, local time); older files pruned automatically |
| `latest.md` | Current week’s working weekly draft (gitignored while in progress) |
| `archive/YYYY-MM-DD.md` | Prior weekly snapshots — kept indefinitely |
| `index.md` | Auto-generated index of dailies, latest, and archive |

## Automation

- **sessionStart:** catch-up yesterday’s digest if missing; brief digest summary on first prompt; Monday suggestion nudge; surface pending weekly approval.
- **stop:** when ≥7 daily digests and repo edits, write `latest.md` and ask the agent to fill proposed diffs (Pro quota).
- **launchd (optional):** daily backup at 06:00 local — see [../launchd/README.md](../launchd/README.md).

## Manual commands

```bash
# Catch-up / yesterday
python3 .cursor/hooks/daily_workflow_digest.py --catch-up

# Backfill today + yesterday
python3 .cursor/hooks/daily_workflow_digest.py --backfill-days 2 --force

# Weekly stub (when ≥7 dailies)
python3 .cursor/hooks/weekly_workflow_report.py --write-stub

# After you approve changes in chat
python3 .cursor/hooks/weekly_workflow_report.py --finalize-approval
```

## Approval

Do **not** apply `AGENTS.md` or `.mdc` edits until you reply **yes**, **no**, or **edit** in chat.

State lives in `.cursor/agent-workflow-state.json` (gitignored).
