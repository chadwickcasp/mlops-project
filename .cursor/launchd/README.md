# launchd — workflow automation backup

Runs scheduled hooks when Cursor may not be open. Logs append to
`.cursor/workflow-review/launchd.log` (gitignored).

| Job | Schedule | Command |
|-----|----------|---------|
| `com.mlops.cursor-daily-digest` | Daily 06:00 | `daily_workflow_digest.py --catch-up` |
| `com.mlops.cursor-monday-suggestions` | Monday 09:00 | `monday_workflow_suggestions.py --due` |

## Install (once)

Replace `REPO_ROOT_PLACEHOLDER` with your repo path, then:

```bash
REPO="/Users/chadcasper/Documents/MLOps Zoomcamp/mlops-project"

for label in com.mlops.cursor-daily-digest com.mlops.cursor-monday-suggestions; do
  sed "s|REPO_ROOT_PLACEHOLDER|$REPO|g" "$REPO/.cursor/launchd/${label}.plist" \
    > ~/Library/LaunchAgents/${label}.plist
  launchctl load ~/Library/LaunchAgents/${label}.plist
done
```

## Manual test (launchd-equivalent, no schedule)

```bash
REPO="/Users/chadcasper/Documents/MLOps Zoomcamp/mlops-project"
/usr/bin/python3 "$REPO/.cursor/hooks/daily_workflow_digest.py" --catch-up
/usr/bin/python3 "$REPO/.cursor/hooks/monday_workflow_suggestions.py" --due
```

## Uninstall

```bash
for label in com.mlops.cursor-daily-digest com.mlops.cursor-monday-suggestions; do
  launchctl unload ~/Library/LaunchAgents/${label}.plist 2>/dev/null || true
  rm -f ~/Library/LaunchAgents/${label}.plist
done
```
