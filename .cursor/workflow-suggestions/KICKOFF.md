# Monday workflow kickoff

Use this in an **Agent mode** chat in mlops-project after `--due` (or Monday 09:00 launchd) has created today's stub.

## Copy-paste prompt

```text
Run the Monday workflow kickoff for this repo.

Read .cursor/workflow-suggestions/KICKOFF.md and the active *-monday.md stub (oldest unviewed in queue).
Read external directories listed under ## External directories in the stub and configured in .cursor/workflow-paths.json.
For Workflow Templates, prefer the newest dated subfolders (e.g. 2026-6-4_*).
Read recent .cursor/workflow-review/daily/ if present, plus AGENTS.md and .cursor/rules/.

Reply using exactly these markdown headings (required for auto-viewed on stop):
## Big change candidate
## Small change 1
## Small change 2
## Small change 3
## Sources
## Ship now vs later

Under each heading: name, why it fits this repo, and ship-now vs later. Use real content—not placeholders.

Do not edit repo files. Ask which items I want in a separate implementation plan.
```

## Required headings

The `stop` hook marks the active suggestion **viewed** only when the assistant reply includes
all of:

- `## Big change candidate`
- `## Small change 1`
- `## Small change 2`
- `## Small change 3`

with non-empty content (not `_(awaiting kickoff chat)_`).

## Out of scope for this chat

- Implementing eval loops, scope drills, skills, or rule edits unless the user explicitly asks in a follow-up plan.
