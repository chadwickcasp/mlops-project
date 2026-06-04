#!/usr/bin/env python3
"""Emit sessionStart hook JSON: digest nudge, Monday queue, pending weekly."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

_HOOKS_DIR = Path(__file__).resolve().parent
if str(_HOOKS_DIR) not in sys.path:
    sys.path.insert(0, str(_HOOKS_DIR))

from workflow_common import (
    DAILY_DIR,
    LATEST_PATH,
    REPO_ROOT,
    digest_summary_lines,
    get_timezone,
    load_state,
    prune_daily_digests,
    save_state,
)

from monday_workflow_suggestions import get_active_suggestion


def _run_catch_up() -> None:
    subprocess.run(
        [sys.executable, "daily_workflow_digest.py", "--catch-up"],
        cwd=str(_HOOKS_DIR),
        timeout=120,
        check=False,
        capture_output=True,
    )


def _digest_nudge(state: dict) -> str | None:
    ready = state.get("digest_ready_date")
    if not ready:
        return None
    notified = state.get("digest_notified_date")
    if notified == ready:
        return None
    path = DAILY_DIR / f"{ready}.md"
    if not path.is_file():
        return None
    bullets = digest_summary_lines(path, max_lines=5)
    if not bullets:
        return None
    lines = ["## Yesterday's workflow digest", ""] + [f"- {b}" for b in bullets]
    state["digest_notified_date"] = ready
    save_state(state)
    return "\n".join(lines)


def _weekly_nudge(state: dict) -> str | None:
    if not state.get("pending_approval"):
        return None
    if not LATEST_PATH.is_file():
        return (
            "A workflow rules review is pending. Read .cursor/workflow-review/latest.md "
            "when available. Ask: Apply these changes? (yes / no / edit). "
            "Do not edit AGENTS.md or .mdc until approved."
        )
    summary = digest_summary_lines(LATEST_PATH, max_lines=4)
    parts = [
        "## Weekly workflow review (pending approval)",
        "",
        "Read `.cursor/workflow-review/latest.md` for full detail.",
        "",
    ]
    if summary:
        parts.extend(f"- {s}" for s in summary)
    parts.append("")
    parts.append("Ask the user: Apply these changes? (yes / no / edit).")
    return "\n".join(parts)


def _monday_nudge(state: dict) -> str | None:
    active = get_active_suggestion()
    if not active:
        return None
    sid = active.get("id", "")
    rel_file = active.get("file", "")
    status = active.get("status", "unviewed")
    state["active_suggestion_id"] = sid
    save_state(state)
    parts = [
        "## Monday workflow suggestions",
        "",
        f"- **Active stub:** `{rel_file}` (id `{sid}`, status `{status}`)",
        "",
    ]
    if status == "unviewed":
        parts.extend(
            [
                "Run kickoff in **Agent mode** using `.cursor/workflow-suggestions/KICKOFF.md`.",
                "Reply must use headings: `## Big change candidate`, `## Small change 1`–`3`.",
                "Queue marks **viewed** automatically when that reply completes (`stop` hook).",
                "",
            ]
        )
    else:
        parts.append("Latest suggestion week was viewed. Check queue for older unviewed stubs if needed.")
        parts.append("")
    return "\n".join(parts)


def main() -> int:
    _run_catch_up()
    prune_daily_digests()
    state = load_state()
    chunks: list[str] = []
    digest = _digest_nudge(state)
    if digest:
        chunks.append(digest)
    monday = _monday_nudge(state)
    if monday:
        chunks.append(monday)
    weekly = _weekly_nudge(state)
    if weekly:
        chunks.append(weekly)
    if not chunks:
        print("{}")
        return 0
    print(json.dumps({"additional_context": "\n\n".join(chunks)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
