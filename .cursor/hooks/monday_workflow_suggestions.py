#!/usr/bin/env python3
"""Monday workflow suggestion stubs and viewed queue.

Run (launchd-equivalent): /usr/bin/python3 .cursor/hooks/monday_workflow_suggestions.py --due
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date, datetime
from pathlib import Path

_HOOKS_DIR = Path(__file__).resolve().parent
if str(_HOOKS_DIR) not in sys.path:
    sys.path.insert(0, str(_HOOKS_DIR))

from workflow_common import (
    DAILY_DIR,
    REPO_ROOT,
    SUGGESTIONS_DIR,
    QUEUE_PATH,
    append_launchd_log,
    get_timezone,
    load_state,
    load_workflow_directories,
    resolve_workflow_path,
    save_state,
    scan_directory_entry,
)

_REQUIRED_HEADINGS = (
    "## Big change candidate",
    "## Small change 1",
    "## Small change 2",
    "## Small change 3",
)

_KICKOFF_MARKERS = (
    "monday workflow kickoff",
    "monday kickoff",
    "workflow kickoff",
    "KICKOFF.md",
)


def _load_queue() -> dict:
    if QUEUE_PATH.is_file():
        try:
            return json.loads(QUEUE_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {"entries": []}


def _save_queue(queue: dict) -> None:
    SUGGESTIONS_DIR.mkdir(parents=True, exist_ok=True)
    QUEUE_PATH.write_text(json.dumps(queue, indent=2) + "\n", encoding="utf-8")


def _recent_daily_paths(limit: int = 7) -> list[Path]:
    if not DAILY_DIR.is_dir():
        return []
    paths = sorted(DAILY_DIR.glob("*.md"), key=lambda p: p.stem, reverse=True)
    return paths[:limit]


def _external_directory_sections() -> str:
    entries = load_workflow_directories()
    if not entries:
        return (
            "### _(not configured)_\n\n"
            "- Add directories in `.cursor/workflow-paths.json` "
            "(see `workflow-paths.example.json`)."
        )
    sections: list[str] = []
    for entry in entries:
        label = entry.get("label") or entry.get("id") or "External directory"
        raw_path = entry.get("path", "")
        if not isinstance(raw_path, str) or not raw_path.strip():
            sections.append(f"### {label}\n\n- _(path not set in workflow-paths.json)_")
            continue
        scanned = scan_directory_entry(entry)
        if scanned:
            lines = [f"- `{p}`" for p in scanned]
        elif not resolve_workflow_path(raw_path.strip()).is_dir():
            lines = [f"- _(path not found: `{raw_path}`)_"]
        else:
            lines = ["- _(no matching entries for configured mode)_"]
        sections.append(f"### {label}\n\n{chr(10).join(lines)}")
    return "\n\n".join(sections)


def _render_stub(suggestion_id: str) -> str:
    dailies = _recent_daily_paths()
    daily_lines = (
        [f"- `.cursor/workflow-review/daily/{p.name}`" for p in dailies]
        if dailies
        else ["- _(no daily digests yet)_"]
    )
    external_sections = _external_directory_sections()

    return f"""# Monday workflow suggestions — {suggestion_id}

> Status: **awaiting_review** — fill via Agent chat using `.cursor/workflow-suggestions/KICKOFF.md` (not by this script).

## Metadata (auto)

## External directories

{external_sections}

### Recent daily digests

{chr(10).join(daily_lines)}

---

## Big change candidate

_(awaiting kickoff chat)_

## Small change 1

_(awaiting kickoff chat)_

## Small change 2

_(awaiting kickoff chat)_

## Small change 3

_(awaiting kickoff chat)_

## Sources

_(template paths + dates — fill in kickoff chat)_

## Ship now vs later

_(fill in kickoff chat)_
"""


def _enqueue(suggestion_id: str, file_path: Path, *, force: bool = False) -> bool:
    queue = _load_queue()
    entries = queue.setdefault("entries", [])
    for entry in entries:
        if entry.get("id") == suggestion_id:
            if entry.get("status") == "unviewed" and not force:
                return False
            entry["file"] = str(file_path.relative_to(REPO_ROOT))
            entry["status"] = "unviewed"
            _save_queue(queue)
            return True
    entries.append(
        {
            "id": suggestion_id,
            "file": str(file_path.relative_to(REPO_ROOT)),
            "status": "unviewed",
        }
    )
    _save_queue(queue)
    return True


def get_active_suggestion() -> dict | None:
    """Oldest unviewed, else newest entry."""
    queue = _load_queue()
    entries = queue.get("entries", [])
    if not entries:
        return None
    unviewed = [e for e in entries if e.get("status") == "unviewed"]
    if unviewed:
        return sorted(unviewed, key=lambda e: e.get("id", ""))[0]
    return sorted(entries, key=lambda e: e.get("id", ""))[-1]


def mark_viewed(suggestion_id: str) -> dict:
    queue = _load_queue()
    found = False
    for entry in queue.get("entries", []):
        if entry.get("id") == suggestion_id:
            entry["status"] = "viewed"
            found = True
    if not found:
        return {"ok": False, "reason": "id_not_in_queue", "id": suggestion_id}
    _save_queue(queue)
    state = load_state()
    if state.get("active_suggestion_id") == suggestion_id:
        state.pop("active_suggestion_id", None)
    save_state(state)
    return {"ok": True, "id": suggestion_id, "status": "viewed"}


def _section_has_content(text: str, heading: str) -> bool:
    idx = text.find(heading)
    if idx < 0:
        return False
    rest = text[idx + len(heading) :]
    next_h = re.search(r"\n## ", rest)
    body = rest[: next_h.start()] if next_h else rest
    body = body.strip()
    if not body:
        return False
    placeholders = ("awaiting kickoff", "_(awaiting", "_(fill")
    return not any(p in body.lower() for p in placeholders)


def kickoff_delivered_in_text(text: str) -> bool:
    """True when structured Monday kickoff headings have real content."""
    if not all(h in text for h in _REQUIRED_HEADINGS):
        return False
    return all(_section_has_content(text, h) for h in _REQUIRED_HEADINGS)


def _read_transcript_text(transcript_path: Path | None) -> str:
    if not transcript_path or not transcript_path.is_file():
        return ""
    parts: list[str] = []
    try:
        raw = transcript_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    for line in raw.splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            parts.append(line)
            continue
        if row.get("role") != "assistant":
            continue
        message = row.get("message") or {}
        for block in message.get("content") or []:
            if block.get("type") == "text":
                t = block.get("text", "")
                if t:
                    parts.append(t)
    return "\n".join(parts)


def mark_viewed_from_stop(hook_input: dict | None = None) -> dict:
    state = load_state()
    active_id = state.get("active_suggestion_id")
    if not active_id:
        return {"ok": False, "reason": "no_active_suggestion_id"}

    if state.get("kickoff_delivered_for") == active_id:
        state.pop("kickoff_delivered_for", None)
        save_state(state)
        return mark_viewed(active_id)

    transcript_path = None
    if hook_input:
        tp = hook_input.get("transcript_path")
        if tp:
            transcript_path = Path(tp)
    text = _read_transcript_text(transcript_path)
    if kickoff_delivered_in_text(text):
        return mark_viewed(active_id)

    blob = text.lower()
    if not text or (not any(m in blob for m in _KICKOFF_MARKERS) and len(text) < 800):
        return {"ok": False, "reason": "kickoff_not_detected"}
    return {"ok": False, "reason": "headings_missing_or_empty"}


def run_due(*, force: bool = False) -> dict:
    append_launchd_log("[monday-suggestions] --due started")
    tz = get_timezone(load_state())
    today = datetime.now(tz).date()
    suggestion_id = today.isoformat()
    stub_name = f"{suggestion_id}-monday.md"
    stub_path = SUGGESTIONS_DIR / stub_name

    SUGGESTIONS_DIR.mkdir(parents=True, exist_ok=True)
    if stub_path.is_file() and not force:
        enqueued = _enqueue(suggestion_id, stub_path, force=False)
        append_launchd_log(
            f"[monday-suggestions] --due completed (idempotent) id={suggestion_id} enqueued={enqueued}"
        )
        return {
            "id": suggestion_id,
            "path": str(stub_path.relative_to(REPO_ROOT)),
            "enqueued": enqueued,
            "idempotent": True,
        }

    stub_path.write_text(_render_stub(suggestion_id), encoding="utf-8")
    enqueued = _enqueue(suggestion_id, stub_path, force=force)
    append_launchd_log(
        f"[monday-suggestions] --due completed id={suggestion_id} enqueued={enqueued}"
    )
    return {
        "id": suggestion_id,
        "path": str(stub_path.relative_to(REPO_ROOT)),
        "enqueued": enqueued,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--due", action="store_true", help="Launchd-equivalent stub + enqueue")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--mark-viewed", metavar="ID", help="Manual fallback")
    parser.add_argument(
        "--mark-viewed-from-stop",
        action="store_true",
        help="Read hook JSON from stdin; auto-mark after kickoff reply",
    )
    parser.add_argument("--get-active", action="store_true")
    args = parser.parse_args()

    if args.mark_viewed_from_stop:
        hook_input = {}
        if not sys.stdin.isatty():
            try:
                hook_input = json.load(sys.stdin)
            except json.JSONDecodeError:
                hook_input = {}
        print(json.dumps(mark_viewed_from_stop(hook_input)))
        return 0

    if args.mark_viewed:
        print(json.dumps(mark_viewed(args.mark_viewed)))
        return 0

    if args.get_active:
        active = get_active_suggestion()
        print(json.dumps({"active": active}))
        return 0

    if args.due:
        print(json.dumps(run_due(force=args.force)))
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
