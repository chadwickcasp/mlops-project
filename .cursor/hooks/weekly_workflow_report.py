#!/usr/bin/env python3
"""Weekly workflow report: roll up daily digests into latest.md (approval-gated)."""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.request
from datetime import date, datetime, timedelta
from pathlib import Path

_HOOKS_DIR = Path(__file__).resolve().parent
if str(_HOOKS_DIR) not in sys.path:
    sys.path.insert(0, str(_HOOKS_DIR))

from workflow_common import (
    ARCHIVE_DIR,
    DAILY_DIR,
    LATEST_PATH,
    REPO_ROOT,
    REVIEW_DIR,
    get_timezone,
    had_repo_edits_since,
    iso_week_start,
    load_state,
    prune_daily_digests,
    save_state,
    snapshot_latest_to_archive,
    workflow_user_agent,
    write_index,
)

SOURCES_PATH = REVIEW_DIR / "sources.json"
_HINT_RE = re.compile(r"^-\s+(.+)$", re.MULTILINE)


def _daily_paths_since(since: date | None) -> list[Path]:
    if not DAILY_DIR.is_dir():
        return []
    paths: list[Path] = []
    for path in DAILY_DIR.glob("*.md"):
        try:
            d = date.fromisoformat(path.stem)
        except ValueError:
            continue
        if since is None or d > since:
            paths.append(path)
    return sorted(paths, key=lambda p: p.stem)


def _extract_hints(text: str) -> list[str]:
    in_section = False
    hints: list[str] = []
    for line in text.splitlines():
        if line.strip() == "## Candidate rule hints":
            in_section = True
            continue
        if in_section and line.startswith("## "):
            break
        if in_section:
            match = _HINT_RE.match(line.strip())
            if match and not match.group(1).startswith("_("):
                hints.append(match.group(1))
    return hints


def _fetch_sources_appendix(max_bytes: int = 8000) -> str:
    if not SOURCES_PATH.is_file():
        return "_No sources.json configured._"
    try:
        entries = json.loads(SOURCES_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return "_Could not read sources.json._"
    parts: list[str] = []
    for entry in entries:
        url = entry.get("url", "")
        title = entry.get("title", url)
        if not url:
            continue
        snippet = ""
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": workflow_user_agent()},
            )
            with urllib.request.urlopen(req, timeout=12) as resp:
                raw = resp.read(max_bytes)
                snippet = raw.decode("utf-8", errors="replace")
                snippet = re.sub(r"<[^>]+>", " ", snippet)
                snippet = " ".join(snippet.split())[:1200]
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            snippet = f"(fetch failed: {exc})"
        parts.append(f"### {title}\n\nSource: {url}\n\n{snippet}\n")
    return "\n".join(parts) if parts else "_No sources fetched._"


def _render_weekly(daily_paths: list[Path], appendix: str) -> str:
    today = datetime.now(get_timezone(load_state())).date().isoformat()
    themes: list[str] = []
    hints: list[str] = []
    summaries: list[str] = []
    for path in daily_paths:
        text = path.read_text(encoding="utf-8")
        summaries.append(f"### {path.stem}\n\n{_first_summary_bullets(text, 4)}\n")
        for line in text.splitlines():
            if line.startswith("- ") and "session" in line.lower():
                continue
            for kw in (
                "refactor",
                "pytest",
                "hook",
                "rule",
                "gbif",
                "segmentation",
                "mlops",
                "experiment",
            ):
                if kw in line.lower() and kw not in themes:
                    themes.append(kw)
        hints.extend(_extract_hints(text))

    dedup_hints: list[str] = []
    for h in hints:
        if h not in dedup_hints:
            dedup_hints.append(h)

    theme_lines = "\n".join(f"- {t}" for t in themes[:12]) or "- _(rolled up from daily digests)_"
    hint_lines = "\n".join(f"- {h}" for h in dedup_hints[:10]) or "- _(none)_"

    return f"""# Weekly workflow review — {today}

> Draft for approval. Do not apply until the user says **yes** in chat.

## Themes (7-day rollup)

{theme_lines}

## Daily digests included

{"".join(summaries)}

## Aggregated candidate rule hints

{hint_lines}

## External references appendix

{appendix}

## Cursor / tooling recommendations

- Keep repo-wide guidance in `AGENTS.md` and `.cursor/rules/00-core-workflow.mdc` high-level.
- Use scoped rules (`10-ml-experiments.mdc`, etc.) for ML and serving paths.
- Use Plan mode for multi-file refactors; sub-agents for verification after non-trivial edits.

## Proposed diffs

_To be filled by the agent after reading this file and daily digests. Target `AGENTS.md` and relevant `.cursor/rules/*.mdc` files._

## Out of scope

- Monday workflow suggestions in `.cursor/workflow-suggestions/` — separate viewed queue; not auto-applied here.
"""


def _first_summary_bullets(text: str, n: int) -> str:
    bullets: list[str] = []
    for line in text.splitlines():
        if line.startswith("- **") or (line.startswith("- ") and "session" in line.lower()):
            bullets.append(line)
        if len(bullets) >= n:
            break
    return "\n".join(bullets) if bullets else "- _(see full daily file)_"


def should_run_weekly(state: dict | None = None) -> bool:
    state = state or load_state()
    tz = get_timezone(state)
    last = state.get("last_weekly_report_date")
    since = date.fromisoformat(last) if last else None
    dailies = _daily_paths_since(since)
    if len(dailies) < 7:
        return False
    week_start = iso_week_start(tz=tz)
    if not had_repo_edits_since(week_start - timedelta(days=1), tz):
        return False
    return True


def write_weekly_stub(*, force: bool = False) -> dict:
    state = load_state()
    if not force and not should_run_weekly(state):
        return {"skipped": True, "reason": "not_due"}

    last = state.get("last_weekly_report_date")
    since = date.fromisoformat(last) if last else None
    daily_paths = _daily_paths_since(since)
    if len(daily_paths) < 7 and not force:
        return {"skipped": True, "reason": "insufficient_dailies", "count": len(daily_paths)}

    if force and len(daily_paths) < 7:
        daily_paths = sorted(DAILY_DIR.glob("*.md"), key=lambda p: p.stem)[-7:]

    appendix = _fetch_sources_appendix()
    body = _render_weekly(daily_paths, appendix)
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    LATEST_PATH.write_text(body, encoding="utf-8")

    today = datetime.now(get_timezone(state)).date()
    state["pending_approval"] = True
    state["pending_approval_since"] = today.isoformat()
    state["last_workflow_review_prompt_at"] = today.isoformat()
    state["had_repo_edits_this_week"] = True
    save_state(state)
    write_index(state)

    return {
        "path": str(LATEST_PATH.relative_to(REPO_ROOT)),
        "daily_count": len(daily_paths),
        "pending_approval": True,
    }


def finalize_weekly_approval() -> dict:
    """Call after user approves — archive latest and prune dailies."""
    state = load_state()
    today = datetime.now(get_timezone(state)).date()
    archived = snapshot_latest_to_archive(today)
    state["pending_approval"] = False
    state.pop("pending_approval_since", None)
    state["last_weekly_report_date"] = today.isoformat()
    state["had_repo_edits_this_week"] = False
    removed = prune_daily_digests()
    save_state(state)
    write_index(state)
    return {
        "archived": str(archived.relative_to(REPO_ROOT)) if archived else None,
        "pruned_daily": removed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-stub", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--finalize-approval", action="store_true")
    args = parser.parse_args()

    try:
        if args.finalize_approval:
            print(json.dumps(finalize_weekly_approval()))
        elif args.check:
            print(json.dumps({"should_run": should_run_weekly()}))
        elif args.write_stub or args.force:
            print(json.dumps(write_weekly_stub(force=args.force)))
        else:
            parser.print_help()
    except Exception as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        print(json.dumps({}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
