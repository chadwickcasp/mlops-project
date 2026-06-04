#!/usr/bin/env python3
"""Build daily workflow digests from Cursor transcripts and git activity.

Run: python .cursor/hooks/daily_workflow_digest.py --backfill-days 2 --force
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

_HOOKS_DIR = Path(__file__).resolve().parent
if str(_HOOKS_DIR) not in sys.path:
    sys.path.insert(0, str(_HOOKS_DIR))

from workflow_common import (
    DAILY_DIR,
    REPO_ROOT,
    DayDigestData,
    collect_day_data,
    format_query_markdown_lines,
    get_timezone,
    load_state,
    normalize_user_query,
    prune_daily_digests,
    save_state,
    write_index,
)


def _render_digest(data: DayDigestData) -> str:
    d = data.target_date.isoformat()
    session_count = len(data.sessions)
    tool_counts: dict[str, int] = {}
    sample_queries: list[str] = []
    path_hits: list[str] = []
    for session in data.sessions:
        for tool in session.tool_names:
            tool_counts[tool] = tool_counts.get(tool, 0) + 1
        for q in session.user_queries[:2]:
            cleaned = normalize_user_query(q)
            if cleaned and cleaned not in sample_queries:
                sample_queries.append(cleaned)
        for p in session.paths_mentioned:
            if len(p) > 120 or p.startswith("cd ") or "\n" in p:
                continue
            if "/terminals/" in p or p.endswith(".txt") and "terminals" in p:
                continue
            if p.startswith("ssh ") or p.startswith("/Users/") and ".cursor/projects" in p:
                continue
            if p not in path_hits:
                path_hits.append(p)

    lines = [
        f"# Daily workflow digest — {d}",
        "",
        "> Auto-generated from agent transcripts and git. Not applied to rules without weekly approval.",
        "",
        "## Summary",
        "",
        f"- **Agent sessions:** {session_count}",
        f"- **Git commits (day window):** {len(data.git_log_lines)}",
        "",
    ]

    if data.themes:
        lines.extend(["## Dominant topics", ""] + [f"- {t}" for t in data.themes] + [""])

    lines.append("## Files and areas touched")
    lines.append("")
    if path_hits or data.areas:
        for area in data.areas:
            lines.append(f"- {area}")
        for p in path_hits[:15]:
            if p not in data.areas:
                lines.append(f"- `{p}`")
    else:
        lines.append("- _(none detected)_")
    lines.append("")

    lines.append("## Prompting patterns")
    lines.append("")
    prompt_lines = format_query_markdown_lines(sample_queries, max_queries=5)
    if prompt_lines:
        lines.extend(prompt_lines)
    else:
        lines.append("- _(no user queries parsed)_")
    if tool_counts:
        tools = ", ".join(
            f"{k} ({v})" for k, v in sorted(tool_counts.items(), key=lambda x: -x[1])[:8]
        )
        lines.append(f"- **Tools used:** {tools}")
    lines.append("")

    lines.append("## Friction signals")
    lines.append("")
    friction: list[str] = []
    blob = " ".join(sample_queries).lower()
    for signal in ("fix ci", "retry", "error", "failed", "plan mode", "ask mode"):
        if signal in blob:
            friction.append(signal)
    if friction:
        lines.extend(f"- {s}" for s in friction)
    else:
        lines.append("- _(none flagged)_")
    lines.append("")

    lines.append("## Candidate rule hints")
    lines.append("")
    hints = _candidate_hints(data, tool_counts)
    if hints:
        lines.extend(f"- {h}" for h in hints)
    else:
        lines.append("- _(none this day)_")
    lines.append("")

    lines.append("## Git activity")
    lines.append("")
    if data.git_log_lines:
        lines.extend(f"- {line}" for line in data.git_log_lines[:15])
    else:
        lines.append("- _(no commits in day window)_")
    if data.git_diff_stat:
        lines.append("")
        lines.append("```")
        lines.extend(data.git_diff_stat[:25])
        lines.append("```")
    lines.append("")

    return "\n".join(lines)


def _candidate_hints(data: DayDigestData, tool_counts: dict[str, int]) -> list[str]:
    hints: list[str] = []
    if "image_segmentation" in data.areas:
        hints.append("Segmentation work — keep demo path documented in image_segmentation/README.md.")
    if "data_sourcing" in data.areas:
        hints.append("GBIF/data sourcing touched — note cache and API constraints in docs.")
    if tool_counts.get("Task", 0) >= 3:
        hints.append("Heavy subagent use — document when to spawn verification sub-agents.")
    if "pytest" in data.themes or "eval" in data.themes:
        hints.append("Testing/eval mentioned — keep verification steps in AGENTS.md definition of done.")
    if "hook" in data.themes or "rule" in data.themes:
        hints.append("Cursor rules/hooks discussed — prefer scoped .mdc over growing always-on context.")
    if len(data.sessions) >= 4:
        hints.append("High session volume — consider Plan mode for multi-file work.")
    return hints[:6]


def write_digest(
    target: date,
    *,
    force: bool = False,
    mark_ready: bool = False,
    update_state: bool = True,
) -> dict:
    DAILY_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DAILY_DIR / f"{target.isoformat()}.md"
    if out_path.is_file() and not force:
        return {
            "skipped": True,
            "path": str(out_path.relative_to(REPO_ROOT)),
        }

    tz = get_timezone(load_state())
    data = collect_day_data(target, tz)
    body = _render_digest(data)
    out_path.write_text(body, encoding="utf-8")

    removed = prune_daily_digests()
    state = load_state()
    if update_state:
        last = state.get("last_daily_digest_date", "")
        if not last or target.isoformat() > last:
            state["last_daily_digest_date"] = target.isoformat()
        if mark_ready:
            state["digest_ready_date"] = target.isoformat()
            state.pop("digest_notified_date", None)
    state["had_repo_edits_this_week"] = (
        state.get("had_repo_edits_this_week", False)
        or bool(data.git_log_lines)
        or bool(data.sessions)
    )
    save_state(state)
    write_index(state)

    return {
        "path": str(out_path.relative_to(REPO_ROOT)),
        "session_count": len(data.sessions),
        "pruned": removed,
    }


def run_catch_up(*, mark_ready: bool = True) -> list[dict]:
    """Write yesterday's digest if missing (sessionStart / launchd)."""
    state = load_state()
    tz = get_timezone(state)
    today = datetime.now(tz).date()
    yesterday = today - timedelta(days=1)
    last = state.get("last_daily_digest_date")
    results: list[dict] = []
    if last and date.fromisoformat(last) >= yesterday:
        prune_daily_digests()
        write_index(state)
        return results
    results.append(
        write_digest(
            yesterday,
            force=False,
            mark_ready=mark_ready,
            update_state=True,
        )
    )
    prune_daily_digests()
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", type=str, help="YYYY-MM-DD (default: yesterday)")
    parser.add_argument("--backfill-days", type=int, metavar="N")
    parser.add_argument("--catch-up", action="store_true", help="Write yesterday if behind")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--mark-ready", action="store_true")
    parser.add_argument("--no-state", action="store_true", help="Do not update state file")
    args = parser.parse_args()

    if args.catch_up:
        print(json.dumps({"results": run_catch_up(mark_ready=True)}))
        return 0

    tz = get_timezone(load_state())
    today = datetime.now(tz).date()

    dates: list[date] = []
    if args.backfill_days:
        for offset in range(args.backfill_days):
            dates.append(today - timedelta(days=offset))
    elif args.date:
        dates.append(date.fromisoformat(args.date))
    else:
        dates.append(today - timedelta(days=1))

    results = []
    for target in sorted(dates):
        mark = args.mark_ready and target == dates[0] and len(dates) == 1
        results.append(
            write_digest(
                target,
                force=args.force,
                mark_ready=mark,
                update_state=not args.no_state,
            )
        )
    print(json.dumps({"results": results}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
