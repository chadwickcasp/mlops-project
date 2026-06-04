"""Shared helpers for daily/weekly Cursor workflow review hooks."""

from __future__ import annotations

import json
import os
import re
import subprocess
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

REPO_ROOT = Path(__file__).resolve().parents[2]
STATE_PATH = REPO_ROOT / ".cursor" / "agent-workflow-state.json"
REVIEW_DIR = REPO_ROOT / ".cursor" / "workflow-review"
DAILY_DIR = REVIEW_DIR / "daily"
ARCHIVE_DIR = REVIEW_DIR / "archive"
LATEST_PATH = REVIEW_DIR / "latest.md"
INDEX_PATH = REVIEW_DIR / "index.md"
SUGGESTIONS_DIR = REPO_ROOT / ".cursor" / "workflow-suggestions"
QUEUE_PATH = SUGGESTIONS_DIR / "queue.json"
LAUNCHD_LOG = REVIEW_DIR / "launchd.log"
WORKFLOW_PATHS_FILE = REPO_ROOT / ".cursor" / "workflow-paths.json"

DEFAULT_PROJECT_SLUG = "Users-chadcasper-Documents-MLOps-Zoomcamp-mlops-project"

_USER_QUERY_RE = re.compile(r"<user_query>\s*(.*?)\s*</user_query>", re.DOTALL)
_ROLE_USER_RE = re.compile(r'"role"\s*:\s*"user"')
_PATH_IN_TEXT_RE = re.compile(
    r"(?:data_sourcing|image_segmentation|scripts|models|docs|\.cursor)[^\s\"']*"
    r"|[^\s\"']+\.(?:py|md|mdc|sh|json)"
)

AREA_KEYWORDS = (
    ("data_sourcing", ("data_sourcing/", "gbif", "get_gbif")),
    ("image_segmentation", ("image_segmentation/", "segmentation", "sam")),
    ("scripts", ("scripts/",)),
    ("models", ("models/",)),
    ("docs", ("docs/", "feature-plan", "experiment-log")),
    (".cursor", (".cursor/",)),
)

THEME_KEYWORDS = (
    "refactor",
    "pytest",
    "readme",
    "hook",
    "rule",
    "gbif",
    "segmentation",
    "sam",
    "mlops",
    "deploy",
    "experiment",
    "feature-plan",
    "plan",
    "ask mode",
    "agent mode",
    "demo",
    "eval",
)


def normalize_user_query(text: str) -> str:
    """Turn transcript escape sequences into real newlines for readable markdown."""
    text = text.replace("\\n", "\n").replace("\\t", " ")
    text = text.replace('\\"', '"').replace("\\'", "'")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [ln.strip() for ln in text.splitlines()]
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines)


def format_query_markdown_lines(queries: list[str], *, max_queries: int = 5) -> list[str]:
    """Format user prompts as markdown bullets without literal \\n artifacts."""
    out: list[str] = []
    for raw in queries[:max_queries]:
        q = normalize_user_query(raw)
        if not q:
            continue
        parts = [ln.strip() for ln in q.splitlines() if ln.strip()]
        if not parts:
            continue
        if len(parts) == 1:
            out.append(f"- {parts[0]}")
            continue
        out.append(f"- {parts[0]}")
        for part in parts[1:]:
            cleaned = part.lstrip("- ").strip()
            if cleaned:
                out.append(f"  - {cleaned}")
    return out


@dataclass
class TranscriptSession:
    path: Path
    conversation_id: str
    mtime: datetime
    user_queries: list[str] = field(default_factory=list)
    tool_names: list[str] = field(default_factory=list)
    paths_mentioned: list[str] = field(default_factory=list)


@dataclass
class DayDigestData:
    target_date: date
    sessions: list[TranscriptSession]
    git_log_lines: list[str]
    git_diff_stat: list[str]
    themes: list[str] = field(default_factory=list)
    areas: list[str] = field(default_factory=list)


def workflow_paths_file() -> Path:
    override = os.environ.get("WORKFLOW_PATHS_FILE", "").strip()
    if override:
        return Path(override).expanduser()
    return WORKFLOW_PATHS_FILE


def resolve_workflow_path(raw: str) -> Path:
    expanded = Path(raw).expanduser()
    if expanded.is_absolute():
        return expanded.resolve()
    return (REPO_ROOT / expanded).resolve()


def load_workflow_directories() -> list[dict]:
    path = workflow_paths_file()
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    directories = data.get("directories")
    if not isinstance(directories, list):
        return []
    return [d for d in directories if isinstance(d, dict)]


def scan_directory_entry(entry: dict) -> list[Path]:
    raw_path = entry.get("path", "")
    if not isinstance(raw_path, str) or not raw_path.strip():
        return []
    root = resolve_workflow_path(raw_path.strip())
    if not root.is_dir():
        return []

    mode = entry.get("mode", "top_level")
    if mode == "newest_dirs":
        limit = entry.get("limit", 3)
        if not isinstance(limit, int) or limit < 1:
            limit = 3
        dirs = [p for p in root.iterdir() if p.is_dir()]
        dirs.sort(key=lambda p: p.name, reverse=True)
        return dirs[:limit]

    if mode == "named_files":
        files = entry.get("files", [])
        if not isinstance(files, list):
            return []
        found: list[Path] = []
        for name in files:
            if isinstance(name, str) and name.strip():
                p = root / name.strip()
                if p.is_file():
                    found.append(p)
        return found

    children = sorted(root.iterdir(), key=lambda p: p.name)
    return children[:20]


def workflow_user_agent(version: str = "1.0") -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", REPO_ROOT.name.lower()).strip("-") or "project"
    return f"{slug}-workflow-review/{version}"


def get_timezone(state: dict | None = None) -> ZoneInfo:
    name = (state or {}).get("timezone")
    if name:
        try:
            return ZoneInfo(name)
        except Exception:
            pass
    local = datetime.now().astimezone().tzinfo
    if isinstance(local, ZoneInfo):
        return local
    return ZoneInfo("America/Los_Angeles")


def load_state() -> dict:
    if not STATE_PATH.is_file():
        return {}
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")


def append_launchd_log(message: str) -> None:
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().astimezone().isoformat(timespec="seconds")
    with LAUNCHD_LOG.open("a", encoding="utf-8") as fh:
        fh.write(f"[{ts}] {message}\n")


def transcript_root() -> Path | None:
    slug = os.environ.get("CURSOR_PROJECT_SLUG", DEFAULT_PROJECT_SLUG)
    root = Path.home() / ".cursor" / "projects" / slug / "agent-transcripts"
    return root if root.is_dir() else None


def day_bounds(target: date, tz: ZoneInfo) -> tuple[datetime, datetime]:
    start = datetime.combine(target, time.min, tzinfo=tz)
    end = start + timedelta(days=1)
    return start, end


def iso_week_start(for_day: date | None = None, tz: ZoneInfo | None = None) -> date:
    tz = tz or get_timezone()
    d = for_day or datetime.now(tz).date()
    return d - timedelta(days=d.weekday())


def prune_daily_digests(repo_root: Path | None = None) -> list[str]:
    """Delete daily/*.md before current ISO week. Returns removed filenames."""
    root = repo_root or REPO_ROOT
    daily_dir = root / ".cursor" / "workflow-review" / "daily"
    if not daily_dir.is_dir():
        return []
    week_start = iso_week_start(tz=get_timezone(load_state()))
    removed: list[str] = []
    for path in daily_dir.glob("*.md"):
        try:
            file_date = date.fromisoformat(path.stem)
        except ValueError:
            continue
        if file_date < week_start:
            path.unlink(missing_ok=True)
            removed.append(path.name)
    return removed


def list_transcript_paths(*, exclude_subagents: bool = True) -> list[Path]:
    root = transcript_root()
    if not root:
        return []
    paths: list[Path] = []
    for path in root.rglob("*.jsonl"):
        if exclude_subagents and "/subagents/" in path.as_posix():
            continue
        paths.append(path)
    return paths


def sessions_for_day(target: date, tz: ZoneInfo | None = None) -> list[TranscriptSession]:
    tz = tz or get_timezone()
    start, end = day_bounds(target, tz)
    sessions: list[TranscriptSession] = []
    for path in list_transcript_paths():
        mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=tz)
        if not (start <= mtime < end):
            continue
        conv_id = path.parent.name if path.parent != path.parent.parent else path.stem
        session = _parse_transcript_file(path, conv_id, mtime)
        if session.user_queries or session.tool_names:
            sessions.append(session)
    sessions.sort(key=lambda s: s.mtime)
    return sessions


def _parse_transcript_file(path: Path, conv_id: str, mtime: datetime) -> TranscriptSession:
    session = TranscriptSession(path=path, conversation_id=conv_id, mtime=mtime)
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return session
    if not _ROLE_USER_RE.search(text):
        return session
    for match in _USER_QUERY_RE.finditer(text):
        query = normalize_user_query(match.group(1))
        flat_len = len(query.replace("\n", " "))
        if flat_len > 20:
            session.user_queries.append(query[:500])
    for line in text.splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("role") != "assistant":
            continue
        message = row.get("message") or {}
        for block in message.get("content") or []:
            if block.get("type") == "tool_use":
                name = block.get("name")
                if name:
                    session.tool_names.append(name)
                inp = block.get("input") or {}
                for key in ("path", "target_directory", "command"):
                    val = inp.get(key)
                    if isinstance(val, str) and val.strip():
                        session.paths_mentioned.append(val.strip())
    blob = text.lower()
    for area, needles in AREA_KEYWORDS:
        if any(n in blob for n in needles):
            session.paths_mentioned.append(area)
    for match in _PATH_IN_TEXT_RE.finditer(text):
        session.paths_mentioned.append(match.group(0)[:120])
    return session


def git_activity_for_day(target: date, tz: ZoneInfo | None = None) -> tuple[list[str], list[str]]:
    tz = tz or get_timezone()
    start, end = day_bounds(target, tz)
    since = start.isoformat()
    until = end.isoformat()
    log_lines: list[str] = []
    diff_stat: list[str] = []
    try:
        log = subprocess.run(
            [
                "git",
                "-C",
                str(REPO_ROOT),
                "log",
                f"--since={since}",
                f"--until={until}",
                "--oneline",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if log.returncode == 0 and log.stdout.strip():
            log_lines = log.stdout.strip().splitlines()[:30]
        stat = subprocess.run(
            [
                "git",
                "-C",
                str(REPO_ROOT),
                "log",
                f"--since={since}",
                f"--until={until}",
                "--stat",
                "--oneline",
                "-5",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if stat.returncode == 0 and stat.stdout.strip():
            diff_stat = stat.stdout.strip().splitlines()[:40]
    except (subprocess.TimeoutExpired, OSError):
        pass
    return log_lines, diff_stat


def had_repo_edits_since(since_date: date, tz: ZoneInfo | None = None) -> bool:
    tz = tz or get_timezone()
    start, _ = day_bounds(since_date, tz)
    try:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(REPO_ROOT),
                "log",
                f"--since={start.isoformat()}",
                "-1",
                "--oneline",
            ],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        return bool(result.returncode == 0 and result.stdout.strip())
    except (subprocess.TimeoutExpired, OSError):
        return False


def collect_day_data(target: date, tz: ZoneInfo | None = None) -> DayDigestData:
    tz = tz or get_timezone()
    sessions = sessions_for_day(target, tz)
    git_log, git_diff = git_activity_for_day(target, tz)
    data = DayDigestData(
        target_date=target,
        sessions=sessions,
        git_log_lines=git_log,
        git_diff_stat=git_diff,
    )
    blob = " ".join(" ".join(s.user_queries) for s in sessions).lower()
    for kw in THEME_KEYWORDS:
        if kw in blob and kw not in data.themes:
            data.themes.append(kw)
    area_hits: Counter[str] = Counter()
    for session in sessions:
        for p in session.paths_mentioned:
            for area, needles in AREA_KEYWORDS:
                if area in p or any(n in p for n in needles):
                    area_hits[area] += 1
    data.areas = [a for a, _ in area_hits.most_common(8)]
    return data


def snapshot_latest_to_archive(archive_date: date | None = None) -> Path | None:
    if not LATEST_PATH.is_file():
        return None
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    d = archive_date or datetime.now(get_timezone()).date()
    dest = ARCHIVE_DIR / f"{d.isoformat()}.md"
    if dest.is_file():
        return dest
    dest.write_text(LATEST_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    return dest


def write_index(state: dict | None = None) -> None:
    state = state or load_state()
    tz = get_timezone(state)
    today = datetime.now(tz).date()
    week_start = iso_week_start(today, tz)
    lines = [
        "# Workflow review index",
        "",
        f"_Updated {today.isoformat()}_",
        "",
        "## Current week dailies",
        "",
    ]
    if DAILY_DIR.is_dir():
        for path in sorted(DAILY_DIR.glob("*.md"), reverse=True):
            try:
                d = date.fromisoformat(path.stem)
            except ValueError:
                continue
            if d >= week_start:
                lines.append(f"- [{path.stem}](daily/{path.name})")
    if lines[-1] == "":
        lines.append("- _(none)_")
    pending = state.get("pending_approval")
    lines.extend(
        [
            "",
            "## Weekly draft",
            "",
            f"- `latest.md`: {'pending approval' if pending else 'see file if present'}",
            "",
            "## Archive (prior weeks)",
            "",
        ]
    )
    if ARCHIVE_DIR.is_dir():
        for path in sorted(ARCHIVE_DIR.glob("*.md"), reverse=True):
            lines.append(f"- [{path.stem}](archive/{path.name})")
    else:
        lines.append("- _(none)_")
    lines.append("")
    INDEX_PATH.write_text("\n".join(lines), encoding="utf-8")


def digest_summary_lines(digest_path: Path, max_lines: int = 5) -> list[str]:
    if not digest_path.is_file():
        return []
    lines: list[str] = []
    for raw in digest_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith(">"):
            continue
        if line.startswith("- ") and not line.startswith("- _"):
            lines.append(line[2:][:200])
        if len(lines) >= max_lines:
            break
    return lines
