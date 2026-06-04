#!/usr/bin/env python3
"""Mark kickoff delivered when structured headings appear in agent response."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_HOOKS_DIR = Path(__file__).resolve().parent
if str(_HOOKS_DIR) not in sys.path:
    sys.path.insert(0, str(_HOOKS_DIR))

from workflow_common import load_state, save_state
from monday_workflow_suggestions import kickoff_delivered_in_text


def _extract_hook_text(hook_input: dict) -> str:
    parts: list[str] = []
    for key in ("text", "response", "message", "content", "assistant_message"):
        val = hook_input.get(key)
        if isinstance(val, str):
            parts.append(val)
        elif isinstance(val, list):
            for item in val:
                if isinstance(item, dict) and item.get("type") == "text":
                    t = item.get("text")
                    if isinstance(t, str):
                        parts.append(t)
    return "\n".join(parts)


def main() -> int:
    hook_input: dict = {}
    if not sys.stdin.isatty():
        try:
            hook_input = json.load(sys.stdin)
        except json.JSONDecodeError:
            hook_input = {}

    state = load_state()
    active_id = state.get("active_suggestion_id")
    if not active_id:
        print("{}")
        return 0

    text = _extract_hook_text(hook_input)
    if kickoff_delivered_in_text(text):
        state["kickoff_delivered_for"] = active_id
        save_state(state)

    print("{}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
