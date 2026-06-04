#!/usr/bin/env bash
# Cursor sessionStart: catch-up daily digest, digest nudge, Monday queue, pending weekly.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT/.cursor/hooks"
exec python3 session_start_context.py
