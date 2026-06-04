"""Core API scraping helpers shared across data sourcing modules."""

from __future__ import annotations

import time
from typing import Any, Callable

import requests


def create_session(user_agent: str) -> requests.Session:
    """Create a requests session with a consistent user agent."""
    session = requests.Session()
    session.headers.update({"User-Agent": user_agent})
    return session


def request_json_with_backoff(
    session: requests.Session,
    url: str,
    params: dict[str, Any],
    *,
    max_retries: int = 5,
    timeout_seconds: int = 60,
) -> dict[str, Any]:
    """Request JSON with retries for transient/network failures."""
    for attempt in range(max_retries):
        try:
            response = session.get(url, params=params, timeout=timeout_seconds)

            if response.status_code == 429 or response.status_code >= 500:
                if attempt == max_retries - 1:
                    response.raise_for_status()
                wait_seconds = min(2**attempt, 30)
                print(
                    f"API request to {url} returned status={response.status_code}. "
                    f"Retrying in {wait_seconds}s (attempt {attempt + 1}/{max_retries})."
                )
                time.sleep(wait_seconds)
                continue

            response.raise_for_status()
            return response.json()
        except requests.RequestException as error:
            if attempt == max_retries - 1:
                raise
            wait_seconds = min(2**attempt, 30)
            print(
                f"API request to {url} failed with '{error}'. "
                f"Retrying in {wait_seconds}s (attempt {attempt + 1}/{max_retries})."
            )
            time.sleep(wait_seconds)

    raise RuntimeError(f"Failed to fetch payload from {url} after retries.")


def fetch_paginated_results(
    session: requests.Session,
    url: str,
    base_params: dict[str, Any],
    *,
    page_limit: int,
    max_retries: int = 5,
    timeout_seconds: int = 60,
    page_sleep_seconds: float = 0.2,
    results_key: str = "results",
    end_of_records_key: str = "endOfRecords",
    on_first_page: Callable[[dict[str, Any]], None] | None = None,
    on_page: Callable[[int, int, int], None] | None = None,
) -> list[dict[str, Any]]:
    """Fetch all paginated results using limit/offset semantics."""
    all_rows: list[dict[str, Any]] = []
    offset = 0
    first_page = True

    while True:
        params = dict(base_params)
        params["limit"] = page_limit
        params["offset"] = offset

        payload = request_json_with_backoff(
            session=session,
            url=url,
            params=params,
            max_retries=max_retries,
            timeout_seconds=timeout_seconds,
        )

        if first_page and on_first_page is not None:
            on_first_page(payload)
        first_page = False

        results = payload.get(results_key, [])
        if not results:
            break

        all_rows.extend(results)
        if on_page is not None:
            on_page(offset, len(results), len(all_rows))

        if payload.get(end_of_records_key, False):
            break

        offset += page_limit
        time.sleep(page_sleep_seconds)

    return all_rows
