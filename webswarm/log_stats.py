"""WebSwarm runtime-log statistics utilities.

After each sample, the runner calls this module to count tool usage and aggregate it into
run_info. Statistics cover the WebSwarm log tree: root, verb agents, guidance events, and
the internal entity_collect sample/merge subprocesses.
"""

from __future__ import annotations

import json
from typing import Any


TOOL_COUNT_METRICS = ("total_calls", "actual_calls", "task_dedup_calls")
TOOL_COUNT_SCOPES = ("task", "guidance", "total")
TOOL_COUNT_KEYS = ("serper", "jina")


def empty_tool_counts() -> dict[str, dict[str, dict[str, int]]]:
    """Create a run-level tool_counts accumulator."""
    return {
        metric: {
            scope: {key: 0 for key in TOOL_COUNT_KEYS}
            for scope in TOOL_COUNT_SCOPES
        }
        for metric in TOOL_COUNT_METRICS
    }


def add_tool_counts(
    target: dict[str, Any],
    source: dict[str, Any],
) -> None:
    """Add one task's tool_counts in place to the run-level aggregate."""
    for metric in TOOL_COUNT_METRICS:
        for scope in TOOL_COUNT_SCOPES:
            for key in TOOL_COUNT_KEYS:
                target[metric][scope][key] += source[metric][scope][key]


def _get_total_api_counts(tool_states: dict[str, Any]) -> dict[str, int]:
    """Metric (1): total calls initiated by agents, including cache hits."""
    s = tool_states.get("search", {})
    f = tool_states.get("fetch_url", {})
    return {
        "serper": s.get("search_api_cnt", 0),
        "jina": f.get("jina_api_cnt", 0),
    }


def _get_actual_api_counts(tool_states: dict[str, Any]) -> dict[str, int]:
    """Metric (2): actual HTTP requests, excluding cache hits."""
    s = tool_states.get("search", {})
    f = tool_states.get("fetch_url", {})
    return {
        "serper": s.get("search_api_cnt_ignore_cache", 0),
        "jina": f.get("jina_api_cnt_ignore_cache", 0),
    }


def _collect_all_tool_states(node: dict[str, Any], results: list[dict]) -> None:
    """Recursively collect tool_states from all agent nodes in the log tree, excluding guidance."""
    if not isinstance(node, dict):
        return
    # tool_states owned by the current node
    if node.get("tool_states"):
        entry = {
            "agent_kind": node.get("agent_kind", "root"),
            "query": node.get("query") or node.get("task", ""),
            "tool_states": node["tool_states"],
        }
        results.append(entry)

    # Recurse into child_results.
    for child in node.get("child_results") or []:
        _collect_all_tool_states(child, results)

    # Recurse into subtask_results.
    for sub in node.get("subtask_results") or []:
        _collect_all_tool_states(sub, results)


def _collect_ec_sample_calls(
    node: dict[str, Any],
    total_counts: dict[str, int],
    actual_counts: dict[str, int],
    search_keys: set[tuple[str, str]],
    fetch_urls: set[str],
) -> None:
    """Recursively collect tool calls from entity_collect sampling logs.

    The current result tree exposes sample agents as child_results. This function also reads
    internal sampling calls from entity_collect call_logs so tool use in the merge flow is not missed.
    """
    if not isinstance(node, dict):
        return

    ts = node.get("tool_states") or {}
    ec_ts = ts.get("entity_collect", {})
    for call_log in ec_ts.get("call_logs") or []:
        for sample in call_log.get("sample_logs") or []:
            for msg in sample.get("messages") or []:
                for tc in msg.get("tool_calls") or []:
                    fn = tc.get("function", {})
                    name = fn.get("name", "")
                    args_raw = fn.get("arguments", "")
                    if isinstance(args_raw, str):
                        try:
                            args = json.loads(args_raw)
                        except (json.JSONDecodeError, TypeError):
                            continue
                    elif isinstance(args_raw, dict):
                        args = args_raw
                    else:
                        continue

                    if name == "search":
                        total_counts["serper"] += 1
                        actual_counts["serper"] += 1
                        date_range = args.get("date_range") or ""
                        queries = args.get("queries") or []
                        if isinstance(queries, str):
                            queries = [queries]
                        if not queries:
                            q = args.get("query", "")
                            if q:
                                queries = [q]
                        for q in queries:
                            q = q.strip()
                            if q:
                                search_keys.add((q, date_range))

                    elif name == "fetch_url":
                        total_counts["jina"] += 1
                        actual_counts["jina"] += 1
                        urls = args.get("urls") or []
                        if isinstance(urls, str):
                            urls = [urls]
                        if not urls:
                            url = args.get("url", "")
                            if url:
                                urls = [url]
                        for u in urls:
                            u = u.strip()
                            if u:
                                fetch_urls.add(u)

    for child in node.get("child_results") or []:
        _collect_ec_sample_calls(child, total_counts, actual_counts, search_keys, fetch_urls)
    for sub in node.get("subtask_results") or []:
        _collect_ec_sample_calls(sub, total_counts, actual_counts, search_keys, fetch_urls)


def _collect_unique_calls_from_messages(
    node: dict[str, Any],
    search_keys: set[tuple[str, str]],
    fetch_urls: set[str],
) -> None:
    """Recursively extract deduplication keys from tool_calls in messages for metric 3.

    search_keys: Set of (query, date_range) pairs
    fetch_urls:  Set of URL strings
    """
    if not isinstance(node, dict):
        return
    for msg in node.get("messages") or []:
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function", {})
            name = fn.get("name", "")
            args_raw = fn.get("arguments", "")
            if isinstance(args_raw, str):
                try:
                    args = json.loads(args_raw)
                except (json.JSONDecodeError, TypeError):
                    continue
            elif isinstance(args_raw, dict):
                args = args_raw
            else:
                continue

            if name == "search":
                date_range = args.get("date_range") or ""
                # Log analysis handles both batch and single arguments, normalizing them to unique queries.
                queries = args.get("queries") or []
                if isinstance(queries, str):
                    queries = [queries]
                if not queries:
                    q = args.get("query", "")
                    if q:
                        queries = [q]
                for q in queries:
                    q = q.strip()
                    if q:
                        search_keys.add((q, date_range))

            elif name == "fetch_url":
                urls = args.get("urls") or []
                if isinstance(urls, str):
                    urls = [urls]
                if not urls:
                    url = args.get("url", "")
                    if url:
                        urls = [url]
                for u in urls:
                    u = u.strip()
                    if u:
                        fetch_urls.add(u)

    for child in node.get("child_results") or []:
        _collect_unique_calls_from_messages(child, search_keys, fetch_urls)
    for sub in node.get("subtask_results") or []:
        _collect_unique_calls_from_messages(sub, search_keys, fetch_urls)


def get_agent_tree(data: dict[str, Any]) -> dict:
    """Return the nested structure tree of WebSwarm multi-agent delegation.

    Args:
        data: Complete task-log dict.

    Returns:
        Nested dict in which each node contains:
          - "type": Agent type, such as "root", "deep", "atom", "search", or "verify"
          - "steps": Number of steps executed by the agent
          - "children": Child-agent list
    """

    def _build_node(node: dict[str, Any]) -> dict:
        if not isinstance(node, dict):
            return {"type": "unknown", "steps": 0, "children": []}
        # Determine the current node type.
        if "verb" in node:
            node_type = node["verb"]
        elif "agent_kind" in node:
            node_type = node["agent_kind"]
        else:
            node_type = "root"

        result = {
            "type": node_type,
            "steps": node.get("steps", 0),
            "children": [],
        }

        for child in node.get("child_results") or []:
            result["children"].append(_build_node(child))

        for sub in node.get("subtask_results") or []:
            result["children"].append(_build_node(sub))

        return result

    return _build_node(data)


def analysis_token_usage(data: dict[str, Any]) -> dict[str, dict[str, int]]:
    """Count token usage in a WebSwarm log, separating task and guidance usage.

    Returns:
        {
            "task": {"prompt_tokens": N, "completion_tokens": N, "total_tokens": N},
            "guidance": {"prompt_tokens": N, "completion_tokens": N, "total_tokens": N},
            "total": {"prompt_tokens": N, "completion_tokens": N, "total_tokens": N},
        }
    """
    task_totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    guidance_totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def _collect(node: dict[str, Any]) -> None:
        if not isinstance(node, dict):
            return
        for step in node.get("trajectory") or []:
            action = step.get("action")
            if isinstance(action, dict):
                usage = action.get("token_usage", {})
                for key in task_totals:
                    task_totals[key] += usage.get(key, 0)

        for child in node.get("child_results") or []:
            _collect(child)
        for sub in node.get("subtask_results") or []:
            _collect(sub)

    _collect(data)

    # Token usage of web_probing_agent in guidance
    for event in (data.get("guidance") or {}).get("events") or []:
        web_probing_agent = event.get("web_probing_agent") or {}
        for step in web_probing_agent.get("trajectory") or []:
            action = step.get("action")
            if isinstance(action, dict):
                usage = action.get("token_usage", {})
                for key in guidance_totals:
                    guidance_totals[key] += usage.get(key, 0)

    combined = {k: task_totals[k] + guidance_totals[k] for k in task_totals}
    return {"task": task_totals, "guidance": guidance_totals, "total": combined}


def read_json(filepath: str) -> dict[str, Any]:
    """Read one JSON log file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def analysis_tool_usage(data: dict[str, Any]) -> dict[str, Any]:
    """Analyze one task log and return Serper and Jina call statistics.

    Three metric groups:
        (1) total_calls:      All calls, including cache hits, from the agent perspective
        (2) actual_calls:     Actual HTTP requests, excluding cache hits, from the billing perspective
        (3) task_dedup_calls: Distinct requests after task-wide deduplication, assuming an empty cache

    Each metric group is split into task / guidance / total dimensions.
    """
    def _sum(a: dict, b: dict) -> dict:
        return {k: a[k] + b[k] for k in a}

    # ── Collect all tool_states (metrics 1 and 2) ──
    all_states: list[dict] = []
    _collect_all_tool_states(data, all_states)

    task_total = {"serper": 0, "jina": 0}
    task_actual = {"serper": 0, "jina": 0}
    for entry in all_states:
        ts = entry["tool_states"]
        for k, v in _get_total_api_counts(ts).items():
            task_total[k] += v
        for k, v in _get_actual_api_counts(ts).items():
            task_actual[k] += v

    # ── Calls in entity_collect sample_logs (metrics 1, 2, and 3) ──
    ec_search_keys: set[tuple[str, str]] = set()
    ec_fetch_urls: set[str] = set()
    _collect_ec_sample_calls(data, task_total, task_actual, ec_search_keys, ec_fetch_urls)

    # ── Metrics 1 and 2 for the guidance web_probing_agent ──
    guidance_total = {"serper": 0, "jina": 0}
    guidance_actual = {"serper": 0, "jina": 0}
    for event in (data.get("guidance") or {}).get("events") or []:
        web_probing_agent = event.get("web_probing_agent") or {}
        ts = web_probing_agent.get("tool_states") or {}
        if ts:
            for k, v in _get_total_api_counts(ts).items():
                guidance_total[k] += v
            for k, v in _get_actual_api_counts(ts).items():
                guidance_actual[k] += v

    # ── Metric 3: extract tool calls from messages and deduplicate globally ──
    task_search_keys: set[tuple[str, str]] = set()
    task_fetch_urls: set[str] = set()
    _collect_unique_calls_from_messages(data, task_search_keys, task_fetch_urls)
    # Merge deduplication keys from entity_collect.
    task_search_keys |= ec_search_keys
    task_fetch_urls |= ec_fetch_urls

    guidance_search_keys: set[tuple[str, str]] = set()
    guidance_fetch_urls: set[str] = set()
    for event in (data.get("guidance") or {}).get("events") or []:
        web_probing_agent = event.get("web_probing_agent") or {}
        _collect_unique_calls_from_messages(
            web_probing_agent, guidance_search_keys, guidance_fetch_urls
        )

    task_dedup = {"serper": len(task_search_keys), "jina": len(task_fetch_urls)}
    guidance_dedup = {
        "serper": len(guidance_search_keys),
        "jina": len(guidance_fetch_urls),
    }
    # Deduplicated total = length of the merged set, not a simple sum.
    all_dedup = {
        "serper": len(task_search_keys | guidance_search_keys),
        "jina": len(task_fetch_urls | guidance_fetch_urls),
    }

    return {
        "total_calls": {
            "task": task_total,
            "guidance": guidance_total,
            "total": _sum(task_total, guidance_total),
        },
        "actual_calls": {
            "task": task_actual,
            "guidance": guidance_actual,
            "total": _sum(task_actual, guidance_actual),
        },
        "task_dedup_calls": {
            "task": task_dedup,
            "guidance": guidance_dedup,
            "total": all_dedup,
        },
    }


def _count_agents(tree: dict, depth: int = 0,
                   by_depth: dict[int, int] | None = None,
                   by_type: dict[str, int] | None = None) -> None:
    """Recursively traverse agent_tree and count agents by depth and type."""
    if by_depth is None or by_type is None:
        return
    by_depth[depth] = by_depth.get(depth, 0) + 1
    agent_type = tree.get("type", "unknown")
    by_type[agent_type] = by_type.get(agent_type, 0) + 1
    for child in tree.get("children", []):
        _count_agents(child, depth + 1, by_depth, by_type)


def analyze_item(item: dict[str, Any], print_tree: bool = True) -> dict[str, Any]:
    """Perform structured analysis of one WebSwarm task log.

    Args:
        item: Complete task-log dict.
        print_tree: Whether to include the complete agent_tree in the result.

    Returns:
        {
            "summary": {
                "token_usage": {...},
                "api_usage": {...},
                "agent_stats": {"by_depth": {"0": N, ...}, "by_type": {"search": N, ...}},
            },
            "agent_tree": { ... },  # Included only when print_tree=True
        }
    """
    tree = get_agent_tree(item)
    by_depth: dict[int, int] = {}
    by_type: dict[str, int] = {}
    _count_agents(tree, 0, by_depth, by_type)

    result: dict[str, Any] = {
        "summary": {
            "token_usage": analysis_token_usage(item),
            "api_usage": analysis_tool_usage(item),
            "agent_stats": {
                "total": sum(by_type.values()),
                "by_depth": dict(sorted(by_depth.items())),
                "by_type": dict(sorted(by_type.items(), key=lambda x: -x[1])),
            },
        },
    }
    if print_tree:
        result["agent_tree"] = tree
    return result
