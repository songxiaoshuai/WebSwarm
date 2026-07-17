"""Subtask dispatch and concurrent-execution logic for the wide agent.

DispatchMixin handles create_sub_agents tool calls: it validates the task list, applies the
wide-recursion safeguard, optionally runs scout/fanout phases, and delegates subtasks to verb
agents concurrently.
"""

import json
import uuid
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from ..registry import VALID_VERBS, run_verb

# Scout batch size for subtask_experience: run a few siblings first, then transfer experience to the remaining fanout.
_SCOUT_SIZE = 2


class DispatchMixin:
    """Fanout dispatch logic for wide."""

    def _execute_create_sub_agents(
        self, tasks: list[str], verb: str
    ) -> tuple[str, list[dict]]:
        """Execute create_sub_agents once; every subtask in the batch shares one verb type."""
        # Tool arguments come from the LLM; return recoverable text feedback instead of raising immediately.
        def _params_echo() -> str:
            tasks_repr = json.dumps(tasks, ensure_ascii=False)
            if len(tasks_repr) > 400:
                tasks_repr = tasks_repr[:400] + "…"
            return f"You passed: tasks={tasks_repr}, type={verb!r}."

        if not isinstance(tasks, list) or not tasks:
            return (
                "[Error] 'tasks' must be a non-empty list of strings. "
                f"{_params_echo()} "
                "Re-issue create_sub_agents with a list of self-contained task description strings.",
                [],
            )
        if not all(isinstance(t, str) and t.strip() for t in tasks):
            bad_indices = [
                i for i, t in enumerate(tasks) if not (isinstance(t, str) and t.strip())
            ]
            return (
                "[Error] All entries in 'tasks' must be non-empty strings. "
                f"Invalid entries at index(es): {bad_indices}. "
                f"{_params_echo()}",
                [],
            )
        if verb not in VALID_VERBS:
            return (
                f"[Error] Unknown type {verb!r}. Must be one of {VALID_VERBS}. "
                f"{_params_echo()}",
                [],
            )

        # Recursion safeguard: at the depth limit, downgrade further wide expansion to an atom leaf.
        effective_verb = verb
        if verb == "wide" and self.depth + 1 >= self.max_depth:
            print(
                f"[Wide-d{self.depth}] safety net: depth+1 ({self.depth + 1}) "
                f">= max_depth ({self.max_depth}); downgrading type='wide' to "
                f"type='atom' for {len(tasks)} sub-task(s)."
            )
            effective_verb = "atom"

        depth_tag = f"[Wide-d{self.depth}]"
        print(
            f"{depth_tag} create_sub_agents: {len(tasks)} task(s), type={effective_verb}"
        )

        # When subtask_experience is enabled, run scouts first and transfer their experience to the remaining fanout.
        scout_size = 0
        if (
            self.guidance_store is not None
            and self.subtask_experience_enabled
            and effective_verb != "wide"
            and len(tasks) > _SCOUT_SIZE
        ):
            scout_size = min(_SCOUT_SIZE, len(tasks) - 1)

        completed: dict[int, dict] = {}

        if scout_size > 0:
            scout_tasks = tasks[:scout_size]
            rest_tasks = tasks[scout_size:]
            scout_workers = min(len(scout_tasks), self.max_parallel_workers)
            print(
                f"{depth_tag} guidance-scout phase: running first {scout_size} task(s) "
                f"in parallel (workers={scout_workers}) before fanning out "
                f"the remaining {len(rest_tasks)}."
            )
            # Phase 1: scout batch to collect transferable execution experience.
            self._run_batch(
                tasks=scout_tasks,
                start_idx=1,
                total_tasks=len(tasks),
                effective_verb=effective_verb,
                workers=scout_workers,
                completed=completed,
                depth_tag=depth_tag,
                phase_label="scout",
            )

            # Phase 2: extract subtask_experience from scout traces.
            subtask_experience = self._maybe_extract_subtask_experience(
                effective_verb=effective_verb,
                scout_size=scout_size,
                tasks=tasks,
                rest_tasks=rest_tasks,
                completed=completed,
                depth_tag=depth_tag,
            )

            if subtask_experience:
                print(
                    f"{depth_tag} subtask_experience: extracted experience "
                    f"({len(subtask_experience)} chars); will prepend to "
                    f"{len(rest_tasks)} fanout task(s)."
                )
            self._current_subtask_experience = subtask_experience or None
            try:
                self._run_batch(
                    tasks=rest_tasks,
                    start_idx=scout_size + 1,
                    total_tasks=len(tasks),
                    effective_verb=effective_verb,
                    workers=min(len(rest_tasks), self.max_parallel_workers),
                    completed=completed,
                    depth_tag=depth_tag,
                    phase_label="fanout",
                )
            finally:
                self._current_subtask_experience = None
        else:
            # Default behavior: fan out all subtasks concurrently in one batch.
            self._run_batch(
                tasks=tasks,
                start_idx=1,
                total_tasks=len(tasks),
                effective_verb=effective_verb,
                workers=min(len(tasks), self.max_parallel_workers),
                completed=completed,
                depth_tag=depth_tag,
                phase_label=None,
            )

        ordered = [completed[i] for i in sorted(completed.keys())]
        aggregated = self._aggregate_results(ordered)
        print(f"{depth_tag} all {len(tasks)} sub-task(s) completed")
        return aggregated, ordered

    def _run_batch(
        self,
        tasks: list[str],
        start_idx: int,
        total_tasks: int,
        effective_verb: str,
        workers: int,
        completed: dict[int, dict],
        depth_tag: str,
        phase_label: Optional[str] = None,
    ) -> None:
        """Run a subtask batch concurrently and write results to completed by original index."""
        phase_tag = f" [{phase_label}]" if phase_label else ""
        with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            future_to_idx = {}
            for offset, task_desc in enumerate(tasks):
                idx = start_idx + offset
                print(
                    f"{depth_tag}{phase_tag} submit sub-task {idx}/{total_tasks}: "
                    f"{task_desc[:120]}"
                )
                future = executor.submit(
                    self._dispatch_one_subtask, task_desc, effective_verb
                )
                future_to_idx[future] = idx

            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    result = future.result()
                    completed[idx] = result
                    print(
                        f"{depth_tag}{phase_tag} sub-task {idx}/{total_tasks} done — "
                        f"status={result.get('status', '?')}, "
                        f"steps={result.get('steps', '?')}"
                    )
                except Exception as e:
                    print(f"{depth_tag}{phase_tag} sub-task {idx}/{total_tasks} error: {e}")
                    completed[idx] = self._make_error_result(
                        tasks[idx - start_idx], e, effective_verb
                    )

    def _dispatch_one_subtask(
        self,
        task: str,
        verb: str,
    ) -> dict:
        """Route one wide subtask to recursive wide or an ordinary verb agent.

        - verb="wide" → recursively create a child WideNode at depth+1
        - Other verbs → route through run_verb to atom/deep/entity_collect
        - If _current_subtask_experience is not None, prepend the experience to task
        """
        # During fanout, prepend execution experience extracted by scouts to the task text.
        subtask_experience = self._current_subtask_experience
        effective_task = task
        if subtask_experience:
            effective_task = (
                "# Execution Tips from Scout Agents\n"
                f"{subtask_experience}\n\n"
                f"{task}"
            )

        if verb == "wide":
            # A recursive wide call keeps independent node state to avoid mixing parent message history.
            from .wide_node import WideNode

            child = WideNode(
                model=self.model,
                provider=self.provider,
                max_steps=self.max_steps,
                env_config=deepcopy(self.env_config),
                depth=self.depth + 1,
                max_depth=self.max_depth,
                max_parallel_workers=self.max_parallel_workers,
                guidance_store=self.guidance_store,
                root_task=self.root_task,
                web_probing_enabled=self.web_probing_enabled,
                subtask_experience_enabled=self.subtask_experience_enabled,
            )
            raw = child.run(task_info={"task": effective_task})
            return {
                "task": task,
                "verb": "wide",
                "answer": raw.get("prediction_answer") or "",
                "status": "completed" if raw.get("terminated") else "truncated",
                "steps": raw.get("steps", 0),
                "messages": raw.get("messages", []),
                "trajectory": raw.get("trajectory", []),
                "tool_states": raw.get("tool_states", {}),
                "child_results": raw.get("all_child_results", []),
                "node_type": "wide_node",
                "depth": self.depth + 1,
            }

        # Route other verbs through the registry to their specialized agents.
        subtask_id = f"wide-d{self.depth}.{verb}#{uuid.uuid4().hex[:6]}"
        raw = run_verb(
            verb=verb,
            task=effective_task,
            original_task=self._original_task,
            env_config=deepcopy(self.env_config),
            model=self.model,
            provider=self.provider,
            max_steps=self.max_steps,
            guidance_store=self.guidance_store,
            root_task=self.root_task,
            subtask_id=subtask_id,
            web_probing_enabled=self.web_probing_enabled,
            subtask_experience_enabled=self.subtask_experience_enabled,
        )
        return {
            "task": task,
            "verb": verb,
            "answer": raw.get("prediction_answer") or "",
            "status": "completed" if raw.get("terminated") else "truncated",
            "steps": raw.get("steps", 0),
            "messages": raw.get("messages", []),
            "trajectory": raw.get("trajectory", []),
            "tool_states": raw.get("tool_states", {}),
            "child_results": raw.get("all_child_results", []),
            "node_type": "react_agent",
            "depth": self.depth + 1,
        }

    @staticmethod
    def _make_error_result(task: str, error: Exception, verb: str) -> dict:
        return {
            "task": task,
            "verb": verb,
            "answer": f"Error: {error}",
            "status": "error",
            "steps": 0,
            "messages": [],
            "trajectory": [],
            "tool_states": {},
            "child_results": [],
            "node_type": "error",
            "depth": -1,
        }

    @staticmethod
    def _aggregate_results(child_results: list[dict]) -> str:
        if not child_results:
            return "No sub-tasks were executed."

        if len(child_results) == 1:
            r = child_results[0]
            return (
                f"[Sub-Task Result]\n"
                f"verb: {r.get('verb', '?')}\n"
                f"task: {r.get('task', '?')}\n"
                f"status: {r.get('status', '?')} (steps: {r.get('steps', '?')})\n"
                f"answer:\n{r.get('answer') or '(empty)'}"
            )

        parts = []
        for idx, r in enumerate(child_results, 1):
            parts.append(
                f"=== Sub-Task {idx} ===\n"
                f"verb: {r.get('verb', '?')}\n"
                f"task: {r.get('task', '?')}\n"
                f"status: {r.get('status', '?')} (steps: {r.get('steps', '?')})\n"
                f"answer:\n{r.get('answer') or '(empty)'}"
            )
        return "\n\n".join(parts)
