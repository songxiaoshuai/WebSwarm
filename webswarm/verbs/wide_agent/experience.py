"""Guidance mechanisms for the wide agent.

ExperienceMixin implements two assistance signals: web_probing detects webpage-information
topology before wide decomposition, while subtask_experience transfers execution experience
from a few completed siblings to remaining subtasks across scout/fanout phases.
"""

from copy import deepcopy
from typing import Optional


class ExperienceMixin:
    """Provide web_probing and subtask_experience for WideNode."""

    def _should_enable_subtask_experience(
        self, effective_verb: str, scout_size: int, n_tasks: int
    ) -> bool:
        # Trigger subtask_experience only after scouts ran and fanout tasks remain.
        if not self.subtask_experience_enabled:
            return False
        if self.guidance_store is None:
            return False
        if effective_verb != "atom":
            return False
        if scout_size < 1:
            return False
        if n_tasks <= scout_size:
            return False
        return True

    def _maybe_extract_subtask_experience(
        self,
        *,
        effective_verb: str,
        scout_size: int,
        tasks: list[str],
        rest_tasks: list[str],
        completed: dict[int, dict],
        depth_tag: str,
    ) -> Optional[str]:
        """Extract transferable execution experience from scout results."""
        if not self._should_enable_subtask_experience(
            effective_verb, scout_size, len(tasks)
        ):
            return None

        from ...guidance import extract_subtask_experience

        model = self.model
        provider = self.provider

        scout_results = [completed[i] for i in sorted(completed.keys())]
        print(
            f"{depth_tag} subtask_experience: extracting experience "
            f"(model={model}, provider={provider})..."
        )
        try:
            subtask_experience, _ev = extract_subtask_experience(
                root_task=self.root_task or "(root task not provided)",
                wide_task=self.task_info.get("task", ""),
                scout_results=scout_results,
                remaining_tasks=rest_tasks,
                guidance_store=self.guidance_store,
                model=model,
                provider=provider,
            )
        except Exception as e:  # noqa: BLE001
            print(f"{depth_tag} subtask_experience LLM call failed: {e}")
            return None

        if subtask_experience is None:
            print(
                f"{depth_tag} subtask_experience: no useful experience extracted; "
                "default fanout."
            )
        return subtask_experience

    def _maybe_run_web_probing(self):
        """Run web_probing and inject topology findings into wide's initial user message.

        The web_probing agent's prediction_answer, as raw Markdown text, is appended to
        WideNode's first user message under a "# Topology Finding" section. The WideNode LLM
        uses it to choose a decomposition strategy.

        build_web_probing_event records the complete web_probing execution trace
        (messages/trajectory/tool_states) in guidance events for later auditing.
        """
        from ...guidance.web_probing import (
            run_web_probing,
            build_web_probing_event,
        )

        model = self.model
        provider = self.provider

        wide_task = self.task_info.get("task", "")
        print(
            f"[Wide-d{self.depth}] web_probing: running structural exploration "
            f"(model={model}, provider={provider})..."
        )

        raw_web_probing_agent: Optional[dict] = None
        err: Optional[str] = None
        try:
            raw_web_probing_agent = run_web_probing(
                wide_task=wide_task,
                model=model,
                provider=provider,
            )
        except Exception as e:
            err = repr(e)
            print(f"[Wide-d{self.depth}] web_probing: failed: {e}")

        # Record an audit event for both success and failure.
        if self.guidance_store is not None:
            try:
                ev = build_web_probing_event(
                    wide_task=wide_task,
                    raw_web_probing_agent=raw_web_probing_agent,
                    error=err,
                )
                self.guidance_store.append_event(ev)
            except Exception as e:  # noqa: BLE001
                print(f"[Wide-d{self.depth}] web_probing: append_event failed: {e}")

        if err is not None:
            return

        web_probing_answer = (
            raw_web_probing_agent.get("prediction_answer")
            if raw_web_probing_agent
            else None
        )
        if not web_probing_answer:
            print(f"[Wide-d{self.depth}] web_probing: no actionable result, proceeding normally.")
            return

        print(
            f"[Wide-d{self.depth}] web_probing: injecting topology finding "
            f"({len(web_probing_answer)} chars)"
        )

        original_user_msg = self.messages[1]["content"]
        self.messages[1]["content"] = (
            f"# Your Task\n{original_user_msg}\n\n"
            f"# Topology Finding\n{web_probing_answer}"
        )
