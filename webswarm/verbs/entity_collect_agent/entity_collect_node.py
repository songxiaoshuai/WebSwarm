"""Node wrapper for the entity_collect verb.

EntityCollectNode implements entity_collect's four stages: schema inference and format
injection, parallel multi-strategy sampling, answer extraction, and split-verify-merge.
It has no additional ReAct scheduler; the sampling stage reuses the general leaf runner
to launch multiple BaseReactAgent instances.
"""

from __future__ import annotations

import json as _json
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from typing import Optional

from .schema_infer import (
    auto_enhance_task,
    detect_has_format_requirement,
)
from .task_difficulty import check_is_count_task
from .split_verify_merge import merge_tables_with_verification
from .prompts import DIVERSE_SYSTEM_PROMPTS, STRATEGY_NAMES, ENTITY_COLLECT_SYSTEM_PROMPT

from ..runtime import run_with_prompt


VERB_NAME = "entity_collect"

DEFAULT_AUTO_SCHEMA_INFER = True
DEFAULT_ENABLE_TASK_DIFFICULTY_CHECK = True

DEFAULT_NUM_SAMPLES = 3
DEFAULT_SAMPLE_MAX_WORKERS = 5
DEFAULT_DIVERSE_SYSTEM_PROMPT = True

DEFAULT_ENABLE_ORIGINAL_TASK_SUFFIX = True

DEFAULT_FALLBACK_SHRINK_RATIO = 0.5
DEFAULT_FALLBACK_EXPANSION_RATIO = 2.0
DEFAULT_FALLBACK_ABS_THRESHOLD = 10


class EntityCollectNode:
    """Runtime context for one entity_collect call.

    ``self`` stores the task, tool environment, LLM configuration, and intermediate logs
    to avoid repeatedly passing large argument groups among the four stages. Every
    ``run_verb("entity_collect", ...)`` call creates a new instance and returns a uniform
    verb-result dict on completion.
    """

    # Construction and main entry point.

    def __init__(
        self,
        *,
        task: str,
        env_config: dict,
        model: str,
        provider: str,
        original_task: Optional[str] = None,
    ):
        # Task text: task is the current subtask; original_task restores higher-level context.
        self.task = task
        self.original_task = original_task

        # Auxiliary LLM calls and sampling agents share the same model and tool environment.
        self.env_config = env_config
        self.model = model
        self.provider = provider

        # Structured log spanning the four-stage flow, ultimately stored in the verb result.
        self.call_log: dict = {"task": task}

    def run(self) -> dict:
        """Execute one entity_collect verb.

        Sampling stage: launch BaseReactAgent through run_with_prompt for N independent samples.

        Merge stage: LLM Split + Search Verifier.
          1. The LLM divides results into intersection (appearing in at least two paths)
             and diff (appearing in only one path)
          2. A search-verifier agent performs real searches to validate diff entities
          3. Final result = intersection ∪ verified_diff
        """
        print(f"[{VERB_NAME}] num_samples={DEFAULT_NUM_SAMPLES}")

        # Step 1: infer the schema and inject format requirements.
        enhanced_task, schema_log = self._prepare_task(self.task)
        self.call_log["schema_log"] = schema_log

        # Append the root task as a boundary reference to prevent semantic drift in fields.
        enhanced_task = self._maybe_append_original_task_suffix(enhanced_task)

        # Step 1.5: reduce the number of sampling paths when the entity set is naturally closed.
        effective_num_samples = self._maybe_check_difficulty(enhanced_task)

        # Step 2: sample multiple strategies in parallel to improve entity-set recall.
        sample_results = self._parallel_sample(
            enhanced_task,
            num_samples=effective_num_samples,
            sample_max_workers=min(DEFAULT_SAMPLE_MAX_WORKERS, effective_num_samples),
        )
        self.call_log["num_samples"] = len(sample_results)
        self.call_log["sample_logs"] = self._build_sample_logs(sample_results)

        # Step 3: extract the submit_answer table from each ReAct trajectory.
        answers = self._extract_answers(sample_results)
        self.call_log["valid_answers"] = len(answers)

        if not answers:
            self.call_log["error"] = "no_valid_answers"
            return self._to_verb_result(
                merged_table="",
                is_error=True,
                sample_results=sample_results,
                merge_log={},
            )

        # Step 4: split, verify, and merge candidate tables into the final entity set.
        merged_table, merge_log = self._merge_answers_with_verification(enhanced_task, answers)
        self.call_log["merge_log"] = merge_log

        is_error = not merged_table
        if is_error:
            self.call_log["error"] = "merge_empty"

        print(f"[EntityCollect] Done: {len(answers)}/{effective_num_samples} valid, "
              f"merged length={len(merged_table)}")

        return self._to_verb_result(
            merged_table=merged_table,
            is_error=is_error,
            sample_results=sample_results,
            merge_log=merge_log,
        )

    # Step 1: infer the schema and inject format requirements.

    def _prepare_task(self, task: str) -> tuple[str, dict]:
        """Prepare task text with format requirements by reusing schema_infer."""
        log: dict = {"original_task": task}

        if detect_has_format_requirement(task):
            log["source"] = "task_already_formatted"
            log["skipped"] = True
            return task, log

        if DEFAULT_AUTO_SCHEMA_INFER:
            enhanced, infer_log = auto_enhance_task(
                task=task,
                provider=self.provider,
                model=self.model,
            )
            log["source"] = "auto_inferred"
            log["infer_log"] = infer_log
            return enhanced, log

        log["source"] = "passthrough"
        return task, log

    def _maybe_append_original_task_suffix(self, task: str) -> str:
        """Append an original-task reference block according to DEFAULT_ENABLE_ORIGINAL_TASK_SUFFIX.

        The reference only clarifies field meanings and filtering boundaries; sampling agents
        may not use it to expand task scope.
        """
        original_task = self.original_task
        if not (DEFAULT_ENABLE_ORIGINAL_TASK_SUFFIX and original_task and original_task.strip()):
            return task
        appended = (
            f"{task}\n\n"
            f"[Reference Context — DO NOT EXPAND SCOPE]\n"
            f"The sub-task above was derived from the following original user task. "
            f"It is provided ONLY as background context to clarify ambiguous attribute definitions "
            f"(e.g., exact year, entity name, field naming, or filtering constraints).\n\n"
            f"IMPORTANT:\n"
            f"- Execute ONLY the sub-task described above.\n"
            f"- Do NOT collect additional attributes, entities, events, or fields unless they are explicitly required by the sub-task.\n"
            f"- Do NOT expand the scope based on information mentioned in the original task.\n"
            f"- If the original task contains extra information not required for the sub-task, ignore it.\n\n"
            f"[Original User Task]\n"
            f"{original_task}\n\n"
        )
        print(f"[{VERB_NAME}] enable_original_task_suffix=True: appended original_task "
              f"to task (original_task length={len(original_task)})")
        return appended

    # Step 1.5: detect task difficulty.

    def _maybe_check_difficulty(self, enhanced_task: str) -> int:
        """Determine the effective number of sampling paths using DEFAULT_ENABLE_TASK_DIFFICULTY_CHECK.

        Returns:
            effective_num_samples: Number of sampling paths enabled after difficulty assessment.
        """
        effective_num_samples = DEFAULT_NUM_SAMPLES
        is_count_task: bool = False
        if DEFAULT_ENABLE_TASK_DIFFICULTY_CHECK and DEFAULT_NUM_SAMPLES > 1:
            is_count_task, count_task_log = check_is_count_task(
                task=enhanced_task,
                provider=self.provider,
                model=self.model,
            )
            self.call_log["count_task_log"] = count_task_log
            if is_count_task:
                effective_num_samples = 1
                print(f"[{VERB_NAME}] Task classified as SIMPLE (count-deterministic), "
                      f"reducing samples: {DEFAULT_NUM_SAMPLES} → 1")
            else:
                print(f"[{VERB_NAME}] Task classified as HARD (open-ended), "
                      f"keeping {DEFAULT_NUM_SAMPLES} samples")
        self.call_log["is_count_task"] = is_count_task
        self.call_log["effective_num_samples"] = effective_num_samples
        return effective_num_samples

    # Step 2: sample multiple strategies in parallel.

    def _run_single_sample(
        self,
        task_query: str,
        sample_idx: int,
        *,
        diverse_system_prompt: bool,
    ) -> dict:
        """Sample one path by creating a BaseReactAgent through run_with_prompt for one search run."""
        if diverse_system_prompt:
            prompt = DIVERSE_SYSTEM_PROMPTS[sample_idx % len(DIVERSE_SYSTEM_PROMPTS)]
            strategy = STRATEGY_NAMES[sample_idx % len(DIVERSE_SYSTEM_PROMPTS)]
            print(f"[EntityCollect] Sample {sample_idx} using strategy: {strategy}")
        else:
            prompt = ENTITY_COLLECT_SYSTEM_PROMPT

        # Official and DeepPagination strategies need more in-site link clues, so enable keep_links.
        # Deep-copy before mutation to prevent concurrent sampling paths from sharing tool configuration changes.
        local_env_config = deepcopy(self.env_config)
        if sample_idx % len(DIVERSE_SYSTEM_PROMPTS) in (1, 2):
            fetch_url_cfg = local_env_config.get("fetch_url")
            if isinstance(fetch_url_cfg, dict):
                fetch_url_cfg["keep_links"] = True
                print(f"[EntityCollect] Sample {sample_idx}: keep_links=True injected into fetch_url config")

        try:
            result = run_with_prompt(
                system_prompt=prompt,
                task=task_query,
                env_config=local_env_config,
            )
            result["sample_idx"] = sample_idx
            return result
        except Exception as e:
            return {
                "sample_idx": sample_idx,
                "error": repr(e),
                "traceback": traceback.format_exc(),
                "terminated": False,
                "prediction_answer": None,
            }

    def _parallel_sample(
        self,
        task_query: str,
        *,
        num_samples: int,
        sample_max_workers: int,
    ) -> list[dict]:
        """Use ThreadPoolExecutor to launch N parallel sampling paths."""
        diverse_system_prompt = DEFAULT_DIVERSE_SYSTEM_PROMPT
        results: list[dict] = []
        with ThreadPoolExecutor(max_workers=sample_max_workers) as executor:
            futures = {
                executor.submit(
                    self._run_single_sample,
                    task_query,
                    i,
                    diverse_system_prompt=diverse_system_prompt,
                ): i
                for i in range(num_samples)
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    result = future.result()
                except Exception as e:
                    result = {
                        "sample_idx": idx,
                        "error": repr(e),
                        "terminated": False,
                        "prediction_answer": None,
                    }
                results.append(result)
        results.sort(key=lambda r: r.get("sample_idx", 0))
        return results

    # Helper: count data rows in a pipe-delimited table.

    @staticmethod
    def _count_pipe_rows(text: str) -> int:
        """Count data rows in a Markdown pipe table, excluding the header and separator.
        Return -1 for non-pipe format and 0 for a header-only table.
        """
        import re as _re
        text = (text or "").strip()
        if text.startswith("```"):
            lines = text.splitlines()
            end = len(lines) - 1 if lines[-1].strip() == "```" else len(lines)
            text = "\n".join(lines[1:end])
        if not text or "|" not in text:
            return -1
        pipe_lines = [l.strip() for l in text.splitlines()
                      if l.strip().startswith("|") and l.strip().endswith("|")]
        if not pipe_lines:
            return -1
        past_sep = False
        data_rows = 0
        for l in pipe_lines:
            cells = [c.strip() for c in l.split("|") if c.strip()]
            if all(_re.match(r"^[-:]+$", c) for c in cells):
                past_sep = True
                continue
            if past_sep:
                data_rows += 1
        return data_rows if past_sep else -1

    # Step 3: extract answers.

    @staticmethod
    def _extract_answer_from_messages(result: dict) -> Optional[str]:
        """Search backward through message history for submit_answer content."""
        if not result.get("terminated"):
            return None
        for msg in reversed(result.get("messages", [])):
            if msg.get("role") != "assistant" or not msg.get("tool_calls"):
                continue
            for tc in msg["tool_calls"]:
                if tc["function"]["name"] == "submit_answer":
                    args = tc["function"]["arguments"]
                    if isinstance(args, str):
                        try:
                            args = _json.loads(args)
                        except _json.JSONDecodeError:
                            return args
                    return args.get("answer", "")
        return None

    def _extract_answers(self, sample_results: list[dict]) -> list[str]:
        """Extract a valid answer from each sub-agent result."""
        answers: list[str] = []
        for r in sample_results:
            answer = r.get("prediction_answer")
            if not answer:
                answer = self._extract_answer_from_messages(r)
            if answer:
                answers.append(answer)
            else:
                print(f"[EntityCollect] Sample {r.get('sample_idx')} "
                      f"produced no answer (error={r.get('error', 'N/A')})")
        return answers

    # Step 4: merge and verify.

    def _merge_answers_with_verification(
        self,
        task_query: str,
        answers: list[str],
    ) -> tuple[str, str | dict]:
        """Merge multiple tables with the current validation flow: LLM Split + Search Verifier.

        Flow:
          1. The LLM divides results into intersection (appearing in at least two paths)
             and diff (appearing in only one path)
          2. A search-verifier agent performs real searches to validate diff entities
          3. Final result = intersection ∪ verified_diff

        The verifier launches a standard BaseReactAgent through verbs/runtime.run_with_prompt.
        self.env_config determines its tool environment.
        """
        if not answers:
            return "", {"skipped": "no_valid_answers"}
        if len(answers) == 1:
            return answers[0], {"skipped": "single_answer"}

        print(f"[EntityCollect] Merging {len(answers)} tables with split_verify_merge...")
        merged, log = merge_tables_with_verification(
            task=task_query,
            tables=answers,
            provider=self.provider,
            model=self.model,
            return_log=True,
            sub_agent_env_config=self.env_config,
        )
        print(f"[EntityCollect] split_verify_merge complete, answer length: {len(merged)}")

        # Fallback: use the general path if merging causes an abnormal row-count contraction or expansion.
        anchor = answers[0]  # path#0 is the sample produced by the general prompt
        anchor_rows = self._count_pipe_rows(anchor)
        merged_rows = self._count_pipe_rows(merged)

        # Fall back immediately if merged cannot be parsed as a pipe-delimited table.
        if merged_rows == -1:
            print(
                f"[EntityCollect] Fallback triggered (parse_failed): "
                f"merged result is not a valid pipe table → returning path#0"
            )
            log = {
                "fallback": "parse_failed",
                "anchor_rows": anchor_rows,
                "merged_rows": merged_rows,
                "original_log": log,
            }
            return anchor, log

        # Check relative and absolute differences only when the anchor has substantive content.
        if anchor_rows > 0:
            abs_diff = abs(merged_rows - anchor_rows)
            ratio = merged_rows / anchor_rows

            shrink = (
                ratio < DEFAULT_FALLBACK_SHRINK_RATIO
                and abs_diff > DEFAULT_FALLBACK_ABS_THRESHOLD
            )
            expansion = (
                ratio > DEFAULT_FALLBACK_EXPANSION_RATIO
                and abs_diff > DEFAULT_FALLBACK_ABS_THRESHOLD
            )

            if shrink or expansion:
                direction = "shrink" if shrink else "expansion"
                print(
                    f"[EntityCollect] Fallback triggered ({direction}): "
                    f"anchor={anchor_rows} rows, merged={merged_rows} rows, "
                    f"ratio={ratio:.2f}, abs_diff={abs_diff} → returning path#0"
                )
                log = {
                    "fallback": direction,
                    "anchor_rows": anchor_rows,
                    "merged_rows": merged_rows,
                    "ratio": round(ratio, 3),
                    "abs_diff": abs_diff,
                    "original_log": log,
                }
                return anchor, log

        return merged, log

    # Log and result conversion.

    @staticmethod
    def _build_sample_logs(sample_results: list[dict]) -> list[dict]:
        """Build a log entry for each sampling path."""
        logs: list[dict] = []
        for r in sample_results:
            log: dict = {
                "sample_idx":        r.get("sample_idx"),
                "terminated":        r.get("terminated", False),
                "truncated":         r.get("truncated", False),
                "steps":             r.get("steps", 0),
                "prediction_answer": r.get("prediction_answer", ""),
                "messages":          r.get("messages", []),
                "trajectory":        r.get("trajectory", []),
            }
            if r.get("error"):
                log["error"] = r["error"]
            if r.get("traceback"):
                log["traceback"] = r["traceback"]
            logs.append(log)
        return logs

    def _to_verb_result(
        self,
        *,
        merged_table: str,
        is_error: bool,
        sample_results: list[dict],
        merge_log: str | dict,
    ) -> dict:
        """Convert the internal result to a verb-contract dict."""
        call_log = self.call_log
        sample_logs = call_log.get("sample_logs", []) or []
        total_steps = sum(int(s.get("steps", 0) or 0) for s in sample_logs)

        child_results = []
        for r in sample_results:
            child_results.append({
                "task": "(entity_collect sampling)",
                "verb": "atom",
                "answer": r.get("prediction_answer", ""),
                "status": "completed" if r.get("terminated") else "failed",
                "steps": r.get("steps", 0),
                "messages": r.get("messages", []),
                "trajectory": r.get("trajectory", []),
                "tool_states": r.get("tool_states", {}),
                "child_results": [],
                "node_type": "react_agent",
            })

        # merge_log contains the merge-validation flow: {"strategy": "...", "steps": [...], ...}
        # Store the complete merge_log in trajectory to avoid losing information.
        merge_log_dict = merge_log if isinstance(merge_log, dict) else {}
        merge_messages = merge_log_dict.get("messages", [])
        has_content = bool(merge_log_dict.get("steps"))
        child_results.append({
            "task": "(entity_collect merge)",
            "verb": "merge",
            "answer": "" if is_error else merged_table,
            "status": "completed" if not is_error else "failed",
            "steps": 1 if has_content else 0,
            "messages": merge_messages,
            "trajectory": [{"step": 0, "merge_log": merge_log_dict}] if has_content else [],
            "tool_states": {},
            "child_results": [],
            "node_type": "llm_call",
        })

        return {
            "prediction_answer": "" if is_error else merged_table,
            "terminated": not is_error,
            "truncated": False,
            "steps": total_steps,
            "messages": [],
            "trajectory": [{
                "step": 0,
                "verb": VERB_NAME,
                "schema_log": call_log.get("schema_log", {}),
                "is_count_task": call_log.get("is_count_task"),
                "effective_num_samples": call_log.get("effective_num_samples"),
                "count_task_log": call_log.get("count_task_log"),
            }],
            "tool_states": {},
            "all_child_results": child_results,
        }
