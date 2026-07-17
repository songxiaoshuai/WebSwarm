"""WebSwarm experiment runner.

This module connects experiment configuration with concurrent execution:

1. Read the complete configuration for one experiment from experiment_config (model,
   tool environment, dataset, and task list). experiment_config constructs and validates
   it; this file only consumes it.
2. Run a set of task_id values concurrently in a thread pool. Each task uses an independent
   ToolEnv / TaskManager / WebSwarmAgent with a deep-copied configuration and no shared
   mutable state.
3. Write all task results and runtime metadata (run_info) to one JSON file under
   result_debug/. Incremental saves after every N completed samples prevent result loss
   during long runs.

Typical usage: edit the experiment-selection constants at the top of experiment_config,
then run `python run_main.py`.

Output JSON structure:
    {
        "run_info": { ...configuration and statistics for this run... },
        "results": { "<task_id>": { ...task result or error... }, ... }
    }
"""

import json
import time
import os
import traceback

from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from tqdm import tqdm

from experiment_config import (
    load_env_file,
    apply_experiment_env,
    build_experiment,
)
from tool_env.tool_env import ToolEnv
from task_manager.task_manager import (
    TaskManager,
    get_judge_model_config,
    list_task_ids,
)
from webswarm import WebSwarmAgent
from webswarm.log_stats import (
    add_tool_counts,
    analysis_tool_usage,
    empty_tool_counts,
)



def get_current_time():
    """Get current time as formatted string."""
    return time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())


def save_json(path, data):
    """Save data to JSON file."""
    with open(path, "w") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def get_model_name(webswarm_config):
    """Take the final segment of a full model identifier for use in the result filename."""
    return webswarm_config["model"].split("/")[-1]


def solve_single_task(env_config, task_config, webswarm_config, task_id):
    """
    Execute one task and return (task_id, result_dict).
    result_dict contains the task result, or an error field on failure.

    Each task owns an independent ToolEnv / TaskManager / Agent with deep-copied
    configuration so concurrently executing tasks share no mutable state.
    """
    # Create an independent tool environment and task manager for this task.
    tool_env = ToolEnv(config=deepcopy(env_config))

    tm_config = deepcopy(task_config)
    tm_config["task_id"] = task_id
    task_manager = TaskManager(config=tm_config)

    # Catch initialization errors separately because no agent state exists yet to save.
    try:
        agent = WebSwarmAgent(
            tool_env=tool_env,
            task_manager=task_manager,
            webswarm_config=deepcopy(webswarm_config),
        )
    except Exception as e:
        return task_id, {
            "error": f"Agent init error: {repr(e)}",
            "task_info": task_manager.get_task_info(),
        }

    # Run phase: return the result normally, or save accumulated state on error for later review.
    try:
        result = agent.run()
        return task_id, result
    except Exception as e:
        # root_node owns the trajectory/messages inside WebSwarmAgent; fall back to the agent itself if absent.
        inner_agent = getattr(agent, "root_node", agent)
        # Use getattr(..., default) because the error may occur before any of these fields is assigned.
        error_result = {
            "error": repr(e),
            "traceback": traceback.format_exc(),
            "messages": getattr(inner_agent, "messages", []),
            "trajectory": getattr(inner_agent, "trajectory", []),
            "steps": getattr(inner_agent, "step_count", None),
            "total_reward": getattr(agent, "total_reward", None),
            "terminated": getattr(inner_agent, "terminated", None),
            "truncated": getattr(inner_agent, "truncated", None),
            "task_info": task_manager.get_task_info(),
        }
        return task_id, error_result


def run_tasks(
    env_config,
    task_config,
    webswarm_config,
    task_ids,
    max_workers=4,
    save_path=None,
    save_every_n=1,
    resume_from=None,
):
    """
    Execute a set of tasks concurrently and save all results to one JSON file.

    Args:
        resume_from: Optional path to an existing result JSON file. When set,
            every task in task_ids is executed again: existing task results are
            overwritten, new tasks are appended, and unselected old results are
            preserved. If save_path is omitted, the existing file is updated in
            place.

    JSON structure:
    {
        "run_info": { ...configuration for this run... },
        "results": {
            "<task_id>": { ...task result... },
            ...
        }
    }
    """
    run_start = get_current_time()

    # task_ids=None runs all cases for the selected benchmark/version.
    if task_ids is None:
        task_ids = list_task_ids(
            task_config["benchmark"], task_config["benchmark_version"]
        )
        print(
            f"[run_main] task_ids=None -> 展开为 "
            f"{task_config['benchmark']}/{task_config['benchmark_version']} "
            f"全部 {len(task_ids)} 条任务"
        )

    if resume_from is not None:
        with open(resume_from, "r", encoding="utf-8") as f:
            output = json.load(f)

        if save_path is None:
            save_path = resume_from

        run_info = output["run_info"]
        existing_ids = set(output["results"])
        rerun_ids = [tid for tid in task_ids if tid in existing_ids]
        new_ids = [tid for tid in task_ids if tid not in existing_ids]
        if rerun_ids:
            print(
                f"[run_main] Will RERUN {len(rerun_ids)} existing tasks: "
                f"{rerun_ids}"
            )
        if new_ids:
            print(f"[run_main] Will ADD {len(new_ids)} new tasks: {new_ids}")

        # Keep every task recorded by the old run, then append this run's new IDs.
        all_task_ids = list(
            dict.fromkeys(run_info.get("task_ids", []) + task_ids)
        )
        run_info["task_ids"] = all_task_ids
        run_info["total_tasks"] = len(all_task_ids)
        run_info["end_time"] = None

        reward_scores = run_info.setdefault("reward_scores", {})
        for tid in all_task_ids:
            reward_scores.setdefault(tid, None)
        for tid in task_ids:
            # A failed rerun must not retain the previous successful reward.
            reward_scores[tid] = None
            output["results"].setdefault(tid, None)
    else:
        # If save_path is omitted, build a unique filename from benchmark, version, model, and timestamp.
        if save_path is None:
            save_path = f"result_debug/{task_config['benchmark']}_{task_config['benchmark_version']}_{get_model_name(webswarm_config)}_webswarm_{run_start}.json"

        # GISA uses rule-based scoring and needs no judge model; other benchmarks read the judge configuration.
        judge_config = (
            {"judge_model_name": None, "judge_model_provider": None}
            if task_config.get("benchmark") == "gisa"
            else get_judge_model_config()
        )
        # run_info collects all configuration and statistics for this run and is saved alongside results.
        run_info = {
            "task_config": task_config,
            "task_ids": task_ids,
            "agent_name": "webswarm",
            "webswarm_config": {k: v for k, v in webswarm_config.items()},
            "env_config": {k: v for k, v in env_config.items()},
            "judge_model_name": judge_config["judge_model_name"],
            "judge_model_provider": judge_config["judge_model_provider"],
            "max_workers": max_workers,
            "start_time": run_start,
            "end_time": None,
            "total_tasks": len(task_ids),
            "completed_tasks": 0,
            "failed_tasks": 0,
            "avg_reward_score": None,
            "reward_scores": {tid: None for tid in task_ids},
            "tool_counts": empty_tool_counts(),
        }
        output = {"run_info": run_info, "results": {tid: None for tid in task_ids}}

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    file_lock = Lock()  # Protect the results dict and file writes from concurrent races
    _counter = [0]  # Completed-task counter (a list so the closure can mutate it)

    def flush(force=False):
        """Write to disk when force=True or after accumulating save_every_n samples."""
        _counter[0] += 1
        if force or _counter[0] % save_every_n == 0:
            with file_lock:
                save_json(save_path, output)
            if not force:
                print(f"[run_main] Auto-saved ({_counter[0]} tasks done) -> {save_path}")

    print(f"[run_main] Starting {len(task_ids)} tasks with max_workers={max_workers}, save_every_n={save_every_n}")
    print(f"[run_main] Results will be saved to: {save_path}")

    # Create the progress bar
    pbar = tqdm(total=len(task_ids), desc="Tasks", unit="task",
                bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

    # Submit all tasks concurrently, collect them in completion order, and save incrementally.
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                solve_single_task,
                env_config, task_config, webswarm_config, tid
            ): tid
            for tid in task_ids
        }

        for future in as_completed(futures):
            tid = futures[future]
            try:
                task_id, result = future.result()
                # For successful results, also count tool calls; parsing failures do not affect the main result.
                task_tool_counts = None
                if isinstance(result, dict) and "error" not in result:
                    try:
                        task_tool_counts = analysis_tool_usage(result)
                        result["tool_counts"] = task_tool_counts
                    except Exception:
                        task_tool_counts = None
                # Lock result writes and counter updates because multiple worker threads may finish concurrently.
                with file_lock:
                    output["results"][task_id] = result
                    if "error" in result:
                        run_info["failed_tasks"] += 1
                        pbar.write(f"[run_main] FAILED  task_id={task_id}: {result['error']}")
                    else:
                        run_info["completed_tasks"] += 1
                        run_info["reward_scores"][task_id] = result.get("total_reward", None)
                        if task_tool_counts is not None:
                            add_tool_counts(run_info["tool_counts"], task_tool_counts)
                        reward_str = f" (reward: {result.get('total_reward', 'N/A')})" if result.get('total_reward') is not None else ""
                        pbar.write(f"[run_main] DONE    task_id={task_id}{reward_str}")
                    pbar.set_postfix({
                        'completed': run_info["completed_tasks"],
                        'failed': run_info["failed_tasks"]
                    })
                flush()
                pbar.update(1)
            except Exception as e:
                # If future.result() itself raises an uncaught solve_single_task error, record and save the failure.
                with file_lock:
                    output["results"][tid] = {
                        "error": repr(e),
                        "traceback": traceback.format_exc(),
                    }
                    run_info["failed_tasks"] += 1
                    pbar.write(f"[run_main] ERROR   task_id={tid}: {repr(e)}")
                    pbar.set_postfix({
                        'completed': run_info["completed_tasks"],
                        'failed': run_info["failed_tasks"]
                    })
                flush()  # Count errors as completed tasks too
                pbar.update(1)

    pbar.close()  # Close the progress bar

    # After all tasks finish, recompute successes, failures, and the mean score from final results.
    run_info["end_time"] = get_current_time()
    completed_results = [
        r for r in output["results"].values()
        if isinstance(r, dict) and "error" not in r
    ]
    failed_results = [
        r for r in output["results"].values()
        if isinstance(r, dict) and "error" in r
    ]
    run_info["completed_tasks"] = len(completed_results)
    run_info["failed_tasks"] = len(failed_results)
    # Average only valid (non-None) rewards; use None if every reward is None.
    valid_rewards = [v for v in run_info["reward_scores"].values() if v is not None]
    run_info["avg_reward_score"] = (sum(valid_rewards) / len(valid_rewards)) if valid_rewards else None

    # Reaggregate run-level tool statistics from each task's tool_counts to avoid incremental-count drift.
    run_info["tool_counts"] = empty_tool_counts()
    for result in output["results"].values():
        if isinstance(result, dict) and isinstance(result.get("tool_counts"), dict):
            add_tool_counts(run_info["tool_counts"], result["tool_counts"])

    flush(force=True)  # Force a final save at completion
    print(f"[run_main] All done. Results saved to: {save_path}")
    return output


if __name__ == "__main__":
    # 1) Load service endpoints/keys from .env, then inject experiment selections into os.environ.
    load_env_file()
    apply_experiment_env()

    # 2) experiment_config builds the complete run configuration (webswarm/env/task/task_ids)
    #    and validates consistency between benchmark and prompt_version.
    experiment = build_experiment()
    webswarm_config = experiment.webswarm_config
    env_config = experiment.env_config
    task_config = experiment.task_config
    task_ids = experiment.task_ids

    max_workers = 5  # Concurrency level
    save_every_n = 1  # Save after every N completed samples
    # Set this to an existing result JSON to rerun task_ids in that log.
    # When None, a new result file is created.
    resume_from = None

    # 3) Print key settings and allow 30 seconds for manual review; press Ctrl+C to abort.
    judge_config = (
        {"judge_model_name": "N/A", "judge_model_provider": "rule-based"}
        if task_config.get("benchmark") == "gisa"
        else get_judge_model_config()
    )
    print("=" * 60)
    print("  Agent     : webswarm")
    print(f"  Model     : {webswarm_config['model']}  ({webswarm_config['provider']})")
    print(f"  Judge     : {judge_config['judge_model_name']}  ({judge_config['judge_model_provider']})")
    print(f"  Task IDs  : {task_ids}")
    print(f"  Resume    : {resume_from}")
    print("── WebSwarm Config ──────────────────────────────────────────")
    print(json.dumps(webswarm_config, indent=2, ensure_ascii=False))
    print("── Env Config ───────────────────────────────────────────────")
    print(json.dumps(env_config, indent=2, ensure_ascii=False))
    print("── Task Config ──────────────────────────────────────────────")
    print(json.dumps(task_config, indent=2, ensure_ascii=False))
    print("=" * 60)
    print("Starting in 30 seconds... Press Ctrl+C to abort.")
    time.sleep(30)

    # 4) Run all tasks concurrently and save the results.
    run_tasks(
        env_config=env_config,
        task_config=task_config,
        webswarm_config=webswarm_config,
        task_ids=task_ids,
        max_workers=max_workers,
        save_every_n=save_every_n,
        resume_from=resume_from,
    )
