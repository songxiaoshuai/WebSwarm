"""Unified evaluation-task entry point.

TaskManager creates an evaluator for the selected benchmark and exposes uniform interfaces
for retrieving task_info and computing rewards. It stores neither tool-environment nor agent
state; it only converts runner-supplied task configuration into a concrete evaluator instance.
"""
import os

from .benchmark.widesearch_evaluator import WideSearchEval
from .benchmark.browsecomp_plus_evaluator import BrowseCompPlusEval
from .benchmark.deepwidesearch_evaluator import DeepWideSearchEval
from .benchmark.gisa_evaluator import GISAEval


JUDGE_MODEL_NAME_ENV = "JUDGE_MODEL_NAME"
JUDGE_MODEL_PROVIDER_ENV = "JUDGE_MODEL_PROVIDER"


def get_judge_model_config() -> dict:
    """Read the evaluation LLM model/provider from environment variables."""
    missing = [
        name
        for name in (JUDGE_MODEL_NAME_ENV, JUDGE_MODEL_PROVIDER_ENV)
        if not os.environ.get(name)
    ]
    if missing:
        raise RuntimeError(
            f"{', '.join(missing)} is not set. Set it in .env or the environment."
        )
    return {
        "judge_model_name": os.environ[JUDGE_MODEL_NAME_ENV],
        "judge_model_provider": os.environ[JUDGE_MODEL_PROVIDER_ENV],
    }


class TaskManager:

    def __init__(self, config: dict = None):
        """
        Args:
            config: In addition to ToolEnv configuration, requires a 'benchmark' key with
                    one of 'widesearch' / 'browsecomp_plus' / 'deepwidesearch' / 'gisa'.
        """
        # The runner must specify the dataset version and task ID so the evaluator never guesses defaults.
        assert "benchmark" in config and "benchmark_version" in config and "task_id" in config
        benchmark = config["benchmark"]
        benchmark_version = config["benchmark_version"]
        self.task_id = config["task_id"]
        self.evaluator = self.load_evaluator(benchmark, benchmark_version, task_id=self.task_id)
        self.task_info = self.evaluator.task_info
        
    @staticmethod
    def load_evaluator(benchmark: str, benchmark_version: str, task_id: str = None):
        """Load an evaluator for the benchmark name."""
        if benchmark == "widesearch":
            judge_config = get_judge_model_config()
            evaluator = WideSearchEval(config={
                "version": benchmark_version,
                "task_id": task_id,
                **judge_config,
            })
        elif benchmark == "browsecomp_plus":
            judge_config = get_judge_model_config()
            evaluator = BrowseCompPlusEval(config={
                "version": benchmark_version,
                "task_id": task_id,
                **judge_config,
            })
        elif benchmark == "deepwidesearch":
            judge_config = get_judge_model_config()
            evaluator = DeepWideSearchEval(config={
                "task_id": task_id,
                "skip_entity_eval_reuse_widesearch_eval_logic": True,
                "version": benchmark_version,
                **judge_config,
            })
        elif benchmark == "gisa":
            evaluator = GISAEval(config={"version": benchmark_version, "task_id": task_id})
        else:
            raise ValueError(f"Unsupported benchmark: {benchmark}")
        return evaluator

    def get_task_info(self) -> dict:
        """Return information for the current task."""
        return self.task_info

    # ── Evaluation ───────────────────────────────────────────────────────

    def calculate_reward(self, prediction_answer) -> tuple[float, dict]:
        """Invoke the evaluator to compute a reward."""
        reward, reward_info = self.evaluator.evaluate(prediction_answer)
        return reward, reward_info


def list_task_ids(benchmark: str, benchmark_version: str) -> list[str]:
    """List all task_id values for a benchmark/version in dataset order.

    When the runner receives task_ids=None, it uses this function to expand to every sample
    in that version. The implementation reuses the evaluator's dataset-loading logic and
    reads the keys of dataset_dict.
    """
    evaluator = TaskManager.load_evaluator(benchmark, benchmark_version, task_id=None)
    return list(evaluator.dataset_dict.keys())
