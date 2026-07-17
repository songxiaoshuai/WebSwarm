"""Data-loading, task-switching, and table-evaluation entry point for the WideSearch benchmark.

This file only wraps WideSearch dataset samples and gold tables in a uniform evaluator.
The widesearch/evaluate/ subpackage handles table parsing, column alignment, and metric computation.
"""
import json
import os
import random
import pandas as pd

from .widesearch.evaluate.schema import WideSearchEvaluation, LLMEvalConfig
from .widesearch.evaluate.table_metrics import table_evaluate
from .widesearch.evaluate.llm import call_claude
from .base_evaluator import BaseEvaluator


class WideSearchEval(BaseEvaluator):
    """WideSearch evaluation environment."""

    def __init__(self, config: dict = None):
        super().__init__(config)
        # Initialize data paths, judge model, and current task state. TaskManager injects the judge model.
        self.version = config["version"]
        self.dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "widesearch/data")  # Dataset path
        self.judge_model_name = config["judge_model_name"]
        self.judge_model_provider = config["judge_model_provider"]
        self.cur_task_id = None                      # Current task ID
        self.load_dataset(dataset_path=self.dataset_path)                             # Load the dataset
        self.task_info = self.reset(task_id=config.get("task_id"))

    def load_dataset(self, dataset_path):
        """Load the dataset and build an instance_id -> item mapping."""
        version_map = {
            "all": "widesearch.json",
            "en_subset": "widesearch_en_subset.json",
        }
        if self.version not in version_map:
            raise ValueError(f"Unsupported WideSearch version: {self.version}")
        dataset_file = os.path.join(dataset_path, version_map[self.version])
        print(f"[EVAL] 加载数据集: {dataset_file}")
        self.gold_dir = os.path.join(dataset_path, "widesearch_gold")
        with open(dataset_file, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        self.dataset_dict = {item["instance_id"]: item for item in dataset}
        print(f"[EVAL] 数据集加载完成，共 {len(dataset)} 条任务")
        print(f"[EVAL] Gold 答案目录: {self.gold_dir}")


    def reset(self, task_id=None):
        """Reset the environment and select a task randomly or by task_id."""
        self.cur_task_id = None                    # Clear the old task ID to avoid reusing prior state.
        if not self.dataset_dict:
            raise RuntimeError("数据集未加载，请先调用 load_dataset()")
        if task_id is not None:
            if task_id not in self.dataset_dict:
                raise ValueError(f"task_id '{task_id}' 不存在于数据集中")
            self.cur_task_id = task_id
        else:
            self.cur_task_id = random.choice(list(self.dataset_dict.keys()))
        print(f"[EVAL] 环境已重置，当前任务 ID: {self.cur_task_id}")
        task_info = {"dataset": "widesearch", "task_id": self.cur_task_id, "task": self.dataset_dict[self.cur_task_id]["query"]}
        return task_info


    def load_target_result(self, task_id: str) -> WideSearchEvaluation:
        """Load the gold answer for a task_id and return a table-evaluation object."""
        if not self.dataset_dict:
            raise RuntimeError("数据集未加载，请先调用 load_dataset()")
        if task_id not in self.dataset_dict:
            raise ValueError(f"task_id '{task_id}' 不存在于数据集中")
        item = self.dataset_dict[task_id]
        # Read the gold CSV and store it in the pydantic evaluation object.
        answer_file = os.path.join(self.gold_dir, f"{task_id}.csv")
        if not os.path.exists(answer_file):
            raise FileNotFoundError(f"找不到 Gold 答案文件: {answer_file}")
        eval_obj = item["evaluation"].copy()
        eval_obj["answer_table"] = pd.read_csv(answer_file)
        return WideSearchEvaluation.model_validate(eval_obj)


    def evaluate(self, prediction: str) -> dict:
        """Evaluate a prediction and return (reward, reward_info)."""
        def call_llm(template: str, kwargs: dict) -> str:
            # table_evaluate expects a (template, kwargs) callback; the closure injects judge settings.
            return call_claude(
                template,
                kwargs,
                judge_model_name=self.judge_model_name,
                judge_model_provider=self.judge_model_provider,
            )

        task_id = self.cur_task_id
        answer = self.load_target_result(task_id)                              # Load the gold answer
        metrics = table_evaluate(prediction, answer, LLMEvalConfig(), call_llm)  # Run evaluation
        reward_info = {}
        reward_info.update(metrics.model_dump())
        reward = reward_info["f1_by_item"]
        return reward, reward_info
