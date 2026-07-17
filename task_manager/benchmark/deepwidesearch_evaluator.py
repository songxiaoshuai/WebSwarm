"""Data-loading and evaluation entry point for the DeepWideSearch benchmark.

The default path reuses WideSearch table-evaluation logic. The retained original
DeepWideSearch entity-gated evaluation flow remains available through a configuration branch.
"""
import json
import os
import random
import pandas as pd

from .base_evaluator import BaseEvaluator
from .deepwidesearch.evaluate.evaluation.data_loader import WideSearchQuery, WideSearchResponse
from .deepwidesearch.evaluate.evaluation.evaluation import evaluate_single_query
from .deepwidesearch.evaluate.utils.utils import norm_column

# Reuse WideSearch table-evaluation data structures and metric implementations.
from .widesearch.evaluate.schema import WideSearchEvaluation, LLMEvalConfig
from .widesearch.evaluate.table_metrics import table_evaluate
from .widesearch.evaluate.llm import call_claude


class DeepWideSearchEval(BaseEvaluator):
    """DeepWideSearch evaluation environment."""

    VERSION_MAP = {
        "all": "deepwidesearch.jsonl",
        "en_subset": "deepwidesearch_en_subset.jsonl",
    }
    GOLD_DIR_NAME = "gold_answer"

    def __init__(self, config):
        super().__init__(config)
        self.version = config["version"]
        self.dataset_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "deepwidesearch", "data"
        )
        if self.version not in self.VERSION_MAP:
            raise ValueError(f"不支持的 version: {self.version}，可选: {list(self.VERSION_MAP.keys())}")
        self.data_file = os.path.join(self.dataset_path, self.VERSION_MAP[self.version])
        self.gold_dir = os.path.join(self.dataset_path, self.GOLD_DIR_NAME)
        # TaskManager injects the evaluation model; it is not configured separately in this evaluator.
        self.judge_model_name = config["judge_model_name"]
        self.judge_model_provider = config["judge_model_provider"]
        self.cur_task_id = None
        # Bypass the original entity-accuracy gate by default and reuse WideSearch table evaluation directly.
        self.skip_entity_eval_reuse_widesearch_eval_logic = config["skip_entity_eval_reuse_widesearch_eval_logic"]
        if self.skip_entity_eval_reuse_widesearch_eval_logic:
            print(f"[EVAL] skip_entity_eval_reuse_widesearch_eval_logic: {self.skip_entity_eval_reuse_widesearch_eval_logic}")
        self.load_dataset()
        self.task_info = self.reset(task_id=config.get("task_id") if config else None)

    def load_dataset(self):
        """Load a JSONL dataset and build an instance_id -> item mapping."""
        if not os.path.exists(self.data_file):
            raise FileNotFoundError(f"数据集文件不存在: {self.data_file}")
        self.dataset_dict = {}
        with open(self.data_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                # The evaluation field may be a JSON string or already a dict.
                if isinstance(item.get("evaluation"), str):
                    item["evaluation"] = json.loads(item["evaluation"])
                instance_id = item["instance_id"]
                self.dataset_dict[instance_id] = item
        print(f"[EVAL] DeepWideSearch 数据集加载完成，共 {len(self.dataset_dict)} 条任务")
        print(f"[EVAL] Gold 答案目录: {self.gold_dir}")

    def reset(self, task_id=None):
        """Reset the environment and select a task randomly or by task_id."""
        self.cur_task_id = None
        if not self.dataset_dict:
            raise RuntimeError("数据集未加载，请先调用 load_dataset()")
        if task_id is not None:
            if task_id not in self.dataset_dict:
                raise ValueError(f"task_id '{task_id}' 不存在于数据集中")
            self.cur_task_id = task_id
        else:
            self.cur_task_id = random.choice(list(self.dataset_dict.keys()))
        print(f"[EVAL] 环境已重置，当前任务 ID: {self.cur_task_id}")
        item = self.dataset_dict[self.cur_task_id]
        task_info = {
            "dataset": "deepwidesearch",
            "task_id": self.cur_task_id,
            "task": item["question"],
        }
        return task_info

    def load_target_result(self, task_id: str) -> WideSearchQuery:
        """Load the gold answer for a task_id and return a WideSearchQuery object."""
        if task_id not in self.dataset_dict:
            raise ValueError(f"task_id '{task_id}' 不存在于数据集中")
        item = self.dataset_dict[task_id]
        answer_file = os.path.join(self.gold_dir, f"{task_id}.csv")
        if not os.path.exists(answer_file):
            raise FileNotFoundError(f"找不到 Gold 答案文件: {answer_file}")
        evaluation = item["evaluation"]
        required_columns = evaluation.get("required", [])
        answer_df = pd.read_csv(answer_file)
        # Keep norm_column consistent with evaluation.py (strip + lower + remove spaces).
        answer_df.columns = [norm_column(c) for c in answer_df.columns]
        answer_df = answer_df[[c for c in required_columns if c in answer_df.columns]]
        return WideSearchQuery(
            instance_id=task_id,
            query=item["question"],
            entity=item.get("entity", ""),
            language=item.get("language", ""),
            topic=item.get("topic", ""),
            evaluation=evaluation,
            answer=answer_df,
        )

    def _load_widesearch_answer(self, task_id: str) -> WideSearchEvaluation:
        """Adapt DeepWideSearch data to a WideSearch table-evaluation object."""
        item = self.dataset_dict[task_id]
        evaluation = item["evaluation"]
        answer_file = os.path.join(self.gold_dir, f"{task_id}.csv")
        if not os.path.exists(answer_file):
            raise FileNotFoundError(f"找不到 Gold 答案文件: {answer_file}")
        answer_table = pd.read_csv(answer_file)
        answer_table.columns = [norm_column(c) for c in answer_table.columns]
        eval_obj = {
            "unique_columns": evaluation["unique_columns"],
            "required": evaluation["required"],
            "eval_pipeline": evaluation["eval_pipeline"],
            "answer_table": answer_table,
        }
        return WideSearchEvaluation.model_validate(eval_obj)

    def evaluate(self, prediction: str) -> tuple[float, dict]:
        """
        Evaluate a prediction and return (reward, reward_info).

        prediction: The agent's final output string in Markdown-table format.
        """
        task_id = self.cur_task_id

        if self.skip_entity_eval_reuse_widesearch_eval_logic:
            # Skip entity accuracy and reuse WideSearch table evaluation directly.
            answer = self._load_widesearch_answer(task_id)

            def call_llm(template: str, kwargs: dict) -> str:
                # table_evaluate only needs a uniform callback signature; the closure injects judge settings.
                return call_claude(
                    template,
                    kwargs,
                    judge_model_name=self.judge_model_name,
                    judge_model_provider=self.judge_model_provider,
                )

            metrics = table_evaluate(prediction, answer, LLMEvalConfig(), call_llm)
            reward_info = {
                "instance_id": task_id,
                "entity_acc": -1.0,  # Mark evaluation as skipped
                **metrics.model_dump(),
            }
            reward = reward_info["f1_by_item"]
            return reward, reward_info

        # Original DeepWideSearch evaluation flow, including the entity-accuracy gate
        query = self.load_target_result(task_id)

        # Build a WideSearchResponse without messages to skip tool-call counting.
        response = WideSearchResponse(
            instance_id=task_id,
            response=prediction,
            messages=[],   # No messages, so skip tool-call statistics
        )

        # evaluate_single_query calls llm_completion internally as the LLM judge.
        eval_result = evaluate_single_query(
            query=query,
            response=response,
            judge_model_name=self.judge_model_name,
            judge_model_provider=self.judge_model_provider,
        )

        reward_info = {
            "instance_id": eval_result.instance_id,
            "score": eval_result.score,
            "entity_acc": eval_result.entity_acc,
            "precision_by_row": eval_result.precision_by_row,
            "recall_by_row": eval_result.recall_by_row,
            "f1_by_row": eval_result.f1_by_row,
            "precision_by_item": eval_result.precision_by_item,
            "recall_by_item": eval_result.recall_by_item,
            "f1_by_item": eval_result.f1_by_item,
            "column_precision": eval_result.column_precision,
            "column_recall": eval_result.column_recall,
            "column_f1": eval_result.column_f1,
            "msg": eval_result.msg,
        }
        reward = reward_info["f1_by_item"]
        return reward, reward_info
