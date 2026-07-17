"""Data loader and LLM-judge evaluator for the BrowseComp Plus benchmark.

This evaluator retrieves the question/answer by task_id and asks the dedicated
BrowseComp Plus LLM judge whether the final answer is correct.
"""

import json
import os
import random


from .base_evaluator import BaseEvaluator
from .browsecomp_plus.llm_judge import browsecomp_plus_llm_judge



class BrowseCompPlusEval(BaseEvaluator):
    """BrowseComp Plus evaluation environment with an interface matching WideSearchEval."""

    def __init__(self, config: dict = None):
        super().__init__(config)
        config = config or {}
        # TaskManager injects the evaluation model; this evaluator has no separate model switch.
        self.judge_model_name = config["judge_model_name"]
        self.judge_model_provider = config["judge_model_provider"]
        self.version = config["version"]

        # Dataset
        self.dataset_dict: dict[str, dict] = {}
        self.load_dataset()

        # Current task state
        self.cur_task_id = None
        self.cur_question = None
        self.cur_answer = None
        self.task_info = self.reset(task_id=config.get("task_id"))

    def load_dataset(self) -> None:
        """Load the BrowseComp Plus dataset and build a task_id -> item mapping."""

        version_map = {
            "bc_all": "bc.jsonl",
            "all": "bc_plus.jsonl",
            "plus_subset": "bc_plus_subset.jsonl",
        }        
        if self.version not in version_map:
            raise ValueError(f"Unsupported BrowseComp Plus version: {self.version}")
        
        file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "browsecomp_plus/data", version_map[self.version])
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"找不到 BrowseComp Plus 数据集文件: {file_path}")
    
        with open(file_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        dataset_list = [json.loads(line) for line in lines]
        self.dataset_dict = {item['task_id']: item for item in dataset_list}
        print(f"[BROWSECOMP_PLUS] BrowseComp Plus 数据集加载完成，共 {len(self.dataset_dict)} 条任务")


    def reset(self, task_id: str = None) -> dict:
        """Reset the environment and select a task randomly or by task_id."""
        # Clear current task state so a new task cannot inherit the previous question/answer.
        self.cur_task_id = None
        self.cur_question = None
        self.cur_answer = None

        if not self.dataset_dict:
            raise RuntimeError("数据集未加载，请先调用 load_dataset()")
        if task_id is not None:
            if task_id not in self.dataset_dict:
                raise ValueError(f"task_id '{task_id}' 不存在于数据集中")
            self.cur_task_id = task_id
        else:
            self.cur_task_id = random.choice(list(self.dataset_dict.keys()))
        item = self.dataset_dict[self.cur_task_id]
        self.cur_question = item["question"]
        self.cur_answer = item["answer"]
        print(f"[BROWSECOMP_PLUS] 环境已重置，当前任务 ID: {self.cur_task_id}")

        return {
            "dataset": "browsecomp_plus",
            "task_id": self.cur_task_id,
            "task": self.cur_question,
            "additional_info": {
                "answer": self.cur_answer,
            }
        }

    def evaluate(self, prediction: str) -> tuple[float, dict]:
        """Evaluate the prediction for the current task."""
        if self.cur_task_id is None:
            raise RuntimeError("请先调用 reset() 选择一条任务")
        # The BrowseComp Plus reward comes directly from the LLM judge's yes/no decision.
        result = browsecomp_plus_llm_judge(
            self.cur_question,
            prediction,
            self.cur_answer,
            judge_model_name=self.judge_model_name,
            judge_model_provider=self.judge_model_provider,
        )
        reward = float(result["llm_equal"])
        return reward, result
