"""Local data-loading and rule-based evaluation entry point for the GISA benchmark.

GISA evaluation environment (General Information-Seeking Assistant Benchmark)
Data source: Hugging Face RUC-NLPIR/GISA
  - Question file: gisa/data/question.jsonl
  - Answer files: gisa/data/answers/{qid}.csv

Evaluation metrics:
  - item:  item_em (exact match)
  - set:   set_f1 / set_precision / set_recall
  - list:  list_content_f1 / list_order_score
  - table: table_row_f1 / table_item_f1, etc.
  - global: global_em

Data format:
  The question file contains one JSON object per line with these fields:
    id           (int)  Question ID
    question     (str)  Question text
    answer_type  (str)  item / set / list / table
    question_type (str) stable / live
    topic        (str)  Topic

This file does not call an LLM; all rewards are computed by local rule-based metrics.
"""
import json
import os
import random
import re
import difflib
from collections import Counter
from io import StringIO
from typing import Optional, Tuple, Union

import pandas as pd
import numpy as np

from .base_evaluator import BaseEvaluator


# ---------------------------------------------------------------------------
# Core evaluation logic (ported inline from GISA/eval_script/run_evaluation.py)
# ---------------------------------------------------------------------------

class _SimpleEvaluator:
    """Original GISA evaluator ported from eval_script/run_evaluation.py."""

    def _normalize_val(self, val: Union[str, int, float]) -> str:
        """Normalize answer values into strings suitable for exact comparison."""
        val_str = str(val).strip()
        if not val_str or val_str.lower() in ['nan', 'none', 'null']:
            return ""

        # First try numeric handling to normalize dollar signs, thousands separators, and percent signs.
        clean_num = val_str.replace(',', '').replace('$', '')
        is_percent = False
        if clean_num.endswith('%'):
            is_percent = True
            clean_num = clean_num[:-1]

        try:
            f_val = float(clean_num)
            if is_percent:
                f_val /= 100.0
            if f_val.is_integer():
                return str(int(f_val))
            else:
                formatted = "{:.6f}".format(f_val).rstrip('0').rstrip('.')
                return formatted if formatted else "0"
        except ValueError:
            pass

        # Normalize nonnumeric content as strings to eliminate formatting differences where possible.
        normed = val_str.lower().replace(" ", "").replace("*", "").replace("\n", "")
        return normed

    def _extract_model_output(self, model_output: str) -> Optional[pd.DataFrame]:
        """Extract a TSV table from model output and normalize its column names and values."""
        pattern = r"```(?:tsv)?\s*(.*?)```"
        match = re.search(pattern, model_output, re.DOTALL)
        if match:
            raw_content = match.group(1)
        else:
            raw_content = model_output
        try:
            # GISA requires TSV; remove empty lines here before parsing with pandas.
            raw_content = "\n".join([line for line in raw_content.split('\n') if line.strip()])
            if not raw_content:
                return None
            output = pd.read_csv(StringIO(raw_content), sep="\t")
            output.columns = [str(col).strip().lower().replace(" ", "") for col in output.columns]
            output = output.map(self._normalize_val)
        except Exception as e:
            print(f"Extract Error: {e}")
            output = None
        return output

    def _load_gt(self, file_path: str, question_type: str) -> pd.DataFrame:
        """Read a gold-answer CSV and decide whether to infer headers based on question type."""
        # Preserve CSV headers for table questions; read item/set/list answer files without headers.
        header = 'infer' if question_type == 'table' else None
        try:
            df = pd.read_csv(file_path, header=header)
        except Exception as e:
            if 'codec' in str(e):
                df = pd.read_csv(file_path, header=header, encoding='gbk')
            else:
                raise e
        df.columns = [str(col).strip().lower().replace(" ", "") for col in df.columns]
        df = df.map(self._normalize_val)
        return df

    def _calculate_f1(self, tp: int, n_pred: int, n_gt: int) -> Tuple[float, float, float]:
        """Compute precision, recall, and F1 from TP, prediction count, and gold-answer count."""
        precision = tp / n_pred if n_pred > 0 else 0.0
        recall = tp / n_gt if n_gt > 0 else 0.0
        if (precision + recall) == 0:
            f1 = 0.0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)
        return precision, recall, f1

    def _flatten_table(self, df: pd.DataFrame):
        """Flatten a table into (column name, cell value) pairs for cell-level matching."""
        items = []
        for col in df.columns:
            for val in df[col]:
                items.append((col, val))
        return items

    def evaluate_item(self, pred_df, gt_df):
        """Evaluate a single-value answer by concatenating all cells and applying exact match."""
        if pred_df is None or pred_df.empty:
            return {"item_em": 0}
        pred_item = "".join(pred_df.iloc[0, :].tolist())
        gt_item = "".join(gt_df.iloc[0, :].tolist())
        return {"item_em": int(pred_item == gt_item)}

    def evaluate_set(self, pred_df, gt_df):
        """Evaluate a set answer using only the set formed by the final column."""
        if pred_df is None or pred_df.empty:
            return {"set_precision": 0.0, "set_recall": 0.0, "set_f1": 0.0}
        pred_set = set(pred_df.iloc[:, -1].tolist())
        gt_set = set(gt_df.iloc[:, -1].tolist())
        tp = len(pred_set.intersection(gt_set))
        p, r, f1 = self._calculate_f1(tp, len(pred_set), len(gt_set))
        return {"set_precision": p, "set_recall": r, "set_f1": f1}

    def evaluate_list(self, pred_df, gt_df):
        """Evaluate a list answer using both content F1 and order similarity."""
        if pred_df is None or pred_df.empty:
            return {"list_content_f1": 0.0, "list_order_score": 0.0}
        pred_list = pred_df.iloc[:, -1].tolist()
        gt_list = gt_df.iloc[:, -1].tolist()
        gt_counter = Counter(gt_list)
        pred_counter = Counter(pred_list)
        intersection = gt_counter & pred_counter
        num_common = sum(intersection.values())
        len_gt = len(gt_list)
        len_pred = len(pred_list)
        precision = num_common / len_pred if len_pred > 0 else 0.0
        recall = num_common / len_gt if len_gt > 0 else 0.0
        content_f1 = (2 * precision * recall / (precision + recall)
                      if (precision + recall) > 0 else 0.0)
        matcher = difflib.SequenceMatcher(None, gt_list, pred_list)
        order_score = matcher.ratio()
        return {
            "list_content_f1": round(content_f1, 4),
            "list_order_score": round(order_score, 4),
        }

    def evaluate_table(self, pred_df, gt_df):
        """Evaluate a table answer with separate row-level and cell-level metrics."""
        default = {
            "table_row_f1": 0.0, "table_row_precision": 0.0, "table_row_recall": 0.0,
            "table_item_f1": 0.0, "table_item_precision": 0.0, "table_item_recall": 0.0,
        }
        if pred_df is None or pred_df.empty:
            return default
        common_cols = [c for c in gt_df.columns if c in pred_df.columns]
        if not common_cols:
            row_p, row_r, row_f1 = 0.0, 0.0, 0.0
        else:
            pred_rows = set(tuple(r) for r in pred_df[common_cols].fillna('__NAN__').astype(str).to_numpy())
            gt_rows = set(tuple(r) for r in gt_df[common_cols].fillna('__NAN__').astype(str).to_numpy())
            tp_rows = len(pred_rows.intersection(gt_rows))
            row_p, row_r, row_f1 = self._calculate_f1(tp_rows, len(pred_rows), len(gt_rows))

        pred_items = self._flatten_table(pred_df)
        gt_items = self._flatten_table(gt_df)
        pred_counter = Counter(pred_items)
        gt_counter = Counter(gt_items)
        intersection = pred_counter & gt_counter
        tp_items = sum(intersection.values())
        item_p, item_r, item_f1 = self._calculate_f1(
            tp_items, sum(pred_counter.values()), sum(gt_counter.values())
        )
        return {
            "table_row_f1": row_f1, "table_row_precision": row_p, "table_row_recall": row_r,
            "table_item_f1": item_f1, "table_item_precision": item_p, "table_item_recall": item_r,
        }

    def evaluate_one(self, prediction: str, gt_path: str, question_type: str, qid=None) -> dict:
        """Dispatch by question type to the appropriate evaluator and add a global exact-match metric."""
        pred_df = self._extract_model_output(prediction)
        if pred_df is None:
            print(f"qid:{qid} prediction is empty or unparseable")

        gt_df = self._load_gt(gt_path, question_type.lower())
        q_type = question_type.lower()

        # Different answer_type values use different primary metrics, but produce the full metric dict here.
        if q_type == 'item':
            metrics = self.evaluate_item(pred_df, gt_df)
        elif q_type == 'set':
            metrics = self.evaluate_set(pred_df, gt_df)
        elif q_type == 'list':
            metrics = self.evaluate_list(pred_df, gt_df)
        elif q_type == 'table':
            metrics = self.evaluate_table(pred_df, gt_df)
        else:
            print(f"Unknown question type: {question_type}, treating as item")
            metrics = self.evaluate_item(pred_df, gt_df)

        if pred_df is not None:
            if q_type != 'set':
                metrics['global_em'] = int(np.array_equal(pred_df.to_numpy(), gt_df.to_numpy()))
            else:
                pred_set = set(pred_df.iloc[:, 0].tolist())
                gt_set = set(gt_df.iloc[:, 0].tolist())
                metrics['global_em'] = int(pred_set == gt_set)
        else:
            metrics['global_em'] = 0

        metrics['question_type'] = question_type
        return metrics


# ---------------------------------------------------------------------------
# Main evaluator
# ---------------------------------------------------------------------------

class GISAEval(BaseEvaluator):
    """GISA evaluation environment.

    Data-directory layout (task_manager/benchmark/gisa/data/):
        question.jsonl          — Question file
        answers/{qid}.csv       — Gold answer for each question
    """

    # Mapping from version names to question filenames.
    VERSION_MAP = {
        "all": "question.jsonl",
    }

    def __init__(self, config: dict = None):
        super().__init__(config)
        self.version = (config or {}).get("version", "all")
        if self.version not in self.VERSION_MAP:
            raise ValueError(f"Unsupported version '{self.version}'; available: {list(self.VERSION_MAP.keys())}")
        self.data_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "gisa", "data"
        )
        self.answers_dir = os.path.join(self.data_dir, "answers")
        self._evaluator = _SimpleEvaluator()
        self.cur_task_id = None
        self.dataset_dict: dict = {}
        self.load_dataset()
        self.task_info = self.reset(task_id=config.get("task_id") if config else None)

    # ── Data loading ──────────────────────────────────────────────────────

    def load_dataset(self):
        """Load the question file corresponding to version."""
        question_file = self.VERSION_MAP[self.version]
        question_path = os.path.join(self.data_dir, question_file)

        if not os.path.exists(question_path):
            raise FileNotFoundError(
                f"Question file '{question_file}' not found; place it at: {question_path}"
            )
        print(f"[GISA] Loading question file (version={self.version}): {question_path}")

        self.dataset_dict = {}
        with open(question_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                qid = f"gisa_{item['id']}"
                self.dataset_dict[qid] = item

        print(f"[GISA] Dataset loaded: {len(self.dataset_dict)} tasks")
        print(f"[GISA] Answer directory: {self.answers_dir}")

    # ── Reset ─────────────────────────────────────────────────────────────

    def reset(self, task_id=None) -> dict:
        """Reset the current GISA sample and return task information visible to the agent."""
        self.cur_task_id = None
        if not self.dataset_dict:
            raise RuntimeError("Dataset is not loaded")
        if task_id is not None:
            task_id = str(task_id)
            if task_id not in self.dataset_dict:
                raise ValueError(f"task_id '{task_id}' does not exist in the dataset")
            self.cur_task_id = task_id
        else:
            self.cur_task_id = random.choice(list(self.dataset_dict.keys()))

        item = self.dataset_dict[self.cur_task_id]
        task_info = {
            "dataset": "gisa",
            "task_id": self.cur_task_id,
            "task": item["question"],
            "answer_type": item["answer_type"],
            "question_type": item.get("question_type", ""),
            "topic": item.get("topic", ""),
        }
        print(f"[GISA] Current task ID: {self.cur_task_id} | Type: {item['answer_type']}")
        return task_info

    # ── Evaluation ────────────────────────────────────────────────────────

    def evaluate(self, prediction: str) -> tuple[float, dict]:
        """Evaluate a prediction.

        Args:
            prediction: The agent's final output string.
                - For table answers, it should contain a TSV code block (```tsv ... ```).
                - item/set/list answers also use TSV format with one or more columns.

        Returns:
            (reward, reward_info)
            reward is the primary metric:
              - table: table_item_f1
              - set:   set_f1
              - list:  list_content_f1
              - item:  item_em (float)
        """
        task_id = self.cur_task_id
        item = self.dataset_dict[task_id]
        answer_type = item["answer_type"].lower()

        # Answer filenames use the dataset's original numeric ID, not the wrapped gisa_xxx form.
        raw_id = item["id"]
        gt_path = os.path.join(self.answers_dir, f"{raw_id}.csv")
        if not os.path.exists(gt_path):
            raise FileNotFoundError(f"Answer file not found: {gt_path}")

        metrics = self._evaluator.evaluate_one(
            prediction=prediction,
            gt_path=gt_path,
            question_type=answer_type,
            qid=task_id,
        )

        # Use the most representative metric per answer type as the reward; retain all metrics in reward_info.
        reward_map = {
            "table": metrics.get("table_item_f1", 0.0),
            "set":   metrics.get("set_f1", 0.0),
            "list":  metrics.get("list_content_f1", 0.0),
            "item":  float(metrics.get("item_em", 0)),
        }
        reward = reward_map.get(answer_type, float(metrics.get("global_em", 0)))

        reward_info = {
            "task_id": task_id,
            "answer_type": answer_type,
            **metrics,
        }
        return reward, reward_info
