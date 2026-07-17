"""Data models and prompt-template configuration for WideSearch table evaluation.

This module stores evaluation input/output structures and the prompt templates used for
table-column alignment and column-level LLM judging. TaskManager supplies the actual judge model.
"""

from pydantic import BaseModel, Field, ConfigDict, field_validator, field_serializer
import pandas as pd
from typing import Annotated, Any, Literal
from enum import Enum


class WideSearchEvaluation(BaseModel):
    """Gold table, evaluation columns, and metric-pipeline configuration for one WideSearch query."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    unique_columns: list[str] = Field(..., description="唯一列")
    required: list[str] = Field(..., description="必填列")
    eval_pipeline: dict[str, dict[str, Any]] = Field(..., description="评估管道")
    answer_table: pd.DataFrame = Field(..., description="标准答案表格")
    detail_save_path: str | None = Field(None, description="详细结果保存路径")

    @field_validator("answer_table", mode="before")
    @classmethod
    def validate_dataframe(cls, v: Any) -> pd.DataFrame:
        """Restore a two-dimensional array to a DataFrame for evaluating intermediate results."""
        if isinstance(v, pd.DataFrame):
            return v
        if isinstance(v, list):
            if len(v) == 0:
                return pd.DataFrame()
            columns = v[0]
            data = v[1:] if len(v) > 1 else []
            return pd.DataFrame(data, columns=columns)
        raise ValueError(f"answer_table必须是DataFrame或二维数组，当前类型: {type(v)}")

    @field_serializer("answer_table")
    def serialize_dataframe(self, df: pd.DataFrame) -> list[list[Any]]:
        """Serialize a DataFrame to a two-dimensional array with column names in the first row."""
        if df is None or df.empty:
            return []
        result = [df.columns.tolist()]
        result.extend(df.values.tolist())
        return result


class TableMetricObj(BaseModel):
    """Evaluation result for a table answer, covering row-, cell-, and primary-key-level metrics."""

    msg: str | dict[str, str] = Field(default="", description="评估结果注释")
    warnings: list[str] = Field(default_factory=list, description="评估过程中的警告信息")
    score: float = Field(default=0.0, description="评估得分")
    precision_by_row: float = Field(default=0.0, description="行级精确率")
    recall_by_row: float = Field(default=0.0, description="行级召回率")
    f1_by_row: float = Field(default=0.0, description="行级F1分数")
    precision_by_item: float = Field(default=0.0, description="单元格级精确率")
    recall_by_item: float = Field(default=0.0, description="单元格级召回率")
    f1_by_item: float = Field(default=0.0, description="单元格级F1分数")
    precision_by_unique_col: float = Field(default=0.0, description="主键列级精确率")
    recall_by_unique_col: float = Field(default=0.0, description="主键列级召回率")
    f1_by_unique_col: float = Field(default=0.0, description="主键列级F1分数")


class LLMEvalConfig:
    """Prompt-template configuration used by the LLM judge.

    Note: this class stores only prompt templates, not a model name, provider, or key.
    """

    table_column_align_prompt: str = """Your task is to align two vocabularies. The inputs are the vocabulary to be aligned and the reference vocabulary respectively. Note that you need to perform semantic alignment (not positional alignment). If two strings are exactly the same, they must correspond to each other. These two strings are supposed to represent the same entity, with differences only in the expression forms and formats.


The vocabulary to be aligned is as follows:
{response}

The reference vocabulary is as follows:
{reference}

The alignment rules are as follows:
List the values in the vocabulary to be aligned one by one. If there is a value in the reference vocabulary that has the same meaning as this value, `transform` should be represented as the value from the reference vocabulary; otherwise, `transform` should be represented as the original value from the vocabulary to be aligned.

Note that `origin` must be taken from the vocabulary to be aligned keeping the original format, and `transform` must be taken from the reference vocabulary. For example: Some words in the vocabulary to be aligned might be the words in the reference vocabulary with Markdown formatting added, keep the to be aligned format in `origin` and the reference format in `transform`.

For the `origin`, first find the `transform` that is the closest in meaning and then judge whether they correspond to each other. Those entities not correspond to each other could not output.

Please output the alignment results in the following format:
```json
{{
    "origin_str1": "transform_str1",
    "origin_str2": "transform_str2"
}}
```"""
    table_eval_column_prompt: str = """You are an expert in grading answers. Your task is to score the responses to a certain question. Below, you will be provided with a set of standard answers, a set of responses to be graded, and specific grading criteria.

Each answer and each response has an idx. Please score each pair of answers and responses in this set according to the following methods:
1. The scoring range is from 0 to 1. A score of 1 indicates a completely correct answer. For deduction items, please refer to the specific grading criteria section.
2. After reading the standard answers, responses to be graded, and grading criteria, please first analyze and judge them item by item according to the grading criteria.
3. The score can only be an integer of 0 or 1.
4. After the analysis and judgment, please provide the final scoring results. Each pair should have a score. Output in Markdown JSON format, as shown below:
```json
{{
    "idx_xxx": score,
    "idx_yyy": score,
    ...
}}
```

====== criterion-start ======
{criterion}
====== criterion-end ======

====== response-start ======
{response}
====== response-end ======

Now start scoring. Please make sure to analyze each item step by step before providing the final scoring results."""
