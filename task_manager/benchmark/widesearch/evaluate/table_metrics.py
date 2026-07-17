"""
Core table-evaluation module.

Score matches between a predicted Markdown table and the gold table at multiple granularities.
Evaluation flow: parse table → align column names → normalize types → deduplicate primary keys
         → align primary-key values → preprocess/normalize → inner-join matching rows
         → score each column → compute precision/recall/F1.

Metric granularities:
  - by_unique_col: measures only primary-key hit rate (whether a row was found)
  - by_row:        counts a TP only when every column in the row is correct (strictest)
  - by_item:       aggregates per-cell scores (used as the final reward)
"""

import pandas as pd
from copy import deepcopy
from typing import Callable

from .schema import TableMetricObj, WideSearchEvaluation
from .table_utils import extract_dataframe, primary_key_preprocess
from .table_utils_metrics import llm_judge_column, metric_call
from .table_utils_norm import preprocess_call
from .schema import LLMEvalConfig


def _apply_metric(
    col: str,
    item: dict,
    df_inner: pd.DataFrame,
    df_inner_score: pd.DataFrame,
    df_inner_msg: pd.DataFrame,
    call_llm: Callable[[str, dict], str],
    tool_config: "LLMEvalConfig",
) -> None:
    """Apply a metric to one column of an inner-join result and write scores/messages to its DataFrame.

    Supports two evaluation modes:
      - Rule-based metrics (exact_match / number_near / date_near, etc.): evaluate cells independently
      - LLM judge (llm_judge): send a complete column to the LLM for batch scoring
    """
    metric_func_name_list = item.get("metric", [])
    criterion = item.get("criterion")
    for metric_func_name in metric_func_name_list:
        if df_inner.empty:
            print(f"Empty df_inner for col `{col}`!")
            df_inner_score[f"{col}_{metric_func_name}"] = pd.Series(dtype="float64", index=df_inner.index)
            df_inner_msg[f"{col}_{metric_func_name}_eval_msg"] = pd.Series(dtype="object", index=df_inner.index)
            continue

        if metric_func_name != "llm_judge":
            # Rule-based metric: call metric_call per cell and return (score, msg).
            metric_info_series = df_inner.apply(
                lambda x: metric_call(
                    x[col + "_response"],
                    x[col + "_query"],
                    criterion,
                    metric_func_name,
                ),
                axis=1,
            )
        else:
            # LLM judge: send the entire column in one batch and return score_list and msg_list.
            score_list, msg_list = llm_judge_column(
                df_inner[col + "_response"].tolist(),
                df_inner[col + "_query"].tolist(),
                criterion or "None",
                tool_config.table_eval_column_prompt,
                call_llm,
            )
            metric_info_series = pd.Series(
                zip(score_list, msg_list),
                index=df_inner.index,
            )
        df_inner_score[f"{col}_{metric_func_name}"] = metric_info_series.apply(lambda x: x[0])
        df_inner_msg[f"{col}_{metric_func_name}_eval_msg"] = metric_info_series.apply(lambda x: x[1])
        print(
            f"col {col}, metric_func_name {metric_func_name}, score {df_inner_score[f'{col}_{metric_func_name}'].tolist()}"
        )


def table_evaluate(
    prediction: str,
    answer: WideSearchEvaluation,
    tool_config: "LLMEvalConfig",
    call_llm: Callable[[str, dict], str],
) -> TableMetricObj:
    """Main table-evaluation function.

    Args:
        prediction: Raw model output containing a Markdown table
        answer: Gold-answer object with the gold table, primary-key definition, and evaluation-pipeline configuration
        tool_config: Prompt-template configuration for the LLM judge
        call_llm: LLM call function with signature (prompt_template, kwargs) -> str

    Returns:
        TableMetricObj: Precision/recall/F1 at three granularities, per-column scores, and warnings
    """
    score = 0.0
    precision_by_row = 0.0
    recall_by_row = 0.0
    f1_by_row = 0.0
    precision_by_item = 0.0
    recall_by_item = 0.0
    f1_by_item = 0.0

    try:
        required_columns = answer.required
        unique_columns = answer.unique_columns
        answer_df = answer.answer_table.copy(deep=True)
        eval_warnings: list[str] = []

        # ── Step 1: Determine the evaluation column set ───────────────────
        eval_columns = required_columns

        # ── Step 2: Extract a Markdown table from the prediction text ─────
        prediction_df = extract_dataframe(prediction)
        if prediction_df is None:
            msg = "Markdown table in Prediction not found"
            print(f"msg: {msg} | prediction: {prediction}")
            return TableMetricObj(msg=msg, warnings=["No Markdown table found in the prediction"])

        # ── Step 3: Align column names semantically ───────────────────────
        # Predicted and gold column names may use different wording (such as "City" vs "Host City");
        # use an LLM to create a semantic rename mapping.
        if set(eval_columns) != set(prediction_df.columns):
            column_map, col_warnings = primary_key_preprocess(
                prediction_df.columns.to_list(),
                eval_columns,
                tool_config.table_column_align_prompt,
                call_llm,
            )
            eval_warnings.extend([f"[Column alignment] {w}" for w in col_warnings])
            print(f"column map: {column_map}")
            print(f"befor mapping: {prediction_df.columns}")
            prediction_df.rename(columns=column_map, inplace=True)
            print(f"after mapping: {prediction_df.columns}")

        if set(eval_columns) != set(prediction_df.columns):
            msg = f"eval_columns {eval_columns} != prediction_df {prediction_df.columns}"
            print(f"msg: {msg} | prediction: {prediction}")
            eval_warnings.append(f"Column alignment failed: {msg}")
            return TableMetricObj(msg=msg, warnings=eval_warnings)

        # ── Step 4: Convert all types to strings ──────────────────────────
        # Convert every column to str to eliminate int/float representation differences (such as 1 vs 1.0).
        for col in eval_columns:
            try:
                answer_type = answer_df[col].dtype
                response_type = prediction_df[col].dtype
            except Exception:
                answer_type = None
                response_type = None
            if (response_type == float and answer_type == int) or (response_type == int and answer_type == float):
                if response_type == int:
                    prediction_df[col] = prediction_df[col].astype(float)
                elif answer_type == int:
                    answer_df[col] = answer_df[col].astype(float)

            answer_df[col] = answer_df[col].astype(str)
            prediction_df[col] = prediction_df[col].astype(str)

        # ── Step 5: Deduplicate by primary key before mapping ─────────────
        # The predicted table may contain identical rows; deduplicate to prevent a Cartesian product in the inner join.
        prediction_df.drop_duplicates(subset=unique_columns, inplace=True)
        answer_df.drop_duplicates(subset=unique_columns, inplace=True)

        # ── Step 6: Align primary-key values semantically ─────────────────
        # Predicted primary-key values may differ in format from gold values (such as "Feb 4, 2010" vs "4 February, 2010");
        # use an LLM to build a prediction_value → gold_value mapping.
        for col in unique_columns:
            item = answer.eval_pipeline.get(col, None)
            if item is None:
                continue
            metric_func_name_list = item.get("metric", [])
            if "llm_judge" in metric_func_name_list or "exact_match" in metric_func_name_list:
                primary_key_map, pk_warnings = primary_key_preprocess(
                    prediction_df[col].tolist(),
                    answer_df[col].tolist(),
                    tool_config.table_column_align_prompt,
                    call_llm,
                )
                eval_warnings.extend([f"[Primary key alignment:{col}] {w}" for w in pk_warnings])
                print(f"col: {col}, primary_key_map {primary_key_map}")
                prediction_df[col + "_before_map"] = prediction_df[col]
                prediction_df[col] = prediction_df[col].apply(lambda x: primary_key_map.get(x, x))

        # ── Step 7: Deduplicate by primary key after mapping ──────────────
        # Mapping may be many-to-one, with several predictions pointing to one gold row, so deduplicate again
        # to prevent the inner join from producing more rows than the gold table.
        prediction_df.drop_duplicates(subset=unique_columns, inplace=True)

        # ── Step 8: Normalize values through preprocessing ────────────────
        # Apply functions such as norm_str, norm_date, and extract_number to each column
        # to eliminate superficial differences in case, whitespace, date formats, and so on.
        for col, item in answer.eval_pipeline.items():
            if col not in eval_columns:
                continue
            preprocess_func_name_list = item.get("preprocess", [])
            for preprocess_func_name in preprocess_func_name_list:
                prediction_df[col] = prediction_df[col].apply(lambda x: preprocess_call(x, preprocess_func_name))
                answer_df[col] = answer_df[col].apply(lambda x: preprocess_call(x, preprocess_func_name))

        # ── Step 9: Fast exact-match check ─────────────────────────────────
        # Compare sorted elements and set score=1.0 for an exact match (score field only; does not affect F1).
        temp_score = 0.0
        if answer_df[eval_columns].shape == prediction_df[eval_columns].shape:
            gt_sorted = answer_df[eval_columns].sort_values(by=eval_columns).reset_index(drop=True)
            pred_sorted = prediction_df[eval_columns].sort_values(by=eval_columns).reset_index(drop=True)
            if gt_sorted.equals(pred_sorted):
                temp_score = 1.0
        score = temp_score

        # ── Step 10: Inner join on primary keys to obtain matching rows ───
        # Inner-join rows are entities shared by the prediction and gold tables (TP candidates).
        df_inner = pd.merge(
            answer_df,
            prediction_df,
            on=unique_columns,
            how="inner",
            suffixes=("_query", "_response"),
        )

        # Use an outer join for detailed reporting to identify FN and FP rows.
        answer_df_outer = deepcopy(answer_df)
        answer_df_outer["exist_flag_gt"] = 1
        response_df_outer = deepcopy(prediction_df)
        response_df_outer["exist_flag_response"] = 1

        df_outer = pd.merge(
            answer_df_outer,
            response_df_outer,
            on=unique_columns,
            how="outer",
            suffixes=("_query", "_response"),
        )
        df_outer_wo_inner = df_outer[df_outer["exist_flag_gt"].isna() | df_outer["exist_flag_response"].isna()]

        print(
            f"df_inner shape: {df_inner.shape}, "
            f"answer_df shape: {answer_df.shape}, response_df shape: {prediction_df.shape}"
        )
        if len(df_inner) == 0:
            eval_warnings.append(
                f"Primary-key match count is 0 (prediction rows={len(prediction_df)}, ground-truth rows={len(answer_df)}), "
                f"possibly due to a failed primary-key alignment mapping or truncated LLM output"
            )

        # ── Step 11: Compute cell-level match scores by column ────────────
        # Score each column in the inner join with the metric configured in eval_pipeline.
        # Primary-key columns already match through the join, so assign them 1.0 directly.
        df_inner_score = pd.DataFrame(index=df_inner.index)
        df_inner_msg = pd.DataFrame(index=df_inner.index)
        for col in eval_columns:
            if col in unique_columns:
                df_inner_score[f"{col}_exact_match"] = 1.0
                df_inner_msg[f"{col}_exact_match_eval_msg"] = "key_match"
                continue

            item = answer.eval_pipeline[col]
            _apply_metric(
                col=col,
                item=item,
                df_inner=df_inner,
                df_inner_score=df_inner_score,
                df_inner_msg=df_inner_msg,
                call_llm=call_llm,
                tool_config=tool_config,
            )

        # ── Step 12: Save a detailed comparison CSV (optional) ────────────
        if answer.detail_save_path is not None:
            result_df = pd.concat([df_inner, df_inner_score, df_inner_msg], axis=1)
            result_df = pd.concat([result_df, df_outer_wo_inner])
            result_columns = result_df.columns.tolist()
            key_cols = (
                unique_columns
                + [col + "_before_map" for col in unique_columns]
                + ["exist_flag_gt", "exist_flag_response"]
            )

            cols1 = sorted([col for col in result_columns if col in key_cols])
            cols2 = sorted([col for col in result_columns if col not in key_cols])
            result_df = result_df[cols1 + cols2]
            result_df.to_csv(answer.detail_save_path, index=False)

        # ── Step 13: Compute Precision / Recall / F1 at three granularities ──
        #
        # Definitions:
        #   N_pred = number of rows in the deduplicated prediction table
        #   N_gt   = number of rows in the deduplicated gold table
        #   N_inner = number of matching inner-join rows
        #   C      = number of eval_columns
        #
        # by_row (row level, strictest):
        #   Take the minimum score across columns in each row; one incorrect column makes the row score 0.
        #   TP_row = sum(min(row_scores))
        #   P_row = TP_row / N_pred,  R_row = TP_row / N_gt
        #
        # by_item (cell level, final reward):
        #   TP_item = sum of all cell scores
        #   P_item = TP_item / (N_pred * C),  R_item = TP_item / (N_gt * C)
        #
        # by_unique_col (primary-key level; checks only whether the row was found):
        #   TP_unique = N_inner
        #   P_unique = N_inner / N_pred,  R_unique = N_inner / N_gt

        row_scores = df_inner_score.min(axis=1)
        tp_by_row = row_scores.sum()
        tp_by_item = df_inner_score.sum().sum()

        num_pred_rows = len(prediction_df)
        num_gt_rows = len(answer_df)
        num_pred_items = num_pred_rows * len(eval_columns)
        num_gt_items = num_gt_rows * len(eval_columns)

        precision_by_row = tp_by_row / num_pred_rows if num_pred_rows > 0 else 0.0
        recall_by_row = tp_by_row / num_gt_rows if num_gt_rows > 0 else 0.0

        precision_by_item = tp_by_item / num_pred_items if num_pred_items > 0 else 0.0
        recall_by_item = tp_by_item / num_gt_items if num_gt_items > 0 else 0.0

        def calc_f1(precision, recall):
            epsilon = 1e-9
            return (2 * precision * recall / (precision + recall)) if (precision + recall > epsilon) else 0.0

        f1_by_row = calc_f1(precision_by_row, recall_by_row)
        f1_by_item = calc_f1(precision_by_item, recall_by_item)

        tp_by_unique_col = len(df_inner)
        precision_by_unique_col = tp_by_unique_col / num_pred_rows if num_pred_rows > 0 else 0.0
        recall_by_unique_col = tp_by_unique_col / num_gt_rows if num_gt_rows > 0 else 0.0
        f1_by_unique_col = calc_f1(precision_by_unique_col, recall_by_unique_col)

        print(f"P/R/F1 by unique_col: {precision_by_unique_col:.4f}/{recall_by_unique_col:.4f}/{f1_by_unique_col:.4f}")
        print(f"P/R/F1 by row: {precision_by_row:.4f}/{recall_by_row:.4f}/{f1_by_row:.4f}")
        print(f"P/R/F1 by item: {precision_by_item:.4f}/{recall_by_item:.4f}/{f1_by_item:.4f}")

        dict_msg = {col: ", ".join(df_inner_score[col].astype(str).tolist()) for col in df_inner_score.columns}
        if (
            precision_by_item == recall_by_item == f1_by_item == 1.0
            and precision_by_row == recall_by_row == f1_by_row == 1.0
        ):
            dict_msg["All items match perfectly."] = "1"
            score = 1

        print(f"final table score: {score}")
        return TableMetricObj(
            msg=dict_msg,
            warnings=eval_warnings,
            score=score,
            precision_by_row=precision_by_row,
            recall_by_row=recall_by_row,
            f1_by_row=f1_by_row,
            precision_by_item=precision_by_item,
            recall_by_item=recall_by_item,
            f1_by_item=f1_by_item,
            precision_by_unique_col=precision_by_unique_col,
            recall_by_unique_col=recall_by_unique_col,
            f1_by_unique_col=f1_by_unique_col,
        )

    except Exception:
        import traceback
        err_msg = traceback.format_exc()
        print(err_msg)
        print("Unexpected error in table evaluation")
        return TableMetricObj(msg=f"Evaluator error:\n{err_msg}", warnings=[f"Evaluation error: {err_msg}"])
