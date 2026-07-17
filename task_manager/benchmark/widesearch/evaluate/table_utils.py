"""
Table-parsing and primary-key-alignment utilities.

Provides three categories of functionality:
  1. Markdown table parsing: extract a structured DataFrame from raw LLM output
  2. JSON-block extraction: extract and parse a ```json ... ``` block from an LLM response
  3. Semantic primary-key/column alignment: use an LLM to map predicted terms to reference terms
"""

import re
import json
import pandas as pd
from io import StringIO
from typing import Callable


def _fallback_parse_markdown_table(lines: list[str]) -> pd.DataFrame | None:
    """Fallback parser that splits on pipes and merges excess fields when cell pipes break pd.read_csv."""
    data_lines: list[list[str]] = []
    for line in lines:
        parts = line.split("|")
        # Remove empty fields introduced by leading or trailing pipes.
        if parts and parts[0].strip() == "":
            parts = parts[1:]
        if parts and parts[-1].strip() == "":
            parts = parts[:-1]
        parts = [p.strip() for p in parts]
        data_lines.append(parts)

    if not data_lines:
        return None

    # Determine the expected column count from the header.
    expected_cols = len(data_lines[0])
    parsed_rows: list[list[str]] = []
    for parts in data_lines:
        # Pipes inside some cells create excess fields, which must be merged.
        while len(parts) > expected_cols:
            merge_pos = max(len(parts) - 2, 1)
            parts[merge_pos - 1] = parts[merge_pos - 1] + " | " + parts[merge_pos]
            parts.pop(merge_pos)
        parsed_rows.append(parts)

    if len(parsed_rows) > 1:
        return pd.DataFrame(parsed_rows[1:], columns=parsed_rows[0])
    else:
        return pd.DataFrame(columns=parsed_rows[0])


def extract_dataframe(prediction: str) -> pd.DataFrame | None:
    """Extract a Markdown table from a raw LLM response and convert it to a DataFrame.

    Parsing strategy, in priority order:
      1. Prefer a table inside a ```markdown ... ``` code block
      2. Without a code block, search the full text for consecutive pipe rows containing at least four pipes
      3. Remove separator rows containing only pipes, hyphens, colons, and spaces, then parse with pd.read_csv
      4. If pipes inside cells break pd.read_csv, fall back to _fallback_parse_markdown_table
    """
    response_df = None
    # Strategy 1: match a ```markdown ... ``` code block.
    markdown_str_list: list[str] = re.findall(r"```markdown(.*?)```", prediction, re.DOTALL)
    if not markdown_str_list:
        # Strategy 2: find a sequence of pipe-delimited lines in the full text.
        pipe_positions = [m.start() for m in re.finditer(r"\|", prediction)]
        if len(pipe_positions) >= 4:
            first_pipe = pipe_positions[0]
            last_pipe = pipe_positions[-1]
            start = prediction.rfind("\n", 0, first_pipe)
            start = 0 if start == -1 else start
            end = prediction.find("\n", last_pipe)
            end = len(prediction) if end == -1 else end
            table_candidate = prediction[start:end]
            markdown_str_list = re.findall(r"((?:\|.*\n?)+)", table_candidate)
    if markdown_str_list:
        print(f"find markdown_str {markdown_str_list[0][:64]} ...")
        markdown_str = markdown_str_list[0].strip()
        lines = markdown_str.split("\n")
        lines = [line.strip() for line in lines]
        # Filter separator rows (containing only pipes, hyphens, colons, and spaces) and rows without pipes.
        new_lines: list[str] = []
        for line in lines:
            if set(line.strip()).issubset(set("|- :")) or "|" not in line:
                continue
            new_lines.append("|".join([_line.strip() for _line in line.split("|")]))
        markdown_str = "\n".join(new_lines)
        try:
            response_df = pd.read_csv(StringIO(markdown_str), sep="|")
            response_df = response_df.loc[:, ~response_df.columns.str.startswith("Unnamed")]
            # Validation: NaN in columns or the index indicates parsing issues caused by pipes inside rows.
            if response_df.columns.isna().any() or response_df.index.isna().any():
                raise ValueError("pd.read_csv 解析结果存在列偏移")
        except Exception as e:
            print(f"[WARN] pd.read_csv 解析失败({e})，使用兜底解析...")
            response_df = _fallback_parse_markdown_table(new_lines)
    else:
        print(f"Markdown table in Prediction not found :\n {prediction}")
    return response_df


def parse_markdown_json(completion: str) -> dict | None:
    """Extract a ```json { ... } ``` code block from an LLM response and parse it into a dict.

    Use the final matching JSON block because the LLM may give an example before its final result.
    """
    pat = r"```json\s*(\{.*?\})\s*```"
    matches = re.findall(pat, completion, re.DOTALL)
    if not matches:
        print(f"[parse_markdown_json] WARNING: 未找到 JSON 块, 原始文本末尾: ...{completion[-200:]}")
        return None
    json_str = matches[-1]
    try:
        json_obj = json.loads(json_str)
    except Exception as e:
        print(f"[parse_markdown_json] WARNING: JSON 解析失败: {e}, 原始片段: {json_str[:200]}...")
        return None
    return json_obj


def primary_key_preprocess(
    response: list[str],
    reference: list[str],
    prompt_id: str,
    call_llm: Callable[[str, dict], str],
    max_retries: int = 5,
) -> tuple[dict[str, str], list[str]]:
    """Use an LLM to align two vocabularies and generate a response→reference semantic mapping.

    Uses: column-name alignment (such as "City" → "Host City") and primary-key value
    alignment (such as "Feb 4, 2010" → "4 February 2010").

    Retry up to max_retries times when the LLM returns empty output or JSON parsing fails.
    Record the failure reason in warnings only after all retries are exhausted.
    """
    primary_key_map: dict[str, str] = {}
    warnings: list[str] = []
    prompt_kwargs = {"response": response, "reference": reference}
    last_fail_reason = ""

    for attempt in range(max_retries):
        result = call_llm(prompt_id, prompt_kwargs)
        if not result:
            last_fail_reason = f"LLM 返回空结果, response 数量={len(response)}, reference 数量={len(reference)}"
            print(f"[primary_key_preprocess] WARNING: {last_fail_reason} (attempt {attempt + 1}/{max_retries})")
            continue

        try:
            print(f"primary_key_preprocess result (attempt {attempt + 1}): {result[:500]}...")
            transform_map = parse_markdown_json(result)
            if transform_map is None:
                last_fail_reason = f"JSON 解析失败, response 数量={len(response)}, reference 数量={len(reference)}, LLM输出长度={len(result)}"
                print(f"[primary_key_preprocess] WARNING: {last_fail_reason} (attempt {attempt + 1}/{max_retries})")
                continue
            primary_key_map.update(transform_map)
            return primary_key_map, warnings
        except Exception as e:
            last_fail_reason = f"异常: {e}"
            print(f"[primary_key_preprocess] WARNING: {last_fail_reason} (attempt {attempt + 1}/{max_retries})")
            continue

    warnings.append(f"重试 {max_retries} 次后仍失败: {last_fail_reason}")
    return primary_key_map, warnings
