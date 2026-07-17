"""LLM-judge wrapper for BrowseComp Plus.

This file only assembles the fixed evaluation prompt, invokes the unified llm_infer,
and parses the `correct: yes/no` result. The prompt body retains the benchmark's original text.
"""

import re

from llm_infer.llm_infer import llm_infer


def call_judge_llm(messages, model_name: str, provider: str) -> str:
    """Call the evaluation model using judge environment variables prepared by the runner."""
    response = llm_infer(
        provider=provider,
        model=model_name,
        messages=messages,
        temperature=0.0,
        enable_thinking=False,
        generation_config={"max_tokens": 16384},
        extra_params={
            "api_key_env": "JUDGE_MODEL_API_KEY",
            "base_url_env": "JUDGE_MODEL_BASE_URL",
        },
    )
    return response.get("content") or ""


# BrowseComp Plus evaluation prompt template (from prompt/eval/bc.base.en)
EVAL_PROMPT_TEMPLATE = """Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.


confidence: The extracted confidence score between 0|%| and 100|%| from [response]. Put 100 if there is no confidence score available.
"""

def browsecomp_plus_llm_judge(
    question: str,
    prediction: str,
    answer: str,
    judge_model_name: str,
    judge_model_provider: str,
) -> dict:
    """Use an LLM to evaluate answer correctness for one BrowseComp Plus sample."""
    prompt_kwargs = {
        "question": question,
        "response": prediction,
        "correct_answer": answer,
    }
    input_content = EVAL_PROMPT_TEMPLATE.format(**prompt_kwargs)
    messages = [{"role": "user", "content": input_content}]

    response_text = call_judge_llm(
        messages,
        model_name=judge_model_name,
        provider=judge_model_provider,
    )

    # Parse only the final yes/no field required by the benchmark; treat parse failures as errors.
    match = re.search(r"correct: (yes|no)", response_text)
    judge_text = match.group(1) if match else "no"
    is_correct = judge_text == "yes"

    return {
        "is_correct": is_correct,
        "llm_equal": int(is_correct),
        "llm_response": response_text,
    }
