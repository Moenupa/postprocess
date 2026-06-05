import json

import numpy as np

from postprocess.verifier.code.code_utils import extract_code, get_successful_tests_fast

TRUNCATED_FEEDBACK = "Truncated Attempt: Your previous response was too long and truncated because it reached the maximum response length. Try again with a shorter response."
NO_CODEBLOCK_FEEDBACK = (
    "Incorrect Format: Put your code inside a ```python ... ``` block."
)


def compute_score(
    solution: str,
    ground_truth: str,
    extra_info: dict | None = None,
    sparse_rewards: bool = False,
    max_test_cases: int | None = None,
) -> dict:
    extra_info = extra_info or {}
    split = extra_info.get("split", "train")
    was_truncated = extra_info.get("truncated", False)

    if split == "test":
        sparse_rewards = True

    test_cases = json.loads(ground_truth)
    program = extract_code(solution)

    if program is None:
        # handle invalid format
        return {
            "score": 0,
            "acc": 0,
            "pred": "",
            "incorrect_format": 1,
            "error_in_test_cases": 0,
            "timed_out": 0,
            "truncated": 1 if was_truncated else 0,
            "truncated_and_missing_answer": 1 if was_truncated else 0,
            "feedback": TRUNCATED_FEEDBACK if was_truncated else NO_CODEBLOCK_FEEDBACK,
        }

    try:
        if isinstance(test_cases, list):
            results, runtimes, errors = get_successful_tests_fast(
                program=program,
                tests=test_cases,
                max_execution_time=1.0,
            )
        elif isinstance(test_cases, dict):
            pass

    except AssertionError as e:
        raise e

    accuracy = np.mean(results)
    if sparse_rewards:
        reward = all(results) * 1.0
    else:
        reward = accuracy

    return {
        "score": reward,
        "acc": accuracy,
        "pred": program,
        "incorrect_format": 0,
        "error_in_test_cases": 1 if any(errors) else 0,
        "timed_out": 1 if any("timeout" in err.lower() for err in errors) else 0,
        "truncated": 1 if was_truncated else 0,
        "truncated_and_missing_answer": 0,
        "feedback": format_test_feedback(errors),
    }


def format_test_feedback(
    errors: list[str],
    max_tests_to_show: int = 2,
    sort_test_cases_by_length: bool = True,
    max_length: int = 2000,
):
    """
    Render test feedback in a LeetCode-like style.

    Rules:
    - Only show failing cases.
    - For runtime errors/timeouts: show a concise error header and the last
      executed input.
    - For wrong answers: show Input, optional Stdout (debug_print), Output and
      Expected.
    - If all cases pass, return an empty string.
    """
    if not errors:
        return "No test execution information available."

    def _truncate_str(value, max_chars):
        if not isinstance(value, str):
            value = str(value)
        if max_chars is not None and len(value) > max_chars:
            return value[:max_chars] + "..."
        return value

    failing = [r for r in errors if r]

    # Sort wrong-answer cases by length (input + output), shortest first
    if sort_test_cases_by_length:
        failing = sorted(failing, key=len)

    if max_tests_to_show is not None:
        failing = failing[: int(max_tests_to_show)]

    result = "\n".join(failing).rstrip()
    if len(result) > max_length:
        result = result[:max_length]
    return result
