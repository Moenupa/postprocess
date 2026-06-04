def extract_boxed_content(text: str) -> str | None:
    """
    Extracts answers in \\boxed{}.
    """
    depth = 0
    start_pos = text.rfind(r"\boxed{")
    end_pos = -1
    if start_pos != -1:
        content = text[start_pos + len(r"\boxed{") :]
        for i, char in enumerate(content):
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1

            if depth == -1:  # exit
                end_pos = i
                break

    if end_pos != -1:
        return content[:end_pos].strip()

    return None


def compute_score(
    solution: str,
    ground_truth: str,
    extra_info: dict,
    format_feedback: bool = True,
    correctness_feedback: bool = False,
) -> dict:
    multiple_choice_answer = extract_boxed_content(solution)

    reward = float(multiple_choice_answer == ground_truth)
    incorrect_format = multiple_choice_answer not in [
        "A",
        "B",
        "C",
        "D",
        "E",
        "F",
        "G",
        "H",
        "I",
        "J",
    ]
    was_truncated = extra_info.get("truncated", False)

    feedback = ""
    if format_feedback and incorrect_format:
        feedback = "Your answer had the wrong format. The solution must be given in the format: \\boxed{your_answer}."
    elif format_feedback and was_truncated:
        feedback = "Your response was truncated because it exceeded the maximum length."
    elif correctness_feedback and not reward:
        feedback = f"Your answer is incorrect. The correct answer is {ground_truth}."

        # optionally include explanation if available
        if desc := extra_info.get("description", None):
            feedback += f" Explanation: {desc}"

    return {
        "score": reward,
        "acc": reward,
        "pred": multiple_choice_answer or "",
        "incorrect_format": 1 if incorrect_format else 0,
        "truncated": 1 if was_truncated else 0,
        "truncated_and_missing_answer": 1 if incorrect_format and was_truncated else 0,
        "feedback": feedback,
    }
