from postprocess.verifier import (
    code,
    math,
    mcq,
    tooluse,
)
from postprocess.verifier.code import lcb


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict | None = None,
) -> dict | float:
    if extra_info is None:
        extra_info = {}

    if not isinstance(data_source, str):
        raise ValueError(f"data_source should be a string, but got {type(data_source)}")

    if data_source in ["code", "livecodebench", "humanevalplus"]:
        results = lcb.compute_score(
            solution_str,
            ground_truth,
            extra_info,
            sparse_rewards=True,
            max_test_cases=None,
        )
    elif data_source in ["code_rlvr_mixture_dpo"]:
        results = code.compute_score(
            solution_str,
            ground_truth,
            extra_info,
            sparse_rewards=True,
            max_test_cases=None,
        )
    elif data_source in [
        "math",
        "math500",
        "dapo_math",
        "math_dapo",
        "gsm8k",
        "aime24",
        "aime25",
        "amc23",
        "minervamath",
        "math_rlvr_mixture_dpo",
    ]:
        results = math.compute_score(
            solution_str,
            ground_truth,
            extra_info,
            format_feedback=True,
            correctness_feedback=True,
        )
    elif data_source in [
        "gpqa",
        "sciknoweval",
        "medmcqa",
        "ZebraLogicBench-private",
        "MedQA-USMLE-4-options",
        "MedXpertQA",
        "mmlu-redux-2.0",
    ] or data_source.startswith("sciknoweval"):
        results = mcq.compute_score(
            solution_str,
            ground_truth,
            extra_info,
            format_feedback=True,
            correctness_feedback=True,
        )
    elif data_source in ["tooluse"]:
        results = tooluse.compute_score(solution_str, ground_truth)
    else:
        raise ValueError(f"Reward style {data_source} not found.")
    return results


def pre_check(data_dir: str):
    from datasets import Dataset

    for split in ["train", "test"]:
        ds = Dataset.from_parquet(f"{data_dir}/{split}.parquet")
        for each_source in set(ds["data_source"]):
            compute_score(each_source, "A", "A")
