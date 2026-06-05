import os.path as osp
from functools import partial
from glob import glob
from pathlib import Path
from typing import Any

import pandas as pd
import typer
from datasets import Dataset
from tqdm.contrib.concurrent import process_map
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from postprocess.analysis.word_count import count_word
from postprocess.io.path import parse_model_path
from postprocess.verifier import compute_score

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Thinking-2507", use_fast=True)


def per_score_reward(score: dict[str, Any] | int | float | bool) -> float:
    if isinstance(score, dict):
        if "acc" in score:
            return float(score["acc"])
        if "score" in score:
            return float(score["score"])
        raise ValueError(f"Unexpected score dict format: {score}")
    return float(score)


def per_example_reward(
    e: dict,
    first_n: int,
    fast: bool,
    tokenizer: "PreTrainedTokenizerBase | None" = None,
) -> dict:
    # it contains N responses for the same input
    responses: list[str] = e["responses"][: max(1, first_n)]
    per_resp_reward = [
        compute_score(
            e["data_source"],
            resp,
            ground_truth=e["reward_model"]["ground_truth"],
            extra_info=e["extra_info"],
        )
        for resp in responses
    ]

    fast_out = {
        "data_source": e["data_source"],
        f"pass@{first_n}": any(
            per_score_reward(score) > 0 for score in per_resp_reward
        ),
        f"acc@{first_n}": sum(per_score_reward(score) for score in per_resp_reward)
        / len(per_resp_reward),
    }

    if fast:
        return fast_out

    return fast_out | count_word(
        responses,
        [
            "wait",
            "hmm",
            "perhaps",
            "maybe",
            "actually",
            "alternatively",
            "seems",
            "might",
            "likely",
            "check",
        ],
        tokenizer,
    )


def per_pq_reward(
    parquet_path: str,
    first_n: int,
    fast: bool,
    num_workers: int,
) -> list[dict[str, Any]]:
    pq_path = Path(parquet_path)

    assert pq_path.is_file(), f"Parquet file not found: {parquet_path!r}"
    ds: Dataset = Dataset.from_parquet(pq_path.as_posix())

    train_method, model_name, model_id = parse_model_path(
        pq_path.parent.as_posix(), skip_validate=True
    )
    results: list[dict] = []

    ds: Dataset = ds.map(
        partial(per_example_reward, first_n=first_n, fast=fast, tokenizer=tokenizer),
        num_proc=num_workers,
        remove_columns=ds.column_names,
        keep_in_memory=True,
    )

    for row in ds:
        row: dict

        data_source: str = row.pop("data_source")
        if data_source in ["medmcqa", "MedQA-USMLE-4-options", "MedXpertQA"]:
            continue

        per_row_result = {
            "train_method": train_method,
            "model_name": model_name,
            "model_id": model_id,
            "parquet_path": str(parquet_path),
            "data_source": data_source,
        }
        for k in list(row.keys()):
            if "@" in k:
                per_row_result[k] = row.pop(k)

        if fast:
            results.append(per_row_result)
            continue

        # each row (example input) may have multiple responses (predictions)
        for i in range(first_n):
            resp_counts = {k: v[i] for k, v in row.items()}
            results.append(
                per_row_result
                | resp_counts
                | {
                    # if x is None, then it has </think> tag in its response
                    "unclosed_think_ratio": resp_counts.get("#tokens_if_unclosed", None)
                    is not None,
                }
            )

    return results


def main(
    pq_paths: list[str],
    first_n: int = 8,
    num_workers: int = 4,
    fast: bool = True,
    output_dir: Path = Path("results"),
    target: str = "v1.parquet",
) -> None:
    for i in pq_paths[:]:
        if osp.isdir(i):
            pq_paths.remove(i)
            pq_paths.extend(glob(f"{i}/**/{target}", recursive=True))
    output_dir = output_dir / target

    reward_lists: list[list[dict[str, Any]]] = process_map(
        partial(
            per_pq_reward, first_n=first_n, fast=fast, num_workers=32 // num_workers
        ),
        pq_paths,
        max_workers=num_workers,
        chunksize=1,
    )

    rows: list[dict[str, Any]] = [row for rewards in reward_lists for row in rewards]
    if len(rows) == 0:
        raise typer.BadParameter("No rows found in parquet files.")

    df: pd.DataFrame = pd.DataFrame(rows)
    grouped_df = df.groupby(
        ["model_id", "model_name", "train_method", "data_source"], as_index=False
    )
    for each_col in df.columns:
        # if dtype is not bool or int or float, skip
        if df[each_col].dtype not in [bool, int, float]:
            typer.echo(f"[warn] skipped non-numeric column: {each_col}")
            continue

        agg_df: pd.DataFrame = (
            grouped_df[each_col]
            .mean()
            .sort_values(["model_id", "model_name", "train_method", "data_source"])
        )

        pivot_df: pd.DataFrame = agg_df.pivot_table(
            index=["model_id", "model_name", "train_method"],
            columns="data_source",
            values=[each_col],
            aggfunc="mean",
        )

        if "ratio" in each_col or "@" in each_col:
            pivot_df *= 100
        pivot_df = pivot_df.round(2)

        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{each_col}.tsv"
        pivot_df.to_csv(output_path, sep="\t")
        typer.echo(f"[saved] {output_path}")


if __name__ == "__main__":
    typer.run(main)
