import os
import time
from functools import partial
from pathlib import Path

import typer
from datasets import Dataset, concatenate_datasets
from openai import OpenAI

VLLM_SERVING = os.getenv("VLLM_SERVING", "localhost:8000")


def wait_for_vllm_server() -> bool:
    # 50*10s is a generous timeout for vLLM start up
    count = 50
    while count > 0:
        if (os.system(f"curl http://{VLLM_SERVING}/v1/models 2> /dev/null")) == 0:  # ty:ignore[deprecated]
            typer.echo(f"\n[vllm] curl good for {VLLM_SERVING}", err=True)
            return True
        # sleep
        typer.echo(f"[vllm] curl failed for {VLLM_SERVING}", err=True)
        time.sleep(10)
        count -= 1

    raise RuntimeError(f"Failed to connect to vLLM {VLLM_SERVING} after 50 attempts.")


def get_response(
    e: dict,
    model: str,
    n: int,
    max_tokens: int | None,
):
    client = OpenAI(api_key="1", base_url=f"http://{VLLM_SERVING}/v1")
    # if we want only 1 response, set temperature to 0 for deterministic output
    add_kwargs = {"temperature": 0} if n == 1 else {}
    completions = client.chat.completions.create(
        model=model,
        messages=e["prompt"],
        n=n,
        max_tokens=max_tokens,
        **add_kwargs,
    )  # ty:ignore[no-matching-overload]
    out = {"responses": [choice.message.content for choice in completions.choices]}
    return out


def main(
    parquets: list[str],
    model: str = "saves/Qwen/Qwen3-4B-Thinking-2507",
    n: int = 8,
    max_tokens: int = 14336,
    out_dir: Path | None = None,
    out_name: str = "v1",
    num_workers: int = 16,
):
    wait_for_vllm_server()

    if len(parquets) == 1:
        ds = Dataset.from_parquet(parquets[0])
    else:
        ds_list = []
        for p in parquets:
            _ds = Dataset.from_parquet(p)
            if len(_ds) > 500:
                _ds = _ds.shuffle(42).select(range(500))
            ds_list.append(_ds)
        ds = concatenate_datasets(ds_list)

    if out_dir is None:
        out_dir = Path(model)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{out_name}.parquet"
    if out_path.exists():
        typer.Exit(1)

    ds: Dataset = ds.map(
        partial(get_response, model=model, n=n, max_tokens=max_tokens),
        num_proc=num_workers,
        keep_in_memory=True,
    )
    ds.to_parquet(out_path)


if __name__ == "__main__":
    typer.run(main)
