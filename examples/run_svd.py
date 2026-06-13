import logging
import math
from functools import partial
from pathlib import Path

import pandas as pd
import torch
import typer
from tqdm.contrib.concurrent import thread_map
from transformers import AutoModelForCausalLM

from postprocess.analysis.svd import (
    _to_tensor,
    overlap_ratio,
    param_changed_mask,
    principal_angles,
    principal_mask,
    spectrum_drift,
)
from postprocess.io.parse import filter_by_fnmatch
from postprocess.io.path import parse_model_path

logger = logging.getLogger(__name__)


def analyze_layer(
    model_a: dict,
    model_b: dict,
    layer_key: str,
    k: int,
    rank: int,
    alpha: float,
    eta: float,
) -> dict | None:
    if layer_key not in model_a or layer_key not in model_b:
        logging.error(f"[skip layer] Key mismatch: {layer_key!r}.")
        return None

    # release GPU memory before heavy computation; may help reduce OOM risk
    torch.cuda.empty_cache()

    try:
        W0 = _to_tensor(model_a[layer_key], dtype=torch.float32)
        W1 = _to_tensor(model_b[layer_key], dtype=torch.float32)
        U0, S0, Vh0 = torch.linalg.svd(W0, full_matrices=False)
        U1, S1, Vh1 = torch.linalg.svd(W1, full_matrices=False)

        r = min(rank, min(W0.shape), min(W1.shape))
        sd = spectrum_drift(S0, S1)
        pa = principal_angles(U0, U1, k=k)
        pm0 = principal_mask(u=U0, s=S0, v=Vh0, rank=r, alpha=alpha)
        pm1 = principal_mask(u=U1, s=S1, v=Vh1, rank=r, alpha=alpha)
        changed = param_changed_mask(W0, W1, eta=eta)

        if pa.nelement() > 0:
            pa_mean_deg = float(pa.mean().item() * 180.0 / math.pi)
            pa_max_deg = float(pa.max().item() * 180.0 / math.pi)
        else:
            pa_mean_deg = float("nan")
            pa_max_deg = float("nan")
        pa_overlap_ratio = overlap_ratio(pm0, pm1)
        principal_changed_overlap_ratio = overlap_ratio(pm0, changed)

        return {
            "layer": layer_key,
            "spectrum_drift": sd,
            "pa_deg_mean": pa_mean_deg,
            "pa_deg_max": pa_max_deg,
            "pa_overlap_ratio": pa_overlap_ratio,
            "principal_changed_overlap_ratio": principal_changed_overlap_ratio,
        }
    except Exception as e:
        logging.error(f"[skip layer] Error analyzing layer {layer_key!r}: {e}")
        return None


def analyze_model_pair(
    model_a: dict,
    model_b: dict,
    path_a: str,
    path_b: str,
    out_dir: Path,
    layer_keys: list[str],
    k: int,
    rank: int,
    alpha: float,
    eta: float,
    max_workers: int,
):
    results = thread_map(
        partial(
            analyze_layer,
            model_a,
            model_b,
            k=k,
            rank=rank,
            alpha=alpha,
            eta=eta,
        ),
        layer_keys,
        desc=f"{path_a!r}->{path_b!r}",
        max_workers=min(max_workers, len(layer_keys)),
    )

    if not results:
        typer.echo(f"[skipped] No results for {path_a} -> {path_b}.")
        return

    csv_path = (
        out_dir
        / ".".join(parse_model_path(path_a, keep_step=True))
        / f"{'.'.join(parse_model_path(path_b, keep_step=True))}.csv"
    )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(list(filter(lambda e: e is not None, results))).to_csv(
        csv_path, index=False
    )
    typer.echo(f"[saved] {csv_path}")


def get_layer_keys(
    model_dict: dict[str, torch.Tensor], layer_pattern: str | None, list_keys: bool
) -> list[str]:
    selected_keys = filter_by_fnmatch(
        list(model_dict.keys()),
        [] if layer_pattern is None else [layer_pattern],
        ["*norm*"],  # exclude all norm layers, where SVD is not meaningful
        fallback=layer_pattern is None,
    )
    if not selected_keys:
        raise typer.Exit(code=1)
    elif list_keys:
        logging.warning("Available Keys (-), Selected Keys (+):")
        for _k, _v in model_dict.items():
            logging.warning(f"{'+' if _k in selected_keys else '-'}{_k:60s} {_v.shape}")
        raise typer.Exit()
    return selected_keys


def main(
    model_paths: list[str],
    out_dir: Path = Path("svd"),
    layer_pattern: str | None = None,
    k: int = 64,
    rank: int = 64,
    alpha: float = 0.1,
    eta: float = 1e-3,
    list_keys: bool = False,
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu",
    max_workers: int = 4,
):
    model0 = AutoModelForCausalLM.from_pretrained(model_paths[0], dtype=torch.bfloat16)
    selected_keys = get_layer_keys(model0.state_dict(), layer_pattern, list_keys)
    if len(model_paths) != 2:
        logging.error(
            "Invalid argument: exactly 2 model paths are required for pairwise analysis."
        )
        raise typer.Exit(code=2)

    model1 = AutoModelForCausalLM.from_pretrained(model_paths[-1], dtype=torch.bfloat16)
    analyze_model_pair(
        model0.to(device).state_dict(),  # ty:ignore[invalid-argument-type]
        model1.to(device).state_dict(),  # ty:ignore[invalid-argument-type]
        model_paths[0],
        model_paths[-1],
        out_dir,
        selected_keys,
        k=k,
        rank=rank,
        alpha=alpha,
        eta=eta,
        max_workers=max_workers,
    )


if __name__ == "__main__":
    typer.run(main)
