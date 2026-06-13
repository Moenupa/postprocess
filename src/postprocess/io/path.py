import re
from functools import cache
from pathlib import Path


@cache
def get_valid_config(name: str, config_dir: str = "verl/trainer/config") -> str:
    return find_file(config_dir, name, "yaml")


@cache
def get_valid_data(path: str, name: str = "test") -> str:
    return find_file(path, name, "parquet")


@cache
def parse_model_path(
    path: str,
    skip_validate: bool = False,
    remove_parts_regex: list[str] = ["actor", "huggingface", r"global_step_\d+"],
    keep_step: bool = False,
) -> tuple[str, str, str]:
    r"""Parse model path into (train_method, model_name, model_id).

    E.g.:
    - Qwen/Qwen3-8B -> ("hf", "Qwen", "Qwen3-8B")
    - saves/grpo/Qwen:math/Qwen3-8B/actor/huggingface -> ("grpo", "Qwen:math", "Qwen3-8B")

    Args:
        path (str): Huggingface path or local path to the model.
        skip_validate (bool, optional): Whether to skip validation of the model path. Defaults to False.
        remove_parts_regex (list[str], optional): List of regex patterns to remove from the path parts. Defaults to ["actor", "huggingface", r"global_step_\d+"].

    Raises:
        ValueError: If the model path is invalid.

    Returns:
        tuple[str, str, str]: (train_method, model_name, model_id)
    """
    if keep_step:
        path = path.replace("/global_step_", "@")
    parts = path.strip("/").split("/")
    if len(parts) < 2:
        raise ValueError(f"Invalid model path: {path!r}")

    if len(parts) == 2:
        return "hf", parts[0], parts[1]

    # otherwise verify it is a correct local path
    if not skip_validate and not keep_step:
        model_metadata = Path(path) / "model.safetensors.index.json"
        assert model_metadata.is_file(), f"Model metadata not found: {model_metadata!r}"

    parts = [
        p for p in parts if not any(re.match(regex, p) for regex in remove_parts_regex)
    ]

    assert len(parts) >= 3, f"Expected >= 3 parts (e.g. Qwen/Qwen3-8B): {path!r}"
    return parts[-3], parts[-2], parts[-1]


def find_file(dir_path: str, filename: str, extension: str) -> str:
    _d = Path(dir_path)
    if not _d.is_dir():
        raise ValueError(f"No such dir: {dir_path!r}")

    file = _d / f"{filename}.{extension}"
    if not file.is_file():
        available_files = [p.stem for p in _d.glob(f"*.{extension}")]
        raise ValueError(
            f"File not found: '{filename}.{extension}' in {dir_path!r}. Available files: {available_files}"
        )
    return file.absolute().as_posix()
