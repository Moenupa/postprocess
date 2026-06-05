import pytest

from postprocess.io.path import parse_model_path


@pytest.mark.parametrize(
    "model_path, expected",
    [
        ("Qwen/Qwen3-8B", ("hf", "Qwen", "Qwen3-8B")),
        ("saves/Qwen/Qwen3-8B", ("saves", "Qwen", "Qwen3-8B")),
        (
            "saves/grpo/Qwen:math/Qwen3-8B/actor/huggingface/global_step_50",
            ("grpo", "Qwen:math", "Qwen3-8B"),
        ),
    ],
)
def test_parse_model_path(model_path: str, expected: tuple[str, str, str]):
    assert parse_model_path(model_path, skip_validate=True) == expected
