import pytest

from postprocess.io.parse import filter_by_regex


@pytest.mark.parametrize(
    "items,incl,excl,whole_match,expected",
    [
        (["apple", "banana", "cherry"], ["a"], [], False, ["apple", "banana"]),
        (["apple", "banana", "cherry"], [], ["a"], False, ["cherry"]),
        (["apple", "banana", "cherry"], ["a"], ["e"], False, ["banana"]),
        (["apple", "banana", "cherry"], ["a"], ["e"], True, []),
        (
            ["apple", "banana", "cherry"],
            ["apple", "banana"],
            [],
            True,
            ["apple", "banana"],
        ),
        ([r"\n\n", r"\n", r"\t"], [r"\n"], [], False, [r"\n\n", r"\n"]),
        ([r"\n\n", r"\n", r"\t"], [r"\n"], [], True, [r"\n"]),
    ],
)
def test_filter_by_regex(
    items: list[str],
    incl: list[str],
    excl: list[str],
    whole_match: bool,
    expected: list[str],
):
    assert filter_by_regex(items, incl, excl, whole_match) == expected
