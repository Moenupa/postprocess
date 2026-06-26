#!/usr/bin/env python3
import re
from collections import Counter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

THINK_END = "</think>"


def count_word(
    solutions: list[str],
    words: list[str],
    tokenizer: "PreTrainedTokenizerBase | None" = None,
) -> dict:
    """Count the occurrences of words in the solutions, and also count the
    number of tokens if the solution is not closed by </think> tag
    (which may indicate being capped by length).

    Args:
        solutions (list[str]): predictions from model.
        words (list[str]): words to count in the solutions.
        tokenizer (PreTrainedTokenizerBase | None, optional): tokenizer to count tokens. Defaults to None.

    Returns:
        dict: {'word1': [count_in_sol1, count_in_sol2, ...], 'word2': [...], ..., '#tokens_if_unclosed': [ntokens_if_unclosed_or_None_for_closed, ...]}
    """

    WORD_RE = re.compile(r"\b(" + "|".join(map(re.escape, words)) + r")\b", re.IGNORECASE)

    # </think> is in answer, then we only count the think part,
    # otherwise we regard as capped by length, where entire answer is think part
    solutions_thinking = []
    solutions_closed = []

    for _s in solutions:
        _closed_think = THINK_END in _s
        solutions_closed.append(_closed_think)
        solutions_thinking.append(_s.split(THINK_END, 1)[0] if _closed_think else _s)

    # if it does not contain </think> tag
    # we need to report the length as evidence for 'being capped by length'
    solutions_thinking_tokens = []
    if tokenizer is not None:
        encoded = tokenizer(
            solutions_thinking,
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        solutions_thinking_tokens = list(map(len, encoded["input_ids"]))

    return {word: [Counter(map(str.lower, WORD_RE.findall(s))).get(word, 0) for s in solutions] for word in words} | {
        "#tokens_if_unclosed": [
            None if closed else ntokens for closed, ntokens in zip(solutions_closed, solutions_thinking_tokens)
        ]
        + [0],
    }
