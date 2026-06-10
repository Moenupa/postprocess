import fnmatch
import re
from functools import partial


def filter_by_regex(
    items: list[str], incl: list[str], excl: list[str], whole_match: bool = False
):
    incl = [re.escape(pattern) for pattern in incl]
    excl = [re.escape(pattern) for pattern in excl]

    incl_regex = "|".join(incl) if incl else None
    excl_regex = "|".join(excl) if excl else None

    if whole_match:
        incl_regex = f"^({incl_regex})$" if incl_regex else None
        excl_regex = f"^({excl_regex})$" if excl_regex else None

    incl_pat = re.compile(incl_regex) if incl_regex else None
    excl_pat = re.compile(excl_regex) if excl_regex else None

    return list(
        filter(
            partial(_excl_priority_filter, incl_pat=incl_pat, excl_pat=excl_pat), items
        )
    )


def filter_by_fnmatch(
    items: list[str], incl: list[str], excl: list[str], fallback: bool = True
):
    def fnmatch_filter(item: str) -> bool:
        if incl and any(fnmatch.fnmatchcase(item, pattern) for pattern in incl):
            return True
        if excl and any(fnmatch.fnmatchcase(item, pattern) for pattern in excl):
            return False
        return fallback

    return list(filter(fnmatch_filter, items))


def _incl_priority_filter(
    item: str,
    incl_pat: re.Pattern | None,
    excl_pat: re.Pattern | None,
    fallback: bool = True,
) -> bool:
    # incl > excl > fallback; item EITHER: match incl, OR not match excl
    if incl_pat and incl_pat.search(item):
        return True
    if excl_pat and excl_pat.search(item):
        return False
    return fallback


def _excl_priority_filter(
    item: str,
    incl_pat: re.Pattern | None,
    excl_pat: re.Pattern | None,
    fallback: bool = True,
) -> bool:
    # incl=excl > fallback; item must satisfy both incl and excl to be True
    if incl_pat and not incl_pat.search(item):
        return False
    if excl_pat and excl_pat.search(item):
        return False
    return fallback
