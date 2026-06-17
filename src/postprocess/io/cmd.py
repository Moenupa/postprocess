from rich.console import Console
from rich.syntax import Syntax


def run_cmd(cmd: list[str], exe: bool = False, bg: bool = False) -> None:
    Console().print(Syntax(" \\\n\t".join(cmd), "sh", word_wrap=True))
    if exe:
        import os

        exe_cmd = " ".join(cmd)
        if bg:
            exe_cmd += " &"
        os.system(exe_cmd)  # ty:ignore[deprecated]


def job_dependency(dep: int | None) -> str:
    if dep is None:
        return ""
    else:
        return f" -d afterok:{dep}"
