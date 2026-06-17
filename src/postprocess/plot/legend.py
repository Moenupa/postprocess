from collections.abc import Iterable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import matplotlib.pyplot as plt


def unify_legends(
    fig: "plt.Figure",
    axes: "Iterable[plt.Axes]",
    remove_only: bool = True,
    ncol: int = 4,
    bbox_to_anchor: tuple[float, float] = (0.5, 0.0),
):
    first_ax = next(iter(axes))
    handles, labels = first_ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles, strict=False))
    first_ax.legend().remove()
    for ax in axes:
        _legend = ax.get_legend()
        if _legend is not None:
            _legend.remove()
    if remove_only:
        return

    if unique:
        fig.legend(
            unique.values(),
            unique.keys(),
            loc="lower center",
            ncol=min(len(unique), ncol),
            bbox_to_anchor=bbox_to_anchor,
        )
