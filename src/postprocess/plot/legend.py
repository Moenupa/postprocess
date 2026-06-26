from collections.abc import Iterable
from typing import TYPE_CHECKING, TypedDict, Unpack

if TYPE_CHECKING:
    import matplotlib.pyplot as plt
    import matplotlib.transforms as mpl_transforms


class LegendKwargs(TypedDict, total=False):
    bbox_to_anchor: tuple[float, float]
    bbox_transform: "mpl_transforms.Transform"
    frameon: bool
    fancybox: bool
    shadow: bool
    framealpha: float
    facecolor: str
    edgecolor: str
    fontsize: float | str
    title: str
    title_fontsize: float | str
    labelspacing: float
    handlelength: float
    handleheight: float
    handletextpad: float
    borderpad: float
    borderaxespad: float
    columnspacing: float
    markerscale: float
    markerfirst: bool
    reverse: bool
    mode: str
    alignment: str
    draggable: bool


def unify_legends(
    fig: "plt.Figure",
    axes: "Iterable[plt.Axes]",
    remove_only: bool = False,
    loc: str = "lower center",
    ncol: int = 4,
    **legend_kwargs: Unpack[LegendKwargs],
):
    axes = tuple(axes)

    unique = {}
    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        unique.update({label: handle for handle, label in zip(handles, labels) if label})

        _legend = ax.get_legend()
        if _legend is not None:
            _legend.remove()

    # no-op
    if remove_only or not unique:
        return

    fig.legend(
        unique.values(),
        unique.keys(),
        loc=loc,
        ncol=min(len(unique), ncol),
        **legend_kwargs,
    )
