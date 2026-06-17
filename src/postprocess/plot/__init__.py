import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from .legend import unify_legends
from .tnse import plot_cluster_gt_heatmap, plot_tsne_against_selection


def _sci_e_fmt(x, pos=None):
    s = f"{x:.1e}"
    mantissa, exp = s.split("e")
    exp = int(exp)  # removes leading zeros
    return f"{mantissa}e{exp}"


sci_e_formatter = mtick.FuncFormatter(_sci_e_fmt)


__all__ = [
    "plot_cluster_gt_heatmap",
    "plot_tsne_against_selection",
    "sci_e_formatter",
    "unify_legends",
]

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Latin Modern Roman", "Computer Modern Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
    }
)
