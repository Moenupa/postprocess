import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from ..constant import SEED

logger = logging.getLogger(__name__)


def compute_tsne(
    embeddings: np.ndarray,
    pca_kwargs: dict | None = None,
    tsne_kwargs: dict | None = None,
) -> np.ndarray:
    pca_kwargs: dict = (
        pca_kwargs
        or {
            "n_components": 50,
            "svd_solver": "randomized",
        }
    ) | {"random_state": SEED}
    logger.info(f"Running PCA with {pca_kwargs}")
    embeddings = PCA(**pca_kwargs).fit_transform(embeddings)

    tsne_kwargs: dict = (tsne_kwargs or {}) | {"random_state": SEED, "n_jobs": -1}
    logger.info(f"Running t-SNE with {tsne_kwargs}")
    coords = TSNE(n_components=2, **tsne_kwargs).fit_transform(embeddings)
    return coords


def auto_alpha(
    n_samples: int, min_alpha: float = 1 / 128, max_alpha: float = 1.0
) -> float:
    # 100 -> 1, 10k -> 0.1, 1M -> 0.01, 100M -> 0.001
    alpha = 10 / np.sqrt(n_samples)
    return float(np.clip(alpha, min_alpha, max_alpha))


def plot_tsne_against_selection(
    embeddings: np.ndarray,
    save_to: str,
    selection_mask: np.ndarray | None = None,
    fake_labels: list[str] | None = None,
    gt_labels: list[str] | None = None,
    save_h: int | float = 8,
    save_w: int | float = 8,
    dot_size: int = 12,
    pca_kwargs: dict | None = None,
    tsne_kwargs: dict | None = None,
) -> None:
    """t-SNE map highlighting selected vs unselected samples.

    Args:
        embeddings (np.ndarray): array of shape (n_samples, dim) containing the embeddings to project.
        save_to (str, optional): where to save the figure.
        selection_mask (np.ndarray): boolean mask of length n_samples, True for selected samples.
        fake_labels (list[str] | None, optional):
            optional per-sample clustering labels to style the points. Defaults to None.
        gt_labels (list[str] | None, optional):
            optional per-sample string gt labels to color the points. Defaults to None.
        save_h (int, optional): height of the saved figure in inches. Defaults to 8.
        save_w (int, optional): width of the saved figure in inches. Defaults to 8.
        dot_size (int, optional): size of the scatter points. Defaults to 12.
        pca_kwargs (dict | None, optional): Extra keyword arguments forwarded to :class:`sklearn.decomposition.PCA`. Defaults to None.
        tsne_kwargs (dict | None, optional): Extra keyword arguments forwarded to :class:`sklearn.manifold.TSNE`. Defaults to None.
    """
    coords = compute_tsne(embeddings, pca_kwargs=pca_kwargs, tsne_kwargs=tsne_kwargs)

    fig, ax = plt.subplots(figsize=(save_w, save_h))
    dot_alpha = auto_alpha(len(embeddings))
    logger.warning(
        f"Plotting t-SNE with {len(embeddings)} points, alpha={dot_alpha:.3f}"
    )
    sns.scatterplot(
        x=coords[:, 0],
        y=coords[:, 1],
        hue=gt_labels,
        # style=fake_labels,
        s=dot_size,
        alpha=dot_alpha,
        linewidth=0,
        edgecolor="none",
        ax=ax,
    )

    if selection_mask is not None:
        ax.scatter(
            x=coords[selection_mask, 0],
            y=coords[selection_mask, 1],
            s=dot_size,
            facecolors="none",
            edgecolors="black",
            linewidths=0.2,
            label="Selected",
        )

    # fix legends: dont respect alpha, always opaque
    handles, labels = ax.get_legend_handles_labels()
    for handle in handles:
        handle.set_alpha(1.0)
    ax.legend(handles=handles, labels=labels, loc="best")

    Path(save_to).parent.mkdir(parents=True, exist_ok=True)
    logger.warning(f"Saving t-SNE plot to {save_to!r} with size {save_w}x{save_h}in.")

    fig.savefig(save_to, bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_cluster_gt_heatmap(
    n_clusters: int,
    fake_labels: list[str],
    gt_labels: list[str],
    save_to: str,
) -> None:
    """Heatmap of cluster assignment vs ground-truth labels.

    Each row represents a ground-truth class; each column a cluster.
    Cell values show the fraction of that GT class found in each cluster.

    Args:
        n_clusters: number of clusters (i.e. unique values in fake_labels).
        fake_labels: (n_samples,) integer array of predicted cluster IDs.
        gt_labels: (n_samples,) array of ground-truth class labels (int or str).
        save_to: where to save the figure.
    """
    gt_to_row = {g: i for i, g in enumerate(sorted(set(gt_labels)))}
    n_gt = len(gt_to_row)

    contingency = np.zeros((n_gt, n_clusters), dtype=int)
    for g, c in zip(gt_labels, fake_labels):
        contingency[gt_to_row[g], int(c)] += 1
    row_sums = contingency.sum(axis=1, keepdims=True)
    contingency_norm = np.divide(
        contingency,
        row_sums,
        where=row_sums > 0,
        out=np.zeros_like(contingency, dtype=float),
    )

    fig, ax = plt.subplots(figsize=(n_clusters, n_gt))
    sns.heatmap(
        contingency_norm,
        annot=True,
        fmt=".2f",
        xticklabels=[f"C{i}" for i in range(n_clusters)],
        yticklabels=[str(g) for g in gt_to_row.keys()],
        ax=ax,
        vmin=0,
        vmax=1,
    )
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Ground Truth")

    Path(save_to).parent.mkdir(parents=True, exist_ok=True)
    logger.warning(f"Saving Cluster v.s. GT plot to {save_to!r}.")

    fig.savefig(save_to, bbox_inches="tight", dpi=100)
    plt.close(fig)
