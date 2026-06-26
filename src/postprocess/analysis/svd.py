import numpy as np
import torch


def _to_tensor(
    x: torch.Tensor,
    dtype: torch.dtype | None = None,
):
    if isinstance(x, torch.nn.Parameter):
        x = x.data
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x)
    if dtype is not None:
        x = x.to(dtype=dtype)
    return x


def spectrum_drift(s0: torch.Tensor, s1: torch.Tensor) -> float:
    denom = torch.norm(s0)
    if denom == 0:
        return float("nan")

    drift = float(torch.norm(s1 - s0) / denom)
    return drift


def principal_angles(u0: torch.Tensor, u1: torch.Tensor, k: int = 64) -> torch.Tensor:
    max_k = min(u0.shape[1], u1.shape[1], k)
    if max_k <= 0:
        return torch.tensor([])

    U0k = u0[:, :max_k]
    U1k = u1[:, :max_k]

    M = U0k.T @ U1k
    cos_theta = torch.linalg.svdvals(M)
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
    theta = torch.arccos(cos_theta)
    return theta


def principal_mask(
    u: torch.Tensor,
    s: torch.Tensor,
    v: torch.Tensor,
    rank: int,
    alpha: float = 0.1,
) -> torch.Tensor:
    Wk = u[:, :rank] @ torch.diag(s[:rank]) @ v[:rank]
    score = torch.abs(Wk)
    try:
        threshold = float(torch.quantile(score.flatten(), 1 - alpha))
    except Exception:
        threshold = float(np.quantile(score.detach().cpu().numpy().ravel(), 1 - alpha))
    mask = score >= threshold
    return mask


def param_changed_mask(W0: torch.Tensor, W1: torch.Tensor, eta: float = 1e-3) -> torch.Tensor:
    tol = eta * torch.maximum(W0.abs(), W1.abs())
    changed = (W1 - W0).abs() > tol
    return changed


def overlap_ratio(principal_mask: torch.Tensor, update_mask: torch.Tensor) -> float:
    overlap = (principal_mask & update_mask).sum()
    updated = update_mask.sum()
    if int(updated) == 0:
        return 0.0
    return float(overlap.float() / updated.float())
