from __future__ import annotations

import torch

from .schedules import DiffusionSchedules


def marginal_kernel(marginal: torch.Tensor, *, with_del_states: bool = True) -> torch.Tensor:
    """Return the GrIDDD/DiGress marginal transition B.

    ``marginal`` spans all ordinary states (including states with zero terminal
    probability). When ``with_del_states`` is true, two structural states are
    appended in the order DEL, DEL*.
    """

    marginal = marginal.float()
    marginal = marginal / marginal.sum().clamp_min(1e-12)
    d0 = int(marginal.numel())
    if not with_del_states:
        return marginal.unsqueeze(0).expand(d0, -1).clone()

    d = d0 + 2
    B = torch.zeros((d, d), dtype=marginal.dtype, device=marginal.device)
    B[:d0, :d0] = marginal.unsqueeze(0).expand(d0, -1)
    B[-2, -2] = 1.0       # DEL is absorbing.
    B[-1, -2] = 1.0       # DEL* deterministically moves to DEL forward.
    return B


def structural_matrices(marginal: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Construct the A/B/C/D matrices used by GrIDDD's DEL/DEL* process."""

    B = marginal_kernel(marginal, with_del_states=True)
    d = B.shape[0]
    d0 = d - 2

    A = torch.eye(d, dtype=B.dtype, device=B.device)
    A[-1, -1] = 0.0
    A[-1, -2] = 1.0

    C = torch.zeros_like(A)
    C[:d0, -1] = 1.0      # ordinary -> DEL* at the deletion boundary
    C[-2:, -2] = 1.0      # special -> DEL

    D = torch.zeros_like(A)
    D[:, -2] = 1.0        # deletion already completed
    return A, B, C, D


def ordinary_qbar(alpha_bar: torch.Tensor | float, marginal: torch.Tensor) -> torch.Tensor:
    """Closed-form categorical cumulative transition without structural deletion."""

    A, B, _, _ = structural_matrices(marginal)
    a = torch.as_tensor(alpha_bar, dtype=A.dtype, device=A.device)
    return a * A + (1.0 - a) * B


def ordinary_qstep(alpha_step: torch.Tensor | float, marginal: torch.Tensor) -> torch.Tensor:
    A, B, _, _ = structural_matrices(marginal)
    a = torch.as_tensor(alpha_step, dtype=A.dtype, device=A.device)
    return a * A + (1.0 - a) * B


def structural_qbar(t: int, schedules: DiffusionSchedules, marginal: torch.Tensor) -> torch.Tensor:
    """Cumulative q(x_t | x_0) for a node/edge eligible for forward deletion.

    This is the activation-time-zero specialization of GrIDDD's generalized
    transition. FragFlex does not use forward insertion, so all clean fragments
    are active at s=0.
    """

    A, B, C, D = structural_matrices(marginal)
    if t <= 0:
        return A

    abar = schedules.alpha_bar[t].to(A.device)
    zbar_t = schedules.zeta_bar[t, 0].to(A.device)
    zbar_tm1 = schedules.zeta_bar[t - 1, 0].to(A.device)
    zt = schedules.zeta[t].to(A.device)

    return (
        zbar_t * (abar * A + (1.0 - abar) * B)
        + ((1.0 - zt) * zbar_tm1) * C
        + (1.0 - zbar_tm1) * D
    )


def structural_qstep(t: int, schedules: DiffusionSchedules, marginal: torch.Tensor) -> torch.Tensor:
    """One-step q(x_t | x_{t-1}) at deletion-capable timestep ``t``."""

    A, B, C, _ = structural_matrices(marginal)
    if t <= 0:
        return A
    alpha = schedules.alpha_step[t].to(A.device)
    zt = schedules.zeta[t].to(A.device)
    return zt * (alpha * A + (1.0 - alpha) * B) + (1.0 - zt) * C


def posterior_from_clean_prediction(
    x_t: torch.Tensor,
    p_clean: torch.Tensor,
    q_step: torch.Tensor,
    qbar_s: torch.Tensor,
    qbar_t: torch.Tensor,
    *,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Marginalize the exact categorical posterior over a denoiser prediction.

    Args:
        x_t: integer current states with shape ``[...]``.
        p_clean: predicted clean distribution with shape ``[..., D]``. States
            that are impossible at t=0 should have probability zero.
        q_step: ``[D,D]`` matrix q(x_t | x_s), rows are x_s.
        qbar_s: ``[D,D]`` matrix q(x_s | x_0), rows are x_0.
        qbar_t: ``[D,D]`` matrix q(x_t | x_0), rows are x_0.

    Returns:
        ``[..., D]`` distribution for x_s.
    """

    if p_clean.shape[:-1] != x_t.shape:
        raise ValueError("p_clean leading dimensions must match x_t")
    D = p_clean.shape[-1]
    if q_step.shape != (D, D) or qbar_s.shape != (D, D) or qbar_t.shape != (D, D):
        raise ValueError("transition matrices and p_clean state dimensions disagree")

    flat_xt = x_t.reshape(-1)
    flat_p0 = p_clean.reshape(-1, D)

    # q(x_t | x_s) for every possible previous state x_s.
    left = q_step[:, flat_xt].T                       # [M, D_s]
    # q(x_t | x_0) for every possible clean state x_0.
    denom = qbar_t[:, flat_xt].T.clamp_min(eps)      # [M, D_0]

    # q(x_s | x_t, x_0) ∝ q(x_t|x_s) q(x_s|x_0).
    # [M, D0, Ds]
    posterior_each_x0 = (
        qbar_s.unsqueeze(0) * left.unsqueeze(1)
    ) / denom.unsqueeze(-1)
    posterior_each_x0 = posterior_each_x0 / posterior_each_x0.sum(-1, keepdim=True).clamp_min(eps)

    out = (flat_p0.unsqueeze(-1) * posterior_each_x0).sum(dim=1)
    out = out.clamp_min(0)
    sums = out.sum(-1, keepdim=True)

    # If numerical zeros arise for an impossible state, fall back to p_clean.
    fallback = flat_p0 / flat_p0.sum(-1, keepdim=True).clamp_min(eps)
    out = torch.where(sums > eps, out / sums.clamp_min(eps), fallback)
    return out.reshape(*x_t.shape, D)
