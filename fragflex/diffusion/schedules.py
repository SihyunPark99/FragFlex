from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch


@dataclass
class DiffusionSchedules:
    alpha_step: torch.Tensor   # [T+1], alpha_step[0]=1
    alpha_bar: torch.Tensor    # [T+1], alpha_bar[0]=1, alpha_bar[T]=0
    zeta: torch.Tensor         # [T+1]
    event_pmf: torch.Tensor    # [T+1], event_pmf[0]=event_pmf[T]=0
    zeta_bar: torch.Tensor     # first index is t; second index is activation time s

    @property
    def T(self) -> int:
        return self.alpha_bar.numel() - 1

    def to(self, device: torch.device | str) -> "DiffusionSchedules":
        return DiffusionSchedules(
            self.alpha_step.to(device),
            self.alpha_bar.to(device),
            self.zeta.to(device),
            self.event_pmf.to(device),
            self.zeta_bar.to(device),
        )

    @classmethod
    def build(
        cls,
        T: int,
        zeta_D: float,
        zeta_w: float,
        cosine_s: float = 0.008,
        *,
        zeta_schedule: str = "legacy_logistic",
        zeta_sampling_peak_step: int = 100,
        zeta_sampling_tau: float = 0.10,
        zeta_event_rel_threshold: float = 1e-3,
    ) -> "DiffusionSchedules":
        """Build categorical and structural schedules.

        Structural time has two coordinate systems in FragFlex:

        * forward diffusion: ``t = 0 -> T``
        * reverse sampling:   ``t = T -> 0``

        The original GrIDDD-style ``legacy_logistic`` schedule uses ``zeta_D``
        in *forward* coordinates.  The two ``sampling_*`` schedules are added to
        make early reverse growth explicit and much harder to misconfigure:

        ``sampling_logistic``
            ``zeta_sampling_peak_step=k`` places the structural-event density
            peak approximately k reverse iterations after sampling starts.  For
            T=500 and k=100 this is forward t=400.

        ``sampling_exponential``
            Structural-event density is maximal at the first legal reverse
            insertion boundary (forward t=T-1) and decays as reverse sampling
            proceeds. ``zeta_sampling_tau`` is a fraction of T, so 0.10 means a
            50-step time constant when T=500.

        For the new schedules, ``zeta`` is derived from the event PMF as a
        discrete survival/hazard process.  Therefore direct event-time sampling,
        q-step transitions, and q-bar posteriors are mutually consistent.
        """

        alpha_step, alpha_bar = _cosine_discrete(T, cosine_s)

        schedule = str(zeta_schedule).lower()
        if schedule == "legacy_logistic":
            # Exact old behavior for checkpoint/config backward compatibility.
            zeta, event = _linear_zeta(T, zeta_D, zeta_w)
        elif schedule == "sampling_logistic":
            event = _sampling_logistic_event_pmf(
                T,
                peak_step=int(zeta_sampling_peak_step),
                width=float(zeta_w),
                rel_threshold=float(zeta_event_rel_threshold),
            )
            zeta = _zeta_from_event_pmf(event)
        elif schedule == "sampling_exponential":
            event = _sampling_exponential_event_pmf(
                T,
                tau=float(zeta_sampling_tau),
                rel_threshold=float(zeta_event_rel_threshold),
            )
            zeta = _zeta_from_event_pmf(event)
        else:
            raise ValueError(
                "zeta_schedule must be one of "
                "{'legacy_logistic', 'sampling_logistic', 'sampling_exponential'}, "
                f"got {zeta_schedule!r}"
            )

        zbar = _zeta_bar_matrix(zeta)
        return cls(
            alpha_step=torch.from_numpy(alpha_step).float(),
            alpha_bar=torch.from_numpy(alpha_bar).float(),
            zeta=torch.from_numpy(zeta).float(),
            event_pmf=torch.from_numpy(event).float(),
            zeta_bar=torch.from_numpy(zbar).float(),
        )


def _cosine_discrete(T: int, s: float) -> tuple[np.ndarray, np.ndarray]:
    # Standard DiGress-style discrete cosine schedule, made explicit at t=0/T.
    x = np.linspace(0, T, T + 1, dtype=np.float64)
    abar = np.cos(((x / T) + s) / (1 + s) * math.pi * 0.5) ** 2
    abar = abar / abar[0]
    abar[-1] = 0.0
    alpha = np.ones(T + 1, dtype=np.float64)
    for t in range(1, T + 1):
        if abar[t - 1] <= 0:
            alpha[t] = 0.0
        else:
            alpha[t] = np.clip(abar[t] / abar[t - 1], 0.0, 0.9999)
    # Reconstruct the cumulative product for numerical consistency.
    abar2 = np.ones(T + 1, dtype=np.float64)
    for t in range(1, T + 1):
        abar2[t] = abar2[t - 1] * alpha[t]
    abar2[-1] = 0.0
    return alpha, abar2


def _linear_zeta(T: int, D: float, w: float, thr: float = 1e-5) -> tuple[np.ndarray, np.ndarray]:
    """Original FragFlex/GrIDDD-style schedule retained byte-for-byte in behavior.

    ``D`` is in forward-diffusion coordinates.  It is intentionally kept as a
    separate legacy mode because existing checkpoints were trained with these
    exact transition probabilities.
    """

    t = np.linspace(0, 1, T)
    sigmaD = 1.0 / (1.0 + np.exp(-(D - t) / w))
    z = sigmaD.copy()
    z[-1] = 0
    z[z < thr] = 0
    z[z > 1 - thr] = 1
    dz = np.abs(np.gradient(z))
    z = np.concatenate(([1.0], z))
    dz = np.concatenate(([0.0], dz))
    z[-1] = 0
    dz[-1] = 0
    dz[dz < thr] = 0
    if dz.sum() <= 0:
        raise ValueError("zeta event PMF is empty; change zeta_D/zeta_w")
    dz /= dz.sum()
    return z.astype(np.float64), dz.astype(np.float64)


def _threshold_and_normalize_event_pmf(
    raw: np.ndarray,
    *,
    rel_threshold: float,
) -> np.ndarray:
    """Restrict an event density to legal times 1..T-1 and normalize it."""

    if raw.ndim != 1 or raw.size < 3:
        raise ValueError("event density must be a 1D array with T+1 >= 3 entries")
    if not (0.0 <= rel_threshold < 1.0):
        raise ValueError("zeta_event_rel_threshold must be in [0, 1)")

    out = np.asarray(raw, dtype=np.float64).copy()
    out[~np.isfinite(out)] = 0.0
    out = np.clip(out, 0.0, None)

    # t=0 is the clean state, and t=T must already be the one-root terminal
    # state.  Hence a structural event is legal only at 1..T-1.
    out[0] = 0.0
    out[-1] = 0.0

    mx = float(out.max())
    if mx <= 0.0:
        raise ValueError("zeta event PMF is empty")
    if rel_threshold > 0.0:
        out[out < mx * rel_threshold] = 0.0

    total = float(out.sum())
    if total <= 0.0:
        raise ValueError("zeta event PMF is empty after thresholding")
    out /= total
    return out


def _sampling_logistic_event_pmf(
    T: int,
    *,
    peak_step: int,
    width: float,
    rel_threshold: float,
) -> np.ndarray:
    """Event PMF with its peak expressed in reverse-sampling iterations.

    ``peak_step=100`` with T=500 means the peak is at forward t=400 because the
    reverse process traverses 500, 499, ..., 400 during its first 100 steps.
    """

    if T < 2:
        raise ValueError("T must be >= 2")
    if not (0 <= peak_step < T):
        raise ValueError(f"zeta_sampling_peak_step must be in [0, {T - 1}]")
    if width <= 0.0:
        raise ValueError("zeta_w must be > 0")

    t = np.arange(T + 1, dtype=np.float64)
    reverse_step = T - t
    center = float(peak_step)
    width_steps = float(width) * float(T)

    x = (reverse_step - center) / width_steps
    # Stable logistic PDF up to a constant scale.  The omitted 1/width factor
    # cancels during normalization.
    sig = np.empty_like(x)
    pos = x >= 0
    sig[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    sig[~pos] = ex / (1.0 + ex)
    raw = sig * (1.0 - sig)
    return _threshold_and_normalize_event_pmf(raw, rel_threshold=rel_threshold)


def _sampling_exponential_event_pmf(
    T: int,
    *,
    tau: float,
    rel_threshold: float,
) -> np.ndarray:
    """Front-load insertion boundaries at the beginning of reverse sampling.

    The conceptual maximum is sampling step 0 (forward t=T), but t=T itself is
    reserved for the one-fragment terminal state.  Therefore the largest legal
    event mass is at t=T-1, i.e. the first reverse insertion boundary.
    """

    if T < 2:
        raise ValueError("T must be >= 2")
    if tau <= 0.0:
        raise ValueError("zeta_sampling_tau must be > 0")

    t = np.arange(T + 1, dtype=np.float64)
    reverse_step = T - t
    tau_steps = float(tau) * float(T)

    # Shift by one so the first legal boundary t=T-1 has raw weight exactly 1.
    distance_from_first_legal = np.maximum(reverse_step - 1.0, 0.0)
    raw = np.exp(-distance_from_first_legal / tau_steps)
    return _threshold_and_normalize_event_pmf(raw, rel_threshold=rel_threshold)


def _zeta_from_event_pmf(event: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    """Convert an event-time PMF into per-step survival probabilities.

    If E is a deletion event time, ``zeta[t] = P(E > t | E >= t)``.  This makes
    ``P(E=t) = prod_{k<t} zeta[k] * (1-zeta[t])`` exactly equal to event[t].
    """

    event = np.asarray(event, dtype=np.float64)
    T = event.size - 1
    if T < 2:
        raise ValueError("event PMF must have at least three entries")
    if abs(float(event.sum()) - 1.0) > 1e-8:
        raise ValueError("event PMF must sum to one")
    if event[0] != 0.0 or event[-1] != 0.0:
        raise ValueError("event PMF endpoints must be zero")

    z = np.ones(T + 1, dtype=np.float64)
    remaining = 1.0
    for t in range(1, T):
        p = float(event[t])
        if remaining <= eps:
            z[t] = 0.0
            remaining = 0.0
            continue
        hazard = min(max(p / remaining, 0.0), 1.0)
        z[t] = 1.0 - hazard
        remaining = max(remaining - p, 0.0)

    # All non-root nodes must be structurally gone before G_T.
    z[T] = 0.0
    return z


def _zeta_bar_matrix(z: np.ndarray, thr: float = 1e-12) -> np.ndarray:
    # Port of the cumulative zeta construction used by GrIDDD; shape [t,s].
    TT = z.shape[0]
    out = np.zeros((TT - 1, TT), dtype=np.float64)
    for s in range(TT - 1):
        a = np.cumprod(z[s + 1 :])
        a[a <= thr] = thr
        a[-1] = 0
        out[s, s + 1 :] = a
        out[s, s] = 1
    return out.T
