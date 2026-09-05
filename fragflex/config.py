from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict


@dataclass
class FragFlexConfig:
    """Configuration for the reference FragFlex implementation.

    The implementation specializes GrIDDD to a deletion-only forward size process:
    every clean fragment graph is reduced to one surviving root fragment at T. The
    structural event schedule can be shifted toward the beginning of reverse sampling,
    and the categorical terminal prior of the protected root can independently be
    chosen as an atom-only prior or the empirical full-fragment prior.
    """

    diffusion_steps: int = 500
    # Structural schedule. Existing checkpoints use legacy_logistic, where
    # zeta_D is a forward-diffusion coordinate (D=0.50 -> t=250 for T=500).
    # sampling_logistic instead expresses the peak as reverse-sampling steps
    # from G_T; sampling_exponential front-loads insertions immediately.
    zeta_schedule: str = "legacy_logistic"
    zeta_D: float = 0.50
    zeta_w: float = 0.05
    zeta_sampling_peak_step: int = 100
    zeta_sampling_tau: float = 0.10
    zeta_event_rel_threshold: float = 1e-3
    cosine_s: float = 0.008

    # Terminal categorical prior for the single protected root at t=T.
    # Keep the dataclass default as ``atom`` for old-checkpoint compatibility.
    # New combined presets explicitly select ``fragment``.
    root_terminal_prior: str = "atom"  # atom | fragment

    # DiGress structural auxiliary features: node C3/C4/C5 counts and
    # graph-level n/max_nodes + C3/C4/C5/C6 counts. Disabled by default for
    # backward compatibility with existing checkpoints.
    use_digress_cycles: bool = False

    max_nodes: int = 32
    max_delta_per_step: int = 31

    d_model: int = 256
    d_edge: int = 64
    d_time: int = 64
    n_layers: int = 5
    n_heads: int = 8
    ff_mult: int = 4
    dropout: float = 0.0

    delta_d_model: int = 64
    delta_d_edge: int = 16
    delta_n_layers: int = 1
    delta_n_heads: int = 8

    lambda_edge: float = 5.0
    lambda_delta: float = 1.0
    # Auxiliary x0 attachment-site prediction loss. Used only when the dataset
    # was prepared with assembly_mode=neural_sites.
    lambda_site: float = 2.0
    site_d_hidden: int = 128
    site_temperature: float = 1.0
    lr: float = 2e-4
    weight_decay: float = 1e-12

    # Optimizer LR schedule.
    lr_scheduler: str = "constant"  # constant | cosine
    warmup_ratio: float = 0.05
    warmup_start_factor: float = 0.1
    min_lr_ratio: float = 0.05

    sample_delta: bool = True
    delta_temperature: float = 1.0
    categorical_temperature: float = 1.0

    seed: int = 0

    def __post_init__(self) -> None:
        self.root_terminal_prior = str(self.root_terminal_prior).lower()
        if self.root_terminal_prior not in {"atom", "fragment"}:
            raise ValueError(
                "root_terminal_prior must be 'atom' or 'fragment', "
                f"got {self.root_terminal_prior!r}"
            )

        valid_schedules = {
            "legacy_logistic",
            "sampling_logistic",
            "sampling_exponential",
        }
        if self.zeta_schedule not in valid_schedules:
            raise ValueError(
                f"zeta_schedule must be one of {sorted(valid_schedules)}, "
                f"got {self.zeta_schedule!r}"
            )
        if self.diffusion_steps < 2:
            raise ValueError("diffusion_steps must be >= 2")
        if self.zeta_w <= 0:
            raise ValueError("zeta_w must be > 0")
        if self.zeta_schedule == "sampling_logistic" and not (
            0 <= self.zeta_sampling_peak_step < self.diffusion_steps
        ):
            raise ValueError(
                "zeta_sampling_peak_step must be in "
                f"[0, {self.diffusion_steps - 1}] for sampling_logistic"
            )
        if self.zeta_sampling_tau <= 0:
            raise ValueError("zeta_sampling_tau must be > 0")
        if not (0 <= self.zeta_event_rel_threshold < 1):
            raise ValueError("zeta_event_rel_threshold must be in [0, 1)")
            
        if self.lr_scheduler not in {"constant", "cosine"}:
            raise ValueError(
                "lr_scheduler must be 'constant' or 'cosine', "
                f"got {self.lr_scheduler!r}"
            )

        if not (0.0 <= self.warmup_ratio < 1.0):
            raise ValueError("warmup_ratio must be in [0, 1)")

        if not (0.0 < self.warmup_start_factor <= 1.0):
            raise ValueError("warmup_start_factor must be in (0, 1]")

        if not (0.0 <= self.min_lr_ratio <= 1.0):
            raise ValueError("min_lr_ratio must be in [0, 1]")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FragFlexConfig":
        return cls(**d)
