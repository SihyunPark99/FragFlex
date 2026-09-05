from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterable, Optional

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import math

from fragflex.config import FragFlexConfig
from fragflex.data.stats import FragFlexStats
from fragflex.diffusion.one_atom import CorruptionBatch, OneAtomFragDiffusion
from fragflex.graph import DenseGraphBatch, FragmentGraph

from .graph_transformer import DeltaCountModel, FragFlexDenoiser


@dataclass
class LossOutput:
    loss: torch.Tensor
    node_loss: torch.Tensor
    edge_loss: torch.Tensor
    delta_loss: torch.Tensor
    site_loss: torch.Tensor
    node_accuracy: torch.Tensor
    edge_accuracy: torch.Tensor
    delta_accuracy: torch.Tensor
    site_accuracy: torch.Tensor
    site_top2_accuracy: torch.Tensor
    site_top4_accuracy: torch.Tensor


class FragFlexModel(pl.LightningModule):
    """FragFlex LightningModule: GrIDDD-style growth + FragDiffusion chemistry.

    The model keeps the same scientific core as the initial implementation, but
    exposes the PyTorch Lightning hooks used by the upstream GrIDDD/FragDiffusion
    codebases: ``training_step``, ``validation_step``, epoch hooks, and
    ``configure_optimizers``.
    """

    def __init__(
        self,
        config: FragFlexConfig | dict,
        stats: FragFlexStats | dict,
    ) -> None:
        super().__init__()

        if isinstance(config, dict):
            config = FragFlexConfig.from_dict(config)
        if isinstance(stats, dict):
            stats = FragFlexStats.from_state_dict(stats)

        self.config = config
        self.stats = stats

        # Save enough metadata for a standalone Lightning .ckpt to reconstruct
        # the model. logger=False keeps large empirical marginals out of W&B's
        # hyperparameter panel; scripts/train.py logs the concise run config.
        self.save_hyperparameters(
            {
                "config": config.to_dict(),
                "stats": stats.state_dict(),
            },
            logger=False,
        )

        self.diffusion = OneAtomFragDiffusion(config, stats)

        self.denoiser = FragFlexDenoiser(
            node_states=self.diffusion.node_state_count,
            edge_states=self.diffusion.edge_state_count,
            num_fragments=stats.num_fragments,
            num_edge_types=stats.num_edge_types,
            max_steps=config.diffusion_steps,
            d_model=config.d_model,
            d_edge=config.d_edge,
            n_layers=config.n_layers,
            n_heads=config.n_heads,
            ff_mult=config.ff_mult,
            dropout=config.dropout,
            max_fragment_atoms=(stats.max_fragment_atoms if stats.uses_neural_sites else 0),
            site_d_hidden=config.site_d_hidden,
            use_digress_cycles=config.use_digress_cycles,
            max_nodes=config.max_nodes,
        )
        self.delta_model = DeltaCountModel(
            node_states=self.diffusion.node_state_count,
            edge_states=self.diffusion.edge_state_count,
            max_steps=config.diffusion_steps,
            max_delta=config.max_delta_per_step,
            d_model=config.delta_d_model,
            d_edge=config.delta_d_edge,
            n_layers=config.delta_n_layers,
            n_heads=config.delta_n_heads,
            ff_mult=config.ff_mult,
            dropout=config.dropout,
        )

        self._epoch_start_time: float | None = None

    def forward(
        self, graph: DenseGraphBatch
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        return self.denoiser(graph)

    def transfer_batch_to_device(
        self,
        batch: list[FragmentGraph],
        device: torch.device,
        dataloader_idx: int,
    ) -> list[FragmentGraph]:
        """Leave graph objects on CPU; ``corrupt`` builds and transfers dense tensors.

        This mirrors the upstream pattern where sparse data are converted into a
        dense diffusion representation inside the model step, while avoiding
        Lightning trying to recursively move the custom FragmentGraph dataclass.
        """

        return batch

    def _edge_mask(self, node_mask: torch.Tensor) -> torch.Tensor:
        _, n = node_mask.shape
        pair = node_mask[:, :, None] & node_mask[:, None, :]
        upper = torch.triu(
            torch.ones((n, n), dtype=torch.bool, device=node_mask.device),
            diagonal=1,
        )
        return pair & upper.unsqueeze(0)

    def loss_from_corruption(self, corr: CorruptionBatch) -> LossOutput:
        node_logits, edge_logits, site_logits = self.denoiser(corr.noisy)
        delta_logits = self.delta_model(corr.delta_input)

        nm = corr.target_node_mask
        if nm.any():
            node_loss = F.cross_entropy(node_logits[nm], corr.target_x[nm])
            node_acc = (node_logits[nm].argmax(-1) == corr.target_x[nm]).float().mean()
        else:
            node_loss = node_logits.sum() * 0
            node_acc = node_loss.detach()

        em = self._edge_mask(corr.target_node_mask)
        if em.any():
            edge_loss = F.cross_entropy(edge_logits[em], corr.target_e[em])
            edge_acc = (edge_logits[em].argmax(-1) == corr.target_e[em]).float().mean()
        else:
            edge_loss = edge_logits.sum() * 0
            edge_acc = edge_loss.detach()

        # Neural atom-level endpoint reconstruction. This is an x0-style
        # auxiliary head: at every noisy timestep it predicts the clean local atom
        # used by each *directed* endpoint of a real fragment edge. The endpoint
        # support depends on the clean source-fragment identity, so padded atom
        # positions are masked before cross entropy.
        if self.stats.uses_neural_sites:
            if site_logits is None or corr.target_sites is None:
                raise RuntimeError("neural_sites mode requires site logits and targets")
            mmax = site_logits.shape[-1]
            counts_table = self.stats.fragment_atom_counts.to(site_logits.device)
            safe_x = corr.target_x.clamp(min=0, max=max(self.stats.num_fragments - 1, 0))
            counts = counts_table[safe_x].clamp(min=0, max=mmax)
            atom_pos = torch.arange(mmax, device=site_logits.device)
            valid_atom = atom_pos[None, None, :] < counts[:, :, None]
            site_logits_masked = site_logits.masked_fill(
                ~valid_atom[:, :, None, :], -1e9
            )
            pair = corr.target_node_mask[:, :, None] & corr.target_node_mask[:, None, :]
            sm = pair & (corr.target_e > 0) & (corr.target_sites >= 0)
            if sm.any():
                flat_site_logits = site_logits_masked[sm]
                flat_site_targets = corr.target_sites[sm]
                site_loss = F.cross_entropy(flat_site_logits, flat_site_targets)

                # Exact local-atom index accuracy at several candidate depths.
                # top-k is computed after masking padded atom positions, so the
                # metric measures whether the ground-truth endpoint is contained
                # in the model's k highest-scoring chemically available local
                # atom slots for that source fragment.  These diagnostics are
                # especially useful because the neural assembler searches over
                # top-k endpoint candidates rather than using only argmax.
                def _topk_site_accuracy(k: int) -> torch.Tensor:
                    k_eff = min(int(k), int(flat_site_logits.shape[-1]))
                    pred = flat_site_logits.topk(k_eff, dim=-1).indices
                    return pred.eq(flat_site_targets.unsqueeze(-1)).any(-1).float().mean()

                site_acc = _topk_site_accuracy(1)
                site_top2_acc = _topk_site_accuracy(2)
                site_top4_acc = _topk_site_accuracy(4)
            else:
                site_loss = site_logits.sum() * 0
                site_acc = site_loss.detach()
                site_top2_acc = site_loss.detach()
                site_top4_acc = site_loss.detach()
        else:
            site_loss = node_logits.sum() * 0
            site_acc = site_loss.detach()
            site_top2_acc = site_loss.detach()
            site_top4_acc = site_loss.detach()

        # GrIDDD trains the auxiliary count model only where the structural
        # event density is non-zero. Otherwise the target is trivially zero and
        # would overwhelm the rare, informative insertion boundaries.
        event_pmf = self.diffusion.schedules.event_pmf.to(delta_logits.device)
        dm = event_pmf[corr.delta_input.t] > 0
        if dm.any():
            delta_loss = F.cross_entropy(delta_logits[dm], corr.delta_target[dm])
            delta_acc = (delta_logits[dm].argmax(-1) == corr.delta_target[dm]).float().mean()
        else:
            delta_loss = delta_logits.sum() * 0
            delta_acc = delta_loss.detach()

        loss = (
            node_loss
            + self.config.lambda_edge * edge_loss
            + self.config.lambda_delta * delta_loss
            + self.config.lambda_site * site_loss
        )
        return LossOutput(
            loss,
            node_loss,
            edge_loss,
            delta_loss,
            site_loss,
            node_acc,
            edge_acc,
            delta_acc,
            site_acc,
            site_top2_acc,
            site_top4_acc,
        )

    def training_loss(self, graphs: Iterable[FragmentGraph]) -> LossOutput:
        """Legacy/manual API retained for tests and small research probes."""

        corr = self.diffusion.corrupt(graphs, device=self.device)
        return self.loss_from_corruption(corr)

    def _log_losses(self, prefix: str, out: LossOutput, batch_size: int, *, on_step: bool) -> None:
        common = {
            "on_step": on_step,
            "on_epoch": True,
            "batch_size": batch_size,
            "sync_dist": True,
        }
        self.log(f"{prefix}/loss", out.loss, prog_bar=True, **common)
        self.log(f"{prefix}/node_CE", out.node_loss, **common)
        self.log(f"{prefix}/edge_CE", out.edge_loss, **common)
        self.log(f"{prefix}/delta_CE", out.delta_loss, **common)
        if self.stats.uses_neural_sites:
            self.log(f"{prefix}/site_CE", out.site_loss, **common)
            # Keep the old site_acc key for backward-compatible dashboards.
            # site_acc and site_top1_acc are intentionally identical.
            self.log(f"{prefix}/site_acc", out.site_accuracy, **common)
            self.log(f"{prefix}/site_top1_acc", out.site_accuracy, **common)
            self.log(f"{prefix}/site_top2_acc", out.site_top2_accuracy, **common)
            self.log(f"{prefix}/site_top4_acc", out.site_top4_accuracy, **common)
        self.log(f"{prefix}/node_acc", out.node_accuracy, **common)
        self.log(f"{prefix}/edge_acc", out.edge_accuracy, **common)
        self.log(f"{prefix}/delta_acc", out.delta_accuracy, **common)

    def training_step(self, batch: list[FragmentGraph], batch_idx: int) -> torch.Tensor:
        out = self.training_loss(batch)
        self._log_losses("train", out, len(batch), on_step=True)
        return out.loss

    def validation_step(self, batch: list[FragmentGraph], batch_idx: int) -> torch.Tensor:
        corr = self.diffusion.corrupt(batch, device=self.device)
        out = self.loss_from_corruption(corr)
        self._log_losses("val", out, len(batch), on_step=False)
        return out.loss

    def test_step(self, batch: list[FragmentGraph], batch_idx: int) -> torch.Tensor:
        corr = self.diffusion.corrupt(batch, device=self.device)
        out = self.loss_from_corruption(corr)
        self._log_losses("test", out, len(batch), on_step=False)
        return out.loss

    def on_fit_start(self) -> None:
        self.print(
            "FragFlex model: "
            f"fragments={self.stats.num_fragments}, "
            f"atom_latents={self.stats.num_atom_latents}, "
            f"edge_types={self.stats.num_edge_types}, "
            f"assembly={self.stats.assembly_mode}, "
            f"root_prior={self.config.root_terminal_prior}, "
            f"zeta_schedule={self.config.zeta_schedule}, "
            f"digress_cycles={self.config.use_digress_cycles}, "
            f"T={self.config.diffusion_steps}"
        )

    def on_train_epoch_start(self) -> None:
        self.print(f"Starting epoch {self.current_epoch + 1}")
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self._epoch_start_time = time.perf_counter()

    def on_train_epoch_end(self) -> None:
        if self._epoch_start_time is None:
            return
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elapsed = time.perf_counter() - self._epoch_start_time
        self.log(
            "train/epoch_time_sec",
            elapsed,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=self.logger is not None,
            rank_zero_only=True,
        )
        self.print(f"Epoch {self.current_epoch + 1} training time: {elapsed:.2f}s")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )

        if self.config.lr_scheduler == "constant":
            return optimizer

        if self.config.lr_scheduler != "cosine":
            raise ValueError(
                f"Unknown lr_scheduler: {self.config.lr_scheduler}"
            )

        total_steps = int(self.trainer.estimated_stepping_batches)
        warmup_steps = int(total_steps * self.config.warmup_ratio)
        warmup_steps = max(warmup_steps, 1)

        min_factor = float(self.config.min_lr_ratio)
        start_factor = float(self.config.warmup_start_factor)

        def lr_lambda(step: int) -> float:
            # Linear warmup.
            if step < warmup_steps:
                progress = step / max(warmup_steps, 1)
                return start_factor + (1.0 - start_factor) * progress

            # Cosine decay after warmup.
            decay_steps = max(total_steps - warmup_steps, 1)
            progress = (step - warmup_steps) / decay_steps
            progress = min(max(progress, 0.0), 1.0)

            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))

            return min_factor + (1.0 - min_factor) * cosine

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lr_lambda,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
                "name": "train/lr",
            },
        }

    def optimizer(self) -> torch.optim.Optimizer:
        """Backward-compatible raw optimizer for legacy/manual scripts."""

        return torch.optim.AdamW(
            self.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )

    def _sample_delta_count(self, logits: torch.Tensor, t: int) -> torch.Tensor:
        # ``event_active`` is a CPU/Python tuple, so this branch never forces a
        # CUDA scalar synchronization. ``sample`` skips the delta-model forward
        # entirely when it is False.
        if not self.diffusion.event_active[t]:
            return torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        temp = max(self.config.delta_temperature, 1e-6)
        probs = torch.softmax(logits / temp, dim=-1)
        if self.config.sample_delta:
            n = torch.multinomial(probs, 1).squeeze(-1)
        else:
            n = probs.argmax(-1)
        return n

    @torch.inference_mode()
    def sample(
        self,
        batch_size: int,
        *,
        return_trace: bool = False,
        enforce_connectivity: bool = True,
    ) -> list[FragmentGraph] | tuple[list[FragmentGraph], list[DenseGraphBatch]]:
        """Generate with the Neural Assembly v1 semantics and optimized hot path.

        Scientific behavior is unchanged: the same delta model, denoiser, exact
        categorical posterior, connectivity repair, and t=1 neural-site logits
        are used. Sampling-only work whose output was discarded in v1 is skipped.
        """

        was_training = self.training
        self.eval()

        # ``OneAtomFragDiffusion`` is not an nn.Module. Keep all schedules and
        # empirical marginals resident on the sampling device instead of copying
        # them inside every reverse update.
        self.diffusion.to(self.device)
        state = self.diffusion.sample_limit(batch_size, self.device)
        trace: list[DenseGraphBatch] = [state.clone()] if return_trace else []
        final_site_logits: torch.Tensor | None = None

        ctemp_inv = 1.0 / max(self.config.categorical_temperature, 1e-6)

        for t in range(self.config.diffusion_steps, 0, -1):
            state.t.fill_(t)

            # The GrIDDD structural-event PMF is exactly zero on a subset of
            # timesteps. v1 still ran DeltaCountModel there and then discarded
            # its output. Skipping those forwards is mathematically exact.
            if self.diffusion.event_active[t]:
                delta_logits = self.delta_model(state)
                n_add = self._sample_delta_count(delta_logits, t)
                capacity = self.config.max_nodes - state.node_mask.sum(-1)
                n_add = torch.minimum(n_add, capacity.clamp_min(0).long())
                state = self.diffusion.insert_delt(state, n_add)
                state.t.fill_(t)

            # Neural-site logits never enter the diffusion posterior. v1 retains
            # only the t=1 logits for atom-level assembly, so computing the site
            # head on t=500..2 was pure discarded work.
            need_sites = bool(t == 1 and self.stats.uses_neural_sites)
            node_logits, edge_logits, site_logits = self.denoiser(
                state, compute_sites=need_sites
            )
            if need_sites:
                final_site_logits = site_logits

            node_p0 = torch.softmax(node_logits * ctemp_inv, dim=-1)
            edge_p0 = torch.softmax(edge_logits * ctemp_inv, dim=-1)
            state = self.diffusion.reverse_categorical(
                state,
                node_p0,
                edge_p0,
                s=t - 1,
                enforce_connectivity=enforce_connectivity,
            )

            if return_trace:
                trace.append(state.clone())

        graphs = self.diffusion.finalize(
            state,
            site_logits=final_site_logits,
            site_temperature=self.config.site_temperature,
        )
        if was_training:
            self.train()
        return (graphs, trace) if return_trace else graphs

    def checkpoint(self) -> dict:
        """Legacy compact checkpoint payload retained for backward compatibility."""

        return {
            "config": self.config.to_dict(),
            "stats": self.stats.state_dict(),
            "model": self.state_dict(),
        }

    @classmethod
    def from_checkpoint(
        cls,
        payload: dict,
        map_location: Optional[str | torch.device] = None,
    ) -> "FragFlexModel":
        """Load either an old FragFlex .pt payload or a Lightning .ckpt payload."""

        if "state_dict" in payload:
            hparams = payload.get("hyper_parameters", {})
            if "config" not in hparams or "stats" not in hparams:
                raise KeyError("Lightning checkpoint is missing FragFlex config/stats hyperparameters")
            config = FragFlexConfig.from_dict(hparams["config"])
            stats = FragFlexStats.from_state_dict(hparams["stats"])
            state = payload["state_dict"]
        else:
            config = FragFlexConfig.from_dict(payload["config"])
            stats = FragFlexStats.from_state_dict(payload["stats"])
            state = payload["model"]

        model = cls(config, stats)
        if map_location is not None:
            state = {k: v.to(map_location) for k, v in state.items()}
        model.load_state_dict(state)
        return model
