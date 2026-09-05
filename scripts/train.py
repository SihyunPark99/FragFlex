#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Any

import hydra
import pytorch_lightning as pl
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from fragflex.config import FragFlexConfig
from fragflex.data import FragFlexDataModule
from fragflex.models import FragFlexModel


def _build_model_config(cfg: DictConfig) -> FragFlexConfig:
    """Convert Hydra's model/train groups into the compact FragFlexConfig.

    Upstream GrIDDD/FragDiffusion keep optimizer and seed settings in the train
    group, while architecture/diffusion settings live in model. FragFlex keeps
    the same public Hydra layout, then merges those fields only at the point
    where the LightningModule is constructed.
    """

    model_dict: dict[str, Any] = OmegaConf.to_container(
        cfg.model,
        resolve=True,
        throw_on_missing=True,
    )
    model_dict.pop("type", None)

    if model_dict.get("max_delta_per_step") is None:
        model_dict["max_delta_per_step"] = max(1, int(model_dict["max_nodes"]) - 1)

    model_dict.update(
        lr=float(cfg.train.lr),
        weight_decay=float(cfg.train.weight_decay),

        lr_scheduler=str(cfg.train.lr_scheduler),
        warmup_ratio=float(cfg.train.warmup_ratio),
        warmup_start_factor=float(cfg.train.warmup_start_factor),
        min_lr_ratio=float(cfg.train.min_lr_ratio),

        seed=int(cfg.train.seed),
    )
    return FragFlexConfig.from_dict(model_dict)


def _make_logger(cfg: DictConfig) -> WandbLogger | bool:
    mode = str(cfg.general.wandb)
    if mode == "disabled":
        return False
    if mode not in {"online", "offline"}:
        raise ValueError(f"general.wandb must be online/offline/disabled, got {mode!r}")

    logger = WandbLogger(
        project=str(cfg.general.project),
        entity=None if cfg.general.entity is None else str(cfg.general.entity),
        name=str(cfg.general.name),
        save_dir=str(cfg.general.wandb_dir),
        offline=(mode == "offline"),
        log_model=False,
    )

    # Hydra already writes .hydra/config.yaml into every run directory; logging
    # the same fully resolved config to W&B makes the dashboard self-contained.
    logger.log_hyperparams(
        OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    )
    return logger


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    print("Config:")
    print(OmegaConf.to_yaml(cfg, resolve=True))

    # Match the upstream projects' convention that cfg.train.seed is the single
    # experiment seed. workers=True also seeds DataLoader workers reproducibly.
    pl.seed_everything(int(cfg.train.seed), workers=True)

    data_dir = Path(to_absolute_path(str(cfg.dataset.datadir)))
    datamodule = FragFlexDataModule(
        data_dir=data_dir,
        batch_size=int(cfg.train.batch_size),
        num_workers=int(cfg.train.num_workers),
        split_seed=int(cfg.dataset.split_seed),
        test_fraction=float(cfg.dataset.test_fraction),
        train_fraction_of_remaining=float(cfg.dataset.train_fraction_of_remaining),
        pin_memory=bool(cfg.dataset.pin_memory),
    )
    datamodule.setup("fit")

    model_cfg = _build_model_config(cfg)
    model = FragFlexModel(model_cfg, datamodule.stats)

    # Hydra changes into outputs/<date>/<time>-<run_name>/, so checkpoints for
    # each experiment remain naturally isolated inside that run directory.
    ckpt_dir = Path(str(cfg.general.checkpoint_dir))
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="best-epoch{epoch:04d}",
        monitor="val/loss",
        mode="min",
        save_top_k=1,
        save_last=True,
        save_weights_only=False,
        every_n_epochs=1,
        save_on_train_epoch_end=False,
        auto_insert_metric_name=False,
    )

    logger = _make_logger(cfg)
    callbacks = [checkpoint_callback]
    if logger is not False:
        callbacks.append(LearningRateMonitor(logging_interval="epoch"))

    accelerator = str(cfg.general.accelerator)
    if accelerator == "gpu" and not torch.cuda.is_available():
        raise RuntimeError("general.accelerator=gpu but torch.cuda.is_available() is False")

    trainer = Trainer(
        accelerator=accelerator,
        devices=int(cfg.general.gpus),
        strategy=str(cfg.general.strategy),
        max_epochs=int(cfg.train.n_epochs),
        precision=str(cfg.general.precision),
        gradient_clip_val=None if cfg.train.clip_grad is None else float(cfg.train.clip_grad),
        check_val_every_n_epoch=int(cfg.general.check_val_every_n_epochs),
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=int(cfg.general.log_every_steps),
        enable_progress_bar=bool(cfg.general.progress_bar),
        accumulate_grad_batches=int(cfg.train.accumulate_grad_batches),
    )

    resume = cfg.general.resume
    ckpt_path = None if resume is None else to_absolute_path(str(resume))
    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)

    if trainer.is_global_zero:
        print(f"Run directory: {Path.cwd()}")
        print(f"Best checkpoint: {checkpoint_callback.best_model_path}")
        print(f"Best val/loss: {checkpoint_callback.best_model_score}")
        print(f"Last checkpoint: {checkpoint_callback.last_model_path}")


if __name__ == "__main__":
    main()
