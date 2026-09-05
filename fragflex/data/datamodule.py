from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split

from fragflex.graph import FragmentGraph

from .dataset import FragmentGraphDataset
from .stats import FragFlexStats


def collate_fragment_graph_list(batch: list[FragmentGraph]) -> list[FragmentGraph]:
    """Keep variable-size FragmentGraph objects as a Python list.

    FragFlex performs its own dense batching after drawing a corruption timestep,
    so the DataLoader should not stack the graph tensors itself.
    """

    return batch


class FragFlexDataModule(pl.LightningDataModule):
    """LightningDataModule for preprocessed FragFlex graphs.

    The default split mirrors FragDiffusion's fragment DataModule:
    20% test, then 80% of the remaining 80% for train, giving approximately
    64/16/20 train/validation/test. The split is deterministic by default.
    """

    def __init__(
        self,
        data_dir: str | Path,
        batch_size: int = 128,
        num_workers: int = 0,
        split_seed: int = 1234,
        test_fraction: float = 0.20,
        train_fraction_of_remaining: float = 0.80,
        pin_memory: bool = False,
    ) -> None:
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.split_seed = split_seed
        self.test_fraction = test_fraction
        self.train_fraction_of_remaining = train_fraction_of_remaining
        self.pin_memory = pin_memory

        self.stats = FragFlexStats.from_state_dict(
            torch.load(self.data_dir / "stats.pt", map_location="cpu", weights_only=False)
        )

        self.dataset: Optional[FragmentGraphDataset] = None
        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None

    def setup(self, stage: str | None = None) -> None:
        if self.dataset is None:
            self.dataset = FragmentGraphDataset.load(self.data_dir / "graphs.pt")

            n = len(self.dataset)
            if n < 3:
                raise ValueError("FragFlexDataModule needs at least 3 graphs for train/val/test splits")

            test_len = int(round(n * self.test_fraction))
            train_len = int(round((n - test_len) * self.train_fraction_of_remaining))
            val_len = n - train_len - test_len

            if min(train_len, val_len, test_len) <= 0:
                raise ValueError(
                    f"invalid split sizes train={train_len}, val={val_len}, test={test_len} for n={n}"
                )

            split_path = self.data_dir / "split_idxs.npz"
            if split_path.exists():
                split_idxs = np.load(split_path)
                required = {"train_idxs", "val_idxs", "test_idxs"}
                missing = required.difference(split_idxs.files)
                if missing:
                    raise ValueError(
                        f"{split_path} is missing split arrays: {sorted(missing)}"
                    )

                train_idxs = split_idxs["train_idxs"].astype(np.int64).tolist()
                val_idxs = split_idxs["val_idxs"].astype(np.int64).tolist()
                test_idxs = split_idxs["test_idxs"].astype(np.int64).tolist()
                all_idxs = train_idxs + val_idxs + test_idxs
                if len(all_idxs) != n or len(set(all_idxs)) != n:
                    raise ValueError(
                        f"{split_path} does not define a one-to-one split of all {n} graphs"
                    )
                if min(all_idxs) < 0 or max(all_idxs) >= n:
                    raise ValueError(f"{split_path} contains out-of-range graph indices")

                self.train_dataset = Subset(self.dataset, train_idxs)
                self.val_dataset = Subset(self.dataset, val_idxs)
                self.test_dataset = Subset(self.dataset, test_idxs)
            else:
                generator = torch.Generator().manual_seed(self.split_seed)
                self.train_dataset, self.val_dataset, self.test_dataset = random_split(
                    self.dataset,
                    [train_len, val_len, test_len],
                    generator=generator,
                )

    def _loader(self, dataset: Dataset, *, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=collate_fragment_graph_list,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            self.setup("fit")
        assert self.train_dataset is not None
        return self._loader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            self.setup("fit")
        assert self.val_dataset is not None
        return self._loader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        if self.test_dataset is None:
            self.setup("test")
        assert self.test_dataset is not None
        return self._loader(self.test_dataset, shuffle=False)
