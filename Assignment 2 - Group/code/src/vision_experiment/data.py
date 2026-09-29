"""Deterministic CIFAR data and once-sampled incorrect-label noise."""

from pathlib import Path
from typing import Literal

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10, CIFAR100


class CIFARExperimentDataset(Dataset[tuple[Tensor, int, int]]):
    def __init__(
        self,
        dataset: Literal["cifar10", "cifar100"],
        root: str | Path,
        train: bool,
        download: bool,
        sample_count: int | None,
        label_noise: float,
        augmentation: bool,
        seed: int,
    ) -> None:
        source = CIFAR10(root=root, train=train, download=download) if dataset == "cifar10" else CIFAR100(
            root=root, train=train, download=download
        )
        if not 0 <= label_noise <= 1:
            raise ValueError("label_noise must be between zero and one")
        if not train and (label_noise != 0 or sample_count is not None or augmentation):
            raise ValueError("test data must keep clean labels and the full split")
        self.images = source.data
        self.clean_labels = torch.tensor(source.targets, dtype=torch.long)
        self.num_classes = 10 if dataset == "cifar10" else 100
        self.seed = seed
        self.epoch = 0
        self.augmentation = augmentation
        total = len(self.clean_labels)
        if sample_count is not None and not 1 <= sample_count <= total:
            raise ValueError(f"sample_count must be in [1, {total}]")
        subset_rng = torch.Generator().manual_seed(seed)
        self.indices = torch.randperm(total, generator=subset_rng)[:sample_count] if sample_count else torch.arange(total)
        self.labels = self.clean_labels.clone()
        if label_noise:
            noise_rng = torch.Generator().manual_seed(seed + 10_000_019)
            corrupt = torch.rand(total, generator=noise_rng) < label_noise
            offsets = torch.randint(1, self.num_classes, (total,), generator=noise_rng)
            self.labels[corrupt] = (self.labels[corrupt] + offsets[corrupt]) % self.num_classes

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int) -> tuple[Tensor, int, int]:
        index = int(self.indices[position])
        image = torch.from_numpy(self.images[index].copy()).permute(2, 0, 1).float() / 255.0
        if self.augmentation:
            rng = torch.Generator().manual_seed(self.seed + self.epoch * 1_000_003 + index)
            image = F.pad(image, (4, 4, 4, 4))
            top = int(torch.randint(0, 9, (1,), generator=rng))
            left = int(torch.randint(0, 9, (1,), generator=rng))
            image = image[:, top : top + 32, left : left + 32]
            if bool(torch.randint(0, 2, (1,), generator=rng)):
                image = torch.flip(image, dims=(2,))
        return image, int(self.labels[index]), index
