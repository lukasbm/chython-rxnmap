# -*- coding: utf-8 -*-
#
#  Copyright 2021, 2022 Ramil Nugmanov <nougmanoff@protonmail.com>
#  This file is part of chytorch.
#
#  chytorch is free software; you can redistribute it and/or modify
#  it under the terms of the GNU Lesser General Public License as published by
#  the Free Software Foundation; either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#  GNU Lesser General Public License for more details.
#
#  You should have received a copy of the GNU Lesser General Public License
#  along with this program; if not, see <https://www.gnu.org/licenses/>.
#
from functools import partial
from importlib.resources import files
from math import cos, pi
from typing import Callable, Iterator

from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from torch import rand
from torch.nn import LazyLinear, Parameter
from torch.nn.functional import cross_entropy
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader

from chytorch.nn import ReactionEncoder
from chytorch.utils.data import ReactionDataset, collate_reactions


class WarmUpCosine(_LRScheduler):
    """Learning rate scheduler with warmup followed by cosine annealing."""

    def __init__(
        self,
        optimizer: Optimizer,
        warmup: int = 10000,
        period: int = 500000,
        decrease_coef: float = 0.01,
        last_epoch: int = -1,
    ):
        self.warmup = warmup
        self.period = period
        self.decrease_coef = decrease_coef
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        current_step = self.last_epoch
        if current_step < self.warmup:
            # Linear warmup
            return [base_lr * (current_step / self.warmup) for base_lr in self.base_lrs]
        else:
            # Cosine annealing
            progress = (current_step - self.warmup) / (self.period - self.warmup)
            progress = min(progress, 1.0)
            return [
                base_lr
                * (
                    self.decrease_coef
                    + (1 - self.decrease_coef) * (1 + cos(pi * progress)) / 2
                )
                for base_lr in self.base_lrs
            ]


class Model(LightningModule):
    def __init__(
        self,
        *,
        masking_rate=0.15,
        lr_scheduler: Callable[[Optimizer], _LRScheduler] = None,
        optimizer: Callable[[Iterator[Parameter]], Optimizer] = None,
        **kwargs,
    ):
        super().__init__()
        self.encoder = ReactionEncoder(**kwargs)
        self.mlma = LazyLinear(118)
        self.mlmn = LazyLinear(
            self.encoder.molecule_encoder.centrality_encoder.num_embeddings - 2
        )

        if lr_scheduler is None:
            lr_scheduler = partial(
                WarmUpCosine, decrease_coef=0.01, warmup=int(1e4), period=int(5e5)
            )
        if optimizer is None:
            optimizer = partial(AdamW, lr=1e-4)

        self.lr_scheduler = lr_scheduler
        self.optimizer = optimizer
        self.masking_rate = masking_rate
        self.save_hyperparameters(kwargs)

    @classmethod
    def pretrained(cls, **kwargs):
        weights_path = files(__package__).joinpath("weights.pt")
        model = cls.load_from_checkpoint(
            str(weights_path), map_location="cpu", **kwargs
        )
        model.eval()
        return model

    def prepare_dataloader(self, reactions, **kwargs):
        """
        Prepare dataloader for training.

        :param reactions: chython packed reactions list.
        """
        ds = ReactionDataset(
            reactions, distance_cutoff=self.encoder.max_distance, unpack=True
        )
        return DataLoader(ds, collate_fn=collate_reactions, **kwargs)

    def forward(self, batch, *, mapping_task=False):
        if mapping_task:
            return self.encoder(batch, need_embedding=False, need_weights=True)
        return self.encoder(batch)

    def training_step(self, batch, batch_idx):
        a, n, d, r = batch
        m = r > 1  # atoms only
        ma = a.masked_fill((rand(a.shape, device=a.device) < self.masking_rate) & m, 2)
        mn = n.masked_fill((rand(n.shape, device=n.device) < self.masking_rate) & m, 1)

        x = self.encoder((ma, mn, d, r))[m]  # atoms only embedding
        atoms = self.mlma(x)
        neighbors = self.mlmn(x)

        l1 = cross_entropy(atoms, a[m].long() - 3)
        l2 = cross_entropy(neighbors, n[m].long() - 2)
        self.log("trn_loss_mlm_a", l1.item(), sync_dist=True)
        self.log("trn_loss_mlm_n", l2.item(), sync_dist=True)
        self.log("trn_loss_tot", l1.item() + l2.item(), sync_dist=True)
        return l1 + l2

    def configure_callbacks(self):
        return [
            ModelCheckpoint(
                save_weights_only=True, save_last=True, every_n_train_steps=10000
            ),
            LearningRateMonitor(logging_interval="step"),
        ]

    def configure_optimizers(self):
        o = self.optimizer(self.parameters())
        s = self.lr_scheduler(o)
        return [o], [{"scheduler": s, "interval": "step", "name": "lr_scheduler"}]


__all__ = ["Model"]
