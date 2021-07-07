# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict
import uuid

import torch
import wandb

from egg.core import Callback, ConsoleLogger, EarlyStopperAccuracy, Interaction
from egg.core.callbacks import WandbLogger
from egg.zoo.emcom_as_ssl.game_callbacks import (
    BestStatsTracker,
)


class VisionModelSaver(Callback):
    """A callback that stores vision module(s) in trainer's checkpoint_dir, if any."""

    def save_vision_model(self, epoch=""):
        if hasattr(self.trainer, "checkpoint_path"):
            self.trainer.checkpoint_path.mkdir(exist_ok=True, parents=True)
            vision_module = self.trainer.game.sender.vision_encoder

            model_name = f"vision_module_{epoch if epoch else 'final'}.pt"
            torch.save(
                vision_module.state_dict(),
                self.trainer.checkpoint_path / model_name,
            )

    def on_train_end(self):
        self.save_vision_model()

    def on_epoch_end(self, loss: float, _logs: Interaction, epoch: int):
        self.save_vision_model(epoch=epoch)


class MyWandbLogger(WandbLogger):
    def __init__(self, *args, **kwargs):
        super(MyWandbLogger, self).__init__(*args, **kwargs)

    @staticmethod
    def log_to_wandb(metrics: Dict[str, Any], commit: bool = False, **kwargs):
        wandb.log(metrics, commit=commit, **kwargs)

    def on_train_begin(self, trainer_instance: "Trainer"):  # noqa: F821
        self.trainer = trainer_instance
        wandb.watch(self.trainer.game, log="all")

    def on_batch_end(
        self, logs: Interaction, loss: float, batch_id: int, is_training: bool = True
    ):
        if is_training and self.trainer.distributed_context.is_leader:
            self.log_to_wandb(
                {"batch_loss": loss, "batch_accuracy": logs.aux["acc"]}, commit=True
            )

    def on_epoch_end(self, loss: float, logs: Interaction, epoch: int):
        if self.trainer.distributed_context.is_leader:
            self.log_to_wandb(
                {"train_loss": loss, "epoch_accuracy": logs.aux["acc"], "epoch": epoch},
                commit=True,
            )


def get_callbacks(opts):
    callbacks = [
        ConsoleLogger(as_json=True, print_train_loss=True),
        BestStatsTracker(),
        EarlyStopperAccuracy(0.95, validation=False),
    ]

    if opts.wandb:
        run_name = opts.checkpoint_dir.split("/")[-1] if opts.checkpoint_dir else ""
        run_id = f"{run_name}_{str(uuid.uuid4())}"
        callbacks.append(MyWandbLogger(opts, "sender_only_baseline", run_id))

    return callbacks
