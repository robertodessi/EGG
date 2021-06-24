# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse

import torch

import egg.core as core
from egg.zoo.referential_language.data import get_dataloader
from egg.zoo.referential_language.game_callbacks import get_callbacks
from egg.zoo.referential_language.archs import build_game
from egg.zoo.emcom_as_ssl.LARC import LARC


def get_common_opts(params):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=10e-6,
        help="Weight decay used for SGD",
    )
    parser.add_argument(
        "--no_larc", action="store_true", default=False, help="Use plain SGD"
    )
    parser.add_argument(
        "--pdb",
        action="store_true",
        default=False,
        help="Run the game with pdb enabled",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        default=False,
        help="Run the game with wandb enabled",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="resnet50",
        choices=["resnet50", "resnet101", "resnet152"],
        help="Model name for the encoder",
    )
    parser.add_argument(
        "--projection_hidden_dim",
        type=int,
        default=128,
        help="Hidden dimension for class prediction",
    )
    parser.add_argument(
        "--num_classes", type=int, default=80, help="Num of prediction layer"
    )
    parser.add_argument(
        "--pretrain_vision",
        default=False,
        action="store_true",
        help="If set, pretrained vision modules will be used",
    )
    parser.add_argument("--image_size", type=int, default=224, help="Image size")
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Workers used in the dataloader"
    )

    opts = core.init(arg_parser=parser, params=params)
    opts.use_larc = not opts.no_larc
    return opts


def add_weight_decay(model, weight_decay=1e-5, skip_name=""):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or skip_name in name:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": no_decay, "weight_decay": 0.0},
        {"params": decay, "weight_decay": weight_decay},
    ]


def main(params):
    opts = get_common_opts(params=params)
    print(f"{opts}\n")
    assert (
        not opts.batch_size % 2
    ), f"Batch size must be multiple of 2. Found {opts.batch_size} instead"
    print(
        f"Running a distruted training is set to: {opts.distributed_context.is_distributed}. "
        f"World size is {opts.distributed_context.world_size}. "
        f"Using batch of size {opts.batch_size} on {opts.distributed_context.world_size} device(s)\n"
        f"Image size: {opts.image_size}.\n"
    )
    if not opts.distributed_context.is_distributed and opts.pdb:
        breakpoint()

    train_loader = get_dataloader(
        image_size=opts.image_size,
        batch_size=opts.batch_size,
        num_workers=opts.num_workers,
        is_distributed=opts.distributed_context.is_distributed,
        seed=opts.random_seed,
    )

    game = build_game(opts)

    model_parameters = add_weight_decay(game, opts.weight_decay, skip_name="bn")

    optimizer = torch.optim.SGD(
        model_parameters,
        lr=opts.lr,
        momentum=0.9,
    )
    optimizer_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=opts.n_epochs
    )

    if (
        opts.distributed_context.is_distributed
        and opts.distributed_context.world_size > 2
        and opts.use_larc
    ):
        optimizer = LARC(optimizer, trust_coefficient=0.001, clip=False, eps=1e-8)

    trainer = core.Trainer(
        game=game,
        optimizer=optimizer,
        optimizer_scheduler=optimizer_scheduler,
        train_data=train_loader,
        callbacks=get_callbacks(opts),
    )
    trainer.train(n_epochs=opts.n_epochs)

    print("| FINISHED JOB")


if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    import sys

    main(sys.argv[1:])
