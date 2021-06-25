# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from egg.core.interaction import LoggingStrategy


def get_vision_module(encoder_arch: str = "resnet50"):
    """Loads ResNet encoder from torchvision along with features number"""
    resnets = {
        "resnet50": torchvision.models.resnet50(),
        "resnet101": torchvision.models.resnet101(),
        "resnet152": torchvision.models.resnet152(),
    }
    if encoder_arch not in resnets:
        raise KeyError(f"{encoder_arch} is not a valid ResNet architecture")

    model = resnets[encoder_arch]
    n_features = model.fc.in_features
    model.fc = nn.Identity()

    return model, n_features


class ProjectSender(nn.Module):
    def __init__(
        self,
        vision_encoder: nn.Module,
        visual_features_dim: int,
        hidden_dim: int = 128,
        output_dim: int = 80,
        merge_mode: str = "sum",
    ):
        super(ProjectSender, self).__init__()

        self.vision_encoder = vision_encoder

        self.fc_coord = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
        )

        self.fc_img = nn.Sequential(
            nn.Linear(visual_features_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
        )

        if merge_mode not in ["sum", "cat", "mul"]:
            raise ValueError(f"Cannot recognize merge mode {merge_mode}")

        self.fc_out = (
            nn.Linear(hidden_dim * 2, output_dim)
            if merge_mode == "cat"
            else nn.Linear(hidden_dim, output_dim)
        )

        self.merge_mode = merge_mode

    def forward(self, image, coordinates):
        resnet_output = self.vision_encoder(image)
        visual_feats = self.fc_img(resnet_output)

        coord_feats = self.fc_coord(coordinates)

        if self.merge_mode == "cat":
            vision_and_coord = torch.cat([visual_feats, coord_feats], dim=-1)
        elif self.merge_mode == "sum":
            vision_and_coord = visual_feats + coord_feats
        elif self.merge_mode == "mul":
            vision_and_coord = torch.mul(visual_feats, coord_feats)
        else:
            raise RuntimeError(f"Cannot recognize merge_mode {self.merge_mode}")

        return self.fc_out(vision_and_coord)


class CatSender(nn.Module):
    def __init__(
        self,
        vision_encoder: nn.Module,
        visual_features_dim: int,
        hidden_dim: int = 128,
        output_dim: int = 80,
    ):
        super(CatSender, self).__init__()

        self.vision_encoder = vision_encoder

        self.fc_out = nn.Sequential(
            nn.Linear(visual_features_dim + 4, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, output_dim, bias=False),
        )

    def forward(self, image, coordinates):
        resnet_output = self.vision_encoder(image)
        vision_and_coord = torch.cat([resnet_output, coordinates], dim=-1)
        return torch.tanh(self.fc_out(vision_and_coord))


class ClassPredictionGame(nn.Module):
    def __init__(
        self,
        sender: nn.Module,
        loss: Callable,
        train_logging_strategy: Optional[LoggingStrategy] = None,
        test_logging_strategy: Optional[LoggingStrategy] = None,
    ):
        super(ClassPredictionGame, self).__init__()

        self.sender = sender
        self.loss = loss

        self.train_logging_strategy = (
            LoggingStrategy()
            if train_logging_strategy is None
            else train_logging_strategy
        )
        self.test_logging_strategy = (
            LoggingStrategy()
            if test_logging_strategy is None
            else test_logging_strategy
        )

    def forward(self, sender_input, labels, receiver_input=None):
        image, coordinates = sender_input
        sender_prediction = self.sender(image, coordinates)
        loss, aux_info = self.loss(sender_prediction, labels)

        logging_strategy = (
            self.train_logging_strategy if self.training else self.test_logging_strategy
        )
        interaction = logging_strategy.filtered_interaction(
            sender_input=sender_input,
            receiver_input=receiver_input,
            labels=labels,
            receiver_output=None,
            message=sender_prediction,
            message_length=torch.ones(sender_prediction.size(0)),
            aux=aux_info,
        )

        return loss.mean(), interaction


def loss(predictions, labels):
    acc = (predictions.argmax(dim=1) == labels).detach().float()
    loss = F.cross_entropy(predictions, labels, reduction="none")
    return loss, {"acc": acc}


def build_game(opts):
    sender_vision_module, visual_features_dim = get_vision_module(
        encoder_arch=opts.model_name
    )

    logging_strategy = LoggingStrategy(False, False, True, True, True, False)

    if opts.sender_type == "cat":
        sender = CatSender(
            vision_encoder=sender_vision_module,
            visual_features_dim=visual_features_dim,
            hidden_dim=opts.projection_hidden_dim,
            output_dim=opts.num_classes,
        )
    elif opts.sender_type == "proj":
        sender = ProjectSender(
            vision_encoder=sender_vision_module,
            visual_features_dim=visual_features_dim,
            hidden_dim=opts.projection_hidden_dim,
            output_dim=opts.num_classes,
            merge_mode=opts.merge_mode,
        )
    else:
        raise ValueError(f"Cannot recognize sender type {opts.sender_type}")

    game = ClassPredictionGame(
        sender,
        loss,
        train_logging_strategy=logging_strategy,
        test_logging_strategy=logging_strategy,
    )

    return game
