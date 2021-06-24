# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torchvision import datasets, transforms


def get_dataloader(
    train_dataset_dir: str = "/datasets01/COCO/022719/val2017",
    train_annotation_path: str = "/datasets01/COCO/022719/annotations/instances_val2017.json",
    image_size: int = 32,
    batch_size: int = 32,
    num_workers: int = 4,
    is_distributed: bool = False,
    seed: int = 111,
):
    transform = transforms.Compose(
        [transforms.Resize(size=(image_size, image_size)), transforms.ToTensor()]
    )
    target_transform = BoxResize((image_size, image_size))
    train_dataset = MyCocoDetection(
        train_dataset_dir,
        train_annotation_path,
        transform=transform,
        target_transform=target_transform,
    )

    # valid_dataset_dir: str = "/datasets01/COCO/022719/val2017",
    # valid_annotation_path: str = "/datasets01/COCO/022719/annotations/instances_val2017.json",

    train_sampler = None
    if is_distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, shuffle=True, drop_last=True, seed=seed
        )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    return train_loader


class MyCocoDetection(datasets.CocoDetection):
    def __init__(self, *args, **kwargs):
        super(MyCocoDetection, self).__init__(*args, **kwargs)

    def __getitem__(self, index):
        img, target = self._load_image_and_target(index)
        idx = index
        # avoiding cases where target is an empty list through linear search
        while len(target) == 0:
            idx = (idx + 1) % len(self.ids)
            img, target = self._load_image_and_target(idx)

        img_size = list(img.size)
        img = self.transform(img)

        coords = torch.Tensor(target[0]["bbox"])
        coords[2] = coords[0] + coords[2]
        coords[3] = coords[1] + coords[3]

        resized_bbox_coord = self.target_transform(coords, original_size=img_size)

        label = target[0]["category_id"]

        return (img, resized_bbox_coord), label

    def _load_image_and_target(self, index):
        img_id = self.ids[index]
        img = self._load_image(img_id)
        target = self.coco.loadAnns(self.coco.getAnnIds(img_id))
        return img, target


class BoxResize:
    def __init__(self, new_size):
        self.new_size = new_size

    @staticmethod
    def resize_boxes(boxes, original_size, new_size):
        ratios = []
        for s, s_orig in zip(new_size, original_size):
            ratios.append(
                torch.tensor(s, dtype=torch.float32, device=boxes.device)
                / torch.tensor(s_orig, dtype=torch.float32, device=boxes.device)
            )

        ratio_height, ratio_width = ratios
        xmin, ymin, xmax, ymax = boxes

        xmin = xmin * ratio_width
        xmax = xmax * ratio_width
        ymin = ymin * ratio_height
        ymax = ymax * ratio_height
        return torch.Tensor((xmin, ymin, xmax, ymax))

    def __call__(self, bbox, original_size):
        return self.resize_boxes(bbox, original_size, self.new_size)
