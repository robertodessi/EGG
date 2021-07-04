# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import Counter, defaultdict
from typing import Callable, List, Optional, Tuple

import torch
from torchvision import datasets, transforms

from egg.zoo.referential_language.utils import cat_id2id_and_name
from pycocotools.coco import COCO


def get_dataloader(
    train_dataset_dir: str = "/datasets01/COCO/022719/train2017",
    train_annotation_path: str = "/datasets01/COCO/022719/annotations/instances_train2017.json",
    image_size: int = 32,
    batch_size: int = 32,
    num_workers: int = 4,
    random_coord: bool = False,
    filter_bbox: str = "minority",
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
        random_coord=random_coord,
        filter_bbox=filter_bbox,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    return train_loader


class MyCOCO(COCO):
    def createIndex(self):
        # create index
        print("creating index...")
        anns, cats, imgs = {}, {}, {}
        imgToAnns, catToImgs = defaultdict(list), defaultdict(list)

        removed = 0
        if "annotations" in self.dataset:
            for ann in self.dataset["annotations"]:
                # removing group annotation, when iscrowd is 1 there's a single bbox around
                # a group of instances of the same category
                if ann["iscrowd"]:
                    removed += 1
                    continue

                imgToAnns[ann["image_id"]].append(ann)
                anns[ann["id"]] = ann
        print(f"removed {removed}")

        if "images" in self.dataset:
            for img in self.dataset["images"]:
                if not imgToAnns[img["id"]]:
                    continue
                imgs[img["id"]] = img

        if "categories" in self.dataset:
            for cat in self.dataset["categories"]:
                cats[cat["id"]] = cat

        if "annotations" in self.dataset and "categories" in self.dataset:
            for ann in self.dataset["annotations"]:
                if ann["iscrowd"]:
                    continue
                catToImgs[ann["category_id"]].append(ann["image_id"])

        print("index created!")

        # create class members
        self.anns = anns
        self.imgToAnns = imgToAnns
        self.catToImgs = catToImgs
        self.imgs = imgs
        self.cats = cats


class MyCocoDetection(datasets.CocoDetection):
    def __init__(
        self,
        root: str,
        annFile: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
        random_coord: Optional[bool] = None,
        filter_bbox: str = "minority",
    ) -> None:
        super(MyCocoDetection, self).__init__(
            root, transforms, transform, target_transform
        )

        self.coco = MyCOCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.random_coord = random_coord
        if filter_bbox == "random":
            self.filter_bbox = self.random_target
        elif filter_bbox == "smallest":
            self.filter_bbox = self.smallesst_target
        elif filter_bbox == "minority":
            self.filter_bbox = self.minority_voting
        else:
            raise RuntimeError("I don't know how to filter targets with {filter_bbox}")

    def random_target(self, targets: List):
        return targets[torch.randint(len(targets), size=(1,)).item()]

    def smallest_target(self, targets: List):
        area_list = [elem["area"] for elem in targets]
        return area_list.index(min(area_list))

    def minority_voting(self, targets: List):
        if len(targets) > 1:
            counter = defaultdict(int)
            for elem in targets:
                counter[elem["category_id"]] += 1
            c = Counter(counter)
            cat_to_pick = c.most_common()[-1]
            for i, elem in enumerate(targets):
                if elem["category_id"] == cat_to_pick[0]:
                    break
            return targets[i]
        return targets[0]

    def __getitem__(self, index):
        img, target = self._load_image_and_target(index)

        img_size = (
            img.size[1],
            img.size[0],
        )  # pillow images report size in (W, H) format whereas pytorh uses (H, W)
        img = self.transform(img)

        picked_target = self.filter_bbox(target)

        if self.random_coord:
            resized_bbox_coord = torch.rand(4) * min(img.shape[1:])
        else:
            coords = torch.Tensor(picked_target["bbox"])
            coords[2] = coords[0] + coords[2]
            coords[3] = coords[1] + coords[3]

            resized_bbox_coord = self.target_transform(coords, original_size=img_size)

        label = cat_id2id_and_name[str(picked_target["category_id"])][0]

        return (img, resized_bbox_coord), label

    def _load_image_and_target(self, index):
        img_id = self.ids[index]
        img = self._load_image(img_id)
        target = self.coco.loadAnns(self.coco.getAnnIds(img_id))
        return img, target


class BoxResize:
    def __init__(self, new_size: Tuple[int, int]):
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
