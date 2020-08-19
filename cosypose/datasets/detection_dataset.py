import torch
import numpy as np
import random
from cosypose.config import LOCAL_DATA_DIR

from .wrappers.visibility_wrapper import VisibilityWrapper
from .augmentations import (
    CropResizeToAspectAugmentation, VOCBackgroundAugmentation,
    PillowBlur, PillowSharpness, PillowContrast, PillowBrightness, PillowColor, to_torch_uint8,
    GrayScale
)


class DetectionDataset(torch.utils.data.Dataset):
    def __init__(self,
                 scene_ds,
                 label_to_category_id,
                 min_area=50,
                 resize=(640, 480),
                 gray_augmentation=False,
                 rgb_augmentation=False,
                 background_augmentation=False):

        self.scene_ds = VisibilityWrapper(scene_ds)

        self.resize_augmentation = CropResizeToAspectAugmentation(resize=resize)

        self.background_augmentation = background_augmentation
        self.background_augmentations = VOCBackgroundAugmentation(
            voc_root=LOCAL_DATA_DIR / 'VOCdevkit/VOC2012', p=0.3)

        self.rgb_augmentation = rgb_augmentation
        self.rgb_augmentations = [
            PillowBlur(p=0.4, factor_interval=(1, 3)),
            PillowSharpness(p=0.3, factor_interval=(0., 50.)),
            PillowContrast(p=0.3, factor_interval=(0.2, 50.)),
            PillowBrightness(p=0.5, factor_interval=(0.1, 6.0)),
            PillowColor(p=0.3, factor_interval=(0., 20.))
        ]

        self.label_to_category_id = label_to_category_id
        self.min_area = min_area

    def __len__(self):
        return len(self.scene_ds)

    def get_data(self, idx):
        rgb, mask, state = self.scene_ds[idx]

        rgb, mask, state = self.resize_augmentation(rgb, mask, state)

        if self.background_augmentation:
            rgb, mask, state = self.background_augmentations(rgb, mask, state)

        if self.rgb_augmentation and random.random() < 0.8:
            for augmentation in self.rgb_augmentations:
                rgb, mask, state = augmentation(rgb, mask, state)

        rgb, mask = to_torch_uint8(rgb), to_torch_uint8(mask)

        categories = torch.tensor([self.label_to_category_id[obj['name']] for obj in state['objects']])
        obj_ids = np.array([obj['id_in_segm'] for obj in state['objects']])
        boxes = np.array([torch.as_tensor(obj['bbox']).tolist() for obj in state['objects']])
        boxes = torch.as_tensor(boxes, dtype=torch.float32).view(-1, 4)
        area = torch.as_tensor((boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]))
        mask = np.array(mask)
        masks = mask == obj_ids[:, None, None]
        masks = torch.as_tensor(masks)

        keep = (area > self.min_area)
        boxes = boxes[keep]
        area = area[keep]
        categories = categories[keep]
        masks = masks[keep, :, :]
        num_objs = len(keep)

        num_objs = len(obj_ids)
        area = torch.as_tensor(area)
        boxes = torch.as_tensor(boxes)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        image_id = torch.tensor([idx])
        iscrowd = torch.zeros((num_objs), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = categories
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        return rgb, target

    def __getitem__(self, index):
        try_index = index
        valid = False
        n_attempts = 0
        while not valid:
            if n_attempts > 10:
                raise ValueError('Cannot find valid image in the dataset')
            im, target = self.get_data(try_index)
            valid = len(target['boxes']) > 0
            if not valid:
                try_index = random.randint(0, len(self.scene_ds) - 1)
                n_attempts += 1
        return im, target
