import json
import os
from typing import Optional, Callable, Any

from PIL import Image
from torchvision.datasets import VisionDataset

import experiments.data.util.constants as const


def _included_ids(id_subset_json):
    with open(id_subset_json, 'r') as json_file:
        included_id_json = json.load(json_file)
        ids = included_id_json[const.INCLUDED_COCO_IDS]
    return ids


class CustomCoco(VisionDataset):

    def __init__(self,
                 root: str,
                 ann_file: str = const.COCO_META_JSON,
                 id_subset_json: str = None,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 transforms: Optional[Callable] = None,
                 ) -> None:
        super(CustomCoco, self).__init__(root, transforms, transform, target_transform)
        self.images_path = os.path.join(self.root, const.IMAGES)
        self.ann_file = os.path.join(self.root, ann_file)

        if id_subset_json:
            self.included_ids = _included_ids(id_subset_json)
        else:
            self.included_ids = list(range(const.COCO_CLASSES + 1))

        self._items = []
        with open(self.ann_file) as f:
            ann_data = json.load(f)
            coco_meta = ann_data[const.COCO_META]
            for e in coco_meta:
                coco_cat = e[const.COCO_CATEGORY_ID]
                if coco_cat in self.included_ids:
                    self._items.append(e)

    def __getitem__(self, index: int) -> Any:
        item = self._items[index]
        image_path = os.path.join(self.images_path, item[const.FILE_NAME])
        img = Image.open(image_path).convert('RGB')
        label = item[const.IMAGENET_CLASS_ID]

        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            label = self.target_transform(label)

        return img, label

    def __len__(self) -> int:
        return len(self._items)
