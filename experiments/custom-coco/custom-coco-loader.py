import json
import os
from typing import Optional, Callable, Any

from PIL import Image
from torchvision.datasets import VisionDataset


class CustomCocoLoader(VisionDataset):

    def __init__(self,
                 root: str,
                 ann_file: str,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 transforms: Optional[Callable] = None,
                 ) -> None:
        super(CustomCocoLoader, self).__init__(root, transforms, transform, target_transform)
        self.annFile = ann_file

        self._items = []
        with open(ann_file) as f:
            ann_data = json.load(f)
            coco_meta = ann_data['coco_meta']
            for e in coco_meta:
                self._items.append(e)

    def __getitem__(self, index: int) -> Any:
        item = self._items[index]
        image_path = os.path.join(self.root, 'images', item['file_name'])
        img = Image.open(image_path).convert('RGB')
        label = item['imagenet_class_id']

        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            label = self.target_transform(label)

        return img, label

    def __len__(self) -> int:
        return len(self._items)
