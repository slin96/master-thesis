import json
import os
from typing import Optional, Callable, Any

from PIL import Image
from torchvision.datasets import VisionDataset

COCO_CLASSES = 91


def _included_ids(id_subset_json):
    with open(id_subset_json, 'r') as json_file:
        included_id_json = json.load(json_file)
        ids = included_id_json['included-coco-ids']
    return ids


class CustomCocoLoader(VisionDataset):

    def __init__(self,
                 root: str,
                 ann_file: str,
                 id_subset_json: str = None,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 transforms: Optional[Callable] = None,
                 ) -> None:
        super(CustomCocoLoader, self).__init__(root, transforms, transform, target_transform)
        self.annFile = ann_file

        if id_subset_json:
            self.included_ids = _included_ids(id_subset_json)
        else:
            self.included_ids = list(range(COCO_CLASSES + 1))

        self._items = []
        with open(ann_file) as f:
            ann_data = json.load(f)
            coco_meta = ann_data['coco_meta']
            for e in coco_meta:
                coco_cat = e['coco_category_id']
                if coco_cat in self.included_ids:
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


if __name__ == '__main__':
    test = CustomCocoLoader(root='/hpi/fs00/home/nils.strassenburg/test/matched-data',
                            ann_file='/hpi/fs00/home/nils.strassenburg/test/matched-data/coco_meta.json',
                            id_subset_json='/hpi/fs00/home/nils.strassenburg/remote/code/experiments/custom-coco/coco-id-groups/coco-indoor-ids.json')
    print(len(test))
