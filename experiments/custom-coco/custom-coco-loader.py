import json
from typing import Optional, Callable, Any

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
        pass

    def __len__(self) -> int:
        return len(self._items)


if __name__ == '__main__':
    test = CustomCocoLoader(root='/hpi/fs00/home/nils.strassenburg/test/matched-data',
                            ann_file='/hpi/fs00/home/nils.strassenburg/test/matched-data/coco_meta.json')
    print(len(test))