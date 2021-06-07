from typing import Any

from torchvision import datasets, transforms


class ImagenetTrainLoader(datasets.ImageNet):

    def __init__(self, root: str, **kwargs: Any):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

        kwargs['transform'] = train_transforms
        super().__init__(root, **kwargs)