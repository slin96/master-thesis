import torch
from torch import nn
from torch.backends import cudnn
from torchvision import datasets
from torchvision.models import resnet18

from experiments.data.custom.custom_coco import CustomCoco
from experiments.imagenet.imagenet_utils import inference_transforms, train_transforms
from experiments.imagenet.processing import train_epoch, validate

if __name__ == '__main__':
    model = resnet18(pretrained=True)
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), 1e-4,
                                momentum=0.9,
                                weight_decay=1e-4)

    # Data loading code
    imagenet_root = '/Users/nils/Studium/master-thesis/repo/tmp/imgnet'
    coco_root = '/Users/nils/Studium/master-thesis/repo/tmp/cutsom-coco-data'
    coco_annotations = '/Users/nils/Studium/master-thesis/repo/tmp/cutsom-coco-data/coco_meta.json'

    imgnet_val_data = datasets.ImageNet(imagenet_root, 'val', transform=inference_transforms)
    coco_val_data = CustomCoco(coco_root, coco_annotations, transform=inference_transforms)

    # because train data is to big for local machine, and because we just want to see if code runs -> use val split
    imgenet_train_data = datasets.ImageNet(imagenet_root, 'val', transform=train_transforms)
    coco_train_data = CustomCoco(coco_root, coco_annotations, transform=train_transforms)

    num = 3
    print('imagenet train')
    img_train_output = train_epoch(model, imgenet_train_data, loss_func, optimizer, get_outputs=True,
                                   number_batches=num)
    print('coco train')
    coco_train_output = train_epoch(model, coco_train_data, loss_func, optimizer, get_outputs=True, number_batches=num)

    print('imagenet val')
    img_val_output = validate(model, imgnet_val_data, loss_func, get_outputs=True, number_batches=num)
    print('coco val')
    coco_val_output = validate(model, coco_val_data, loss_func, get_outputs=True, number_batches=num)
