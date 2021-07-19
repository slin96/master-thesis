import argparse

import torch
from torch import nn
from torchvision import datasets
from torchvision.models import resnet18

from data.custom.custom_coco import CustomCoco
from experiments.imagenet.imagenet_utils import inference_transforms, train_transforms
from experiments.imagenet.processing import train_epoch, validate


def main(args):
    loss_func = nn.CrossEntropyLoss()

    print('load imagenet data, if zip has to be unpacked this can take a while ...')
    imagenet_val_data = datasets.ImageNet(args.imagenet_root, 'val', transform=inference_transforms)
    coco_val_data = CustomCoco(args.coco_root, args.coco_annotations, transform=inference_transforms)
    print('done loading data')

    # because train data is to big for local machine, and because we just want to see if code runs -> use val split
    imagenet_train_data = datasets.ImageNet(args.imagenet_root, 'val', transform=train_transforms)
    coco_train_data = CustomCoco(args.coco_root, args.coco_annotations, transform=train_transforms)

    model = resnet18(pretrained=True)
    optimizer = torch.optim.SGD(model.parameters(), 1e-4, momentum=0.9, weight_decay=1e-4)

    print('imagenet train')
    img_train_output = train_epoch(model, imagenet_train_data, loss_func, optimizer, get_outputs=True,
                                   number_batches=args.num_epochs)
    print('imagenet val')
    img_val_output = validate(model, imagenet_val_data, loss_func, get_outputs=True, number_batches=args.num_epochs)

    model = resnet18(pretrained=True)
    optimizer = torch.optim.SGD(model.parameters(), 1e-4, momentum=0.9, weight_decay=1e-4)
    print('coco train')
    coco_train_output = train_epoch(model, coco_train_data, loss_func, optimizer, get_outputs=True,
                                    number_batches=args.num_epochs)

    print('coco val')
    coco_val_output = validate(model, coco_val_data, loss_func, get_outputs=True, number_batches=args.num_epochs)


def parse_args():
    parser = argparse.ArgumentParser(description='Example script for loading and using imagenet data')
    parser.add_argument('--num-epochs', type=int, help='the number of epochs')
    parser.add_argument('--imagenet-root', type=str,
                        help='root dir for the imagenet data, should contain the .tar files for the dataset to load')
    parser.add_argument('--coco-root', type=str, help='root directory for the coco data')
    parser.add_argument('--coco-annotations', type=str, help='path to the coco_meta.json file')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)
