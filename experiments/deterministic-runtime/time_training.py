import argparse

import torch
from torch import nn
from torchvision.models import mobilenet_v2, googlenet, resnet18, resnet50, resnet152

from experiments.imagenet.processing import in_number_of_batches

MOBILENET = "mobilenet"
GOOGLENET = "googlenet"
RESNET_18 = "resnet18"
RESNET_50 = "resnet50"
RESNET_152 = "resnet152"

MODELS_DICT = {MOBILENET: mobilenet_v2, GOOGLENET: googlenet, RESNET_18: resnet18, RESNET_50: resnet50,
               RESNET_152: resnet152}


def main(args):
    # TODO log args
    # TODO log pytorch info

    # TODO set deterministic if true

    # load the custom coco dataset
    coco_train_data = CustomCoco(args.coco_root, args.coco_annotations, transform=train_transforms)

    # specify the model to use
    model_class = MODELS_DICT[args.model]
    model = model_class()
    # specify the loss function and optimizer
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), 1e-4, momentum=0.9, weight_decay=1e-4)

    # TODO loop
    # TODO measure time before and after epoch
    train_epoch(model, coco_train_data, loss_func, optimizer)


def train_epoch(model, data, loss_func, optimizer, batch_size=64, num_workers=0, device=None,
                number_batches=None):
    # TODO check how to get device for this experiment
    # device = get_device(device)

    train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                               pin_memory=True)

    # load model to the given device, for this experiment most likely a GPU
    model.to(device)
    # switch to train mode
    model.train()

    # TODO measure time per batch
    for i, (images, target) in enumerate(train_loader):
        images = images.to(device)
        target = target.to(device)

        # compute output
        output = model(images)
        loss = loss_func(output, target)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if not in_number_of_batches(i, number_batches):
            break


def parse_args():
    parser = argparse.ArgumentParser(description='Example script for loading and using imagenet data')
    parser.add_argument('--num-epochs', type=int, help='the number of epochs')
    parser.add_argument('--coco-root', type=str, help='root directory for the coco data')
    parser.add_argument('--coco-annotations', type=str, help='path to the coco_meta.json file')
    parser.add_argument('--model', help='The model to use for the run',
                        choices=[MOBILENET, GOOGLENET, RESNET_18, RESNET_50, RESNET_152])
    parser.add_argument('--deterministic', help='Indicates if we set the deterministic flag or not', type=bool)

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)
