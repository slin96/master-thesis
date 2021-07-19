import argparse
import time

import torch
from mmlib.deterministic import set_deterministic
from torch import nn
from torch.utils.collect_env import get_pretty_env_info
from torchvision.models import mobilenet_v2, googlenet, resnet18, resnet50, resnet152

from data.custom.custom_coco import TrainCustomCoco
from imagenet.processing import in_number_of_batches

TO_DEVICE = 'to_device'

BACKWARD_PATH = 'backward_path'
FORWARD_PATH = 'forward_path'
BATCH = 'batch-time'
LOAD_DATA = 'load_data'
STOP = 'STOP'
EPOCH = 'epoch'
START = 'START'

MOBILENET = "mobilenet"
GOOGLENET = "googlenet"
RESNET_18 = "resnet18"
RESNET_50 = "resnet50"
RESNET_152 = "resnet152"

MODELS_DICT = {MOBILENET: mobilenet_v2, GOOGLENET: googlenet, RESNET_18: resnet18, RESNET_50: resnet50,
               RESNET_152: resnet152}


def get_device(device=None):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return device


def log_time(start_stop, event_key, epoch_number=None, batch_number=None):
    t = time.time_ns()
    print('{};{};epoch-{};batch-{};time.time_ns-{}'.format(start_stop, event_key, epoch_number, batch_number, t))


def main(args):
    print('Given args')
    print(args)
    print('Pytorch Environment info')
    print(torch.utils.collect_env.get_env_info())

    if args.deterministic:
        print("Use mmlib set_deterministic()")
        set_deterministic()

    # load the custom coco dataset
    coco_train_data = TrainCustomCoco(args.coco_root, args.coco_annotations)

    # specify the model to use and load it to the device (GPU or CPU)
    model_class = MODELS_DICT[args.model]
    if model_class == googlenet:
        model = model_class(aux_logits=False)
    else:
        model = model_class()
    device = get_device()
    model.to(device)
    # specify the loss function and optimizer
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), 1e-4, momentum=0.9, weight_decay=1e-4)

    for epoch in range(args.num_epochs):
        log_time(START, EPOCH, epoch)
        train_epoch(model, coco_train_data, loss_func, optimizer, device, epoch=epoch)
        log_time(STOP, EPOCH, epoch)


def train_epoch(model, data, loss_func, optimizer, device, batch_size=64, num_workers=0, number_batches=None, epoch=0):
    train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                               pin_memory=True)
    # switch to train mode
    model.train()

    log_time(START, LOAD_DATA, epoch, 0)
    log_time(START, BATCH, epoch, 0)

    for i, (images, target) in enumerate(train_loader):
        log_time(STOP, LOAD_DATA, epoch, i)

        log_time(START, TO_DEVICE, epoch, i)
        images = images.to(device)
        target = target.to(device)
        log_time(STOP, TO_DEVICE, epoch, i)

        log_time(START, FORWARD_PATH, epoch, i)
        # compute output
        output = model(images)
        loss = loss_func(output, target)
        log_time(STOP, FORWARD_PATH, epoch, i)

        log_time(START, BACKWARD_PATH, epoch, i)
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        log_time(STOP, BACKWARD_PATH, epoch, i)

        if not in_number_of_batches(i, number_batches):
            break

        log_time(START, LOAD_DATA, epoch, i + 1)
        log_time(STOP, BATCH, epoch, i)
        log_time(START, BATCH, epoch, i + 1)


def parse_args():
    parser = argparse.ArgumentParser(description='Script to measure the time used for training a model')
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
