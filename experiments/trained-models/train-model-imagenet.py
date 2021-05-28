import argparse
import os.path
import time

import torch
from mmlib.deterministic import set_deterministic
from torch import nn
from torch.utils.collect_env import get_pretty_env_info
from torchvision import datasets
from torchvision.models import mobilenet_v2, googlenet, resnet18, resnet50, resnet152

from experiments.data.custom.custom_coco import train_transforms
from experiments.imagenet.processing import in_number_of_batches

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

IMAGENET_VAL = 'imagenet_val'
# m-<model-name>_e-<epoch>_d-<dataset>
MODEL_SAVE_TEMPLATE = 'model_m-{}_e-{}_d-{}.pt'
OPTIMIZER_SAVE_TEMPLATE = 'optimizer_m-{}_e-{}_d-{}.pt'

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

    print("Use mmlib set_deterministic()")
    set_deterministic()

    print('load imagenet data, if zip has to be unpacked this can take a while ...')
    # For this experiment we train on the validation data
    # For details see thesis
    imagenet_val_data = datasets.ImageNet(args.imagenet_root, 'val', transform=train_transforms)
    print('done loading data')

    # specify the model to use and load it to the device (GPU or CPU)
    model_class = MODELS_DICT[args.model]
    model = model_class()
    device = get_device()
    model.to(device)
    # specify the loss function and optimizer
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), 1e-4, momentum=0.9, weight_decay=1e-4)

    for epoch in range(args.num_epochs):
        if epoch % args.save_freq == 0:
            model_name = MODEL_SAVE_TEMPLATE.format(args.model, epoch, IMAGENET_VAL)
            torch.save(model.state_dict(), os.path.join(args.save_root, model_name))

            optimizer_name = OPTIMIZER_SAVE_TEMPLATE.format(args.model, epoch, IMAGENET_VAL)
            torch.save(optimizer.state_dict(), os.path.join(args.save_root, optimizer_name))

        log_time(START, EPOCH, epoch)
        train_epoch(model, imagenet_val_data, loss_func, optimizer, device, epoch=epoch, num_workers=args.workers)
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
    parser.add_argument('--save-root', type=str, help='the root directory to save snapshots')
    parser.add_argument('--workers', type=int, help='the number fo workers to use for data loading')
    parser.add_argument('--imagenet-root', type=str,
                        help='root dir for the imagenet data, should contain the .tar files for the dataset to load')
    parser.add_argument('--model', help='The model to use for the run',
                        choices=[MOBILENET, GOOGLENET, RESNET_18, RESNET_50, RESNET_152])

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)
