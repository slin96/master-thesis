import argparse
from pathlib import Path

import torch
from mmlib.deterministic import deterministic
from torch import nn
from torchvision import datasets

from experiments.imagenet.imagenet_utils import inference_transforms
from experiments.imagenet.processing import train_epoch
from experiments.repeatability.args import add_shared_args
from experiments.repeatability.util import save_output, MODELS, save_model_weights


def experiment_training(model, data, optimizer, number_batches):
    loss_func = nn.CrossEntropyLoss()

    output = train_epoch(model, data, loss_func, optimizer, get_outputs=True, number_batches=number_batches)

    return output


def main(args):
    imgnet_val_data = datasets.ImageNet(args.imagenet_root, 'val', transform=inference_transforms)

    # create output dir
    Path(args.tmp_output_root).mkdir(parents=True, exist_ok=False)

    for mod_getter in MODELS:
        model = mod_getter(pretrained=True)
        optimizer = torch.optim.SGD(model.parameters(), 1e-4, momentum=0.9, weight_decay=1e-4)

        params = [model, imgnet_val_data, optimizer, args.number_batches]
        out = deterministic(experiment_training, f_args=params)

        save_output(args.tmp_output_root, mod_getter, out)
        save_model_weights(args.tmp_output_root, mod_getter, model)


def parse_args():
    parser = argparse.ArgumentParser(description='Script that generates and saves results of model training')
    add_shared_args(parser)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
