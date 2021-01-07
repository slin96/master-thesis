import argparse
from pathlib import Path

import torch
from mmlib.deterministic import set_deterministic
from torchvision import datasets

from experiments.imagenet.imagenet_utils import inference_transforms
from experiments.imagenet.processing import train_epoch
from experiments.repeatability.args import add_shared_args
from experiments.repeatability.util import save_output, MODELS, save_model_weights


def experiment_training(model, data, optimizer, loss_func, args):
    output = train_epoch(model, data, loss_func, optimizer, get_outputs=True, number_batches=args.number_batches,
                         device=args.devcie)

    return output


def main(args):
    imgnet_val_data = datasets.ImageNet(args.imagenet_root, 'val', transform=inference_transforms)
    loss_func = torch.nn.CrossEntropyLoss()

    # create output dir
    Path(args.tmp_output_root).mkdir(parents=True, exist_ok=False)

    for mod_getter in MODELS:
        model = mod_getter(pretrained=True)
        # make the execution deterministic
        set_deterministic()
        # generate output for training
        optimizer = torch.optim.SGD(model.parameters(), 1e-4, momentum=0.9, weight_decay=1e-4)
        out = experiment_training(model, imgnet_val_data, optimizer, loss_func, args)
        # save output to the output root to compare later
        save_output(args.tmp_output_root, mod_getter, out)
        save_model_weights(args.tmp_output_root, mod_getter, model)


def parse_args():
    parser = argparse.ArgumentParser(description='Script that generates and saves results of model training')
    add_shared_args(parser)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
