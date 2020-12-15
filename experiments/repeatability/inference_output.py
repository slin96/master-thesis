import argparse
import os
from pathlib import Path

import torch
from mmlib.deterministic import deterministic
from torch import nn
from torch.backends import cudnn
from torchvision import models, datasets

from experiments.imagenet.imagenet_utils import inference_transforms
from experiments.imagenet.processing import validate

MODELS = [models.alexnet, models.vgg19, models.resnet18, models.resnet50, models.resnet152]


def experiment_inference(model, data, number_batches):
    loss_func = nn.CrossEntropyLoss()

    # TODO check what this var does
    cudnn.benchmark = True

    print('imagenet val')
    output = validate(model, data, loss_func, get_outputs=True, number_batches=number_batches)

    return output


def save_output(args, model_getter, out):
    output_file = os.path.join(args.tmp_output_root, 'model-{}-output'.format(model_getter.__name__))
    torch.save(out, output_file)


def main(args):
    imgnet_val_data = datasets.ImageNet(args.imagenet_root, 'val', transform=inference_transforms)

    for mod_getter in MODELS:
        model = mod_getter(pretrained=True)
        params = [model, imgnet_val_data, args.number_batches]
        out = deterministic(experiment_inference, f_args=params, f_kwargs={})

        # create output dir
        Path(args.tmp_output_root).mkdir(parents=True, exist_ok=False)
        save_output(args, mod_getter, out)


def parse_args():
    parser = argparse.ArgumentParser(description='Script that generates and saves resulst of model inference')
    parser.add_argument('--imagenet-root', help='imagenet root path for')
    parser.add_argument('--tmp-output-root', help='dir where tmp output is written to')
    parser.add_argument('--number-batches', type=int,
                        help='the number of batches that should be included in the output')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)
