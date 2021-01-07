import argparse
from pathlib import Path

from mmlib.deterministic import set_deterministic
from torch import nn
from torchvision import datasets

from experiments.imagenet.imagenet_utils import inference_transforms
from experiments.imagenet.processing import validate
from experiments.repeatability.args import add_shared_args
from experiments.repeatability.util import save_output, MODELS


def experiment_inference(model, data, loss_func, number_batches):
    output = validate(model, data, loss_func, get_outputs=True, number_batches=number_batches)

    return output


def main(args):
    imgnet_val_data = datasets.ImageNet(args.imagenet_root, 'val', transform=inference_transforms)
    loss_func = nn.CrossEntropyLoss()

    # create output dir
    Path(args.tmp_output_root).mkdir(parents=True, exist_ok=False)

    for mod_getter in MODELS:
        model = mod_getter(pretrained=True)
        # make the execution deterministic
        set_deterministic()
        # generate output for inference
        out = experiment_inference(model, imgnet_val_data, loss_func, args.number_batches)
        # save output to the output root to compare later
        save_output(args.tmp_output_root, mod_getter, out)


def parse_args():
    parser = argparse.ArgumentParser(description='Script that generates and saves results of model inference')
    add_shared_args(parser)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
