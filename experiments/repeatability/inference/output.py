import argparse
from pathlib import Path

from mmlib.deterministic import set_deterministic
from torch import nn
from torchvision import datasets

from experiments.imagenet.imagenet_utils import inference_transforms
from experiments.imagenet.processing import validate
from experiments.repeatability.util import save_output, MODELS


def experiment_inference(model, data, number_batches):
    loss_func = nn.CrossEntropyLoss()

    output = validate(model, data, loss_func, get_outputs=True, number_batches=number_batches)

    return output


def main(args):
    imgnet_val_data = datasets.ImageNet(args.imagenet_root, 'val', transform=inference_transforms)

    # create output dir
    Path(args.tmp_output_root).mkdir(parents=True, exist_ok=False)

    for mod_getter in MODELS:
        model = mod_getter(pretrained=True)
        # make the execution deterministic
        set_deterministic()
        # generate output for inference
        out = experiment_inference(model, imgnet_val_data, args.number_batches
        # save output to the output root to compare later
        save_output(args.tmp_output_root, mod_getter, out)


def parse_args():
    parser = argparse.ArgumentParser(description='Script that generates and saves resulst of model inference')
    parser.add_argument('--imagenet-root', help='imagenet root path for')
    parser.add_argument('--tmp-output-root', help='dir where tmp output is written to')
    parser.add_argument('--number-batches', type=int,
                        help='the number of batches that should be included in the output')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], help='the to execute on')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)
