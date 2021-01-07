import argparse

import torch

from experiments.repeatability.args import add_compare_args
from experiments.repeatability.util import MODELS, get_output


def compare(out1, out2):
    # we assume the outputs are lists of tensors
    zipped = zip(out1, out2)
    for (tensor1, tensor2) in zipped:
        if not torch.equal(tensor1, tensor2):
            return False

    return True


def compare_all_outputs(args):
    for mod_getter in MODELS:
        out1 = get_output(args.input_root, mod_getter)
        out2 = get_output(args.compare_to_root, mod_getter)

        if not compare(out1, out2):
            return False

    return True


def main(args):
    if compare_all_outputs(args):
        print('ALL OUTPUTS ARE THE SAME')
    else:
        print('OUTPUTS DIFFER')


def parse_args():
    parser = argparse.ArgumentParser(description='Inference experiment script')
    add_compare_args(parser)
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)
