import argparse

from mmlib.helper import imagenet_input
from mmlib.model_equal import equal

from experiments.repeatability.args import add_compare_args
from experiments.repeatability.inference.compare import compare_all_outputs
from experiments.repeatability.util import MODELS, load_state_dict


def load_model(mod_getter, root_dir):
    mod = mod_getter()
    weights = load_state_dict(root_dir, mod_getter)
    mod.load_state_dict(weights)

    return mod


def compare_all_models(args):
    for mod_getter in MODELS:
        mod1 = load_model(mod_getter, args.input_root)
        mod2 = load_model(mod_getter, args.compare_to_root)

        if not equal(mod1, mod2, imagenet_input):
            return False

    return True


def main(args):
    if compare_all_outputs(args):
        print('ALL OUTPUTS ARE THE SAME')
    else:
        print('OUTPUTS DIFFER')
    if compare_all_models(args):
        print('ALL MODELS ARE THE SAME')
    else:
        print('MODELS DIFFER')


def parse_args():
    parser = argparse.ArgumentParser(description='Training experiment script')
    add_compare_args(parser)
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)
