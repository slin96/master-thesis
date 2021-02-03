import argparse
import os

import torch
from mmlib.equal import tensor_equal, state_dict_equal

from experiments.workflows.shared import add_paths

global_args = None


def main(args):
    # compare model outputs
    node_model_output = torch.load(os.path.join(args.log_dir, 'node-model-output'))
    server_model_output = torch.load(os.path.join(args.log_dir, 'server-model-output'))

    output_equal = tensor_equal(node_model_output, server_model_output)
    print('output_equal: {}'.format(output_equal))

    # compare model state dicts
    node_state_dict = torch.load(os.path.join(args.log_dir, 'node-model-state-dict'))
    server_state_dict = torch.load(os.path.join(args.log_dir, 'server-model-state-dict'))

    state_dict_eq = state_dict_equal(node_state_dict, server_state_dict)
    print('state_dict_equal: {}'.format(state_dict_eq))


def parse_args():
    parser = argparse.ArgumentParser(description='Script for evaluating')

    add_paths(parser)

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)
