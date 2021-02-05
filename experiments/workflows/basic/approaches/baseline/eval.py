import argparse
import os

import torch
from mmlib.equal import tensor_equal, state_dict_equal
from mmlib.save import FileSystemMongoSaveRecoverService

from experiments.workflows.shared import add_paths, add_mongo_ip

global_args = None

version_strings = ['init', 'updated']


def main(args):
    save_service = FileSystemMongoSaveRecoverService(args.tmp_dir, args.mongo_ip)
    model_ids = save_service.saved_model_ids()

    print(model_ids)

    for model_id in model_ids:
        check(args, model_id)


def check(args, model_id):
    print(model_id)
    # compare model outputs
    node_output_name = 'node-{}-output'.format(model_id)
    node_model_output = torch.load(os.path.join(args.log_dir, node_output_name))
    server_output_name = 'server-{}-output'.format(model_id)
    server_model_output = torch.load(os.path.join(args.log_dir, server_output_name))
    output_equal = tensor_equal(node_model_output, server_model_output)
    print('output_equal: {}'.format(output_equal))

    # compare model state dicts
    node_state_dict_name = 'node-{}-state-dict'.format(model_id)
    node_state_dict = torch.load(os.path.join(args.log_dir, node_state_dict_name))
    server_state_dict_name = 'server-{}-state-dict'.format(model_id)
    server_state_dict = torch.load(os.path.join(args.log_dir, server_state_dict_name))
    state_dict_eq = state_dict_equal(node_state_dict, server_state_dict)
    print('state_dict_equal: {}'.format(state_dict_eq))
    print()

    # storage consumption
    sizes = measure_storage_consumption(args)
    print(sizes)


def measure_storage_consumption(args):
    result = []
    save_recover_service = FileSystemMongoSaveRecoverService(args.tmp_dir, args.mongo_ip)
    model_ids = save_recover_service.saved_model_ids()
    for model_id in model_ids:
        save_size = save_recover_service.model_save_size(model_id)
        result.append((model_id, save_size))

    return result


def parse_args():
    parser = argparse.ArgumentParser(description='Script for evaluating')

    add_paths(parser)
    add_mongo_ip(parser)

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)


