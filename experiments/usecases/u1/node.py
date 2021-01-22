import argparse

from mmlib.log import use_model
from mmlib.save import RecoverService
from mmlib.util import extract_mongo_id

from experiments.usecases.shared import listen, add_connection_arguments, add_tmp_dir_path


def main(args):
    # wait for new model to be ready
    listen((args.node_ip, args.node_port), react_to_new_model)


def react_to_new_model(msg):
    print(msg)
    model_id = extract_mongo_id(msg)
    # as soon as new model is available
    save_service = RecoverService(args.tmp_dir, args.server_ip)
    recovered_model = save_service.recover_model(model_id)
    # use recovered model
    use_model(recovered_model)


def parse_args():
    parser = argparse.ArgumentParser(description='Script modeling the node for usecase 1')

    add_connection_arguments(parser)
    add_tmp_dir_path(parser)

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)
