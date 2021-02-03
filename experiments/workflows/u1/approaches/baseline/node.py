import argparse

from mmlib.log import use_model
from mmlib.recover import FileSystemMongoRecoverService

from experiments.workflows.node_shared import listen
from experiments.workflows.shared import add_connection_arguments, add_paths, save_compare_info

global_args = None


def main(args):
    global global_args
    global_args = args
    # wait for new model to be ready
    listen((args.node_ip, args.node_port), react_to_new_model)


def react_to_new_model(msg):
    print(msg)
    model_id = msg[0].decode("utf-8")
    # as soon as new model is available
    save_service = FileSystemMongoRecoverService(args.tmp_dir, args.mongo_ip)
    recovered_model = save_service.recover_model(model_id)
    # use recovered model
    use_model(model_id)

    # save state_dict and output to compare restored model
    save_compare_info(recovered_model, 'node', args.log_dir)


def parse_args():
    parser = argparse.ArgumentParser(description='Script modeling the node for usecase 1')

    add_connection_arguments(parser)
    add_paths(parser)

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)
