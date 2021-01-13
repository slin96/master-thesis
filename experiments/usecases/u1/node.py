import argparse

from mmlib.log import use_model
from mmlib.recover import recover_model

from experiments.usecases.shared import listen, add_connection_arguments


def main(args):
    # wait for new model to be ready
    model_info = listen((args.node_ip, args.node_port), react_to_new_model)
    # as soon as new model is available
    recovered_model = recover_model(model_info)
    # use recovered model
    use_model(recovered_model)


def react_to_new_model(msg):
    print(msg)


def parse_args():
    parser = argparse.ArgumentParser(description='Script modeling the node for usecase 1')

    add_connection_arguments(parser)

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)
