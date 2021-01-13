import argparse

from mmlib.save import save_model
from torchvision import models

from experiments.usecases.shared import initial_train, add_connection_arguments, inform, ENCODING

MOBILENET_V2 = "mobilenet_v2"
GOOGLENET = "googlenet"
RESNET_18 = "resnet18"
RESNET_50 = "resnet50"
RESNET_152 = "resnet152"

models_dict = {MOBILENET_V2: models.mobilenet_v2, GOOGLENET: models.googlenet, RESNET_18: models.resnet18,
               RESNET_50: models.resnet50, RESNET_152: models.resnet152}


def main(args):
    print('model used: {}'.format(args.model))
    # initially train the model in full dataset
    init_model = initial_train(models_dict[args.model])
    # save the initially trained model and get info (id, filepath, size, ...) back
    # TODO for now hidden by mmlib and not implemented
    saved_info = save_model(init_model)
    # inform that a new model is available in the DB ready to use
    inform(bytes("saved_info", encoding=ENCODING), (args.server_ip, args.server_port), (args.node_ip, args.node_port))
    print('server done')


def parse_args():
    parser = argparse.ArgumentParser(description='Script modeling the server for usecase 1')
    parser.add_argument('--model', help='the model to use for the run',
                        choices=[MOBILENET_V2, GOOGLENET, RESNET_18, RESNET_50, RESNET_152])

    add_connection_arguments(parser)

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)
