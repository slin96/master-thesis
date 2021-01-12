import argparse

import mmlib
from mmlib.save import save_model
from torchvision import models

from experiments.usecases.shared import initial_train

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
    # TODO probably we also have to store the model code or somehow reference, could also pickle the hole model
    # saved_info = mmlib.save.save_model(init_model)
    # inform that a new model is available in the DB ready to use
    # inform(saved_info, receiver)
    print('server done')


def parse_args():
    parser = argparse.ArgumentParser(description='Script modeling the server for Usecase 1')
    parser.add_argument('--model', help='the model to use for the run',
                        choices=[MOBILENET_V2, GOOGLENET, RESNET_18, RESNET_50, RESNET_152])

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)
