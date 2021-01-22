import argparse

from mmlib.save import SaveService

from experiments.usecases.shared import *


# to run this make sure mongoDB is running:
# docker run --rm --name mongo-test -it -p 27017:27017 -d  mongo:latest
def main(args):
    save_service = SaveService(args.tmp_dir)

    print('model used: {}'.format(args.model))
    # initially train the model in full dataset
    init_model = initial_train(models_dict[args.model])
    # save the initially trained model and get info (id, filepath, size, ...) back
    model_id = save_service.save_model(args.model, init_model, args.model_code, args.import_root)
    # inform that a new model is available in the DB ready to use
    inform(bytes(str(model_id), encoding=ENCODING), (args.server_ip, args.server_port), (args.node_ip, args.node_port))
    print('server done')


def parse_args():
    parser = argparse.ArgumentParser(description='Script modeling the server for usecase 1')
    parser.add_argument('--model', help='The model to use for the run',
                        choices=[MOBILENET_V2, GOOGLENET, RESNET_18, RESNET_50, RESNET_152])
    parser.add_argument('--model_code', help='The path to the code defining the model')
    parser.add_argument('--import_root', help='The root path for imports')

    add_connection_arguments(parser)
    add_tmp_dir_path(parser)

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)
