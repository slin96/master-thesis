import argparse
from time import sleep

from mmlib.save import FileSystemMongoSaveService

from experiments.workflows.server_shared import *
from experiments.workflows.shared import *


# to run this make sure mongoDB is running:
# docker run --rm --name mongo-test -it -p 27017:27017 -d  mongo:latest


def main(args):
    save_service = FileSystemMongoSaveService(args.tmp_dir, args.mongo_ip)

    print('model used: {}'.format(args.model))
    # initially train the model in full dataset
    init_model = initial_train(models_dict[args.model])
    # save the initially trained model
    init_model_id = save_service.save_model(args.model, init_model, args.model_code, args.import_root)

    # NOT TIMED: save state_dict and output to compare restored model
    save_compare_info(init_model, 'server', init_model_id, args.log_dir)

    # inform that a new model is available in the DB ready to use
    inform(bytes(str(init_model_id), encoding=ENCODING), (args.server_ip, args.server_port),
           (args.node_ip, args.node_port))

    print('informed about init model')
    # NOT TIMED: wait some time
    sleep(2)

    # update model
    updated_model = update_model()
    # save the updated model
    updated_model_name = args.model + '-updated'
    updated_model_id = save_service.save_model(updated_model_name, updated_model, args.model_code, args.import_root)

    # NOT TIMED: save state_dict and output to compare restored model
    save_compare_info(updated_model, 'server', updated_model_id, args.log_dir)

    # inform that a new model is available in the DB ready to use
    inform(bytes(str(updated_model_id), encoding=ENCODING), (args.server_ip, args.server_port),
           (args.node_ip, args.node_port))

    print('informed about updated model')

    print('server done')


def parse_args():
    parser = argparse.ArgumentParser(description='Script modeling server for basic workflow using baseline approach')
    parser.add_argument('--model', help='The model to use for the run',
                        choices=[MOBILENET, GOOGLENET, RESNET_18, RESNET_50, RESNET_152])
    parser.add_argument('--model_code', help='The path to the code defining the model')
    parser.add_argument('--import_root', help='The root path for imports')

    add_connection_arguments(parser)
    add_paths(parser)

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)
