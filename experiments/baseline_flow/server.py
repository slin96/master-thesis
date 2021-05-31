import argparse
import os

from mmlib.persistence import FileSystemPersistenceService, MongoDictPersistenceService
from mmlib.save import BaselineSaveService

from experiments.baseline_flow.shared import save_model, add_connection_arguments, add_paths, inform, generate_message, \
    listen, reusable_udp_socket
from experiments.models.mobilenet import mobilenet_v2

socket = None


def main(args):
    global socket
    socket = reusable_udp_socket()
    socket.bind((args.server_ip, args.server_port))

    # initialize a service to save files
    abs_tmp_path = os.path.abspath(args.tmp_dir)
    file_pers_service = FileSystemPersistenceService(abs_tmp_path)

    # initialize service to store dictionaries (JSON),
    dict_pers_service = MongoDictPersistenceService(host=args.mongo_host)

    # initialize baseline save service
    save_service = BaselineSaveService(file_pers_service, dict_pers_service)

    # U1 - server : initialize_model, and save it
    model = mobilenet_v2(pretrained=True)
    init_model_id = save_model(model, save_service)

    # inform node about available_model
    message = generate_message(init_model_id, False)
    inform(message, socket, (args.node_ip, args.node_port))

    print('listen')
    listen(sock=socket, callback=react_to_new_model)


def react_to_new_model(msg):
    print('informed about new model')
    print(msg)
    listen(sock=socket, callback=react_to_new_model)


def parse_args():
    parser = argparse.ArgumentParser(description='Script modeling server for workflow using baseline appraoch')
    # TODO make model configurable later
    # parser.add_argument('--model', help='The model to use for the run',
    #                     choices=[MOBILENET, GOOGLENET, RESNET_18, RESNET_50, RESNET_152])
    add_connection_arguments(parser)
    add_paths(parser)
    _args = parser.parse_args()

    return _args


if __name__ == '__main__':
    args = parse_args()
    main(args)
