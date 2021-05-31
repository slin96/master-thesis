import argparse
import os
import time

from mmlib.persistence import FileSystemPersistenceService, MongoDictPersistenceService
from mmlib.save import BaselineSaveService

from experiments.baseline_flow.shared import save_model, add_connection_arguments, add_paths, inform, generate_message, \
    listen, reusable_udp_socket, extract_fields
from experiments.models.mobilenet import mobilenet_v2

socket = None
save_service = None
saved_model_ids = []


def main(args):
    global socket, saved_model_ids, save_service
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
    saved_model_ids.append(init_model_id)

    # inform node about available_model
    message = generate_message(init_model_id, False)
    inform(message, socket, (args.node_ip, args.node_port))

    print('listen')
    listen(sock=socket, callback=react_to_new_model)


def send_updated_model():
    # TODO redundant code
    # U2 - server
    # TODO implement loading new model
    model = mobilenet_v2(pretrained=True)
    init_model_id = save_model(model, save_service)
    saved_model_ids.append(init_model_id)

    # inform node about available_model
    message = generate_message(init_model_id, False)
    inform(message, socket, (args.node_ip, args.node_port))

    print('listen')
    listen(sock=socket, callback=react_to_new_model)


def react_to_new_model(msg):
    print('informed about new model')
    print(msg)
    last, model_id = extract_fields(msg)
    saved_model_ids.append(model_id)
    if last:
        # if this is the last message that will reach from the node for now U2 is finished
        # we transition to U2 and the server send an updated model
        print('send updated model')
        send_updated_model()
    else:
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
