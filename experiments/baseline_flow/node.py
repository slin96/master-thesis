import argparse
import os

from mmlib.persistence import FileSystemPersistenceService, MongoDictPersistenceService
from mmlib.save import BaselineSaveService

from experiments.baseline_flow.shared import recover_model, listen, extract_fields, add_connection_arguments, add_paths, \
    save_model, generate_message, inform, reusable_udp_socket
from experiments.models.mobilenet import mobilenet_v2

U3_REPEAT = 3

save_service = None
state = 'U1'
counter = 0

socket = None


def main(args):
    global socket
    socket = reusable_udp_socket()
    socket.bind((args.node_ip, args.node_port))

    # initialize a service to save files
    abs_tmp_path = os.path.abspath(args.tmp_dir)
    file_pers_service = FileSystemPersistenceService(abs_tmp_path)

    # initialize service to store dictionaries (JSON),
    dict_pers_service = MongoDictPersistenceService(host=args.mongo_host)

    # initialize baseline save service
    global save_service
    save_service = BaselineSaveService(file_pers_service, dict_pers_service)

    # U1- node: listen for models to be in DB
    listen(sock=socket, callback=react_to_new_model)


def load_model():
    # TODO
    return mobilenet_v2(pretrained=True)


def train_and_save_model(last=False):
    global save_service

    # model training model by loading it from checkpoint
    # TODO somehow specify what model to load
    model = load_model()

    # save the model
    model_id = save_model(model, save_service)

    # notify server
    message = generate_message(model_id, last)
    inform(message, socket, (args.server_ip, args.server_port))

    next_state()


def next_state():
    global state, counter

    print('CURRENT STATE: {}'.format(state))

    if state == 'U1':
        state = 'U3_1'
        counter += 1
        train_and_save_model()
    elif state == 'U3_1':
        if counter < U3_REPEAT:
            counter += 1
            train_and_save_model(last=(counter==U3_REPEAT))
        else:
            state = 'U2'
            counter = 0
            listen(sock=socket, callback=react_to_new_model)
    elif state == 'U2':
        state = 'U3_2'
        counter += 1
        train_and_save_model()
    elif state == 'U3_2':
        if counter < U3_REPEAT:
            counter += 1
            train_and_save_model(last=(counter==U3_REPEAT))
        else:
            print('DONE')


def react_to_new_model(msg):
    print(msg)
    last, model_id = extract_fields(msg)
    model = recover_model(model_id, save_service)
    assert model is not None
    next_state()


def parse_args():
    parser = argparse.ArgumentParser(description='Script modeling node for workflow using baseline appraoch')
    add_connection_arguments(parser)
    add_paths(parser)
    _args = parser.parse_args()

    return _args


if __name__ == '__main__':
    args = parse_args()
    main(args)
