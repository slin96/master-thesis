import argparse
import json
import os

import torch
from mmlib.persistence import FileSystemPersistenceService, MongoDictPersistenceService
from mmlib.save import BaselineSaveService

from experiments.baseline_flow.shared import save_model, add_paths, inform, generate_message, \
    listen, reusable_udp_socket, extract_fields, add_mongo_ip, add_server_connection_arguments, \
    add_node_connection_arguments, NEW_MODEL, add_model_arg, MODELS_DICT, \
    add_model_snapshot_arg, U_3_1, U_4, U_2, U_3_2, U_1, log_start, log_stop

RECOVER_MODELS = 'recover_models'

EXTRACT_NOTIFY_MESSAGE = 'extract_notify_message'

SAVE_MODEL = 'save_model'

START_USECASE = 'start_usecase'

SERVER = 'server'

USE_CASE_1_PT = 'use-case-1.pt'
USE_CASE_2_PT = 'use-case-2.pt'


class ServerState:
    def __init__(self, tmp_dir, mongo_host, ip, port, model_class, model_snapshots):
        # initialize a socket to communicate with other nodes
        self.socket = reusable_udp_socket()
        self.socket.bind((ip, port))

        # initialize a service to save files
        abs_tmp_path = os.path.abspath(tmp_dir)
        file_pers_service = FileSystemPersistenceService(abs_tmp_path)

        # initialize service to store dictionaries (JSON),
        dict_pers_service = MongoDictPersistenceService(host=mongo_host)

        # initialize baseline save service
        self.save_service = BaselineSaveService(file_pers_service, dict_pers_service, logging=True)

        # list of all models that have been saved by the node or have been communicated to be available
        self.saved_model_ids = {}

        self.state_description = U_1
        self.u3_counter = 0

        self.model_class = model_class
        self.model_snapshots = model_snapshots


server_state: ServerState = None


def main(args):
    global server_state
    server_state = ServerState(args.tmp_dir, args.mongo_host, args.server_ip, args.server_port, MODELS_DICT[args.model],
                               args.model_snapshots)

    use_case_1()


def use_case_1():
    # load model from snapshot
    model = _load_model_snapshot(USE_CASE_1_PT)

    log = log_start(SERVER, server_state.state_description, SAVE_MODEL)
    init_model_id = save_model(model, server_state.save_service)
    log_stop(log)

    server_state.saved_model_ids[init_model_id] = server_state.state_description

    _inform_node_about_model(init_model_id)

    next_state()


def _load_model_snapshot(snapshot_name):
    snapshot_path = os.path.join(server_state.model_snapshots, snapshot_name)
    print('load model: {}'.format(snapshot_path))
    state_dict = torch.load(snapshot_path)
    model: torch.nn.Module = server_state.model_class()
    model.load_state_dict(state_dict)
    return model


def _inform_node_about_model(init_model_id):
    message = generate_message(text=NEW_MODEL, model_id=init_model_id)
    inform(message, server_state.socket, (args.node_ip, args.node_port))


def use_case_3(msg):
    log = log_start(SERVER, server_state.state_description, EXTRACT_NOTIFY_MESSAGE)
    print(msg)
    text, model_id = extract_fields(msg)
    state_with_counter = _state_with_counter()
    server_state.saved_model_ids[model_id] = state_with_counter
    log_stop(log)

    next_state(text)


def _state_with_counter():
    return '{}_{}'.format(server_state.state_description, server_state.u3_counter)


def use_case_2():
    model = _load_model_snapshot(USE_CASE_2_PT)

    log = log_start(SERVER, server_state.state_description, SAVE_MODEL)
    model_id = save_model(model, server_state.save_service)
    log_stop(log)

    server_state.saved_model_ids[model_id] = server_state.state_description

    _inform_node_about_model(model_id)

    next_state()


def use_case_4():
    log = log_start(SERVER, server_state.state_description, RECOVER_MODELS)
    for model_id in server_state.saved_model_ids.keys():
        state_with_counter = _state_with_counter()
        model_recover = 'recover-{}-{}'.format(server_state.saved_model_ids[model_id], model_id)
        log_i = log_start(SERVER, state_with_counter, model_recover)
        server_state.save_service.recover_model(model_id, execute_checks=True)
        log_stop(log_i)

    log_stop(log)
    next_state()


def log_sizes():
    for model_id in server_state.saved_model_ids.keys():
        size_info = server_state.save_service.model_save_size(model_id)
        print('size-info-{}-{}'.format(model_id, json.dumps(size_info)))


def next_state(text=None):
    if server_state.state_description == U_1:
        server_state.state_description = U_3_1
        server_state.u3_counter += 1
        listen(sock=server_state.socket, callback=use_case_3)
    elif server_state.state_description == U_3_1:
        if 'last' in text:
            server_state.state_description = U_2
            server_state.u3_counter = 0
            use_case_2()
        else:
            server_state.u3_counter += 1
            listen(sock=server_state.socket, callback=use_case_3)
    elif server_state.state_description == U_2:
        server_state.state_description = U_3_2
        server_state.u3_counter += 1
        listen(sock=server_state.socket, callback=use_case_3)
    elif server_state.state_description == U_3_2:
        if 'done' in text:
            server_state.state_description = U_4
            use_case_4()
        else:
            server_state.u3_counter += 1
            listen(sock=server_state.socket, callback=use_case_3)
    elif server_state.state_description == U_4:
        log_sizes()
        print('DONE')


def parse_args():
    parser = argparse.ArgumentParser(description='Script modeling server for workflow using baseline appraoch')

    add_server_connection_arguments(parser)
    add_node_connection_arguments(parser)
    add_model_arg(parser)
    add_model_snapshot_arg(parser)
    add_paths(parser)
    add_mongo_ip(parser)
    _args = parser.parse_args()

    return _args


if __name__ == '__main__':
    args = parse_args()
    main(args)
