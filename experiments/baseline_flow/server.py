import argparse
import os

import torch
from mmlib.persistence import FileSystemPersistenceService, MongoDictPersistenceService
from mmlib.save import BaselineSaveService

from experiments.baseline_flow.shared import save_model, add_paths, inform, generate_message, \
    listen, reusable_udp_socket, extract_fields, add_mongo_ip, add_server_connection_arguments, \
    add_node_connection_arguments, NEW_MODEL, add_model_arg, MODELS_DICT, \
    add_model_snapshot_arg

USE_CASE_1_PT = 'use-case-1.pt'
USE_CASE_2_PT = 'use-case-2.pt'


class ServerState:
    def __init__(self, tmp_dir, mongo_host, ip, port, admin_ip, admin_port, model_class, model_snapshots):
        # initialize a socket to communicate with otehr nodes
        self.socket = reusable_udp_socket()
        self.socket.bind((ip, port))
        self.admin_address = (admin_ip, admin_port)

        # initialize a service to save files
        abs_tmp_path = os.path.abspath(tmp_dir)
        file_pers_service = FileSystemPersistenceService(abs_tmp_path)

        # initialize service to store dictionaries (JSON),
        dict_pers_service = MongoDictPersistenceService(host=mongo_host)

        # initialize baseline save service
        self.save_service = BaselineSaveService(file_pers_service, dict_pers_service)

        # list of all models that have been saved by the node or have been communicated to be available
        self.saved_model_ids = []

        self.model_class = model_class
        self.model_snapshots = model_snapshots


server_state: ServerState = None


def main(args):
    global server_state
    server_state = ServerState(args.tmp_dir, args.mongo_host, args.server_ip, args.server_port, args.admin_ip,
                               args.admin_port, MODELS_DICT[args.model], args.model_snapshots)

    use_case_1()

    print('wait for node ...')
    listen(sock=server_state.socket, callback=use_case_3)


def use_case_1():
    print('use case 1')
    # load model from snapshot
    model = _load_model_snapshot(USE_CASE_1_PT)

    init_model_id = save_model(model, server_state.save_service)
    server_state.saved_model_ids.append(init_model_id)

    _inform_node_about_model(init_model_id)


def _load_model_snapshot(snapshot_name):
    snapshot_path = os.path.join(server_state.model_snapshots, snapshot_name)
    state_dict = torch.load(snapshot_path)
    model: torch.nn.Module = server_state.model_class()
    model.load_state_dict(state_dict)
    return model


def _inform_node_about_model(init_model_id):
    message = generate_message(text=NEW_MODEL, model_id=init_model_id)
    inform(message, server_state.socket, (args.node_ip, args.node_port))


def use_case_3(msg):
    print('use case 3')
    print(msg)
    text, model_id = extract_fields(msg)
    server_state.saved_model_ids.append(model_id)
    if 'done' in text:
        use_case_4()
    elif 'last' in text:
        # if this is the last message that will reach from the node for now U2 is finished
        # we transition to U2 and the server send an updated model
        print('send updated model')
        use_case_2()
    else:
        listen(sock=server_state.socket, callback=use_case_3)


def use_case_2():
    print('use case 2')
    model = _load_model_snapshot(USE_CASE_2_PT)

    model_id = save_model(model, server_state.save_service)
    server_state.saved_model_ids.append(model_id)

    _inform_node_about_model(model_id)

    print('wait for node ...')
    listen(sock=server_state.socket, callback=use_case_3)


def use_case_4():
    print('use case 4')
    for model_id in server_state.saved_model_ids:
        print('recover: {}'.format(model_id))
        server_state.save_service.recover_model(model_id, execute_checks=True)
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
