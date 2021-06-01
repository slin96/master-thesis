import argparse
import os

import torch
from mmlib.persistence import FileSystemPersistenceService, MongoDictPersistenceService
from mmlib.save import BaselineSaveService

from experiments.baseline_flow.shared import recover_model, listen, extract_fields, add_paths, \
    save_model, generate_message, inform, reusable_udp_socket, add_mongo_ip, add_server_connection_arguments, \
    add_node_connection_arguments, NEW_MODEL, add_model_arg, MODELS_DICT, add_model_snapshot_arg, U_1, U_3_1, U_2, U_3_2

USE_CASE_TEMPLATE = 'use-case-{}-{}.pt'


class NodeState:
    def __init__(self, u3_repeat, ip, port, model_class, model_snapshots):
        self.socket = reusable_udp_socket()
        self.socket.bind((ip, port))

        # initialize a service to save files
        abs_tmp_path = os.path.abspath(args.tmp_dir)
        file_pers_service = FileSystemPersistenceService(abs_tmp_path)

        # initialize service to store dictionaries (JSON),
        dict_pers_service = MongoDictPersistenceService(host=args.mongo_host)

        # initialize baseline save service
        self.save_service = BaselineSaveService(file_pers_service, dict_pers_service)

        self.state_description = U_1
        self.u3_repeat = u3_repeat
        self.u3_counter = 0

        self.model_class = model_class
        self.model_snapshots = model_snapshots


node_sate: NodeState = None


def main(args):
    global node_sate
    node_sate = NodeState(2, args.node_ip, args.node_port, MODELS_DICT[args.model], args.model_snapshots)

    # U1- node: listen for models to be in DB
    listen(sock=node_sate.socket, callback=use_case_1)


def _react_to_new_model(msg):
    print('message received')
    print(msg)
    text, model_id = extract_fields(msg)
    model = recover_model(model_id, node_sate.save_service)
    assert model is not None
    next_state()


def use_case_1(msg):
    print('use case 1')
    _react_to_new_model(msg)


def use_case_2(msg):
    print('use case 2')
    _react_to_new_model(msg)


def use_case_3(last_time=False, done=False):
    print('use case 3')
    # simulate model training by loading model from checkpoint
    model = _load_model_snapshot(node_sate.state_description, node_sate.u3_counter)

    # save the model
    model_id = save_model(model, node_sate.save_service)

    # notify server
    text = NEW_MODEL
    if last_time:
        text = 'last'
    if done:
        text = 'done'
    message = generate_message(model_id=model_id, text=text)
    inform(message, node_sate.socket, (args.server_ip, args.server_port))

    next_state()


def next_state():
    if node_sate.state_description == U_1:
        node_sate.state_description = U_3_1
        node_sate.u3_counter += 1
        use_case_3()
    elif node_sate.state_description == U_3_1:
        if node_sate.u3_counter < node_sate.u3_repeat:
            node_sate.u3_counter += 1
            use_case_3(last_time=node_sate.u3_counter == node_sate.u3_repeat)
        else:
            node_sate.state_description = U_2
            node_sate.u3_counter = 0
            listen(sock=node_sate.socket, callback=use_case_2)
    elif node_sate.state_description == U_2:
        node_sate.state_description = U_3_2
        node_sate.u3_counter += 1
        use_case_3(node_sate.u3_counter == node_sate.u3_repeat)
    elif node_sate.state_description == U_3_2:
        if node_sate.u3_counter < node_sate.u3_repeat:
            node_sate.u3_counter += 1
            use_case_3(done=node_sate.u3_counter == node_sate.u3_repeat)
        else:
            print('DONE')


def _load_model_snapshot(state, counter):
    state = state.replace('U_', '').replace('_', '-')
    snapshot_name = USE_CASE_TEMPLATE.format(state, counter)
    snapshot_path = os.path.join(node_sate.model_snapshots, snapshot_name)
    print('load model: {}'.format(snapshot_path))
    state_dict = torch.load(snapshot_path)
    model: torch.nn.Module = node_sate.model_class()
    model.load_state_dict(state_dict)
    return model


def parse_args():
    parser = argparse.ArgumentParser(description='Script modeling node for workflow using baseline appraoch')
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
