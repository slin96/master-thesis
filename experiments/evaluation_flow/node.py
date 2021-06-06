import argparse
import os

import torch
from mmlib.persistence import FileSystemPersistenceService, MongoDictPersistenceService
from mmlib.track_env import track_current_environment

from experiments.evaluation_flow.shared import recover_model, listen, extract_fields, add_paths, \
    save_model, generate_message, inform, reusable_udp_socket, add_mongo_ip, add_server_connection_arguments, \
    add_node_connection_arguments, NEW_MODEL, add_model_arg, MODELS_DICT, add_model_snapshot_arg, U_1, U_3_1, U_2, \
    U_3_2, log_start, log_stop, add_approach, get_save_service, add_u3_count, PROVENANCE, save_provenance_model, \
    get_dummy_train_kwargs, dummy_train_service_wrapper

SAVE_MODEL = 'save_model'

RECOVER_MODEL = 'recover_model'

NODE = 'node'

USE_CASE_TEMPLATE = 'use-case-{}-{}.pt'


class NodeState:
    def __init__(self, approach, u3_repeat, ip, port, model_class, model_snapshots):
        self.socket = reusable_udp_socket()
        self.socket.bind((ip, port))

        # initialize a service to save files
        abs_tmp_path = os.path.abspath(args.tmp_dir)
        file_pers_service = FileSystemPersistenceService(abs_tmp_path)

        # initialize service to store dictionaries (JSON),
        dict_pers_service = MongoDictPersistenceService(host=args.mongo_host)

        # initialize save service
        self.save_service = get_save_service(approach, dict_pers_service, file_pers_service)
        self.approach = args.approach

        self.state_description = U_1
        self.u3_repeat = u3_repeat
        self.u3_counter = 0

        self.model_class = model_class
        self.model_snapshots = model_snapshots

        self.last_model_id = None
        self.last_recovered_model = None


node_state: NodeState = None


def main(args):
    global node_state
    node_state = NodeState(
        args.approach, args.u3_count, args.node_ip, args.node_port, MODELS_DICT[args.model], args.model_snapshots)

    # U1- node: listen for models to be in DB
    listen(sock=node_state.socket, callback=use_case_1)


def _react_to_new_model(msg):
    print(msg)
    log = log_start(NODE, node_state.state_description, RECOVER_MODEL)
    text, model_id = extract_fields(msg)
    node_state.last_model_id = model_id
    model = recover_model(model_id, node_state.save_service)
    assert model is not None
    node_state.last_recovered_model = model
    log_stop(log)
    next_state()


def use_case_1(msg):
    _react_to_new_model(msg)


def use_case_2(msg):
    _react_to_new_model(msg)


def _state_with_counter():
    return '{}_{}'.format(node_state.state_description, node_state.u3_counter)


def use_case_3(last_time=False, done=False):
    state_w_counter = _state_with_counter()
    log = log_start(NODE, state_w_counter, SAVE_MODEL)
    if node_state == PROVENANCE:
        pass
        model_id = save_provenance_model(
            save_service=node_state.save_service,
            base_model_id=node_state.last_model_id,
            train_kwargs=get_dummy_train_kwargs(),
            prov_env=track_current_environment(),
            raw_data=None,
            ts_wrapper=dummy_train_service_wrapper(node_state.last_recovered_model),
        )
    else:
        # simulate model training by loading model from checkpoint
        model = _load_model_snapshot(node_state.state_description, node_state.u3_counter)
        model_id = save_model(model, node_state.save_service, base_model_id=node_state.last_model_id)

    node_state.last_model_id = model_id
    log_stop(log)

    # notify server
    text = NEW_MODEL
    if last_time:
        text = 'last'
    if done:
        text = 'done'
    message = generate_message(model_id=model_id, text=text)
    inform(message, node_state.socket, (args.server_ip, args.server_port))

    next_state()


def next_state():
    if node_state.state_description == U_1:
        node_state.state_description = U_3_1
        node_state.u3_counter += 1
        use_case_3()
    elif node_state.state_description == U_3_1:
        if node_state.u3_counter < node_state.u3_repeat:
            node_state.u3_counter += 1
            use_case_3(last_time=node_state.u3_counter == node_state.u3_repeat)
        else:
            node_state.state_description = U_2
            node_state.u3_counter = 0
            listen(sock=node_state.socket, callback=use_case_2)
    elif node_state.state_description == U_2:
        node_state.state_description = U_3_2
        node_state.u3_counter += 1
        use_case_3(node_state.u3_counter == node_state.u3_repeat)
    elif node_state.state_description == U_3_2:
        if node_state.u3_counter < node_state.u3_repeat:
            node_state.u3_counter += 1
            use_case_3(done=node_state.u3_counter == node_state.u3_repeat)
        else:
            print('DONE')


def _load_model_snapshot(state, counter):
    state = state.replace('U_', '').replace('_', '-')
    snapshot_name = USE_CASE_TEMPLATE.format(state, counter)
    snapshot_path = os.path.join(node_state.model_snapshots, snapshot_name)
    print('load model: {}'.format(snapshot_path))
    state_dict = torch.load(snapshot_path)
    model: torch.nn.Module = node_state.model_class()
    model.load_state_dict(state_dict)
    return model


def parse_args():
    parser = argparse.ArgumentParser(description='Script modeling node for workflow using baseline approach')
    add_server_connection_arguments(parser)
    add_node_connection_arguments(parser)
    add_model_arg(parser)
    add_model_snapshot_arg(parser)
    add_paths(parser)
    add_mongo_ip(parser)
    add_approach(parser)
    add_u3_count(parser)
    _args = parser.parse_args()

    return _args


if __name__ == '__main__':
    args = parse_args()
    main(args)
