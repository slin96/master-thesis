import argparse
import json
import os
import time

import torch
from mmlib.constants import MMLIB_CONFIG, CURRENT_DATA_ROOT
from mmlib.deterministic import set_deterministic
from mmlib.persistence import FileSystemPersistenceService, MongoDictPersistenceService
from mmlib.schema.file_reference import FileReference
from mmlib.schema.restorable_object import RestorableObjectWrapper, StateFileRestorableObjectWrapper
from mmlib.track_env import track_current_environment
from mmlib.util.helper import get_device
from torch.utils.data import DataLoader

from experiments.evaluation_flow.create_finetuned import get_fine_tuned_model
from experiments.evaluation_flow.imagenet_optimizer import ImagenetOptimizer
from experiments.evaluation_flow.imagenet_train import ImagenetTrainService, DATA, DATALOADER, OPTIMIZER, \
    ImagenetTrainWrapper
from experiments.evaluation_flow.imagenet_train_loader import ImagenetTrainLoader
from experiments.evaluation_flow.shared import save_model, add_paths, inform, generate_message, \
    listen, reusable_udp_socket, extract_fields, add_mongo_ip, add_server_connection_arguments, \
    add_node_connection_arguments, NEW_MODEL, add_model_arg, MODELS_DICT, \
    add_model_snapshot_args, U_3_1, U_4, U_2, U_3_2, U_1, log_start, log_stop, add_approach, get_save_service, \
    add_config, PROVENANCE, add_training_data_path, get_dummy_train_kwargs, save_provenance_model, FINE_TUNED, VERSION, \
    add_u3_count, add_node_repeat

DONE_TXT = 'done.txt'

RECOVER_MODELS = 'recover_models'
EXTRACT_NOTIFY_MESSAGE = 'extract_notify_message'
SAVE_MODEL = 'save_model'
START_USECASE = 'start_usecase'
SERVER = 'server'

USE_CASE_1_PT = 'use-case-1.pt'
USE_CASE_2_PT = 'use-case-2.pt'


class ServerState:
    def __init__(self, approach, tmp_dir, mongo_host, ip, port, model_class, model_snapshots, snapshot_types,
                 training_data_path=None, config=None):
        # initialize a socket to communicate with other nodes
        self.socket = reusable_udp_socket()
        self.socket.bind((ip, port))

        # initialize a service to save files
        abs_tmp_path = os.path.abspath(tmp_dir)
        file_pers_service = FileSystemPersistenceService(abs_tmp_path)

        # initialize service to store dictionaries (JSON),
        dict_pers_service = MongoDictPersistenceService(host=mongo_host)

        self.save_service = get_save_service(approach, dict_pers_service, file_pers_service)
        self.approach = approach

        # list of all models that have been saved by the node or have been communicated to be available
        self.saved_model_ids = {}

        self.state_description = U_1
        self.u3_counter = 0

        self.model_class = model_class
        self.model_snapshots = model_snapshots
        self.snapshot_types = snapshot_types

        self.u1_model = None
        self.training_data_path = args.training_data_path

        self.server_environment = track_current_environment()
        self.dummy_train_kwargs = get_dummy_train_kwargs()

        # TODO should be read from args
        self.simulated_nodes = 2

        if approach == PROVENANCE:
            os.environ[MMLIB_CONFIG] = config


server_state: ServerState = None
init_model_id = None


def main(args):
    print(args)
    os.system('rm %s' % DONE_TXT)

    global server_state
    server_state = ServerState(args.approach, args.tmp_dir, args.mongo_host, args.server_ip, args.server_port,
                               MODELS_DICT[args.model], args.model_snapshots, args.snapshot_type,
                               training_data_path=args.training_data_path, config=args.config)

    use_case_1()


def use_case_1():
    global init_model_id
    # load model from snapshot
    model = _load_model_snapshot(USE_CASE_1_PT)
    server_state.u1_model = model

    log = log_start(SERVER, server_state.state_description, SAVE_MODEL)
    init_model_id = save_model(SERVER, server_state, model, server_state.save_service, server_state.server_environment)
    log_stop(log)

    server_state.saved_model_ids[init_model_id] = server_state.state_description

    _inform_node_about_model(init_model_id)

    next_state()


def _load_model_snapshot(snapshot_name):
    snapshot_path = os.path.join(server_state.model_snapshots, snapshot_name)

    if not server_state.state_description == U_1 and server_state.snapshot_types == FINE_TUNED:
        print('load model (fine-tuned): {}'.format(snapshot_path))
        base_path = os.path.join(server_state.model_snapshots, USE_CASE_1_PT)
        model = get_fine_tuned_model(server_state.model_class, base_path, snapshot_path)
    elif server_state.state_description == U_1 or server_state.snapshot_types == VERSION:
        device = get_device(None)
        print('load model (version): {}'.format(snapshot_path))
        state_dict = torch.load(snapshot_path, map_location=device)
        model: torch.nn.Module = server_state.model_class()
        model.load_state_dict(state_dict)
    else:
        raise NotImplementedError

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
    if server_state.approach == PROVENANCE:
        ts_wrapper = dummy_custom_imagenet_train_service_wrapper(
            server_state.u1_model, server_state.training_data_path)
        model_id = save_provenance_model(
            save_service=server_state.save_service,
            base_model_id=init_model_id,
            train_kwargs=server_state.dummy_train_kwargs,
            prov_env=server_state.server_environment,
            raw_data=server_state.training_data_path,
            ts_wrapper=ts_wrapper,
            model=model
        )
    else:
        model_id = save_model(SERVER, server_state, model, server_state.save_service, server_state.server_environment,
                              base_model_id=init_model_id)
    log_stop(log)

    server_state.saved_model_ids[model_id] = server_state.state_description

    _inform_node_about_model(model_id)

    next_state()


def use_case_4():
    log = log_start(SERVER, server_state.state_description, RECOVER_MODELS)
    for model_id in server_state.saved_model_ids.keys():
        model_recover = 'recover-{}-{}'.format(server_state.saved_model_ids[model_id], model_id)
        log_i = log_start(SERVER, server_state.state_description, model_recover)
        # if we use the provenance approach we only simulate the training, thus the saved and the recovered models will
        # differ -> we deactivate the checks for this approach
        execute_checks = not (server_state.approach == PROVENANCE)
        server_state.save_service.recover_model(model_id, execute_checks=execute_checks)
        log_stop(log_i)

    log_stop(log)
    next_state()


def log_sizes():
    for model_id in server_state.saved_model_ids.keys():
        size_info = server_state.save_service.model_save_size(model_id)
        print('size-info-{}-{}'.format(model_id, json.dumps(size_info)))


def next_state(text=None):
    time.sleep(5)
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
            if server_state.u3_counter > server_state.simulated_nodes:
                server_state.u3_counter = 1
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
            if server_state.u3_counter > server_state.simulated_nodes:
                server_state.u3_counter = 1
            listen(sock=server_state.socket, callback=use_case_3)
    elif server_state.state_description == U_4:
        log_sizes()
        print('DONE')
        os.system('touch %s' % DONE_TXT)


def dummy_custom_imagenet_train_service_wrapper(model, raw_data):
    imagenet_ts = ImagenetTrainService()

    set_deterministic()

    state_dict = {}

    data_wrapper = ImagenetTrainLoader(raw_data, split='val')
    state_dict[DATA] = RestorableObjectWrapper(
        config_args={'root': CURRENT_DATA_ROOT},
        init_args={'split': 'val'},
        instance=data_wrapper
    )

    data_loader_kwargs = {'batch_size': 4, 'shuffle': True, 'num_workers': 0, 'pin_memory': True}
    dataloader = DataLoader(data_wrapper, **data_loader_kwargs)
    state_dict[DATALOADER] = RestorableObjectWrapper(
        import_cmd='from torch.utils.data import DataLoader',
        init_args=data_loader_kwargs,
        init_ref_type_args=['dataset'],
        instance=dataloader
    )

    optimizer_kwargs = {'lr': 1e-4, 'weight_decay': 1e-4}
    optimizer = ImagenetOptimizer(model.parameters(), **optimizer_kwargs)
    state_dict[OPTIMIZER] = StateFileRestorableObjectWrapper(
        code=FileReference('imagenet_optimizer.py'),
        init_args=optimizer_kwargs,
        init_ref_type_args=['params'],
        instance=optimizer
    )

    imagenet_ts.state_objs = state_dict

    ts_wrapper = ImagenetTrainWrapper(instance=imagenet_ts)

    return ts_wrapper


def parse_args():
    parser = argparse.ArgumentParser(description='Script modeling server for workflow using baseline approach')

    add_server_connection_arguments(parser)
    add_node_connection_arguments(parser)
    add_model_arg(parser)
    add_model_snapshot_args(parser)
    add_paths(parser)
    add_mongo_ip(parser)
    add_approach(parser)
    add_training_data_path(parser)
    add_config(parser)
    add_u3_count(parser)
    add_node_repeat(parser)
    _args = parser.parse_args()

    return _args


if __name__ == '__main__':
    args = parse_args()
    main(args)
