import argparse
import os

import torch
from mmlib.constants import CURRENT_DATA_ROOT, MMLIB_CONFIG
from mmlib.deterministic import set_deterministic
from mmlib.persistence import FileSystemPersistenceService, MongoDictPersistenceService
from mmlib.schema.file_reference import FileReference
from mmlib.schema.restorable_object import RestorableObjectWrapper, StateFileRestorableObjectWrapper
from mmlib.track_env import track_current_environment
from mmlib.util.helper import get_device
from torch.utils.data import DataLoader

from experiments.evaluation_flow.create_finetuned import get_fine_tuned_model
from experiments.evaluation_flow.custom_coco import TrainCustomCoco
from experiments.evaluation_flow.imagenet_optimizer import ImagenetOptimizer
from experiments.evaluation_flow.imagenet_train import ImagenetTrainService, DATA, DATALOADER, OPTIMIZER, \
    ImagenetTrainWrapper
from experiments.evaluation_flow.shared import *

U_2 = 'U_2'

U_1 = 'U1'

SAVE_MODEL = 'save_model'

RECOVER_MODEL = 'recover_model'

NODE = 'node'

USE_CASE_TEMPLATE = 'use-case-{}-{}.pt'
USE_CASE_1_PT = 'use-case-1.pt'


class NodeState:
    def __init__(self, approach, u3_repeat, ip, port, model_class, model_snapshots, snapshot_types,
                 training_data_path=None, config=None):
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
        self.snapshot_types = snapshot_types

        self.last_model_id = None
        self.last_recovered_model = None

        self.u1_model_id = None
        self.u1_last_recovered_model = None
        self.u2_model_id = None
        self.u2_last_recovered_model = None

        if approach == PROVENANCE:
            self.training_data_path = training_data_path
            os.environ[MMLIB_CONFIG] = config

        self.node_environment = track_current_environment()
        self.dummy_train_kwargs = get_dummy_train_kwargs()

        # TODO should be read from args
        self.simulated_nodes = 2
        self.simulated_node_counter = 1


node_state: NodeState = None


def main(args):
    print(args)
    global node_state
    node_state = NodeState(args.approach, args.u3_count, args.node_ip, args.node_port, MODELS_DICT[args.model],
                           args.model_snapshots, args.snapshot_type, training_data_path=args.training_data_path,
                           config=args.config)

    # U1- node: listen for models to be in DB
    listen(sock=node_state.socket, callback=use_case_1)


def _react_to_new_model(msg, use_case=None):
    print(msg)
    log = log_start(NODE, node_state.state_description, RECOVER_MODEL)
    text, model_id = extract_fields(msg)
    node_state.last_model_id = model_id
    if use_case == U_1:
        node_state.u1_model_id = model_id
    # if we use the provenance approach we only simulate the training, thus the saved and the recovered models will
    # differ -> we deactivate the checks for this approach
    execute_checks = not (node_state.approach == PROVENANCE)
    model = recover_model(model_id, node_state.save_service, execute_checks=execute_checks)
    assert model is not None
    node_state.last_recovered_model = model
    log_stop(log)
    next_state()


def use_case_1(msg):
    _react_to_new_model(msg, U_1)


def use_case_2(msg):
    _react_to_new_model(msg, U_2)


def _state_with_counter():
    return '{}_{}'.format(node_state.state_description, node_state.u3_counter)


def use_case_3(last_time=False, done=False):
    state_w_counter = _state_with_counter()
    model = _load_model_snapshot(node_state.state_description, node_state.u3_counter)

    log = log_start(NODE, state_w_counter, SAVE_MODEL)
    if node_state.approach == PROVENANCE:
        ts_wrapper = dummy_custom_coco_train_service_wrapper(
            node_state.last_recovered_model, node_state.training_data_path)
        model_id = save_provenance_model(
            save_service=node_state.save_service,
            base_model_id=node_state.last_model_id,
            train_kwargs=node_state.dummy_train_kwargs,
            prov_env=node_state.node_environment,
            raw_data=node_state.training_data_path,
            ts_wrapper=ts_wrapper,
            model=model
        )
    else:
        # simulate model training by loading model from checkpoint
        model_id = save_model(NODE, node_state, model, node_state.save_service, node_state.node_environment,
                              base_model_id=node_state.last_model_id)
    log_stop(log)

    node_state.last_model_id = model_id

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
    time.sleep(5)
    if node_state.state_description == U_1:
        node_state.state_description = U_3_1
        node_state.u3_counter += 1
        use_case_3()
    elif node_state.state_description == U_3_1:
        if node_state.u3_counter < node_state.u3_repeat:
            node_state.u3_counter += 1
            use_case_3(last_time=node_state.u3_counter == node_state.u3_repeat and
                                 node_state.simulated_node_counter == node_state.simulated_nodes)
        elif node_state.simulated_node_counter < node_state.simulated_nodes:
            node_state.u3_counter = 1
            node_state.simulated_node_counter += 1
            use_case_3()
        else:
            node_state.state_description = U_2
            node_state.u3_counter = 0
            node_state.simulated_node_counter = 1
            listen(sock=node_state.socket, callback=use_case_2)
    elif node_state.state_description == U_2:
        node_state.state_description = U_3_2
        node_state.u3_counter += 1
        use_case_3(node_state.u3_counter == node_state.u3_repeat)
    elif node_state.state_description == U_3_2:
        if node_state.u3_counter < node_state.u3_repeat:
            node_state.u3_counter += 1
            use_case_3(done=node_state.u3_counter == node_state.u3_repeat and
                            node_state.simulated_node_counter == node_state.simulated_nodes)
        elif node_state.simulated_node_counter < node_state.simulated_nodes:
            node_state.u3_counter = 1
            node_state.simulated_node_counter += 1
            use_case_3()
        else:
            print('DONE')


def _load_model_snapshot(state, counter):
    state = state.replace('U_', '').replace('_', '-')
    snapshot_name = USE_CASE_TEMPLATE.format(state, counter)
    snapshot_path = os.path.join(node_state.model_snapshots, snapshot_name)

    if node_state.snapshot_types == FINE_TUNED:
        print('load model (fine-tuned): {}'.format(snapshot_path))
        base_path = os.path.join(node_state.model_snapshots, USE_CASE_1_PT)
        model = get_fine_tuned_model(node_state.model_class, base_path, snapshot_path)
    elif node_state.snapshot_types == VERSION:
        device = get_device(None)
        print('load model (version): {}'.format(snapshot_path))
        state_dict = torch.load(snapshot_path, map_location=device)
        model: torch.nn.Module = node_state.model_class()
        model.load_state_dict(state_dict)
    else:
        raise NotImplementedError

    return model


def dummy_custom_coco_train_service_wrapper(model, raw_data):
    imagenet_ts = ImagenetTrainService()

    set_deterministic()

    state_dict = {}

    data_wrapper = TrainCustomCoco(raw_data)
    state_dict[DATA] = RestorableObjectWrapper(
        config_args={'root': CURRENT_DATA_ROOT},
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
        code=FileReference('imagenet_optimizer.py'),  # NOTE check if we need this, might work automatically
        init_args=optimizer_kwargs,
        init_ref_type_args=['params'],
        instance=optimizer
    )

    imagenet_ts.state_objs = state_dict

    ts_wrapper = ImagenetTrainWrapper(instance=imagenet_ts)

    return ts_wrapper


def parse_args():
    parser = argparse.ArgumentParser(description='Script modeling node for workflow using baseline approach')
    add_server_connection_arguments(parser)
    add_node_connection_arguments(parser)
    add_model_arg(parser)
    add_model_snapshot_args(parser)
    add_paths(parser)
    add_mongo_ip(parser)
    add_approach(parser)
    add_u3_count(parser)
    add_training_data_path(parser)
    add_config(parser)
    _args = parser.parse_args()

    return _args


if __name__ == '__main__':
    args = parse_args()
    main(args)
