import json
import socket
import time
import uuid

from mmlib.save import BaselineSaveService, WeightUpdateSaveService, ProvenanceSaveService
from mmlib.track_env import track_current_environment
from schema.save_info_builder import ModelSaveInfoBuilder

from experiments.models.googlenet import googlenet
from experiments.models.mobilenet import mobilenet_v2
from experiments.models.resnet152 import resnet152
from experiments.models.resnet18 import resnet18
from experiments.models.resnet50 import resnet50

BASELINE = 'baseline'
PARAM_UPDATE = 'param_update'
PARAM_UPDATE_IMPROVED = 'param_update_improved'
PROVENANCE = 'provenance'

TIME = 'time'
START_STOP = 'start-stop'

LOCAL_HOST = "127.0.0.1"

SERVER_IP = LOCAL_HOST
SERVER_PORT = 18196

NODE_IP = LOCAL_HOST
NODE_PORT = 18197

ADMIN_IP = LOCAL_HOST
ADMIN_PORT = 18198

ENCODING = 'utf-8'
MODEL_ID = 'model_id'
TEXT = 'TEXT'
NEW_MODEL = 'new model'

MSG_LEN = 1024

MOBILENET = "mobilenet"
GOOGLENET = "googlenet"
RESNET_18 = "resnet18"
RESNET_50 = "resnet50"
RESNET_152 = "resnet152"

START = 'START'
STOP = 'STOP'

U_1 = 'U_1'
U_2 = 'U_2'
U_3_1 = 'U_3_1'
U_3_2 = 'U_3_2'
U_4 = 'U_4'

MODELS_DICT = {MOBILENET: mobilenet_v2, GOOGLENET: googlenet, RESNET_18: resnet18, RESNET_50: resnet50,
               RESNET_152: resnet152}


def add_server_connection_arguments(parser):
    parser.add_argument('--server_ip', help='The server ip or hostname', default=SERVER_IP)
    parser.add_argument('--server_port', help='The server port', default=SERVER_PORT)


def add_node_connection_arguments(parser):
    parser.add_argument('--node_ip', help='The node ip or hostname', default=NODE_IP)
    parser.add_argument('--node_port', help='The node port', default=NODE_PORT)


def add_admin_connection_arguments(parser):
    parser.add_argument('--admin_ip', help='The db ip or hostname', default=NODE_IP)
    parser.add_argument('--admin_port', help='The db port', default=NODE_PORT)


def add_mongo_ip(parser):
    parser.add_argument('--mongo_host', help='The ip or hostname for the mongoDB.', default=LOCAL_HOST)


def add_paths(parser):
    parser.add_argument('--tmp_dir', help='The directory to write tmp files to')
    parser.add_argument('--log_dir', help='The directory to write log files to')


def add_model_arg(parser):
    parser.add_argument('--model', help='The model to use for the run',
                        choices=[MOBILENET, GOOGLENET, RESNET_18, RESNET_50, RESNET_152])


def add_approach(parser):
    parser.add_argument('--approach', help='The approach to use for the run', required=True,
                        choices=[BASELINE, PARAM_UPDATE, PARAM_UPDATE_IMPROVED, PROVENANCE])


def add_model_snapshot_arg(parser):
    parser.add_argument('--model_snapshots', help='The directory do find the model snapshots in', type=str)


def add_u3_count(parser):
    parser.add_argument('--u3_count', help='The amount of times u3 is repeated', type=int, required=True)


def add_training_data_path(parser):
    parser.add_argument('--training_data_path', help='The path to the data used to retrain the models', type=str)


def add_config(parser):
    parser.add_argument('--config', help='configuration file, only needed for prov appraoch', type=str)


def get_save_service(approach, dict_pers_service, file_pers_service):
    result = None

    # initialize save service
    if approach == BASELINE:
        result = BaselineSaveService(file_pers_service, dict_pers_service, logging=True)
    elif approach == PARAM_UPDATE:
        result = WeightUpdateSaveService(
            file_pers_service, dict_pers_service, improved_version=False, logging=True)
    elif approach == PARAM_UPDATE_IMPROVED:
        result = WeightUpdateSaveService(
            file_pers_service, dict_pers_service, improved_version=True, logging=True)
    elif approach == PROVENANCE:
        result = ProvenanceSaveService(file_pers_service, dict_pers_service, logging=True)
    else:
        raise NotImplementedError

    return result


def save_model(model, save_service, base_model_id=None):
    save_info_builder = ModelSaveInfoBuilder()
    env = track_current_environment()
    save_info_builder.add_model_info(model=model, env=env, base_model_id=base_model_id)
    save_info = save_info_builder.build()

    model_id = save_service.save_model(save_info)

    return model_id


def save_provenance_model(save_service, base_model_id, prov_env, raw_data, train_kwargs, ts_wrapper, model):
    save_info_builder = ModelSaveInfoBuilder()
    save_info_builder.add_model_info(base_model_id=base_model_id, env=prov_env)
    save_info_builder.add_prov_data(
        raw_data_path=raw_data, train_kwargs=train_kwargs, train_service_wrapper=ts_wrapper)
    save_info = save_info_builder.build()

    model_id = save_service.save_model(save_info)
    save_service.add_weights_hash_info(model_id, model)

    return model_id


def recover_model(model_id, save_service):
    # turn execute checks on to validate that saved an recovered model are the same
    restored_model_info = save_service.recover_model(model_id, execute_checks=True)
    return restored_model_info.model


def listen(sock, callback):
    received = sock.recvfrom(MSG_LEN)
    callback(received)


def reusable_udp_socket():
    # socket.SOCK_DGRAM use UDP
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # allow to reuse the socket address
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    return sock


def inform(message, sock, receiver):
    sock.sendto(message, receiver)


def extract_fields(msg):
    json_msg = json.loads(msg[0].decode("utf-8"))
    text = json_msg[TEXT]
    model_id = json_msg[MODEL_ID]
    return text, model_id


def generate_message(model_id=None, text=None):
    msg_json = {MODEL_ID: model_id, TEXT: text}
    msg_string = json.dumps(msg_json)
    return bytes(msg_string, encoding=ENCODING)


def log_start(role, use_case_id, event):
    t = time.time_ns()
    _id = uuid.uuid4()
    log_dict = {
        START_STOP: START,
        '_id': str(_id),
        'use_case': use_case_id,
        'role': role,
        'event': event,
        TIME: t
    }

    print(json.dumps(log_dict))

    return log_dict


def log_stop(log_dict):
    assert log_dict[START_STOP] == START

    t = time.time_ns()

    log_dict[START_STOP] = STOP
    log_dict[TIME] = t

    print(json.dumps(log_dict))
