import json
import socket

from mmlib.track_env import track_current_environment
from schema.save_info_builder import ModelSaveInfoBuilder

from experiments.models.googlenet import googlenet
from experiments.models.mobilenet import mobilenet_v2
from experiments.models.resnet152 import resnet152
from experiments.models.resnet18 import resnet18
from experiments.models.resnet50 import resnet50

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


def add_model_snapshot_arg(parser):
    parser.add_argument('--model_snapshots', help='The directory do find the model snapshots in', type=str)


def save_model(model, save_service):
    save_info_builder = ModelSaveInfoBuilder()
    env = track_current_environment()
    save_info_builder.add_model_info(model=model, env=env)
    save_info = save_info_builder.build()

    model_id = save_service.save_model(save_info)

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
