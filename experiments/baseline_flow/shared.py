import json
import socket

from mmlib.track_env import track_current_environment
from schema.save_info_builder import ModelSaveInfoBuilder

LOCAL_HOST = "127.0.0.1"

SERVER_IP = LOCAL_HOST
SERVER_PORT = 18196

NODE_IP = LOCAL_HOST
NODE_PORT = 18197

ENCODING = 'utf-8'
MODEL_ID = 'model_id'
LAST = 'last'

MSG_LEN = 1024


def add_connection_arguments(parser):
    parser.add_argument('--server_ip', help='The server ip or hostname', default=SERVER_IP)
    parser.add_argument('--server_port', help='The server port', default=SERVER_PORT)
    parser.add_argument('--node_ip', help='The node ip or hostname', default=NODE_IP)
    parser.add_argument('--node_port', help='The node port', default=NODE_PORT)
    add_mongo_ip(parser)


def add_mongo_ip(parser):
    parser.add_argument('--mongo_host', help='The ip or hostname for the mongoDB.', default=LOCAL_HOST)


def add_paths(parser):
    parser.add_argument('--tmp_dir', help='The directory to write tmp files to')
    parser.add_argument('--log_dir', help='The directory to write log files to')


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


def listen(receiver, callback):
    sock = reusable_udp_socket()
    sock.bind(receiver)
    received = sock.recvfrom(MSG_LEN)
    callback(received)
    sock.detach()
    sock.close()


def reusable_udp_socket():
    # socket.SOCK_DGRAM use UDP
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # allow to reuse the socket address
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    return sock


def inform(message, sender, receiver):
    sock = reusable_udp_socket()
    sock.bind(sender)
    sock.sendto(message, receiver)


def extract_fields(msg):
    json_msg = json.loads(msg[0].decode("utf-8"))
    last = json_msg[LAST]
    model_id = json_msg[MODEL_ID]
    return last, model_id


def generate_message(model_id, last_message):
    msg_json = {MODEL_ID: model_id, LAST: last_message}
    msg_string = json.dumps(msg_json)
    return bytes(msg_string, encoding=ENCODING)
