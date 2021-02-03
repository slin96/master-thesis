import os

import torch
from mmlib.deterministic import set_deterministic
from mmlib.helper import imagenet_input

SERVER_IP = "127.0.0.1"
SERVER_PORT = 18196

NODE_IP = "127.0.0.1"
NODE_PORT = 18197

ENCODING = 'utf-8'
MODEL_ID = 'model_id'
LAST = 'last'


def add_connection_arguments(parser):
    parser.add_argument('--server_ip', help='The server ip or hostname', default=SERVER_IP)
    parser.add_argument('--server_port', help='The server port', default=SERVER_PORT)
    parser.add_argument('--node_ip', help='The node ip or hostname', default=NODE_IP)
    parser.add_argument('--node_port', help='The node port', default=NODE_PORT)
    add_mongo_ip(parser)


def add_mongo_ip(parser):
    parser.add_argument('--mongo_ip', help='The ip or hostname for the mongoDB.', default=NODE_PORT)


def add_paths(parser):
    parser.add_argument('--tmp_dir', help='The directory to write tmp files to')
    parser.add_argument('--log_dir', help='The directory to write log files to')


def save_compare_info(model, container, model_id, log_dir):
    state_dict_path = os.path.join(log_dir, '{}-{}-state-dict'.format(container, model_id))
    torch.save(model.state_dict(), state_dict_path)

    # we have to make the input and computation deterministic to make the models comparable
    set_deterministic()
    dummy_input = imagenet_input()
    model.eval()
    dummy_output = model(dummy_input)
    output_path = os.path.join(log_dir, '{}-{}-output'.format(container, model_id))
    torch.save(dummy_output, output_path)
