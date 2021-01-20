import socket

MSG_LEN = 1024

SERVER_IP = "127.0.0.1"
SERVER_PORT = 18196

NODE_IP = "127.0.0.1"
NODE_PORT = 18197

ENCODING = 'utf-8'


def initial_train(model_ref):
    # we model the initial training of the model by just making use of the pretrained version
    return model_ref(pretrained=True)


def add_connection_arguments(parser):
    parser.add_argument('--server_ip', help='The server ip address', default=SERVER_IP)
    parser.add_argument('--server_port', help='The server port', default=SERVER_PORT)
    parser.add_argument('--node_ip', help='The node ip address', default=NODE_IP)
    parser.add_argument('--node_port', help='The node port', default=NODE_PORT)


def add_tmp_dir_path(parser):
    parser.add_argument('--tmp_dir', help='The directory to write tmp files to')


def inform(message, sender, receiver):
    # socket.SOCK_DGRAM use UDP
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(sender)
    sock.sendto(message, receiver)


def listen(receiver, callback):
    # socket.SOCK_DGRAM use UDP
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(receiver)
    received = sock.recvfrom(MSG_LEN)
    callback(received)
