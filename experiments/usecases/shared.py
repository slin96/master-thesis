SERVER_IP = "127.0.0.1"
SERVER_PORT = 18196

NODE_IP = "127.0.0.1"
NODE_PORT = 18197

ENCODING = 'utf-8'


def add_connection_arguments(parser):
    parser.add_argument('--server_ip', help='The server ip or hostname', default=SERVER_IP)
    parser.add_argument('--server_port', help='The server port', default=SERVER_PORT)
    parser.add_argument('--node_ip', help='The node ip or hostname', default=NODE_IP)
    parser.add_argument('--node_port', help='The node port', default=NODE_PORT)
    parser.add_argument('--mongo_ip', help='The ip or hostname for the mongoDB.', default=NODE_PORT)


def add_paths(parser):
    parser.add_argument('--tmp_dir', help='The directory to write tmp files to')
    parser.add_argument('--log_dir', help='The directory to write log files to')
