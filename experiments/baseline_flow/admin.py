import argparse
import os

from experiments.baseline_flow.shared import add_server_connection_arguments, add_admin_connection_arguments, \
    reusable_udp_socket, inform, listen, generate_message

MONGO_CONTAINER_NAME = 'mongo-db'


def start_db():
    os.system('docker run --rm --name %s -it -p 27017:27017 -d  mongo:4.4.3 ' % MONGO_CONTAINER_NAME)


def stop_db():
    os.system('docker kill %s' % MONGO_CONTAINER_NAME)


socket = None
repeat = 0
server_connection = None


def main(args):
    global socket, repeat, server_connection

    repeat = args.repeat

    server_connection = (args.server_ip, args.server_port)

    socket = reusable_udp_socket()
    socket.bind((args.admin_ip, args.admin_port))

    start_over(None)


def start_over(msg):
    global repeat

    if repeat > 0:
        print('start over')
        repeat -= 1

        stop_db()
        start_db()

        print('db ready')
        msg = generate_message(text='ready')
        inform(msg, socket, server_connection)

        print('listen for server to be done')
        listen(sock=socket, callback=start_over)


def parse_args():
    parser = argparse.ArgumentParser(description='Script modeling admin for workflow using baseline appraoch')
    add_server_connection_arguments(parser)
    add_admin_connection_arguments(parser)
    parser.add_argument('--repeat', help='How often to repeat the experiment', default=3)

    _args = parser.parse_args()

    return _args


if __name__ == '__main__':
    args = parse_args()
    main(args)
