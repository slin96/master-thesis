import argparse
import os

from mmlib.persistence import FileSystemPersistenceService, MongoDictPersistenceService
from mmlib.save import BaselineSaveService

from experiments.baseline_flow.shared import save_model, add_paths, inform, generate_message, \
    listen, reusable_udp_socket, extract_fields, add_mongo_ip, add_server_connection_arguments, \
    add_node_connection_arguments, NEW_MODEL, add_admin_connection_arguments
from experiments.models.mobilenet import mobilenet_v2


class ServerState:
    def __init__(self, tmp_dir, mongo_host, ip, port, admin_ip, admin_port):
        # initialize a socket to communicate with otehr nodes
        self.socket = reusable_udp_socket()
        self.socket.bind((ip, port))
        self.admin_address = (admin_ip, admin_port)

        # initialize a service to save files
        abs_tmp_path = os.path.abspath(tmp_dir)
        file_pers_service = FileSystemPersistenceService(abs_tmp_path)

        # initialize service to store dictionaries (JSON),
        dict_pers_service = MongoDictPersistenceService(host=mongo_host)

        # initialize baseline save service
        self.save_service = BaselineSaveService(file_pers_service, dict_pers_service)

        # list of all models that have been saved by the node or have been communicated to be available
        self.saved_model_ids = []


server_state: ServerState = None
global_args = None


def main(args):
    global server_state, global_args
    global_args = args
    server_state = ServerState(args.tmp_dir, args.mongo_host, args.server_ip, args.server_port, args.admin_ip,
                               args.admin_port)

    use_case_1()

    print('wait for node ...')
    listen(sock=server_state.socket, callback=use_case_3)


def use_case_1():
    print('use case 1')
    # TODO parametrize model
    model = mobilenet_v2(pretrained=True)
    init_model_id = save_model(model, server_state.save_service)
    server_state.saved_model_ids.append(init_model_id)

    _inform_node_about_model(init_model_id)


def _inform_node_about_model(init_model_id):
    message = generate_message(text=NEW_MODEL, model_id=init_model_id)
    inform(message, server_state.socket, (args.node_ip, args.node_port))


def use_case_3(msg):
    print('use case 3')
    print(msg)
    text, model_id = extract_fields(msg)
    server_state.saved_model_ids.append(model_id)
    if 'done' in text:
        use_case_4()
    elif 'last' in text:
        # if this is the last message that will reach from the node for now U2 is finished
        # we transition to U2 and the server send an updated model
        print('send updated model')
        use_case_2()
    else:
        listen(sock=server_state.socket, callback=use_case_3)


def use_case_2():
    print('use case 2')
    # TODO implement loading new model
    model = mobilenet_v2(pretrained=True)
    model_id = save_model(model, server_state.save_service)
    server_state.saved_model_ids.append(model_id)

    _inform_node_about_model(model_id)

    print('wait for node ...')
    listen(sock=server_state.socket, callback=use_case_3)


def use_case_4():
    print('use case 4')
    for model_id in server_state.saved_model_ids:
        print('recover: {}'.format(model_id))
        server_state.save_service.recover_model(model_id, execute_checks=True)
    print('DONE')


def parse_args():
    parser = argparse.ArgumentParser(description='Script modeling server for workflow using baseline appraoch')
    # TODO make model configurable later
    # parser.add_argument('--model', help='The model to use for the run',
    #                     choices=[MOBILENET, GOOGLENET, RESNET_18, RESNET_50, RESNET_152])
    add_server_connection_arguments(parser)
    add_node_connection_arguments(parser)
    add_admin_connection_arguments(parser)
    add_paths(parser)
    add_mongo_ip(parser)
    _args = parser.parse_args()

    return _args


if __name__ == '__main__':
    args = parse_args()
    main(args)
