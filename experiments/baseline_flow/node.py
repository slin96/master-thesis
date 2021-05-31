import argparse
import os

from mmlib.persistence import FileSystemPersistenceService, MongoDictPersistenceService
from mmlib.save import BaselineSaveService

from experiments.baseline_flow.shared import recover_model, listen, extract_fields, add_paths, \
    save_model, generate_message, inform, reusable_udp_socket, add_mongo_ip, add_server_connection_arguments, \
    add_node_connection_arguments, NEW_MODEL
from experiments.models.mobilenet import mobilenet_v2


class NodeState:
    def __init__(self, u3_repeat, ip, port):
        self.socket = reusable_udp_socket()
        self.socket.bind((ip, port))

        # initialize a service to save files
        abs_tmp_path = os.path.abspath(args.tmp_dir)
        file_pers_service = FileSystemPersistenceService(abs_tmp_path)

        # initialize service to store dictionaries (JSON),
        dict_pers_service = MongoDictPersistenceService(host=args.mongo_host)

        # initialize baseline save service
        self.save_service = BaselineSaveService(file_pers_service, dict_pers_service)

        self.state_description = 'U1'
        self.u3_repeat = u3_repeat
        self.u3_counter = 0


node_sate: NodeState = None
global_args = None


def main(args):
    global node_sate, global_args
    node_sate = NodeState(2, args.node_ip, args.node_port)
    global_args = args

    # U1- node: listen for models to be in DB
    listen(sock=node_sate.socket, callback=use_case_1)


def _react_to_new_model(msg):
    print('message received')
    print(msg)
    text, model_id = extract_fields(msg)
    model = recover_model(model_id, node_sate.save_service)
    assert model is not None
    next_state()


def use_case_1(msg):
    print('use case 1')
    _react_to_new_model(msg)


def use_case_2(msg):
    print('use case 2')
    _react_to_new_model(msg)


def use_case_3(last_time=False, done=False):
    print('use case 3')
    # model training model by loading it from checkpoint
    # TODO somehow specify what model to load
    model = load_model()

    # save the model
    model_id = save_model(model, node_sate.save_service)

    # notify server
    text = NEW_MODEL
    if last_time:
        text = 'last'
    if done:
        text = 'done'
    message = generate_message(model_id=model_id, text=text)
    inform(message, node_sate.socket, (args.server_ip, args.server_port))

    next_state()


def next_state():
    if node_sate.state_description == 'U1':
        node_sate.state_description = 'U3_1'
        node_sate.u3_counter += 1
        use_case_3()
    elif node_sate.state_description == 'U3_1':
        if node_sate.u3_counter < node_sate.u3_repeat:
            node_sate.u3_counter += 1
            use_case_3(last_time=node_sate.u3_counter == node_sate.u3_repeat)
        else:
            node_sate.state_description = 'U2'
            node_sate.u3_counter = 0
            listen(sock=node_sate.socket, callback=use_case_2)
    elif node_sate.state_description == 'U2':
        node_sate.state_description = 'U3_2'
        node_sate.u3_counter += 1
        use_case_3(node_sate.u3_counter == node_sate.u3_repeat)
    elif node_sate.state_description == 'U3_2':
        if node_sate.u3_counter < node_sate.u3_repeat:
            node_sate.u3_counter += 1
            use_case_3(done=node_sate.u3_counter == node_sate.u3_repeat)
        else:
            print('DONE')


def load_model():
    # TODO
    return mobilenet_v2(pretrained=True)


def parse_args():
    parser = argparse.ArgumentParser(description='Script modeling node for workflow using baseline appraoch')
    add_server_connection_arguments(parser)
    add_node_connection_arguments(parser)
    add_paths(parser)
    add_mongo_ip(parser)
    _args = parser.parse_args()

    return _args


if __name__ == '__main__':
    args = parse_args()
    main(args)
