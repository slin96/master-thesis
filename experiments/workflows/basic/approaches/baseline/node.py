import argparse
import json

from mmlib.log import use_model
from mmlib.save import FileSystemMongoSaveRecoverService

from experiments.workflows.node_shared import listen, update_model
from experiments.workflows.shared import add_connection_arguments, add_paths, save_compare_info, MODEL_ID, LAST

global_args = None


def main(args):
    global global_args
    global_args = args
    # wait for new model to be ready
    listen((args.node_ip, args.node_port), react_to_new_model)


def react_to_new_model(msg):
    print(msg)
    json_msg = json.loads(msg[0].decode("utf-8"))
    model_id = json_msg[MODEL_ID]
    last = json_msg[LAST]

    # as soon as new model is available
    recover_service = FileSystemMongoSaveRecoverService(args.tmp_dir, args.mongo_ip)
    recovered_model = recover_service.recover_model(model_id)
    # use recovered model
    use_model(model_id)

    # NOT TIMED save state_dict and output to compare restored model
    save_compare_info(recovered_model, 'node', model_id, args.log_dir)

    # go back to listen state id there is another message expected
    if not last:
        listen((args.node_ip, args.node_port), react_to_new_model)
    else:
        # if there are no updates coming anymore from server switch to update locally
        # update_model_locally(recovered_model, model_id)
        pass


def update_model_locally(model, model_id):
    locally_trained_model = update_model(model)


def parse_args():
    parser = argparse.ArgumentParser(description='Script modeling the node')

    add_connection_arguments(parser)
    add_paths(parser)

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)
