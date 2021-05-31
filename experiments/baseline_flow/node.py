import argparse
import os

from mmlib.persistence import FileSystemPersistenceService, MongoDictPersistenceService
from mmlib.save import BaselineSaveService

from experiments.baseline_flow.shared import recover_model, listen, extract_fields, add_connection_arguments, add_paths

save_service = None


def main(args):
    # initialize a service to save files
    abs_tmp_path = os.path.abspath(args.tmp_dir)
    file_pers_service = FileSystemPersistenceService(abs_tmp_path)

    # initialize service to store dictionaries (JSON),
    dict_pers_service = MongoDictPersistenceService(host=args.mongo_host)

    # initialize baseline save service
    global save_service
    save_service = BaselineSaveService(file_pers_service, dict_pers_service)

    # U1- node: listen for models to be in DB
    listen(receiver=(args.node_ip, args.node_port), callback=react_to_new_model)


def react_to_new_model(msg):
    print(msg)
    last, model_id = extract_fields(msg)
    model = recover_model(model_id, save_service)
    print(model)


def parse_args():
    parser = argparse.ArgumentParser(description='Script modeling node for workflow using baseline appraoch')
    add_connection_arguments(parser)
    add_paths(parser)
    _args = parser.parse_args()

    return _args


if __name__ == '__main__':
    args = parse_args()
    main(args)
