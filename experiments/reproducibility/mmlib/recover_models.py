import argparse
import os

from mmlib.constants import MMLIB_CONFIG
from mmlib.persistence import FileSystemPersistenceService, MongoDictPersistenceService
from mmlib.save import ProvenanceSaveService

from experiments.evaluation_flow.shared import add_mongo_ip, add_config, add_paths


def main(args):
    saved_model_ids = args.recover_ids.split(',')

    os.environ[MMLIB_CONFIG] = args.config

    abs_tmp_path = os.path.abspath(args.tmp_dir)
    file_pers_service = FileSystemPersistenceService(abs_tmp_path)

    dict_pers_service = MongoDictPersistenceService(host=args.mongo_host)

    save_service = ProvenanceSaveService(file_pers_service, dict_pers_service, logging=False)

    for _id in saved_model_ids:
        print('recover: {}'.format(_id))
        save_service.recover_model(_id, execute_checks=True)


def parse_args():
    parser = argparse.ArgumentParser()
    add_mongo_ip(parser)
    add_config(parser)
    add_paths(parser)
    parser.add_argument('--recover_ids', required=True)
    _args = parser.parse_args()

    return _args


if __name__ == '__main__':
    args = parse_args()
    main(args)
