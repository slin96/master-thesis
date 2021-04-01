import argparse

from mmlib.persistence import FileSystemPersistenceService, MongoDictPersistenceService
from mmlib.recover_validation import RecoverValidationService
from mmlib.save import ProvenanceSaveService

from experiments.workflows.shared import listen, extract_fields


def main(args):
    listen((args.node_ip, args.node_port), react_to_new_model)


def react_to_new_model(msg):
    file_pers_service = FileSystemPersistenceService(args.tmp_dir)
    dict_pers_service = MongoDictPersistenceService(args.mongo_ip)
    provenance_save_service = ProvenanceSaveService(file_pers_service, dict_pers_service)
    recover_val_service = RecoverValidationService(dict_pers_service)

    print(msg)
    last, model_id = extract_fields(msg)

    recovered_model_info = provenance_save_service.recover_model(model_id)

    recovered_model_1 = recovered_model_info.model
    recover_check = recover_val_service.check_recover_val(model_id, recovered_model_1)
    print('Recover Check: {}'.format(recover_check))


def parse_args():
    parser = argparse.ArgumentParser(
        description='Script modeling server for basic workflow using baseline approach')
    parser.add_argument('--model', help='The model to use for the run',
                        choices=[MOBILENET, GOOGLENET, RESNET_18, RESNET_50, RESNET_152])

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)
