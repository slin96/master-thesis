import argparse

import torch
from mmlib.constants import CURRENT_DATA_ROOT
from mmlib.deterministic import set_deterministic
from mmlib.persistence import FileSystemPersistenceService, MongoDictPersistenceService
from mmlib.recover_validation import RecoverValidationService
from mmlib.save import ProvenanceSaveService
from schema.environment import Environment
from schema.restorable_object import OptimizerWrapper, RestorableObjectWrapper
from schema.save_info_builder import ModelSaveInfoBuilder

from experiments.data.custom.custom_coco import TrainCustomCoco
from experiments.models import mobilenet
from experiments.models.resnet152 import resnet152
from experiments.models.resnet18 import resnet18
from experiments.models.resnet50 import resnet50
from experiments.prov_test.imagenet.imagenet_train import ImagenetTrainService
from experiments.workflows.server_shared import MOBILENET, RESNET_18, RESNET_50, RESNET_152
from experiments.workflows.shared import inform, generate_message, add_mongo_ip

MODEL_PATH = '../models/{}.py'
MODEL_LIST = [resnet18, resnet50, resnet152, mobilenet]


def main(args):
    file_pers_service = FileSystemPersistenceService(args.tmp_dir)
    dict_pers_service = MongoDictPersistenceService(args.mongo_ip)
    provenance_save_service = ProvenanceSaveService(file_pers_service, dict_pers_service)
    recover_val_service = RecoverValidationService(dict_pers_service)

    # save model-0 --------------------------------------------------
    print('save model-0')
    model = eval('{}(pretrained=True)'.format(args.model_name))
    code_file = MODEL_PATH.format(args.model_name)

    class_name = args.model_name
    save_info_builder = ModelSaveInfoBuilder()
    save_info_builder.add_model_info(model, code_file, class_name)
    save_info = save_info_builder.build()
    base_model_id = provenance_save_service.save_model(save_info)
    # ---------------------------------------------------------------

    # create ImagenetTrainService
    imagenet_ts = ImagenetTrainService()
    add_imagenet_prov_state_dict(imagenet_ts, model, args)
    prov_train_serv_code = './imagenet/imagenet_train.py'
    prov_train_serv_class_name = 'ImagenetTrainService'
    prov_train_wrapper_code = './imagenet/imagenet_train.py'
    prov_train_wrapper_class_name = 'ImagenetTrainWrapper'
    raw_data = args.data_path
    prov_env = Environment({})  # TODO store proper environment
    train_kwargs = {'number_batches': 2}

    # save provenance-0 ---------------------------------------------
    print('save-prov-0')
    save_info_builder = ModelSaveInfoBuilder()
    save_info_builder.add_model_info(code=code_file, model_class_name=class_name, base_model_id=base_model_id)

    save_info_builder.add_prov_data(
        raw_data_path=raw_data, env=prov_env, train_service=imagenet_ts, train_kwargs=train_kwargs,
        code=prov_train_serv_code, class_name=prov_train_serv_class_name, wrapper_code=prov_train_wrapper_code,
        wrapper_class_name=prov_train_wrapper_class_name)
    save_info = save_info_builder.build()

    model_id = provenance_save_service.save_model(save_info)
    # ---------------------------------------------------------------

    # train model on server to store the recover vl info
    print('generate recover-val')
    imagenet_ts.train(model, **train_kwargs)
    recover_val_service.save_recover_val_info(model, model_id, dummy_input_shape=[10, 3, 300, 400])

    # inform about model --------------------------------------------
    print('inform about new model')
    message = generate_message(model_id, False)
    inform(message, (args.server_ip, args.server_port), (args.node_ip, args.node_port))
    # ---------------------------------------------------------------


def add_imagenet_prov_state_dict(train_service, model, args):
    set_deterministic()

    state_dict = {}

    optimizer = torch.optim.SGD(model.parameters(), 1e-4, momentum=0.9, weight_decay=1e-4)
    state_dict['optimizer'] = OptimizerWrapper(
        import_cmd='import torch',
        class_name='torch.optim.SGD',
        init_args={'lr': 1e-4, 'momentum': 0.9, 'weight_decay': 1e-4},
        config_args={},
        init_ref_type_args=['params'],
        instance=optimizer
    )

    data_wrapper = TrainCustomCoco(args.data_path)
    state_dict['data'] = RestorableObjectWrapper(
        code='../data/custom/custom_coco.py',
        class_name='TrainCustomCoco',
        init_args={},
        config_args={'root': CURRENT_DATA_ROOT},
        init_ref_type_args=[],
        instance=data_wrapper
    )

    # Note use batch size 5 to reduce speed up tests
    dataloader = torch.utils.data.DataLoader(data_wrapper, batch_size=5, shuffle=False, num_workers=0,
                                             pin_memory=True)
    state_dict['dataloader'] = RestorableObjectWrapper(
        import_cmd='from torch.utils.data import DataLoader',
        class_name='DataLoader',
        init_args={'batch_size': 5, 'shuffle': False, 'num_workers': 0, 'pin_memory': True},
        config_args={},
        init_ref_type_args=['dataset'],
        instance=dataloader
    )

    train_service.state_objs = state_dict


def parse_args():
    parser = argparse.ArgumentParser(description='Script modeling server for basic workflow using baseline approach')
    parser.add_argument('--model_name', help='The model to use for the run'
                        , choices=[MOBILENET, RESNET_18, RESNET_50, RESNET_152]
                        )
    parser.add_argument('--tmp_dir', help='The directory to write tmp files to')
    parser.add_argument('--data_path', help='The directory where the train data can be found')
    add_mongo_ip(parser)

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)
