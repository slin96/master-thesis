import argparse

import torch
from mmlib.constants import CURRENT_DATA_ROOT
from mmlib.deterministic import set_deterministic
from mmlib.persistence import FileSystemPersistenceService, MongoDictPersistenceService
from mmlib.save import ProvenanceSaveService
from schema.environment import Environment
from schema.restorable_object import OptimizerWrapper, RestorableObjectWrapper
from schema.save_info_builder import ModelSaveInfoBuilder

from experiments.data.custom.custom_coco import TrainCustomCoco
from experiments.prov_test.imagenet.imagenet_train import ImagenetTrainService
from experiments.workflows.server_shared import MOBILENET, RESNET_18, RESNET_50, RESNET_152

MODEL_PATH = '../models/{}.py'


def main(args):
    file_pers_service = FileSystemPersistenceService(args.tmp_dir)
    dict_pers_service = MongoDictPersistenceService(args.mongo_ip)
    provenance_save_service = ProvenanceSaveService(file_pers_service, dict_pers_service)

    # save model-0 --------------------------------------------------
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
    add_imagenet_prov_state_dict(imagenet_ts, model)
    prov_train_serv_code = './imagenet/imagenet_train.py'
    prov_train_serv_class_name = 'ImagenetTrainService'
    prov_train_wrapper_code = './imagenet/imagenet_train.py'
    prov_train_wrapper_class_name = 'ImagenetTrainWrapper'
    raw_data = './data/reduced-custom-coco-data'  # TODO use args argument here
    prov_env = Environment({})  # TODO store proper environment
    train_kwargs = {'number_batches': 2}

    # save provenance-0 ---------------------------------------------
    save_info_builder = ModelSaveInfoBuilder()
    save_info_builder.add_model_info(code=code_file, model_class_name=class_name, base_model_id=base_model_id)

    save_info_builder.add_prov_data(
        raw_data_path=raw_data, env=prov_env, train_service=imagenet_ts, train_kwargs=train_kwargs,
        code=prov_train_serv_code, class_name=prov_train_serv_class_name, wrapper_code=prov_train_wrapper_code,
        wrapper_class_name=prov_train_wrapper_class_name)
    save_info = save_info_builder.build()

    model_id = provenance_save_service.save_model(save_info)
    # ---------------------------------------------------------------


def add_imagenet_prov_state_dict(train_service, model):
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

    data_wrapper = TrainCustomCoco('./data/reduced-custom-coco-data')
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
    parser.add_argument('--model_name', help='The model to use for the run',
                        choices=[MOBILENET, RESNET_18, RESNET_50, RESNET_152])

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)
