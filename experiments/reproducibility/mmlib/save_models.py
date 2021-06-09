import argparse
import os

from mmlib.constants import MMLIB_CONFIG, CURRENT_DATA_ROOT
from mmlib.deterministic import set_deterministic
from mmlib.persistence import FileSystemPersistenceService, MongoDictPersistenceService
from mmlib.save import ProvenanceSaveService
from mmlib.schema.restorable_object import RestorableObjectWrapper, StateFileRestorableObjectWrapper
from mmlib.schema.save_info_builder import ModelSaveInfoBuilder
from mmlib.track_env import track_current_environment
from torch.utils.data import DataLoader

from experiments.evaluation_flow.custom_coco import TrainCustomCoco
from experiments.evaluation_flow.imagenet_optimizer import ImagenetOptimizer
from experiments.evaluation_flow.imagenet_train import ImagenetTrainService, DATA, DATALOADER, OPTIMIZER, \
    ImagenetTrainWrapper
from experiments.evaluation_flow.shared import add_mongo_ip, add_config, add_training_data_path, add_paths
from experiments.models.mobilenet import mobilenet_v2


def main(args):
    saved_model_ids = []

    os.environ[MMLIB_CONFIG] = args.config

    abs_tmp_path = os.path.abspath(args.tmp_dir)
    file_pers_service = FileSystemPersistenceService(abs_tmp_path)

    dict_pers_service = MongoDictPersistenceService(host=args.mongo_host)

    save_service = ProvenanceSaveService(file_pers_service, dict_pers_service, logging=False)

    model = mobilenet_v2(pretrained=True)

    ###############################################################################
    # as a first step we save model model-0
    ###############################################################################
    # it will be save as a full model since there is no model it was derived from
    save_info_builder = ModelSaveInfoBuilder()
    env = track_current_environment()
    save_info_builder.add_model_info(model=model, env=env)
    save_info = save_info_builder.build()

    base_model_id = save_service.save_model(save_info)
    saved_model_ids.append(base_model_id)

    ################################################################################################################
    # as next we define the provenance data, that can not be automatically inferred
    ################################################################################################################
    # define the environment we will train in
    prov_env = track_current_environment()

    # define the data we use for training
    raw_data = args.training_data_path

    # define the arguments we use for training (here we use only 4 batches to speed up the process)
    train_kwargs = {'number_batches': 4}

    # to train the model we use the imagenet train service specified above
    imagenet_ts = ImagenetTrainService()
    # to make the train service work we have to initialize its state dict containing objects that are required for
    # training, for example: optimizer and dataloader. The objects in the state dict have to be of type
    # RestorableObjectWrapper so that the hold instances and their state can be stored and restored

    set_deterministic()

    state_dict = {}

    # before we can define the data loader, we have to define the data wrapper
    # for this test case we will use the data from our custom coco dataset
    data_wrapper = TrainCustomCoco(raw_data)
    state_dict[DATA] = RestorableObjectWrapper(
        config_args={'root': CURRENT_DATA_ROOT},
        instance=data_wrapper
    )

    # as a dataloader we use the standard implementation provided by pyTorch
    # this is why we instead of specifying the code path, we specify an import cmd
    # also we to track all arguments that have been used for initialization of this objects
    data_loader_kwargs = {'batch_size': 5, 'shuffle': True, 'num_workers': 0, 'pin_memory': True}
    dataloader = DataLoader(data_wrapper, **data_loader_kwargs)
    state_dict[DATALOADER] = RestorableObjectWrapper(
        import_cmd='from torch.utils.data import DataLoader',
        init_args=data_loader_kwargs,
        init_ref_type_args=['dataset'],
        instance=dataloader
    )

    # while the other objects do not have an internal state (other than the basic parameters) an optimizer can have
    # some more extensive state. In pyTorch it offers also the method .state_dict(). To store and restore we use an
    # Optimizer wrapper object.
    optimizer_kwargs = {'lr': 1e-4, 'weight_decay': 1e-4}
    optimizer = ImagenetOptimizer(model.parameters(), **optimizer_kwargs)
    state_dict[OPTIMIZER] = StateFileRestorableObjectWrapper(
        # code=FileReference(OPTIMIZER_CODE),
        init_args=optimizer_kwargs,
        init_ref_type_args=['params'],
        instance=optimizer
    )

    # having created all the objects needed for imagenet training we can plug the state dict into the train servcie
    imagenet_ts.state_objs = state_dict

    # finally we wrap the train service in the corresponding wrapper
    ts_wrapper = ImagenetTrainWrapper(instance=imagenet_ts)

    ################################################################################################################
    # having specified all the provenance information that will be used to train a model, we can save it
    ################################################################################################################
    save_info_builder = ModelSaveInfoBuilder()
    save_info_builder.add_model_info(base_model_id=base_model_id, env=prov_env)
    save_info_builder.add_prov_data(
        raw_data_path=raw_data, train_kwargs=train_kwargs, train_service_wrapper=ts_wrapper)
    save_info = save_info_builder.build()

    ################################################################################################################
    # restoring this model will result in a model that was trained according to the given provenance data
    # in this case it should be equivalent to the initial model trained using the specified train_service using the
    # specified data and train kwargs
    ################################################################################################################
    model_id = save_service.save_model(save_info)
    saved_model_ids.append(model_id)

    imagenet_ts.train(model, **train_kwargs)
    save_service.add_weights_hash_info(model_id, model)

    ################################################################################################################
    # Having defined the provenance information above storing a second version is a lot shorter
    ################################################################################################################
    save_info_builder = ModelSaveInfoBuilder()
    save_info_builder.add_model_info(base_model_id=model_id, env=prov_env)
    save_info_builder.add_prov_data(raw_data_path=raw_data, train_kwargs=train_kwargs,
                                    train_service_wrapper=ts_wrapper)
    save_info = save_info_builder.build()

    model_id_2 = save_service.save_model(save_info)
    saved_model_ids.append(model_id_2)

    imagenet_ts.train(model, **train_kwargs)
    save_service.add_weights_hash_info(model_id_2, model)

    print('saved models ids: {}'.format(saved_model_ids))


def parse_args():
    parser = argparse.ArgumentParser()
    add_mongo_ip(parser)
    add_training_data_path(parser)
    add_config(parser)
    add_paths(parser)
    _args = parser.parse_args()

    return _args


if __name__ == '__main__':
    args = parse_args()
    main(args)
