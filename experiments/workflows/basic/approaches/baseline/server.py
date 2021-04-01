import argparse
from time import sleep

from mmlib.log import use_model
from mmlib.persistence import FileSystemPersistenceService, MongoDictPersistenceService
from mmlib.save import BaselineSaveService
from schema.save_info_builder import ModelSaveInfoBuilder

from experiments.measure.eventtimer import EventTimer
from experiments.workflows.server_shared import *
from experiments.workflows.shared import *

# to run this make sure mongoDB is running:
# docker run --rm --name mongo-test -it -p 27017:27017 -d  mongo:latest

SERVER = 'server'

server_timer = EventTimer()


def main(args):
    dict_pers_service = MongoDictPersistenceService(host=args.mongo_ip)
    file_pers_service = FileSystemPersistenceService(args.tmp_dir)
    save_service = BaselineSaveService(file_pers_service, dict_pers_service)

    print('model used: {}'.format(args.model))
    # initially train the model in full dataset
    init_model = initial_train(models_dict[args.model])

    # time save the initial model
    time_name = 'save-initial'
    server_timer.start_event(time_name)
    # save the initially trained model
    save_info_builder = ModelSaveInfoBuilder()
    save_info_builder.add_model_info(init_model, args.model_code, models_dict[args.model].__name__)
    save_info = save_info_builder.build()
    init_model_id = save_service.save_model(save_info)
    server_timer.stop_event(time_name)
    # -------------------------------------

    # NOT TIMED: save state_dict and output to compare restored model
    save_compare_info(init_model, SERVER, init_model_id, args.log_dir)

    # inform that a new model is available in the DB ready to use
    message = generate_message(init_model_id, False)
    inform(message, (args.server_ip, args.server_port),
           (args.node_ip, args.node_port))

    print('informed about init model')
    # NOT TIMED: wait some time
    sleep(2)

    # update model
    updated_model = update_model(init_model)
    # save the updated model

    # time save the updated model
    time_name = 'save-updated'
    server_timer.start_event(time_name)
    save_info_builder = ModelSaveInfoBuilder()
    save_info_builder.add_model_info(updated_model, args.model_code, models_dict[args.model].__name__,
                                     base_model_id=init_model_id)
    save_info = save_info_builder.build()
    updated_model_id = save_service.save_model(save_info)
    server_timer.stop_event(time_name)
    # -------------------------------------

    # NOT TIMED: save state_dict and output to compare restored model
    save_compare_info(updated_model, SERVER, updated_model_id, args.log_dir)

    # inform that a new model is available in the DB ready to use
    message = generate_message(updated_model_id, True)
    inform(message, (args.server_ip, args.server_port), (args.node_ip, args.node_port))

    print('informed about updated model')

    print('server done, waiting for node updates')
    listen((args.server_ip, args.server_port), react_to_new_model)


def react_to_new_model(msg):
    print(msg)
    last, model_id = extract_fields(msg)

    # as soon as new model is available
    dict_pers_service = MongoDictPersistenceService(host=args.mongo_ip)
    file_pers_service = FileSystemPersistenceService(args.tmp_dir)
    save_service = BaselineSaveService(file_pers_service, dict_pers_service)

    # time save the updated model
    time_name = 'recover-model'
    server_timer.start_event(time_name)
    recovered_model_info = save_service.recover_model(model_id)
    recovered_model = recovered_model_info.model
    server_timer.stop_event(time_name)
    # -------------------------------------

    # use recovered model
    use_model(model_id)

    # NOT TIMED save state_dict and output to compare restored model
    save_compare_info(recovered_model, SERVER, model_id, args.log_dir)

    # go back to listen state id there is another message expected
    if not last:
        listen((args.node_ip, args.node_port), react_to_new_model)
    else:
        print('Server done')
        print(server_timer.get_time_line())
        print(server_timer.get_elapsed_times())


def parse_args():
    parser = argparse.ArgumentParser(description='Script modeling server for basic workflow using baseline approach')
    parser.add_argument('--model', help='The model to use for the run',
                        choices=[MOBILENET, GOOGLENET, RESNET_18, RESNET_50, RESNET_152])
    parser.add_argument('--model_code', help='The path to the code defining the model')
    parser.add_argument('--import_root', help='The root path for imports')

    add_connection_arguments(parser)
    add_paths(parser)

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)
