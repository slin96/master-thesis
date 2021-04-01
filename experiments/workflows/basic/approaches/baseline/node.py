import argparse

from mmlib.log import use_model
from mmlib.persistence import MongoDictPersistenceService, FileSystemPersistenceService
from mmlib.save import BaselineSaveService
from schema.save_info_builder import ModelSaveInfoBuilder

from experiments.measure.eventtimer import EventTimer
from experiments.workflows.node_shared import update_model
from experiments.workflows.shared import add_connection_arguments, add_paths, save_compare_info, listen, \
    extract_fields, generate_message, inform

NODE = 'node'

node_timer = EventTimer()
recover_counter = 0


def main(args):
    # wait for new model to be ready
    listen((args.node_ip, args.node_port), react_to_new_model)


def react_to_new_model(msg):
    print(msg)
    last, model_id = extract_fields(msg)
    global node_timer, recover_counter

    # as soon as new model is available
    dict_pers_service = MongoDictPersistenceService(host=args.mongo_ip)
    file_pers_service = FileSystemPersistenceService(args.tmp_dir)
    save_service = BaselineSaveService(file_pers_service, dict_pers_service)

    # time recover
    time_name = 'recover-{}'.format(recover_counter)
    recover_counter += 1
    node_timer.start_event(time_name)
    recovered_model_info = save_service.recover_model(model_id)
    node_timer.stop_event(time_name)
    # -------------------------------------

    # use recovered model
    use_model(model_id)
    recovered_model = recovered_model_info.model
    # NOT TIMED save state_dict and output to compare restored model
    save_compare_info(recovered_model, NODE, model_id, args.log_dir)

    # go back to listen state id there is another message expected
    if not last:
        listen((args.node_ip, args.node_port), react_to_new_model)
    else:
        # if there are no updates coming anymore from server switch to update locally
        update_model_locally(recovered_model, model_id)


def update_model_locally(model, base_model_id):
    locally_trained_model = update_model(model)

    dict_pers_service = MongoDictPersistenceService(host=args.mongo_ip)
    file_pers_service = FileSystemPersistenceService(args.tmp_dir)
    save_service = BaselineSaveService(file_pers_service, dict_pers_service)

    # time save process
    # -------------------------------------
    time_name = 'save_model'
    node_timer.start_event(time_name)
    save_version_info_builder = ModelSaveInfoBuilder()
    save_version_info_builder.add_model_info(locally_trained_model, base_model_id=base_model_id)
    save_version_info = save_version_info_builder.build()
    model_id = save_service.save_model(save_version_info)
    node_timer.stop_event(time_name)
    # -------------------------------------

    # NOT TIMED save state_dict and output to compare restored model
    save_compare_info(locally_trained_model, NODE, model_id, args.log_dir)

    # inform that a new model is available in the DB ready to use
    message = generate_message(model_id, True)
    inform(message, (args.node_ip, args.node_port), (args.server_ip, args.server_port))

    print(node_timer.get_time_line())
    print(node_timer.get_elapsed_times())


def parse_args():
    parser = argparse.ArgumentParser(description='Script modeling the node')

    add_connection_arguments(parser)
    add_paths(parser)

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)
