import collections
import json
import os
from os import listdir
from os.path import isfile

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from experiments.evaluation_flow.shared import BASELINE, PARAM_UPDATE, PARAM_UPDATE_IMPROVED, PROVENANCE

HPI_ORANGE = '#DE6207'
HPI_LIGHT_ORANGE = '#F7A900'
HPI_RED = '#B1083A'
YELLOW = '#c7c91e'
PURPLE = '#7516b5'

# Accessibility colors
A_RED = '#FA2100'
A_BLUE = '#6F62DB'
A_GREEN = '#6FDBA8'
A_YELLOW = '#E0D523'
A_PINK = '#EB88DB'

colors = {
    'load': A_RED,
    'pickle parameters': PURPLE,
    'recover': A_YELLOW,
    'hash parameters': HPI_RED,
    'check params': A_BLUE,
    'persist': HPI_ORANGE,
    'check env': HPI_LIGHT_ORANGE,
    'save dataset': HPI_ORANGE,
    'other': HPI_LIGHT_ORANGE,
    'training': HPI_LIGHT_ORANGE,
    'recover base': YELLOW,
    'recover full model': HPI_ORANGE,
    'generate update': HPI_RED

}

SAVE_DATASET = 'save_dataset'

GENERATE_PARAM_UPDATE = 'generate_param_update'

CALC_UPDATE = 'calc_update'

PERSIST_MODEL_INFO = 'persist_model_info'

WEIGHTS_HASH_INFO_NS = 'get_weights_hash_info_ns'

PICKLE_WEIGHTS_NS = 'pickle_weights_ns'

RECOVER_BASE_MODEL = 'recover_base_model'

TOTAL_SAVE_TIME_NS = 'total_save_time_ns'

DETAILED_RECOVER_TIMES = 'detailed_recover_times'

DETAILED_SAVE_TIMES = 'detailed_save_times'

GAPTIME = 'GAPTIME'

HIGH_LEVEL_RECOVER_TIMES = 'high_level_recover_times'

HIGH_LEVEL_SAVE_TIMES = 'high_level_save_times'

NODE = 'node'

SERVER = 'server'

LOCATION = 'location'

U_4 = 'U_4'

U_3_1 = 'U_3_1'
U_3_2 = 'U_3_2'

MODEL = 'model'

U_2 = 'U_2'

U_1 = 'U_1'

COLLECTION = 'collection'

TIME = 'time'

SCHEMA_OBJ = 'schema_obj'

SERVICE = 'service'

USE_CASE = 'use_case'

EVENT = 'event'

METHOD = 'method'

STOP = 'stop'

ID = '_id'

START = 'start'

START_STOP = 'start-stop'

EVENT = 'event'
EVENTS = 'events'

FLOAT_TEMPLATE = '{:.10f}'

APPROACH = 'approach'
SNAPSHOT_TYPE = 'snapshot_type'
SNAPSHOT_DIST = 'snapshot_dist'
RUN = 'run'

TOTAL_CONSUMPTIONS = 'total-consumptions'
CONSUMPTIONS = 'consumptions'


def use_case_ids(log_file, id2use_case=False):
    relevant_lines = []
    try:
        with open(log_file) as f:
            lines = f.readlines()
            search_for = 'recover-U'
            search_for_dist = 'recover-N'

            for l in lines:
                if search_for in l or search_for_dist in l:
                    relevant_lines.append(l)

        result = {}

        for l in relevant_lines:
            event_json = json.loads(l)
            split = event_json[EVENT].split('-')
            if id2use_case:
                result[split[2]] = split[1]
            else:
                result[split[1]] = split[2]

        return result
    except:
        print('broken file: {}'.format(log_file))


def id_use_case_dict(log_file):
    return use_case_ids(log_file, id2use_case=True)


def use_case1_id(log_file):
    with open(log_file) as f:
        lines = f.readlines()

        search_for = 'recover-U_1-'

        for l in lines:
            if search_for in l:
                event_json = json.loads(l)
                return event_json[EVENT].replace('recover-U_1-', '')


def storage_info(log_file, model_id):
    with open(log_file) as f:
        lines = f.readlines()

        search_for = 'size-info' + '-' + model_id

        for l in lines:
            if search_for in l:
                json_string = l.replace(search_for + '-', '').strip()
                return json.loads(json_string)


def total_storage_consumption(storage_info: dict):
    result = 0

    for k, v in storage_info.items():
        if isinstance(v, int):
            result += v
        else:
            result += total_storage_consumption(v)

    return result


class Event:

    def __init__(self, start_json, stop_json, children):
        self.start_json = start_json
        self.stop_json = stop_json
        self.children = children
        assert stop_json[ID] == start_json[ID]
        self.event_id = start_json[ID]
        self.use_case = start_json[USE_CASE] if USE_CASE in stop_json else None
        self.event = start_json[EVENT] if EVENT in stop_json else None
        self.method = start_json[METHOD] if METHOD in stop_json else None
        self.service = start_json[SERVICE] if SERVICE in stop_json else None
        self.schema_obj = start_json[SCHEMA_OBJ] if SCHEMA_OBJ in stop_json else None
        self.collection = start_json[COLLECTION] if COLLECTION in stop_json else None
        self.start_time = int(start_json[TIME])
        self.stop_time = int(stop_json[TIME])
        self.duration_ns = int(stop_json[TIME]) - int(start_json[TIME])
        self.duration_s = self.duration_ns * 10 ** -9
        # the times between the events
        self.gap_times_ns = self._calc_gap_times()
        self.gap_times_s = [t * 10 ** -9 for t in self.gap_times_ns]

    @property
    def _event_name(self):
        if self.use_case:
            return self.use_case
        elif self.method and self.event:
            return '{} -- {}'.format(self.method, self.event)
        elif self.service and self.method:
            if self.collection:
                return '{} -- {} -- collection({})'.format(self.service, self.method, self.collection)
            else:
                return '{} -- {}'.format(self.service, self.method)
        elif self.method and self.schema_obj:
            return '{} -- schema_obj({})'.format(self.method, self.schema_obj)

        else:
            return 'no name'

    def __str__(self):
        return self._generate_representation()

    def _generate_representation(self, level=0):
        result = ''
        tabs = '\t' * level

        time_str = FLOAT_TEMPLATE.format(self.duration_s)
        result += '{}: {}s \n'.format(self._event_name, time_str)
        for i, c in enumerate(self.children):

            if len(self.gap_times_ns) > 0:
                # add first gap time to result
                result += tabs + '{}: {}s \n'.format(GAPTIME, self.gap_times_s[i])

            result += tabs + c._generate_representation(level + 1)

        if len(self.gap_times_ns) > 0:
            # add last gap time to result
            result += tabs + '{}: {}s \n'.format(GAPTIME, self.gap_times_s[-1])
        return result

    def _calc_gap_times(self):
        gap_times = []
        if len(self.children) > 0:
            # The first gap is between the own start and the start of the first child
            gap_times.append(abs(self.start_time - self.children[0].start_time))

            # the next gap times are between the children events
            for child1, child2 in zip(self.children[:-1], self.children[1:]):
                gap_times.append(abs(child1.stop_time - child2.start_time))

            # the last gap is between the own stop time and the last child stop time
            gap_times.append(abs(self.stop_time - self.children[-1].stop_time))

            sum_gap_times = sum(gap_times)
            sum_child_times = sum([c.duration_ns for c in self.children])
            assert self.duration_ns == sum_gap_times + sum_child_times

        return gap_times


def parse_events(file):
    with open(file) as f:
        lines = f.readlines()

    filtered_lines = filter_non_events(lines)
    events = []
    while len(filtered_lines) > 0:
        event = _parse_event(filtered_lines)
        events.append(event)
    return events


def filter_non_events(lines):
    return list(filter(lambda l: START_STOP in l, lines))


def _parse_event(lines: list, _continue=True):
    start_line = lines.pop(0)
    start_line_json = json.loads(start_line)
    assert start_line_json[START_STOP].lower() == START
    start_id = start_line_json[ID]

    next_line = lines[0]
    next_line_json = json.loads(next_line)
    next_id = next_line_json[ID]

    if start_id == next_id:
        assert next_line_json[START_STOP].lower() == STOP
        # remove stop line so that next line for parsing is start again
        lines.pop(0)
        e = Event(start_line_json, next_line_json, [])
        return e
    else:
        assert next_line_json[START_STOP].lower() == START
        children_events = []
        while not next_id == start_id:
            child_event = _parse_event(lines, _continue=False)
            children_events.append(child_event)
            next_line = lines[0]
            next_line_json = json.loads(next_line)
            next_id = next_line_json[ID]
        assert next_id == start_id
        assert next_line_json[START_STOP].lower() == STOP
        e = Event(start_line_json, next_line_json, children_events)
        lines.pop(0)
        return e


def all_files_in_dir(directory):
    files = [os.path.join(directory, f) for f in listdir(directory) if isfile(os.path.join(directory, f))]
    return files


def extract_event_and_and_meta(file):
    meta = extract_file_meta(file)
    events = parse_events(file)

    return meta, events


def extract_file_meta(file):
    file_name = os.path.split(file)[-1].replace('.txt', '')
    split_file_name = file_name.split('--')
    meta = {}
    for e in split_file_name:
        spl = e.split(':')
        if len(spl) == 1:
            meta['location'] = spl[0]
        else:
            k, v = spl
            meta[k] = v
    return meta


def parse_all_log_files(root_log_dir):
    all_files = all_files_in_dir(root_log_dir)

    all_events = []
    for f in all_files:
        try:
            meta, events = extract_event_and_and_meta(f)
            all_events.append((meta, events))
        except:
            print('file is broken: {}'.format(f))

    return all_events


def filter_by_attribute(files, attribute):
    result = []
    attribute_key, attribute_value = attribute
    for f in files:
        meta, _ = f
        if meta[attribute_key] == attribute_value:
            result.append(f)

    return result


def filter_by_attributes(files, attributes):
    result = []
    for a in attributes:
        result = filter_by_attribute(files, a)

    return result


def calc_save_times(server_logs, node_logs):
    save_times = {}

    for file in server_logs:
        meta, events = file
        times = {}
        for e in events:
            if e.use_case == U_1:
                times[U_1] = e.duration_s
            elif e.use_case == U_2:
                times[U_2] = e.duration_s
        save_times[meta[MODEL]] = times

    for file in node_logs:
        meta, events = file
        times = {}
        u31_counter = 1
        u32_counter = 1
        for e in events:
            if e.use_case and e.use_case.startswith(U_3_1):
                times[e.use_case] = e.duration_s
                u31_counter += 1
            elif e.use_case and e.use_case.startswith(U_3_2):
                times[e.use_case] = e.duration_s
                u32_counter += 1
        save_times[meta[MODEL]].update(times)

    return save_times


def join_server_and_node_meta(meta_and_files):
    def _custom_id_string(meta):
        _id = '{}-{}-{}-{}-{}'.format(meta[MODEL], meta[APPROACH], meta[SNAPSHOT_TYPE], meta[SNAPSHOT_DIST], meta[RUN])
        return _id

    joined = {}
    for elem in meta_and_files:
        meta, file = elem
        meta_id = _custom_id_string(meta)
        if meta_id not in joined:
            joined[meta_id] = {meta[LOCATION]: (meta, file)}
        else:
            joined[meta_id].update({meta[LOCATION]: (meta, file)})

    joined_data = list(joined.values())
    valid_joined = [i for i in joined_data if len(i.keys()) == 2]

    return valid_joined


def extract_times(valid_joined, num_nodes=1, high_level_only = False):
    save_times = []

    for _dict in valid_joined:
        server_meta, _ = _dict[SERVER]
        node_meta, _ = _dict[NODE]

        combined = {
            MODEL: server_meta[MODEL],
            APPROACH: server_meta[APPROACH],
            SNAPSHOT_TYPE: server_meta[SNAPSHOT_TYPE],
            SNAPSHOT_DIST: server_meta[SNAPSHOT_DIST],
            RUN: server_meta[RUN]
        }


        _high_level_save_times = high_level_save_times(node_meta, server_meta, num_nodes)
        _high_level_recover_times = high_level_recover_times(server_meta, num_nodes)
        combined[HIGH_LEVEL_SAVE_TIMES] = _high_level_save_times
        combined[HIGH_LEVEL_RECOVER_TIMES] = _high_level_recover_times

        if not high_level_only:
            _detailed_save_times = detailed_save_times(node_meta, server_meta, num_nodes)
            _detailed_recover_times = detailed_recover_times(server_meta, num_nodes)
            combined[DETAILED_SAVE_TIMES] = _detailed_save_times
            combined[DETAILED_RECOVER_TIMES] = _detailed_recover_times

        save_times.append(combined)

    return save_times


def get_schema_obj_event(root_event, schema_obj, method):
    if schema_obj == root_event.schema_obj and root_event.method == method:
        return root_event
    else:
        for c in root_event.children:
            event_found = get_schema_obj_event(c, method, schema_obj)
            if event_found is not None:
                return event_found


def get_sub_event(root_event, method=None, event_name=None, starts_with=False, check_event_only=False):
    if check_event_only and root_event.event == event_name:
        return root_event
    elif check_event_only and root_event and root_event.event and root_event.event.startswith(event_name):
        return root_event
    elif not check_event_only and root_event.method == method and root_event.event.startswith(event_name):
        return root_event
    elif not check_event_only and starts_with and root_event and root_event.method \
            and root_event.method.startswith(method) and root_event.event.startswith(event_name):
        return root_event
    else:
        for c in root_event.children:
            event_found = get_sub_event(c, method, event_name, starts_with, check_event_only)
            if event_found is not None:
                return event_found


def _extract_detailed_save_times(event, approach):
    if approach == BASELINE:
        # search for event with event: 'all', method: 'save_full_model'
        save_sub_event = get_sub_event(event, method='_save_full_model', event_name='all')
        total_save_time_ns = event.duration_ns
        pickle_weights_ns = save_sub_event.children[0].duration_ns
        get_weights_hash_info_ns = save_sub_event.children[1].duration_ns
        persist_model_info = save_sub_event.children[2].duration_ns

        detailed_times = {
            TOTAL_SAVE_TIME_NS: total_save_time_ns,
            PICKLE_WEIGHTS_NS: pickle_weights_ns,
            WEIGHTS_HASH_INFO_NS: get_weights_hash_info_ns,
            PERSIST_MODEL_INFO: persist_model_info
        }
        return detailed_times
    elif approach == PARAM_UPDATE or approach == PARAM_UPDATE_IMPROVED:
        if event.use_case == U_1:
            save_sub_event = get_sub_event(event, method='_save_full_model', event_name='all')
            total_save_time_ns = event.duration_ns
            persist_model_info = save_sub_event.children[2].duration_ns
            generate_weights_update = 0
        else:
            save_sub_event = get_sub_event(event, method='_save_updated_model', event_name='all')
            total_save_time_ns = event.duration_ns

            persist_event = get_sub_event(save_sub_event, method='_save_updated_model', event_name='persist')
            persist_model_info = persist_event.duration_ns

            generate_weights_update_event = get_sub_event(
                save_sub_event, method='_save_updated_model', event_name='generate_weights_update', starts_with=True)
            generate_weights_update = generate_weights_update_event.duration_ns

        detailed_times = {
            TOTAL_SAVE_TIME_NS: total_save_time_ns,
            GENERATE_PARAM_UPDATE: generate_weights_update,
            PERSIST_MODEL_INFO: persist_model_info,
        }
        return detailed_times
    elif approach == PROVENANCE:
        if event.use_case == U_1:
            total_save_time_ns = event.duration_ns
            save_dataset_time = 0
        else:
            persist_dataset_event = get_schema_obj_event(event, 'dataset', 'persist')
            total_save_time_ns = event.duration_ns
            save_dataset_time = persist_dataset_event.duration_ns

        detailed_times = {
            TOTAL_SAVE_TIME_NS: total_save_time_ns,
            SAVE_DATASET: save_dataset_time,
        }
        return detailed_times
    else:
        return []


def _extract_detailed_recover_times(event, approach):
    result = []
    if approach == BASELINE:
        load_event = get_sub_event(event, method='recover_model', event_name='load_model_info_rec_files')
        recover_event = get_sub_event(event, method='recover_model', event_name='recover_from_info')
        check_weights = get_sub_event(recover_event, method='_check_weights', event_name='_all')
        check_environment = get_sub_event(recover_event, method='_check_env', event_name='_all')

        load_time = load_event.duration_ns
        check_weights_time = check_weights.duration_ns
        check_env_time = check_environment.duration_ns
        recover_time = recover_event.duration_ns - check_env_time - check_weights_time

        detailed_times = {
            'load_time': load_time,
            'recover_time': recover_time,
            'check_weights_time': check_weights_time,
            'check_env_time': check_env_time
        }
        result = detailed_times
    elif approach == PARAM_UPDATE or approach == PARAM_UPDATE_IMPROVED:
        recover_event = get_sub_event(event, method='recover_model', event_name='_recover_from_weight_update')
        if recover_event is None:
            load_event = get_sub_event(event, method='recover_model', event_name='load_model_info_rec_files')
            recover_event = get_sub_event(event, method='recover_model', event_name='recover_from_info')
            check_weights = get_sub_event(recover_event, method='_check_weights', event_name='_all')
            check_environment = get_sub_event(recover_event, method='_check_env', event_name='_all')

            load_time = load_event.duration_ns
            check_weights_time = check_weights.duration_ns
            check_env_time = check_environment.duration_ns
            recover_time = recover_event.duration_ns - check_weights_time - check_env_time
            load_base_model_time = 0
        else:
            recover_from_patch_event = get_sub_event(event, method='_recover_from_parameter_patch', event_name='all')
            load_base_model_event = get_sub_event(recover_from_patch_event, method='recover_model-', event_name='all',
                                                  starts_with=True)
            check_weights_event = get_sub_event(event, method='_recover_from_weight_update',
                                                event_name='_check_weights')

            check_weights_time = check_weights_event.duration_ns
            load_base_model_time = load_base_model_event.duration_ns
            recover_time = recover_from_patch_event.duration_ns - load_base_model_time
            load_time = recover_event.duration_ns - recover_from_patch_event.duration_ns - check_weights_time
            check_env_time = 0

        detailed_times = {
            'recover_base_model_time': load_base_model_time,
            'load_time': load_time,
            'recover_time': recover_time,
            'check_weights_time': check_weights_time,
            'check_env_time': check_env_time
        }
        result = detailed_times
    elif approach == PROVENANCE:
        u1_event = get_sub_event(event, event_name='recover-U_1', starts_with=True, check_event_only=True)
        if u1_event is not None:
            # if the train event is none then we have use case U1
            load_base_model_time = 0
            load_prov_info_time = 0
            training_time = 0
            recover_time = event.duration_ns
        else:
            recover_event = get_sub_event(event, method='recover_model-', event_name='all', starts_with=True)
            recover_base_model_event = get_sub_event(recover_event, method='recover_model',
                                                     event_name='recover_base_model')
            load_prov_info_event = get_sub_event(recover_event, method='recover_model', event_name='load_model_info')
            training_event = get_sub_event(recover_event, method='recover_model', event_name='train')

            load_base_model_time = recover_base_model_event.duration_ns
            load_prov_info_time = load_prov_info_event.duration_ns
            training_time = training_event.duration_ns
            recover_time = 0

        detailed_times = {
            'recover_base_model_time': load_base_model_time,
            'load_prov_info_time': load_prov_info_time,
            'training_time': training_time,
            'recover_time': recover_time,
        }
        result = detailed_times

    return result


def high_level_recover_times(server_meta, num_nodes=1):
    return _detailed_recover_times(server_meta, lambda x, y: x.duration_ns, num_nodes)


def detailed_recover_times(server_meta, num_nodes=1):
    return _detailed_recover_times(server_meta, _extract_detailed_recover_times, num_nodes)


def _detailed_recover_times(server_meta, extract_method, num_nodes=1):
    node_count = 0
    approach = server_meta[APPROACH]
    times = {}
    for e in server_meta[EVENTS]:
        if e.use_case == U_4:
            sub_events = e.children
            for sub_e in sub_events:
                _, use_case, _ = sub_e.event.split('-')

                if 'U_3' in use_case:
                    key = use_case
                    if "N" not in use_case:
                        key = F"N{node_count}-{use_case}"
                    times[key] = extract_method(sub_e, approach)
                    node_count += 1
                    node_count %= num_nodes
                else:
                    times[use_case] = extract_method(sub_e, approach)



    return times


def detailed_save_times(node_meta, server_meta, num_nodes=1):
    return _detailed_save_times(node_meta, server_meta, _extract_detailed_save_times, num_nodes)


def high_level_save_times(node_meta, server_meta, num_nodes=1):
    return _detailed_save_times(node_meta, server_meta, lambda x, y: x.duration_ns, num_nodes)


def _detailed_save_times(node_meta, server_meta, extract_method, num_nodes=1):
    times = {}

    approach = server_meta[APPROACH]

    for e in server_meta[EVENTS]:
        if e.use_case == U_1:
            times[U_1] = extract_method(e, approach)
        elif e.use_case == U_2:
            times[U_2] = extract_method(e, approach)
    u31_counter = 1
    u32_counter = 1
    node_counter = 0
    for e in node_meta[EVENTS]:
        if e.use_case and e.use_case.startswith(U_3_1):
            times[F"N{node_counter}-{e.use_case}"] = extract_method(e, approach)
            if node_counter == num_nodes:
                u31_counter += 1
        elif e.use_case and e.use_case.startswith(U_3_2):
            times[F"N{node_counter}-{e.use_case}"] = extract_method(e, approach)
            if node_counter == num_nodes:
                u32_counter += 1

        if e.use_case and e.use_case.startswith(U_3_1) or e.use_case.startswith(U_3_2):
            node_counter += 1
            node_counter %= num_nodes
    return times


def rearrange_u2(use_cases):
    use_cases.sort()
    # remove U_2
    u2 = use_cases.pop(1)
    num_cases = len(use_cases)
    new_u2_pos = int((num_cases - 1) / 2) + 1
    use_cases.insert(new_u2_pos, u2)
    return use_cases


def plot_time_one_model(save_times, save_path=None, ignore_use_cases=[], y_min_max=None):
    use_cases = rearrange_u2(list(save_times.keys()))

    for u in ignore_use_cases:
        use_cases.remove(u)

    plt.rc('font', size=12)
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    times = np.array([save_times[k] for k in use_cases]) * 10 ** -9
    ax.bar(use_cases, times)
    ax.set_ylabel('Time in seconds')
    ax.set_xlabel('Use case description')
    plt.xticks(rotation=45)
    if y_min_max:
        axes = plt.gca()
        axes.set_ylim(y_min_max)
    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
        fig.savefig(save_path + '.pdf', bbox_inches='tight')

    plt.show()


MODEL_PARAMETERS = np.array([3504872, 6624904, 11689512, 25557032, 60192808]) * 10 ** -6


def plot_detailed_times(plot_data, labels, x_labels, save_path=None, only_hpi_colors=False, y_min_max=None, size=None,
                        model_params=False, lgd_right=False, reorder_labels=False):
    plt.rc('font', size=40)
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    if size:
        fig.set_size_inches(size[0], size[1])
    else:
        fig.set_size_inches(10, 6)
    ax.set_ylabel('Time in seconds')
    ax.set_xlabel('Use case description')
    plt.xticks(rotation=60)
    bottom = np.zeros(plot_data[0].shape)
    pos = range(len(x_labels))

    for i, l in enumerate(labels):
        d = plot_data[i] * 10 ** -9
        plt.bar(pos, d, label=labels[i], bottom=bottom, color=colors[labels[i]])

        bottom += d

    plt.xticks(pos, x_labels)
    handles, labels = ax.get_legend_handles_labels()

    for i in range(len(labels)):
        if labels[i] == 'recover':
            labels[i] = 'recover model'

    if reorder_labels:
        handles = [handles[2], handles[1], handles[0], handles[3]]
        labels = [labels[2], labels[1], labels[0], labels[3]]

    if lgd_right:
        plt.legend(handles, labels, bbox_to_anchor=(1.04, 1), borderaxespad=0)
    else:
        plt.legend(handles, labels, bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left", mode="expand", borderaxespad=0,
                   ncol=2)

    if model_params:
        ax2 = ax.twinx()
        ax2.plot(x_labels, MODEL_PARAMETERS, color=HPI_RED, marker="o", markersize=10)
        ax2.set_ylabel("Number of parameters (* 10^-6)", color=HPI_RED)

    if y_min_max:
        axes = plt.gca()
        axes.set_ylim(y_min_max)

    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
        fig.savefig(save_path + '.pdf', bbox_inches='tight')

    plt.show()


def aggregate_fields(metas, aggregate, field_key):
    meta_0 = metas[0]
    combined = {
        MODEL: meta_0[MODEL],
        APPROACH: meta_0[APPROACH],
        SNAPSHOT_TYPE: meta_0[SNAPSHOT_TYPE],
        SNAPSHOT_DIST: meta_0[SNAPSHOT_DIST],
        RUN: meta_0[RUN]
    }

    total_cons = []
    for meta in metas:
        total_cons.append(meta[field_key])

    df = pd.DataFrame(total_cons)
    if aggregate == 'avg':
        aggregated_total_cons = dict(df.mean())
    elif aggregate == 'median':
        aggregated_total_cons = dict(df.median())
    else:
        raise NotImplementedError

    combined[field_key] = aggregated_total_cons

    return combined


def median_detailed_save_times(data):
    return median_detailed_times(data, 'detailed_save_times')


def median_detailed_times(data, key):
    all_detailed_times = [d[key] for d in data]
    use_cases = list(all_detailed_times[0].keys())
    result = {}
    for u in use_cases:
        tmp_u = [run[u] for run in all_detailed_times if u in run]
        df = pd.DataFrame(tmp_u)
        result[u] = dict(df.median())

    return result


def median_detailed_recover_times(data):
    return median_detailed_times(data, 'detailed_recover_times')


def combine_avg(dict_list):
    df = pd.DataFrame(dict_list)
    return dict(df.mean())


def flatten_dict(d, parent_key='', sep='_'):
    # form https://stackoverflow.com/questions/6027558/flatten-nested-dictionaries-compressing-keys
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def split_in_params_and_rest(storage_dict):
    other = 'other'
    parameters = 'parameters'
    update = 'update'

    result = {
        other: 0,
        parameters: 0
    }

    for k, v in storage_dict.items():
        if parameters in k or update in k:
            key = parameters
        else:
            key = other

        result[key] += int(v)

    return result


def plot_median_high_level_save_time(metas, save_path=None, ignore_use_cases=[], y_min_max=None):
    agg = aggregate_fields(metas, aggregate='median', field_key=HIGH_LEVEL_SAVE_TIMES)
    plot_time_one_model(agg[HIGH_LEVEL_SAVE_TIMES], save_path, ignore_use_cases, y_min_max)


def plot_median_high_level_recover_time(metas, save_path=None, ignore_use_cases=[], y_min_max=None):
    agg = aggregate_fields(metas, aggregate='median', field_key=HIGH_LEVEL_RECOVER_TIMES)
    plot_time_one_model(agg[HIGH_LEVEL_RECOVER_TIMES], save_path, ignore_use_cases, y_min_max)


# helper function to filer metadata
def filter_meta(to_filter, model=None, approach=None, snapshot_type=None, snapshot_dist=None, run=None):
    result = [f for f in to_filter]
    if model:
        result = [f for f in result if f[MODEL] == model]
    if approach:
        result = [f for f in result if f[APPROACH] == approach]
    if snapshot_type:
        result = [f for f in result if f[SNAPSHOT_TYPE] == snapshot_type]
    if snapshot_dist:
        result = [f for f in result if f[SNAPSHOT_DIST] == snapshot_dist]
    if run:
        result = [f for f in result if f[RUN] == run]

    return result


def split_in_dataset_and_rest(storage_dict):
    other = 'other'
    dataset = 'dataset'
    update = 'update'

    result = {
        other: 0,
        dataset: 0
    }

    for k, v in storage_dict.items():
        if dataset in k or update in k:
            key = dataset
        else:
            key = other

        result[key] += int(v)

    return result
