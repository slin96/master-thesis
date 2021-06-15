import collections
import json
import os
from os import listdir
from os.path import isfile

import pandas as pd

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

FLOAT_TEMPLATE = '{:.10f}'

APPROACH = 'approach'
SNAPSHOT_TYPE = 'snapshot_type'
SNAPSHOT_DIST = 'snapshot_dist'
RUN = 'run'

TOTAL_CONSUMPTIONS = 'total-consumptions'
CONSUMPTIONS = 'consumptions'


def use_case_ids(log_file, id2use_case=False):
    relevant_lines = []
    with open(log_file) as f:
        lines = f.readlines()
        search_for = 'recover-U'

        for l in lines:
            if search_for in l:
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


def create_pie_chart_sizes_and_labels(storage_distribution):
    labels = []
    sizes = []
    total_size = total_storage_consumption(storage_distribution)
    for k, v in storage_distribution.items():
        labels.append(k)
        if isinstance(v, dict):
            size = total_storage_consumption(v)
        else:
            size = v
        sizes.append(size / total_size * 100)

    return labels, sizes


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
        self.duration_ns = int(stop_json[TIME]) - int(start_json[TIME])
        self.duration_s = self.duration_ns * 10 ** -9

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
        time_str = FLOAT_TEMPLATE.format(self.duration_s)
        result = '{}: {}s \n'.format(self._event_name, time_str)
        for c in self.children:
            tabs = '\t' * level
            result += tabs + c._generate_representation(level + 1)
        return result


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


def calc_recover_times(server_logs):
    recover_times = {}

    for file in server_logs:
        meta, events = file
        times = {}
        for e in events:
            if e.use_case == U_4:
                sub_events = e.children
                for sub_e in sub_events:
                    _, use_case, _ = sub_e.event.split('-')
                    times[use_case] = sub_e.duration_s
        recover_times[meta[MODEL]] = times

    return recover_times


def aggregate_total_storage_consumption(metas, aggregate):
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
        total_cons.append(meta[TOTAL_CONSUMPTIONS])

    df = pd.DataFrame(total_cons)
    if aggregate == 'avg':
        aggregated_total_cons = dict(df.mean())
    elif aggregate == 'median':
        aggregated_total_cons = dict(df.median())
    else:
        raise NotImplementedError

    combined[TOTAL_CONSUMPTIONS] = aggregated_total_cons

    return combined


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

    result = {
        other: 0,
        parameters: 0
    }

    for k, v in storage_dict.items():
        if parameters in k:
            key = parameters
        else:
            key = other

        result[key] += int(v)

    return result


if __name__ == '__main__':
    combine_avg(None)
