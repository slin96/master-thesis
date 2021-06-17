import collections
import json
import os
from os import listdir
from os.path import isfile

import pandas as pd

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


def extract_times(valid_joined):
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

        _high_level_save_times = high_level_save_times(node_meta, server_meta)
        _high_level_recover_times = high_level_recover_times(server_meta)

        combined[HIGH_LEVEL_SAVE_TIMES] = _high_level_save_times
        combined[HIGH_LEVEL_RECOVER_TIMES] = _high_level_recover_times
        save_times.append(combined)

    return save_times


def high_level_save_times(node_meta, server_meta):
    times = {}
    for e in server_meta[EVENTS]:
        if e.use_case == U_1:
            times[U_1] = e.duration_s
        elif e.use_case == U_2:
            times[U_2] = e.duration_s
    u31_counter = 1
    u32_counter = 1
    for e in node_meta[EVENTS]:
        if e.use_case and e.use_case.startswith(U_3_1):
            times[e.use_case] = e.duration_s
            u31_counter += 1
        elif e.use_case and e.use_case.startswith(U_3_2):
            times[e.use_case] = e.duration_s
            u32_counter += 1
    return times


def high_level_recover_times(server_meta):
    times = {}
    for e in server_meta[EVENTS]:
        if e.use_case == U_4:
            sub_events = e.children
            for sub_e in sub_events:
                _, use_case, _ = sub_e.event.split('-')
                times[use_case] = sub_e.duration_s
    return times


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


if __name__ == '__main__':
    combine_avg(None)
