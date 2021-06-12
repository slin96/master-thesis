import json

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


def use_case_ids(log_file):
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
        result[split[1]] = split[2]

    return result


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


if __name__ == '__main__':
    events = parse_events('/Users/nils/Studium/master-thesis/repo/tmp/res/server-update-out.txt')
    for e in events:
        print(e)
