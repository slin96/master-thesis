import json

EVENT = 'event'


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
