import os

FILENAME_TEMPLATE = '{}-out.txt'


def train_times(model, log_dir):
    log_file = os.path.join(log_dir, FILENAME_TEMPLATE.format(model))
    with open(log_file) as f:
        lines = f.readlines()

    training_times = {}
    for line in lines:
        if line.startswith('START;epoch'):
            epoch_number = get_epoch_number(line)
            time_ns = get_time_ns(line)
            training_times[epoch_number] = {'start': time_ns}
        elif line.startswith('STOP;epoch'):
            epoch_number = get_epoch_number(line)
            time_ns = get_time_ns(line)
            training_times[epoch_number]['stop'] = time_ns
            training_times[epoch_number]['diff'] = time_ns - training_times[epoch_number]['start']

    for k in list(training_times.keys()):
        if 'diff' not in training_times[k]:
            del training_times[k]

    return training_times


def get_epoch_number(line):
    return int(line.split(';')[2].replace('epoch-', ''))


def get_time_ns(line):
    return int(line.split(';')[4].replace('time.time_ns-', ''))
