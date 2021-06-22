import os

RESNET_152 = 'resnet152'
RESNET_50 = 'resnet50'
RESNET_18 = 'resnet18'
GOOGLENET = 'googlenet'
MOBILENET = 'mobilenet'
U2_LOGS = './snapshot-training-logs/u2'
U3_FOOD_LOGS = './snapshot-training-logs/u3-1-food'
U2_OUTDOOR_LOGS = './snapshot-training-logs/u3-1-outdoor'
FILENAME_TEMPLATE = '{}-out.txt'
MODELS = [MOBILENET, GOOGLENET, RESNET_18, RESNET_50, RESNET_152]


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

    return training_times


def get_epoch_number(line):
    return int(line.split(';')[2].replace('epoch-', ''))


def get_time_ns(line):
    return int(line.split(';')[4].replace('time.time_ns-', ''))


if __name__ == '__main__':
    googlenet_u2_times = train_times('googlenet', U2_LOGS)
    googlenet_u3_times = train_times('googlenet', U3_FOOD_LOGS)
    print('test')
