import pandas as pd


def get_key_and_time(line):
    split_l = line.split(';')
    description = split_l[1]
    epoch = split_l[2]
    batch = split_l[3]
    time = int(split_l[4].split('-')[1])
    key = ','.join([description, epoch, batch])

    return key, time


def to_data_frame(data_file):
    data = {}
    with open(data_file) as f:
        lines = f.readlines()
        for line in lines:
            if 'START' in line or 'STOP' in line:
                key, time = get_key_and_time(line)

                if key not in data:
                    assert 'START' in line
                    data[key] = [time]
                else:
                    assert 'STOP' in line
                    start = data[key][0]
                    stop = time
                    data[key].append(stop)
                    diff_ = stop - start
                    assert diff_ > 0
                    data[key].append(diff_)

        # remove timestamps with no stop (should just be one load data event per epoch)
        for key in list(data.keys()):
            if len(data[key]) < 3:
                del data[key]

    return pd.DataFrame.from_dict(data, orient='index', columns=['start', 'stop', 'diff'])


def extract_data(data_file):
    data_frame = to_data_frame(data_file)
    epochs = data_frame.filter(like='epoch,', axis=0)
    load_data = data_frame.filter(like='load_data', axis=0)
    to_device = data_frame.filter(like='to_device', axis=0)
    forward_path = data_frame.filter(like='forward_path', axis=0)
    backward_path = data_frame.filter(like='backward_path', axis=0)

    return epochs, load_data, to_device, forward_path, backward_path


if __name__ == '__main__':
    epochs, load_data, to_device, forward_path, backward_path = extract_data('dummy-out.txt')

    print(epochs)
    print(load_data)
    print(to_device)
    print(forward_path)
    print(backward_path)
