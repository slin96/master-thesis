import pandas as pd


def get_key_and_time(line):
    split_l = line.split(';')
    description = split_l[1]
    epoch = split_l[2]
    batch = split_l[3]
    time = int(split_l[4].split('-')[1])
    key = ','.join([description, epoch, batch])

    return key, time


def to_data_frame():
    data = {}
    with open('dummy-out.txt') as f:
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

    return pd.DataFrame.from_dict(data, orient='index', columns=['start', 'stop', 'diff'])


if __name__ == '__main__':
    data_frame = to_data_frame()
    print(data_frame)
    print('test')
