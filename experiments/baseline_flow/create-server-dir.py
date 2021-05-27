import os
import shutil

FILE_PATH = os.path.dirname(os.path.realpath(__file__))
SERVER = os.path.join(FILE_PATH, 'server')
EXPERIMENTS = os.path.join(SERVER, 'experiments')
BASELINE_FLOW = os.path.join(EXPERIMENTS, 'baseline_flow')

if __name__ == '__main__':
    os.mkdir(SERVER)
    os.mkdir(EXPERIMENTS)
    os.mkdir(BASELINE_FLOW)

    files_to_copy = [
        (os.path.join(FILE_PATH, './server.py'), os.path.join(BASELINE_FLOW, './server.py')),
        (os.path.join(FILE_PATH, './shared.py'), os.path.join(BASELINE_FLOW, './node.py')),
    ]

    dirs_to_copy = [
        (os.path.join(FILE_PATH, '../models'), os.path.join(EXPERIMENTS, 'models'))
    ]

    for s, d in files_to_copy:
        shutil.copyfile(s, d)

    for s, d in dirs_to_copy:
        shutil.copytree(s, d)
